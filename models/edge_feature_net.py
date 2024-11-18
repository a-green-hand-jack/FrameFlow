import torch
from torch import nn
from torch import Tensor
from typing import List, Any
from models.utils import get_index_embedding, calc_distogram


class EdgeFeatureNet(nn.Module):
    """边特征网络,用于生成残基对之间的特征表示。

    该网络整合节点特征、相对位置信息、距离信息等,生成边级别的特征向量。

    Attributes:
        _cfg: 网络配置参数
        c_s: 输入节点特征维度
        c_p: 输出边特征维度
        feat_dim: 中间特征维度
        linear_s_p: 节点特征投影层
        linear_relpos: 相对位置编码投影层
        edge_embedder: 边特征嵌入网络
    """

    def __init__(self, module_cfg: Any) -> None:
        """初始化EdgeFeatureNet。

        Args:
            module_cfg: 配置对象,必须包含:
                - c_s: 输入节点特征维度
                - c_p: 输出边特征维度
                - feat_dim: 中间特征维度
                - num_bins: 距离直方图的bin数量
                - embed_chain: 是否嵌入链信息
                - embed_diffuse_mask: 是否嵌入扩散掩码
        """
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        # 特征投影层
        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        # 计算总的边特征维度
        total_edge_feats = (
            self.feat_dim * 3  # 节点特征和相对位置特征
            + self._cfg.num_bins * 2  # 两个距离直方图
        )
        if self._cfg.embed_chain:
            total_edge_feats += 1  # 链信息
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2  # 扩散掩码

        # 边特征嵌入网络
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, r: Tensor) -> Tensor:
        """计算相对位置编码。

        基于AlphaFold 2 Algorithm 4 & 5实现。

        Args:
            r: 位置索引张量, shape为[B, N]

        Returns:
            相对位置编码, shape为[B, N, N, feat_dim]

        Notes:
            - 计算每对残基之间的相对位置
            - 使用正弦位置编码
            - 通过线性层投影到目标维度
        """
        d = r[:, :, None] - r[:, None, :]  # [B, N, N]
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d: Tensor, num_batch: int, num_res: int) -> Tensor:
        """将1D特征扩展为2D特征并拼接。

        Args:
            feats_1d: 1D特征张量, shape为[B, N, C]
            num_batch: batch大小
            num_res: 残基数量

        Returns:
            拼接后的2D特征, shape为[B, N, N, 2*C]
        """
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res, num_res, -1])
        )

    def forward(
        self, s: Tensor, t: Tensor, sc_t: Tensor, p_mask: Tensor, diffuse_mask: Tensor
    ) -> Tensor:
        """前向传播函数。

        Args:
            s: 节点特征, shape为[B, N, c_s]
            t: 当前平移向量, shape为[B, N, 3]
            sc_t: 自条件平移向量, shape为[B, N, 3]
            p_mask: 边掩码, shape为[B, N, N]
            diffuse_mask: 扩散掩码, shape为[B, N]

        Returns:
            边特征张量, shape为[B, N, N, c_p]

        Notes:
            - B表示batch大小
            - N表示序列长度
            - 整合多种特征:节点特征、相对位置、距离分布等
        """
        num_batch, num_res, _ = s.shape

        # 节点特征投影和扩展
        p_i = self.linear_s_p(s)  # [B, N, feat_dim]
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # 计算相对位置特征
        r = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(r)

        # 计算距离直方图特征
        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins
        )
        sc_feats = calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins
        )

        # 整合所有特征
        all_edge_feats: List[Tensor] = [
            cross_node_feats,
            relpos_feats,
            dist_feats,
            sc_feats,
        ]
        if self._cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)

        # 生成最终的边特征
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)

        return edge_feats

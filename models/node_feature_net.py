# pyright: reportInvalidTypeForm=false
# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUnusedImport=false
# ruff: noqa: F401, E741, F841

import torch
from torch import nn
from torch import Tensor
from typing import List, Any
from models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):
    """节点特征网络,用于生成残基级别的特征表示。

    该网络将位置编码、时间编码和掩码信息整合成残基级别的特征向量。

    Attributes:
        _cfg: 网络配置参数
        c_s: 输出特征维度
        c_pos_emb: 位置编码维度
        c_timestep_emb: 时间步编码维度
        linear: 线性投影层,将拼接的特征映射到目标维度
    """

    def __init__(self, module_cfg: Any) -> None:
        """初始化NodeFeatureNet。

        Args:
            module_cfg: 配置对象,必须包含:
                - c_s: 输出特征维度
                - c_pos_emb: 位置编码维度
                - c_timestep_emb: 时间步编码维度
                - embed_chain: 是否嵌入链信息
        """
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb

        # 计算总的嵌入维度
        embed_size = (
            self._cfg.c_pos_emb  # 位置编码
            + self._cfg.c_timestep_emb * 2  # SO(3)和R3时间编码
            + 1  # 扩散掩码
        )
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb  # 链编码

        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps: Tensor, mask: Tensor) -> Tensor:
        """生成时间步的编码。

        Args:
            timesteps: 时间步张量, shape为[B, 1]
            mask: 残基掩码, shape为[B, N]

        Returns:
            时间编码张量, shape为[B, N, c_timestep_emb]

        Notes:
            - 使用正弦位置编码
            - 对每个残基位置重复相同的时间编码
            - 应用残基掩码
        """
        timestep_emb = get_time_embedding(
            timesteps[:, 0], self.c_timestep_emb, max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
        self,
        so3_t: Tensor,
        r3_t: Tensor,
        res_mask: Tensor,
        diffuse_mask: Tensor,
        pos: Tensor,
    ) -> Tensor:
        """前向传播函数。

        Args:
            so3_t: SO(3)空间的时间点, shape为[B, 1]
            r3_t: R3空间的时间点, shape为[B, 1]
            res_mask: 残基掩码, shape为[B, N]
            diffuse_mask: 扩散掩码, shape为[B, N]
            pos: 残基位置索引, shape为[B, N]

        Returns:
            节点特征张量, shape为[B, N, c_s]

        Notes:
            - B表示batch大小
            - N表示序列长度
            - c_s表示输出特征维度
        """
        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # 生成位置编码 [B, N, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # 拼接所有特征
        input_feats: List[Tensor] = [
            pos_emb,  # 位置编码
            diffuse_mask[..., None],  # 扩散掩码
            self.embed_t(so3_t, res_mask),  # SO(3)时间编码
            self.embed_t(r3_t, res_mask),  # R3时间编码
        ]

        # 通过线性层映射到目标维度
        return self.linear(torch.cat(input_feats, dim=-1))

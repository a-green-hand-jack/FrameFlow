import torch
from torch import nn
from typing import Dict, Any, Callable
from torch import Tensor

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from data import utils as du


class FlowModel(nn.Module):
    """蛋白质结构生成的流模型。
    
    该模型使用IPA(Invariant Point Attention)架构来生成蛋白质结构,主要包含:
    1. 节点特征网络 - 处理残基级别的特征
    2. 边特征网络 - 处理残基对之间的特征
    3. IPA主干网络 - 通过多个IPA块迭代更新结构
    
    Attributes:
        _model_conf: 模型配置参数
        _ipa_conf: IPA相关配置参数
        rigids_ang_to_nm: 角度到纳米的转换函数
        rigids_nm_to_ang: 纳米到角度的转换函数
        node_feature_net: 节点特征网络
        edge_feature_net: 边特征网络
        trunk: IPA主干网络,包含多个IPA块
    """

    def __init__(self, model_conf: Any) -> None:
        """初始化FlowModel。
        
        Args:
            model_conf: 包含模型配置的对象,必须包含:
                - ipa: IPA配置
                - node_features: 节点特征配置
                - edge_features: 边特征配置
                - edge_embed_size: 边嵌入维度
        """
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        
        # 定义单位转换函数
        self.rigids_ang_to_nm: Callable = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang: Callable = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE)
        
        # 初始化特征网络
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # 构建IPA主干网络
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            # IPA层
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            
            # Transformer层
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, 
                self._ipa_conf.seq_tfmr_num_layers, 
                enable_nested_tensor=False
            )
            
            # 后处理层
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            # 除最后一个块外,添加边特征更新
            if b < self._ipa_conf.num_blocks-1:
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, input_feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """前向传播函数。
        
        Args:
            input_feats: 输入特征字典,包含:
                - res_mask: 残基掩码 [B, N]
                - diffuse_mask: 扩散掩码 [B, N]
                - res_idx: 残基索引 [B, N]
                - so3_t: SO(3)时间点 [B, 1]
                - r3_t: R3时间点 [B, 1]
                - trans_t: 当前平移向量 [B, N, 3]
                - rotmats_t: 当前旋转矩阵 [B, N, 3, 3]
                - trans_sc: (可选)自条件平移向量 [B, N, 3]
                
        Returns:
            包含预测结果的字典:
                - pred_trans: 预测的平移向量 [B, N, 3]
                - pred_rotmats: 预测的旋转矩阵 [B, N, 3, 3]
                
        Notes:
            - B表示batch大小
            - N表示序列长度
        """
        # 提取输入特征
        node_mask = input_feats['res_mask']  # [B, N]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]  # [B, N, N]
        diffuse_mask = input_feats['diffuse_mask']  # [B, N]
        res_index = input_feats['res_idx']  # [B, N]
        so3_t = input_feats['so3_t']  # [B, 1]
        r3_t = input_feats['r3_t']  # [B, 1]
        trans_t = input_feats['trans_t']  # [B, N, 3]
        rotmats_t = input_feats['rotmats_t']  # [B, N, 3, 3]

        # 初始化节点和边的嵌入
        init_node_embed = self.node_feature_net(
            so3_t,
            r3_t,
            node_mask,
            diffuse_mask,
            res_index
        )
        
        # 处理自条件
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
            
        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # 初始化刚体变换
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # 主干网络处理
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)  # 转换到纳米尺度
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        
        # 迭代IPA块
        for b in range(self._ipa_conf.num_blocks):
            # IPA注意力
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            
            # Transformer处理
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, 
                src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            
            # 节点更新
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            
            # 刚体更新
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            
            # 边特征更新(除最后一个块)
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # 转换回角度尺度并提取结果
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
        }

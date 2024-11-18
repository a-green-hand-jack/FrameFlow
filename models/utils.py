import math
import torch
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from typing import Dict, Union, Optional
from data import utils as du


def calc_distogram(
    pos: Tensor, min_bin: float, max_bin: float, num_bins: int
) -> Tensor:
    """计算距离图(distogram)。

    Args:
        pos: 位置坐标张量, shape为[B, N, 3]
        min_bin: 最小距离bin
        max_bin: 最大距离bin
        num_bins: bin的数量

    Returns:
        距离图张量, shape为[B, N, N, num_bins]

    Notes:
        - 计算每对残基之间的欧氏距离
        - 将距离映射到预定义的bin中
        - 返回one-hot编码的距离分布
    """
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[
        ..., None
    ]

    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(
    indices: Tensor, embed_size: int, max_len: int = 2056
) -> Tensor:
    """创建正弦/余弦位置编码。

    Args:
        indices: 整数类型的索引张量, shape为[..., N]
        embed_size: 编码的维度
        max_len: 最大序列长度,默认2056

    Returns:
        位置编码张量, shape为[..., N, embed_size]

    Notes:
        - 使用正弦和余弦函数生成位置编码
        - 编码维度平均分配给正弦和余弦部分
        - 使用不同频率的正弦波捕获不同尺度的位置信息
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(
    timesteps: Tensor, embedding_dim: int, max_positions: int = 2000
) -> Tensor:
    """生成时间步的嵌入表示。

    Args:
        timesteps: 时间步张量, shape为[B]
        embedding_dim: 嵌入维度
        max_positions: 最大位置数,默认2000

    Returns:
        时间编码张量, shape为[B, embedding_dim]

    Notes:
        - 基于diffusion模型的时间编码方法
        - 使用对数空间的正弦/余弦编码
        - 如果维度为奇数,在末尾补零
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # 补零
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def t_stratified_loss(
    batch_t: Union[Tensor, np.ndarray],
    batch_loss: Union[Tensor, np.ndarray],
    num_bins: int = 4,
    loss_name: Optional[str] = None,
) -> Dict[str, float]:
    """将损失按时间步分层统计。

    Args:
        batch_t: 时间步张量/数组, shape为[B, ...]
        batch_loss: 损失张量/数组, shape为[B, ...]
        num_bins: 时间bin的数量,默认4
        loss_name: 损失名称,用于结果字典的键

    Returns:
        包含各时间区间平均损失的字典

    Notes:
        - 将时间步划分为等间隔的bin
        - 计算每个bin内的平均损失
        - 返回格式为{'loss t=[t1,t2)': mean_loss, ...}
    """
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()

    # 创建时间bin
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1

    # 计算每个bin的统计量
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)

    # 生成结果字典
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss

    return stratified_losses

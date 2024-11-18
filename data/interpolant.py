import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from scipy.spatial.transform import Rotation
from torch import Tensor, autograd

from data import all_atom, so3_utils
from data import utils as du
from motif_scaffolding import twisting


def _centered_gaussian(num_batch: int, num_res: int, device: torch.device) -> Tensor:
    """生成中心化的高斯噪声。
    
    Args:
        num_batch: batch大小
        num_res: 残基数量 
        device: 计算设备

    Returns:
        shape为(num_batch, num_res, 3)的中心化高斯噪声张量
    """
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch: int, num_res: int, device: torch.device) -> Tensor:
    """生成均匀分布的SO(3)旋转矩阵。
    
    Args:
        num_batch: batch大小
        num_res: 残基数量
        device: 计算设备
    
    Returns:
        shape为(num_batch, num_res, 3, 3)的随机旋转矩阵张量
    """
    return torch.tensor(
        Rotation.random(num_batch * num_res).as_matrix(),
        device=device, 
        dtype=torch.float32
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t: Tensor, trans_1: Tensor, diffuse_mask: Tensor) -> Tensor:
    """根据扩散掩码混合两个平移向量。
    
    Args:
        trans_t: 当前时刻的平移向量
        trans_1: 目标平移向量
        diffuse_mask: 扩散掩码
        
    Returns:
        混合后的平移向量
    """
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t: Tensor, rotmats_1: Tensor, diffuse_mask: Tensor) -> Tensor:
    """根据扩散掩码混合两个旋转矩阵。
    
    Args:
        rotmats_t: 当前时刻的旋转矩阵, shape为[B, N, 3, 3]
        rotmats_1: 目标旋转矩阵, shape为[B, N, 3, 3]
        diffuse_mask: 扩散掩码, shape为[B, N], 值为0或1
        
    Returns:
        混合后的旋转矩阵, shape为[B, N, 3, 3]。
        对于mask=1的位置使用rotmats_t,
        对于mask=0的位置使用rotmats_1
    """
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (1 - diffuse_mask[..., None, None])


class Interpolant:
    """处理蛋白质结构扩散过程中的插值和采样。
    
    该类实现了蛋白质结构在SO(3)旋转空间和R3平移空间的扩散过程,包括:
    1. 结构的噪声污染
    2. 扩散轨迹的采样
    3. 运动支架(motif scaffolding)的引导
    
    Attributes:
        _cfg: 主配置参数
        _rots_cfg: 旋转相关配置
        _trans_cfg: 平移相关配置
        _sample_cfg: 采样相关配置
        _igso3: SO(3)空间上的各向同性高斯分布采样器
        _device: 计算设备
    """
    
    def __init__(self, cfg: Any) -> None:
        """初始化Interpolant实例。
        
        Args:
            cfg: 包含所有必要配置参数的对象,必须包含rots、trans和sampling子配置
        """
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        
    @property
    def igso3(self) -> Any:
        """获取或初始化SO(3)采样器。
        
        Returns:
            初始化好的SO(3)采样器实例
        """
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3
        
    def set_device(self, device: torch.device) -> None:
        """设置计算设备。
        
        Args:
            device: PyTorch计算设备
        """
        self._device = device
        
    def sample_t(self, num_batch: int) -> Tensor:
        """采样时间点。
        
        Args:
            num_batch: batch大小
            
        Returns:
            shape为[num_batch]的时间点张量,范围在[min_t, 1]之间
        """
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(
        self, 
        trans_1: Tensor,
        t: Tensor, 
        res_mask: Tensor,
        diffuse_mask: Tensor
    ) -> Tensor:
        """对平移向量进行噪声污染。
        
        Args:
            trans_1: 目标平移向量, shape为[B, N, 3]
            t: 时间点, shape为[B, 1]
            res_mask: 残基掩码, shape为[B, N]
            diffuse_mask: 扩散掩码, shape为[B, N]
            
        Returns:
            污染后的平移向量, shape为[B, N, 3]
        """
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]

    def _corrupt_rotmats(
        self,
        rotmats_1: Tensor,
        t: Tensor,
        res_mask: Tensor,
        diffuse_mask: Tensor
    ) -> Tensor:
        """对旋转矩阵进行噪声污染。
        
        Args:
            rotmats_1: 目标旋转矩阵, shape为[B, N, 3, 3]
            t: 时间点, shape为[B, 1]
            res_mask: 残基掩码, shape为[B, N]
            diffuse_mask: 扩散掩码, shape为[B, N]
            
        Returns:
            污染后的旋转矩阵, shape为[B, N, 3, 3]
        """
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_batch * num_res).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        
        # 处理掩码
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (1 - res_mask[..., None, None])
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def corrupt_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """对输入batch进行噪声污染。
        
        对蛋白质结构的旋转矩阵和平移向量添加噪声,实现扩散过程。
        
        Args:
            batch: 包含以下键的字典:
                - trans_1: 目标平移向量 [B, N, 3]
                - rotmats_1: 目标旋转矩阵 [B, N, 3, 3]
                - res_mask: 残基掩码 [B, N]
                - diffuse_mask: 扩散掩码 [B, N]
                
        Returns:
            污染后的batch字典,额外包含:
                - trans_t: 污染后的平移向量
                - rotmats_t: 污染后的旋转矩阵
                - so3_t: SO(3)空间的时间点
                - r3_t: R3空间的时间点
                
        Raises:
            ValueError: 如果生成的张量包含NaN值
        """
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch["trans_1"]  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch["rotmats_1"]

        # [B, N]
        res_mask = batch["res_mask"]
        diffuse_mask = batch["diffuse_mask"]
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        so3_t = t
        r3_t = t
        noisy_batch["so3_t"] = so3_t
        noisy_batch["r3_t"] = r3_t
        
        # 污染平移向量
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during corruption")
        noisy_batch["trans_t"] = trans_t
        
        # 污染旋转矩阵
        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        noisy_batch["rotmats_t"] = rotmats_t
        
        return noisy_batch

    def rot_sample_kappa(self, t: Tensor) -> Tensor:
        """计算旋转采样的kappa参数。
        
        Args:
            t: 时间点张量
            
        Returns:
            根据采样调度计算的kappa值
            
        Raises:
            ValueError: 如果采样调度方式无效
        """
        if self._rots_cfg.sample_schedule == "exp":
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {self._rots_cfg.sample_schedule}")

    def _trans_vector_field(self, t: Tensor, trans_1: Tensor, trans_t: Tensor) -> Tensor:
        """计算平移向量场。
        
        Args:
            t: 时间点
            trans_1: 目标平移向量
            trans_t: 当前时刻平移向量
            
        Returns:
            平移向量场
        """
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t: float, t: Tensor, trans_1: Tensor, trans_t: Tensor) -> Tensor:
        """执行平移的欧拉步进。
        
        Args:
            d_t: 时间步长
            t: 当前时间点
            trans_1: 目标平移向量
            trans_t: 当前时刻平移向量
            
        Returns:
            更新后的平移向量
            
        Raises:
            AssertionError: 如果d_t不为正数
        """
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t: float, t: Tensor, rotmats_1: Tensor, rotmats_t: Tensor) -> Tensor:
        """执行旋转的欧拉步进。
        
        Args:
            d_t: 时间步长
            t: 当前时间点
            rotmats_1: 目标旋转矩阵
            rotmats_t: 当前时刻旋转矩阵
            
        Returns:
            更新后的旋转矩阵
            
        Raises:
            ValueError: 如果采样调度方式无效
        """
        if self._rots_cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(f"Unknown sample schedule {self._rots_cfg.sample_schedule}")
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def sample(
        self,
        num_batch: int,
        num_res: int,
        model: Any,
        num_timesteps: Optional[int] = None,
        trans_potential: Optional[Any] = None,
        trans_0: Optional[Tensor] = None,
        rotmats_0: Optional[Tensor] = None,
        trans_1: Optional[Tensor] = None,
        rotmats_1: Optional[Tensor] = None,
        diffuse_mask: Optional[Tensor] = None,
        chain_idx: Optional[Tensor] = None,
        res_idx: Optional[Tensor] = None,
        verbose: bool = False
    ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor]]]:
        """采样蛋白质结构的扩散轨迹。
        
        该方法实现了从噪声分布开始,通过反向扩散过程生成蛋白质结构。过程包括:
        1. 初始化结构状态
        2. 执行反向扩散步骤
        3. 可选的运动支架引导
        4. 轨迹记录和后处理
        
        Args:
            num_batch: batch大小
            num_res: 残基数量
            model: 用于预测的扩散模型
            num_timesteps: 扩散时间步数,若为None则使用配置中的默认值
            trans_potential: 可选的平移势能函数
            trans_0: 初始平移向量,若为None则随机采样
            rotmats_0: 初始旋转矩阵,若为None则随机采样
            trans_1: 目标平移向量,用于运动支架
            rotmats_1: 目标旋转矩阵,用于运动支架
            diffuse_mask: 扩散掩码,指示哪些残基需要扩散
            chain_idx: 链索引
            res_idx: 残基索引
            verbose: 是否打印详细信息
            
        Returns:
            包含三个元素的元组:
            - atom37_traj: 原子级别的完整轨迹
            - clean_atom37_traj: 去噪后的原子级别轨迹
            - clean_traj: 去噪后的(trans, rot)轨迹
            
        Raises:
            ValueError: 当trans_1为None但trans_cfg.corrupt为False时
                       或当rotmats_1为None但rots_cfg.corrupt为False时
        """
        # 初始化残基掩码
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # 设置初始先验样本
        if trans_0 is None:
            trans_0 = (
                _centered_gaussian(num_batch, num_res, self._device)
                * du.NM_TO_ANG_SCALE
            )
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(num_res, device=self._device, dtype=torch.float32)[
                None
            ].repeat(num_batch, 1)
            
        # 初始化batch字典
        batch = {"res_mask": res_mask, "diffuse_mask": res_mask, "res_idx": res_idx}

        # 处理运动支架相关设置
        motif_scaffolding = False
        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool().squeeze(0)
        else:
            motif_mask = None
            
        # 处理非扭转的运动支架
        if motif_scaffolding and not self._cfg.twisting.use:  
            diffuse_mask = diffuse_mask.expand(num_batch, -1)  
            batch["diffuse_mask"] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            if torch.isnan(trans_0).any():
                raise ValueError("NaN detected in trans_0")

        # 初始化轨迹记录
        logs_traj = defaultdict(list)
        
        # 处理带扭转的运动支架
        if motif_scaffolding and self._cfg.twisting.use:
            assert trans_1.shape[0] == 1  # 假设只有一个motif
            motif_locations = torch.nonzero(motif_mask).squeeze().tolist()
            true_motif_locations, motif_segments_length = (
                twisting.find_ranges_and_lengths(motif_locations)
            )

            # Marginalise both rotation and motif location
            assert len(motif_mask.shape) == 1
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]
            num_res = trans_1.shape[-2]
            with torch.inference_mode(False):
                motif_locations = (
                    true_motif_locations if self._cfg.twisting.motif_loc else None
                )
                F, motif_locations = twisting.motif_offsets_and_rots_vec_F(
                    num_res,
                    motif_segments_length,
                    motif_locations=motif_locations,
                    num_rots=self._cfg.twisting.num_rots,
                    align=self._cfg.twisting.align,
                    scale=self._cfg.twisting.scale_rots,
                    trans_motif=trans_motif,
                    R_motif=R_motif,
                    max_offsets=self._cfg.twisting.max_offsets,
                    device=self._device,
                    dtype=torch.float64,
                    return_rots=False,
                )

        # 扩展motif掩码到batch维度
        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # 设置时间步数
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        # 初始化轨迹列表
        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        
        # 主循环:执行反向扩散步骤
        for i, t_2 in enumerate(ts[1:]):
            if verbose:
                print(f"{i=}, t={t_1.item():.2f}")
                print(
                    torch.cuda.mem_get_info(trans_0.device),
                    torch.cuda.memory_allocated(trans_0.device),
                )
                
            # 准备当前步骤的输入
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if self._trans_cfg.corrupt:
                batch["trans_t"] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError("Must provide trans_1 if not corrupting.")
                batch["trans_t"] = trans_1
            if self._rots_cfg.corrupt:
                batch["rotmats_t"] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError("Must provide rotmats_1 if not corrupting.")
                batch["rotmats_t"] = rotmats_1
            batch["t"] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch["so3_t"] = batch["t"]
            batch["r3_t"] = batch["t"]
            d_t = t_2 - t_1

            # 判断是否使用扭转引导
            use_twisting = (
                motif_scaffolding
                and self._cfg.twisting.use
                and t_1 >= self._cfg.twisting.t_min
            )

            # 执行模型推理和引导
            if use_twisting:  
                with torch.inference_mode(False):
                    batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                    model_out = model(batch)
                    t = batch["r3_t"]  # TODO: different time for SO3?
                    trans_t_1, rotmats_t_1, logs_traj = self.guidance(
                        trans_t_1,
                        rotmats_t_1,
                        model_out,
                        motif_mask,
                        R_motif,
                        trans_motif,
                        Log_delta_R,
                        delta_x,
                        t,
                        d_t,
                        logs_traj,
                    )
            else:
                with torch.no_grad():
                    model_out = model(batch)

            # 处理模型输出
            pred_trans_1 = model_out["pred_trans"]
            pred_rotmats_1 = model_out["pred_rotmats"]
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            
            # 自条件设置
            if self._cfg.self_condition:
                if motif_scaffolding:
                    batch["trans_sc"] = pred_trans_1 * diffuse_mask[
                        ..., None
                    ] + trans_1 * (1 - diffuse_mask[..., None])
                else:
                    batch["trans_sc"] = pred_trans_1

            # 执行反向步骤
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            if trans_potential is not None:
                with torch.inference_mode(False):
                    grad_pred_trans_1 = (
                        pred_trans_1.clone().detach().requires_grad_(True)
                    )
                    pred_trans_potential = autograd.grad(
                        outputs=trans_potential(grad_pred_trans_1),
                        inputs=grad_pred_trans_1,
                    )[0]
                if self._trans_cfg.potential_t_scaling:
                    trans_t_2 -= t_1 / (1 - t_1) * pred_trans_potential * d_t
                else:
                    trans_t_2 -= pred_trans_potential * d_t
                    
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            
            # 处理运动支架的掩码
            if motif_scaffolding and not self._cfg.twisting.use:
                trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # 最后一步处理
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        if self._trans_cfg.corrupt:
            batch["trans_t"] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError("Must provide trans_1 if not corrupting.")
            batch["trans_t"] = trans_1
        if self._rots_cfg.corrupt:
            batch["rotmats_t"] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError("Must provide rotmats_1 if not corrupting.")
            batch["rotmats_t"] = rotmats_1
        batch["t"] = torch.ones((num_batch, 1), device=self._device) * t_1
        
        # 最后一次模型推理
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out["pred_trans"]
        pred_rotmats_1 = model_out["pred_rotmats"]
        clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # 转换轨迹到atom37格式
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        
        return atom37_traj, clean_atom37_traj, clean_traj

    def guidance(
        self,
        trans_t: Tensor,
        rotmats_t: Tensor,
        model_out: Dict[str, Tensor],
        motif_mask: Tensor,
        R_motif: Tensor,
        trans_motif: Tensor,
        Log_delta_R: Tensor,
        delta_x: Tensor,
        t: Tensor,
        d_t: float,
        logs_traj: Dict[str, List[Any]]
    ) -> Tuple[Tensor, Tensor, Dict[str, List[Any]]]:
        """实现运动支架(motif scaffolding)的引导过程。
        
        该方法通过以下步骤实现motif引导:
        1. 提取motif区域的预测结果
        2. 计算motif旋转的边缘化
        3. 估计条件概率p(motif|predicted_motif)
        4. 根据时间点选择合适的缩放方式
        5. 更新结构状态
        
        Args:
            trans_t: 当前时刻的平移向量, shape为[B, N, 3]
            rotmats_t: 当前时刻的旋转矩阵, shape为[B, N, 3, 3]
            model_out: 模型输出字典,包含预测的平移和旋转
            motif_mask: motif区域的掩码, shape为[B, N]
            R_motif: motif的目标旋转矩阵, shape为[1, M, 3, 3]
            trans_motif: motif的目标平移向量, shape为[1, M, 3]
            Log_delta_R: 旋转扰动的对数, 用于梯度计算
            delta_x: 平移扰动, 用于梯度计算
            t: 当前时间点, shape为[B, 1]
            d_t: 时间步长
            logs_traj: 用于记录轨迹信息的字典
            
        Returns:
            包含三个元素的元组:
            - 更新后的平移向量, shape为[B, N, 3]
            - 更新后的旋转矩阵, shape为[B, N, 3, 3]
            - 更新后的轨迹日志字典
            
        Notes:
            - B表示batch大小
            - N表示序列长度
            - M表示motif残基数量
            - 该方法在扩散过程中通过motif引导来改善采样质量
        """
        # 提取motif区域的预测结果
        motif_mask = motif_mask.clone()
        trans_pred = model_out["pred_trans"][:, motif_mask]  # [B, motif_res, 3]
        R_pred = model_out["pred_rotmats"][:, motif_mask]  # [B, motif_res, 3, 3]

        # 计算motif旋转的边缘化
        F = twisting.motif_rots_vec_F(
            trans_motif,
            R_motif,
            self._cfg.twisting.num_rots,
            align=self._cfg.twisting.align,
            scale=self._cfg.twisting.scale_rots,
            device=self._device,
            dtype=torch.float32
        )

        # 估计p(motif|predicted_motif)的梯度
        grad_Log_delta_R, grad_x_log_p_motif, logs = twisting.grad_log_lik_approx(
            R_pred,
            trans_pred,
            R_motif,
            trans_motif,
            Log_delta_R,
            delta_x,
            None,
            None,
            None,
            F,
            twist_potential_rot=self._cfg.twisting.potential_rot,
            twist_potential_trans=self._cfg.twisting.potential_trans,
        )

        with torch.no_grad():
            # 根据时间点选择缩放方式
            t_trans = t
            t_so3 = t
            if self._cfg.twisting.scale_w_t == "ot":
                # 最优传输缩放
                var_trans = ((1 - t_trans) / t_trans)[:, None]
                var_rot = ((1 - t_so3) / t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == "linear":
                # 线性缩放
                var_trans = (1 - t)[:, None]
                var_rot = (1 - t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == "constant":
                # 常数缩放
                num_batch = trans_pred.shape[0]
                var_trans = torch.ones((num_batch, 1, 1)).to(R_pred.device)
                var_rot = torch.ones((num_batch, 1, 1, 1)).to(R_pred.device)
            
            # 添加观测噪声
            var_trans = var_trans + self._cfg.twisting.obs_noise**2
            var_rot = var_rot + self._cfg.twisting.obs_noise**2

            # 计算最终的缩放因子
            trans_scale_t = self._cfg.twisting.scale / var_trans
            rot_scale_t = self._cfg.twisting.scale / var_rot

            # 更新结构状态
            trans_t, rotmats_t = twisting.step(
                trans_t,
                rotmats_t,
                grad_x_log_p_motif,
                grad_Log_delta_R,
                d_t,
                trans_scale_t,
                rot_scale_t,
                self._cfg.twisting.update_trans,
                self._cfg.twisting.update_rot,
            )

        # 清理内存以防止泄漏
        del grad_Log_delta_R
        del grad_x_log_p_motif
        del Log_delta_R
        del delta_x
        for key, value in model_out.items():
            model_out[key] = value.detach().requires_grad_(False)

        return trans_t, rotmats_t, logs_traj

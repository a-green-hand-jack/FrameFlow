from typing import Any, Dict, Optional
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor

from analysis import metrics
from analysis import utils as au
from models.flow_model import FlowModel
from models import utils as mu
from data.interpolant import Interpolant
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu


class FlowModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            "train/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_time = time.time()

    def model_step(
        self, 
        noisy_batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """执行模型的一个训练步骤,计算各种损失。
        
        该方法实现了三个主要的损失计算:
        1. 骨架原子损失(bb_atom_loss) - 评估预测的原子位置准确性
        2. 平移损失(trans_loss) - 评估预测的平移向量准确性
        3. 旋转向量场损失(rots_vf_loss) - 评估预测的旋转变换准确性
        
        Args:
            noisy_batch: 包含噪声数据的batch字典,必须包含:
                - res_mask: 残基掩码 [B, N]
                - diffuse_mask: 扩散掩码 [B, N]
                - trans_1: 目标平移 [B, N, 3]
                - rotmats_1: 目标旋转 [B, N, 3, 3]
                - rotmats_t: 当前旋转 [B, N, 3, 3]
                - r3_t: R3时间点 [B, 1]
                - so3_t: SO3时间点 [B, 1]
                
        Returns:
            Dict[str, Tensor]: 包含各种损失的字典:
                - bb_atom_loss: 骨架原子损失 [B]
                - trans_loss: 平移损失 [B]
                - rots_vf_loss: 旋转向量场损失 [B]
                - se3_vf_loss: 总损失 [B]
                
        Raises:
            ValueError: 当遇到空batch或计算过程中出现NaN值时
            
        Notes:
            - B表示batch大小
            - N表示序列长度
            - 所有损失都会根据diffuse_mask进行掩码处理
            - 使用时间相关的缩放因子来归一化损失
        """
        # 获取训练配置
        training_cfg = self._exp_cfg.training
        
        # 计算有效的loss掩码 [B, N]
        loss_mask = noisy_batch["res_mask"] * noisy_batch["diffuse_mask"]
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError("Empty batch encountered")
        num_batch, num_res = loss_mask.shape

        # 1. 准备真实标签
        # 获取目标平移和旋转 [B, N, 3] 和 [B, N, 3, 3]
        gt_trans_1 = noisy_batch["trans_1"]
        gt_rotmats_1 = noisy_batch["rotmats_1"]
        rotmats_t = noisy_batch["rotmats_t"]
        
        # 计算真实的旋转向量场 [B, N, 3]
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, 
            gt_rotmats_1.type(torch.float32)
        )
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError("NaN encountered in gt_rot_vf")
            
        # 计算真实的骨架原子位置 [B, N, 3, 3]
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # 2. 计算时间归一化因子
        # 获取R3和SO3空间的时间点 [B, 1]
        r3_t = noisy_batch["r3_t"]
        so3_t = noisy_batch["so3_t"]
        
        # 计算归一化缩放因子 [B, 1]
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], 
            torch.tensor(training_cfg.t_normalize_clip)
        )
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], 
            torch.tensor(training_cfg.t_normalize_clip)
        )

        # 3. 获取模型预测
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output["pred_trans"]  # [B, N, 3]
        pred_rotmats_1 = model_output["pred_rotmats"]  # [B, N, 3, 3]
        
        # 计算预测的旋转向量场 [B, N, 3]
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError("NaN encountered in pred_rots_vf")

        # 4. 计算骨架原子损失
        # 获取预测的骨架原子位置 [B, N, 3, 3]
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        
        # 应用时间归一化和缩放
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        
        # 计算MSE损失 [B]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3),
        ) / loss_denom

        # 5. 计算平移损失
        # 计算归一化的平移误差 [B, N, 3]
        trans_error = (
            (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        )
        
        # 计算加权MSE损失 [B]
        trans_loss = (
            training_cfg.translation_loss_weight * torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )
        trans_loss = torch.clamp(trans_loss, max=5)  # 限制最大损失

        # 6. 计算旋转向量场损失
        # 计算归一化的旋转误差 [B, N, 3]
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        
        # 计算MSE损失 [B]
        rots_vf_loss = torch.sum(
            rots_vf_error**2 * loss_mask[..., None], dim=(-1, -2)
        ) / loss_denom

        # 7. 返回所有损失
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": bb_atom_loss + trans_loss + rots_vf_loss,
        }

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """执行验证步骤。

        Args:
            batch: 输入batch字典,必须包含:
                - res_mask: 残基掩码 [B, N]
                - diffuse_mask: 扩散掩码 [B, N]
                - csv_idx: 样本索引
                - trans_1: 目标平移 [B, N, 3]
                - rotmats_1: 目标旋转 [B, N, 3, 3]
                - chain_idx: 链索引
                - res_idx: 残基索引
            batch_idx: batch的索引

        Notes:
            - 生成蛋白质结构样本
            - 保存样本到PDB文件
            - 计算评估指标
            - 记录到wandb(如果启用)
        """
        res_mask = batch["res_mask"]
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch["diffuse_mask"]
        csv_idx = batch["csv_idx"]

        # 采样轨迹
        atom37_traj, _, _ = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch["trans_1"],
            rotmats_1=batch["rotmats_1"],
            diffuse_mask=diffuse_mask,
            chain_idx=batch["chain_idx"],
            res_idx=batch["res_idx"],
        )
        samples = atom37_traj[-1].numpy()

        # 计算每个样本的指标
        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f"sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            # 保存样本到PDB文件
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos, os.path.join(sample_dir, "sample.pdb"), no_indexing=True
            )

            # 记录到wandb
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            # 计算评估指标
            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order["CA"]
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key="valid/samples",
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples,
            )
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f"valid/{metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True,
    ):
        if sync_dist and rank_zero_only:
            raise ValueError("Unable to sync dist when rank_zero_only=True")
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

    def training_step(self, batch: Dict[str, Tensor], stage: int) -> Tensor:
        """执行训练步骤。

        Args:
            batch: 输入batch字典
            stage: 训练阶段

        Returns:
            训练损失标量

        Notes:
            - 对输入数据添加噪声
            - 执行模型前向传播
            - 计算并记录各种损失
            - 记录训练统计信息
        """
        step_start_time = time.time()
        self.interpolant.set_device(batch["res_mask"].device)

        # 添加噪声
        noisy_batch = self.interpolant.corrupt_batch(batch)

        # 自条件采样
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch["trans_sc"] = model_sc["pred_trans"] * noisy_batch["diffuse_mask"][..., None] + noisy_batch["trans_1"] * (1 - noisy_batch["diffuse_mask"][..., None])

        # 执行模型步骤
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses["trans_loss"].shape[0]
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}

        # 记录损失
        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # 记录时间步统计
        so3_t = torch.squeeze(noisy_batch["so3_t"])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False,
            batch_size=num_batch,
        )
        r3_t = torch.squeeze(noisy_batch["r3_t"])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False,
            batch_size=num_batch,
        )

        # 记录分层损失
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == "rots_vf_loss":
                batch_t = so3_t
            else:
                batch_t = r3_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name
            )
            for k, v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # 记录训练统计信息
        scaffold_percent = torch.mean(batch["diffuse_mask"].float()).item()
        self._log_scalar(
            "train/scaffolding_percent",
            scaffold_percent,
            prog_bar=False,
            batch_size=num_batch,
        )
        motif_mask = 1 - batch["diffuse_mask"].float()
        num_motif_res = torch.sum(motif_mask, dim=-1)
        self._log_scalar(
            "train/motif_size",
            torch.mean(num_motif_res).item(),
            prog_bar=False,
            batch_size=num_batch,
        )
        self._log_scalar(
            "train/length",
            batch["res_mask"].shape[1],
            prog_bar=False,
            batch_size=num_batch,
        )
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)

        # 记录训练速度
        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", num_batch / step_time)

        # 返回总损失
        train_loss = total_losses["se3_vf_loss"]
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(), **self._exp_cfg.optimizer
        )

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: Optional[int] = None
    ) -> None:
        """执行预测步骤。

        Args:
            batch: 输入batch字典,必须包含:
                - sample_id: 样本ID
                对于motif-scaffolding还需要:
                    - target: 目标名称
                    - trans_1: 目标平移
                    - rotmats_1: 目标旋转
                    - diffuse_mask: 扩散掩码
                对于无条件生成需要:
                    - num_res: 残基数量
            batch_idx: batch索引(未使用)

        Notes:
            - 支持motif-scaffolding和无条件生成两种模式
            - 生成蛋白质结构轨迹
            - 保存结果到文件
        """
        del batch_idx  # 未使用
        device = f"cuda:{torch.cuda.current_device()}"
        interpolant = Interpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        # 处理样本ID
        sample_ids = batch["sample_id"].squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        # 处理motif-scaffolding模式
        if "diffuse_mask" in batch:
            target = batch["target"][0]
            trans_1 = batch["trans_1"]
            rotmats_1 = batch["rotmats_1"]
            diffuse_mask = batch["diffuse_mask"]
            true_bb_pos = all_atom.atom37_from_trans_rot(
                trans_1, rotmats_1, 1 - diffuse_mask
            )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            _, sample_length, _ = trans_1.shape
            sample_dirs = [
                os.path.join(self.inference_dir, target, f"sample_{str(sample_id)}")
                for sample_id in sample_ids
            ]
        # 处理无条件生成模式
        else:
            sample_length = batch["num_res"].item()
            true_bb_pos = None
            sample_dirs = [
                os.path.join(
                    self.inference_dir,
                    f"length_{sample_length}",
                    f"sample_{str(sample_id)}",
                )
                for sample_id in sample_ids
            ]
            trans_1 = rotmats_1 = diffuse_mask = None
            diffuse_mask = torch.ones(1, sample_length, device=device)

        # 采样batch
        atom37_traj, model_traj, _ = interpolant.sample(
            num_batch,
            sample_length,
            self.model,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            diffuse_mask=diffuse_mask,
        )

        # 处理轨迹数据
        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))

        # 保存结果
        for i in range(num_batch):
            sample_dir = sample_dirs[i]
            bb_traj = bb_trajs[i]
            os.makedirs(sample_dir, exist_ok=True)

            # 处理氨基酸类型
            if "aatype" in batch:
                aatype = du.to_numpy(batch["aatype"].long())[0]
            else:
                aatype = np.zeros(sample_length, dtype=int)

            # 保存轨迹
            _ = eu.save_traj(
                bb_traj[-1],
                bb_traj,
                np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                du.to_numpy(diffuse_mask)[0],
                output_dir=sample_dir,
                aatype=aatype,
            )

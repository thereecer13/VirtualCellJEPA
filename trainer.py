"""
Cell-JEPA Training
===================
Implements the pre-training and fine-tuning training loops described in
Sections 2.3, 2.4, and Appendices E.1–E.2.

Pre-training recipe (Appendix E.1):
    - Optimizer : AdamW, lr=1e-4, weight_decay=2e-4
    - LR decay  : 0.9 per epoch
    - Batch size: 32
    - Mask ratio: 0.15
    - Epochs    : 4

Fine-tuning recipe (Appendix E.2):
    - Optimizer : Adam, lr=1e-4
    - LR decay  : 0.9 per epoch
    - Batch size: 64
    - Mask ratio: 0.40
    - Epochs    : 30
"""

from __future__ import annotations

import time
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from cell_jepa import CellJEPA
from losses import PretrainingLoss, FinetuningLoss, PerturbationLoss, DeltaPerturbationLoss


# ---------------------------------------------------------------------------
# Training configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 2e-4
    lr_decay: float = 0.9

    # Data
    batch_size: int = 32
    mask_ratio: float = 0.15
    num_workers: int = 4

    # Training
    n_epochs: int = 4
    grad_clip: float = 1.0
    log_every: int = 50        # steps between logging

    # Loss weights
    w_rec: float = 1.0
    w_jepa: float = 1000.0

    # EMA
    ema_momentum: float = 0.996


@dataclass
class FinetuneConfig:
    # Optimizer
    lr: float = 1e-4
    lr_decay: float = 0.9

    # Data
    batch_size: int = 64
    mask_ratio: float = 0.40
    val_split: float = 0.1
    num_workers: int = 4

    # Training
    n_epochs: int = 30
    grad_clip: float = 1.0
    log_every: int = 20

    # Loss weights (all = 1 in Appendix E.2; w_jepa inherited from pre-training)
    w_gep: float = 1.0
    w_gepc: float = 1.0
    w_ecs: float = 1.0
    w_jepa: float = 1000.0
    ecs_temperature: float = 0.1
    include_jepa: bool = True


# ---------------------------------------------------------------------------
# Utility: move batch dict to device
# ---------------------------------------------------------------------------

def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Pre-training Trainer
# ---------------------------------------------------------------------------

class Pretrainer:
    """
    Runs Cell-JEPA pre-training.

    Args:
        model:   CellJEPA instance.
        dataset: SingleCellDataset (mask_ratio should match config.mask_ratio).
        config:  PretrainConfig.
        device:  torch.device.
    """

    def __init__(
        self,
        model: CellJEPA,
        dataset,
        config: PretrainConfig = PretrainConfig(),
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.lr_decay
        )
        self.criterion = PretrainingLoss(
            w_rec=config.w_rec, w_jepa=config.w_jepa
        )
        self._start_epoch = 1
        self._current_epoch = 0
        self._history: list[dict] = []

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        model: CellJEPA,
        dataset,
        device: Optional[torch.device] = None,
    ) -> "Pretrainer":
        """
        Restore a Pretrainer from a checkpoint saved by .save().
        Training will resume from the epoch after the one that was saved.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        trainer = cls(model, dataset, config=ckpt["config"], device=device)
        model.load_state_dict(ckpt["model_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
        trainer._start_epoch = ckpt.get("epoch", 0) + 1
        trainer._history = ckpt.get("history", [])
        print(f"Resumed from {path} (next epoch: {trainer._start_epoch})")
        return trainer

    def train(self, epoch_callback=None) -> list[dict]:
        """
        Run the pre-training loop, optionally resuming from a checkpoint.

        Args:
            epoch_callback: Optional callable(epoch_num) called after each epoch.
                            Useful for saving per-epoch checkpoints.

        Returns:
            List of per-epoch metric dicts.
        """
        use_pin = torch.cuda.is_available() and self.config.num_workers > 0
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=use_pin,
        )

        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(self._start_epoch, self.config.n_epochs + 1):
            self._current_epoch = epoch
            metrics = self._train_epoch(epoch, loader, scaler, use_amp)
            self.scheduler.step()
            self._history.append(metrics)
            print(
                f"[Epoch {epoch}/{self.config.n_epochs}]  "
                f"loss={metrics['loss']:.4f}  "
                f"l_jepa={metrics['l_jepa']:.4f}  "
                f"l_rec={metrics['l_rec']:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )
            if epoch_callback is not None:
                epoch_callback(epoch)
        return self._history

    def _train_epoch(self, epoch: int, loader: DataLoader,
                     scaler=None, use_amp: bool = False) -> dict:
        self.model.train()
        total_loss = total_jepa = total_rec = 0.0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(loader, 1):
            batch = batch_to_device(batch, self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = self.model(
                    gene_ids=batch["gene_ids"],
                    values=batch["values"],
                    masked_vals=batch["masked_vals"],
                    is_masked=batch["mask"],
                    is_pad=batch["padding"],
                )

                loss_dict = self.criterion(
                    model_out=out,
                    target_values=batch["values"],
                    is_masked=batch["mask"],
                )

            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            scaler.step(self.optimizer)
            scaler.update()

            # EMA teacher update (after each gradient step)
            self.model.update_teacher()

            total_loss += loss_dict["loss"].item()
            total_jepa += loss_dict["l_jepa"].item()
            total_rec  += loss_dict["l_rec"].item()
            n_steps    += 1

            if step % self.config.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch} | Step {step}/{len(loader)} | "
                    f"loss={total_loss/n_steps:.4f} | "
                    f"l_jepa={total_jepa/n_steps:.4f} | "
                    f"l_rec={total_rec/n_steps:.4f} | "
                    f"{elapsed:.1f}s elapsed"
                )

        return {
            "loss":   total_loss / n_steps,
            "l_jepa": total_jepa / n_steps,
            "l_rec":  total_rec  / n_steps,
        }

    def save(self, path: str):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "epoch": self._current_epoch,
                "config": self.config,
                "history": self._history,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# Fine-tuning Trainer
# ---------------------------------------------------------------------------

class Finetuner:
    """
    Runs Cell-JEPA fine-tuning for cell-type clustering (Section 2.4).

    Args:
        model:   CellJEPA (loaded from pre-trained checkpoint).
        dataset: SingleCellDataset with cell_types populated and
                 mask_ratio set to config.mask_ratio.
        config:  FinetuneConfig.
        device:  torch.device.
    """

    def __init__(
        self,
        model: CellJEPA,
        dataset,
        config: FinetuneConfig = FinetuneConfig(),
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Train / val split
        n_val = int(len(dataset) * config.val_split)
        n_train = len(dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.lr_decay
        )
        self.criterion = FinetuningLoss(
            w_gep=config.w_gep,
            w_gepc=config.w_gepc,
            w_ecs=config.w_ecs,
            w_jepa=config.w_jepa,
            ecs_temperature=config.ecs_temperature,
            include_jepa=config.include_jepa,
        )

    def train(self) -> list[dict]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        history = []
        for epoch in range(1, self.config.n_epochs + 1):
            train_metrics = self._train_epoch(epoch, train_loader)
            val_metrics   = self._val_epoch(val_loader)
            self.scheduler.step()

            metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}":   v for k, v in val_metrics.items()}}
            history.append(metrics)

            print(
                f"[Epoch {epoch}/{self.config.n_epochs}]  "
                f"train_loss={metrics['train_loss']:.4f}  "
                f"val_loss={metrics['val_loss']:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )
        return history

    def _train_epoch(self, epoch: int, loader: DataLoader) -> dict:
        self.model.train()
        total = {k: 0.0 for k in ["loss", "l_gep", "l_gepc", "l_ecs", "l_jepa"]}
        n = 0

        for batch in loader:
            batch = batch_to_device(batch, self.device)
            self.optimizer.zero_grad()

            out = self.model(
                gene_ids=batch["gene_ids"],
                values=batch["values"],
                masked_vals=batch["masked_vals"],
                is_masked=batch["mask"],
                is_pad=batch["padding"],
            )

            loss_dict = self.criterion(
                model_out=out,
                target_values=batch["values"],
                is_masked=batch["mask"],
                cell_types=batch["cell_type"],
            )

            loss_dict["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.model.update_teacher()

            for k in total:
                total[k] += loss_dict[k].item() if k != "loss" else loss_dict["loss"].item()
            n += 1

        return {k: v / n for k, v in total.items()}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total = {k: 0.0 for k in ["loss", "l_gep", "l_gepc", "l_ecs", "l_jepa"]}
        n = 0

        for batch in loader:
            batch = batch_to_device(batch, self.device)
            out = self.model(
                gene_ids=batch["gene_ids"],
                values=batch["values"],
                masked_vals=batch["masked_vals"],
                is_masked=batch["mask"],
                is_pad=batch["padding"],
            )
            loss_dict = self.criterion(
                model_out=out,
                target_values=batch["values"],
                is_masked=batch["mask"],
                cell_types=batch["cell_type"],
            )
            for k in total:
                total[k] += loss_dict[k].item() if k != "loss" else loss_dict["loss"].item()
            n += 1

        return {k: v / n for k, v in total.items()}

    def save(self, path: str):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        print(f"Fine-tune checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# Perturbation Finetuning configuration  (Appendix E.3)
# ---------------------------------------------------------------------------

@dataclass
class PerturbationConfig:
    # Optimizer
    lr: float = 1e-4
    lr_decay: float = 0.9

    # Data
    batch_size: int = 64
    num_workers: int = 4

    # Training
    n_epochs: int = 15
    grad_clip: float = 1.0
    log_every: int = 20

    # Loss weights (Appendix E.3)
    w_pert_rec: float = 1.0
    w_jepa_pert: float = 1.0
    w_ecs: float = 0.8
    ecs_temperature: float = 0.1

    # Ablation flags
    include_jepa: bool = True   # False → zero out JEPA term
    predict_delta: bool = False # True → use DeltaPerturbationLoss instead of absolute rec


# ---------------------------------------------------------------------------
# Perturbation Trainer  (Section 2.5)
# ---------------------------------------------------------------------------

class PerturbationTrainer:
    """
    Finetunes Cell-JEPA for perturbation response prediction (Section 2.5).

    The dataset must supply batches with keys:
        gene_ids    : (B, L) gene vocab IDs
        values      : (B, L) baseline (unperturbed) bin indices
        pert_ids    : (B, L) perturbation vocab IDs (0 for unperturbed genes)
        pert_values : (B, L) ground-truth post-perturbation bin indices
        padding     : (B, L) True at padding positions
        cell_type   : (B,)   integer cell-type label (-1 = unknown)

    Args:
        model:   CellJEPA with n_perturbations > 0.
        dataset: Perturbation dataset (see above).
        config:  PerturbationConfig.
        device:  torch.device.
    """

    def __init__(
        self,
        model: CellJEPA,
        dataset,
        config: PerturbationConfig = PerturbationConfig(),
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.lr_decay
        )
        if config.predict_delta:
            self.criterion = DeltaPerturbationLoss(
                w_delta=config.w_pert_rec,
                w_jepa_pert=config.w_jepa_pert,
                w_ecs=config.w_ecs,
                ecs_temperature=config.ecs_temperature,
                include_jepa=config.include_jepa,
            )
        else:
            self.criterion = PerturbationLoss(
                w_pert_rec=config.w_pert_rec,
                w_jepa_pert=config.w_jepa_pert,
                w_ecs=config.w_ecs,
                ecs_temperature=config.ecs_temperature,
                include_jepa=config.include_jepa,
            )

    def train(self) -> list[dict]:
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        history = []
        for epoch in range(1, self.config.n_epochs + 1):
            metrics = self._train_epoch(epoch, loader)
            self.scheduler.step()
            history.append(metrics)
            rec_key = "l_delta" if self.config.predict_delta else "l_pert_rec"
            print(
                f"[Epoch {epoch}/{self.config.n_epochs}]  "
                f"loss={metrics['loss']:.4f}  "
                f"{rec_key}={metrics[rec_key]:.4f}  "
                f"l_jepa_pert={metrics['l_jepa_pert']:.4f}  "
                f"l_ecs={metrics['l_ecs']:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )
        return history

    def _train_epoch(self, epoch: int, loader: DataLoader) -> dict:
        self.model.train()
        rec_key = "l_delta" if self.config.predict_delta else "l_pert_rec"
        total = {k: 0.0 for k in ["loss", rec_key, "l_jepa_pert", "l_ecs"]}
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(loader, 1):
            batch = batch_to_device(batch, self.device)
            self.optimizer.zero_grad()

            if self.config.predict_delta:
                out = self.model.forward_perturb_delta(
                    gene_ids=batch["gene_ids"],
                    values=batch["values"],
                    pert_ids=batch["pert_ids"],
                    pert_values=batch["pert_values"],
                    is_pad=batch["padding"],
                )
                loss_dict = self.criterion(
                    model_out=out,
                    ctrl_values=batch["values"],
                    pert_values=batch["pert_values"],
                    is_pad=batch["padding"],
                    cell_types=batch.get("cell_type"),
                )
            else:
                out = self.model.forward_perturb(
                    gene_ids=batch["gene_ids"],
                    values=batch["values"],
                    pert_ids=batch["pert_ids"],
                    pert_values=batch["pert_values"],
                    is_pad=batch["padding"],
                )
                loss_dict = self.criterion(
                    model_out=out,
                    pert_values=batch["pert_values"],
                    is_pad=batch["padding"],
                    cell_types=batch.get("cell_type"),
                )

            loss_dict["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.model.update_teacher()

            for k in total:
                total[k] += loss_dict[k].item() if k != "loss" else loss_dict["loss"].item()
            n_steps += 1

            if step % self.config.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch} | Step {step}/{len(loader)} | "
                    f"loss={total['loss']/n_steps:.4f} | "
                    f"{elapsed:.1f}s elapsed"
                )

        return {k: v / n_steps for k, v in total.items()}

    def save(self, path: str):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        print(f"Perturbation checkpoint saved to {path}")

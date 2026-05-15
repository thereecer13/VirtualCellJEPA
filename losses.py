"""
Cell-JEPA Loss Functions
=========================
Implements all objectives from Sections 2.3 and 2.4 of the paper.

Pre-training (Section 2.3):
    L_pre-train = w_rec * L_rec + w_JEPA * L_JEPA

Fine-tuning (Section 2.4):
    L_finetune = w_GEP * L_GEP + w_GEPC * L_GEPC + w_ECS * L_ECS + w_JEPA * L_JEPA

Perturbation (Section 2.5):
    L_pert = w_pert-rec * L_pert-rec + w_JEPA^pert * L_JEPA^pert + w_ECS * L_ECS

Delta-perturbation objective (this project):
    Instead of predicting absolute post-perturbation expression, the model
    predicts the *delta* (perturbed − control) in bin-index space.
    L_delta = MSE(delta_hat, delta_true)  where delta = pert_bins − ctrl_bins
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# JEPA Loss  (Eq. 2.3)
# ---------------------------------------------------------------------------

def jepa_loss(
    e_tilde: torch.Tensor,
    e: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine-distance JEPA loss between the predicted student embedding and the
    (stop-gradient) teacher target embedding.

    L_JEPA = 1 - cosine_similarity(p(e_hat), sg(e))

    Note: stop_gradient is handled by the caller (teacher runs under
    torch.no_grad()), so e is already detached.

    Args:
        e_tilde: (B, D) predictor output  p(e_hat)
        e:       (B, D) teacher <cls> embedding  (already detached)

    Returns:
        Scalar loss.
    """
    # F.cosine_similarity returns (B,)
    cos_sim = F.cosine_similarity(e_tilde, e.detach(), dim=-1)
    return (1.0 - cos_sim).mean()


# ---------------------------------------------------------------------------
# Reconstruction / GEP Loss  (Eq. 2.4)
# ---------------------------------------------------------------------------

def reconstruction_loss(
    v_hat: torch.Tensor,
    target_values: torch.LongTensor,
    is_masked: torch.BoolTensor,
) -> torch.Tensor:
    """
    MSE on masked gene expression bin predictions.

    L_rec = (1/|U_mask|) * sum_{i in U_mask} (v_hat_i - v_i)^2

    The paper treats bin indices as regression targets (following scGPT).
    We use MSE over the raw bin index scalars as described in Eq. 2.4.

    Args:
        v_hat:         (B, L, n_bins+1) logits from value_head
        target_values: (B, L) integer bin targets  (0 at <cls>/<pad>)
        is_masked:     (B, L) True at masked positions

    Returns:
        Scalar MSE loss over masked positions only.
    """
    # Convert logits to scalar predictions via argmax (for MSE)
    # OR use cross-entropy over bins — the paper says "MSE on masked gene token"
    # We follow the paper literally: predict the bin index, MSE loss.
    # Use the expected value of the softmax distribution as the scalar prediction.
    probs = F.softmax(v_hat, dim=-1)                       # (B, L, n_bins+1)
    bin_range = torch.arange(
        v_hat.shape[-1], device=v_hat.device, dtype=torch.float32
    )
    v_pred = (probs * bin_range).sum(-1)                   # (B, L) predicted bin

    v_target = target_values.float()                       # (B, L)

    # Only compute loss over masked positions
    mask = is_masked & (target_values >= 0)                # exclude <cls>/<pad>
    if mask.sum() == 0:
        return v_pred.new_tensor(0.0)

    return F.mse_loss(v_pred[mask], v_target[mask])


# ---------------------------------------------------------------------------
# GEPC Loss  (Section 2.4.2)
# ---------------------------------------------------------------------------

def gepc_loss(
    gepc_scores: torch.Tensor,
    target_values: torch.LongTensor,
    is_masked: torch.BoolTensor,
) -> torch.Tensor:
    """
    Gene Expression Prediction for Cell Modeling loss.

    v_tilde_i = f(y_i)^T W e_hat  (inner product; precomputed in model)
    L_GEPC = MSE over masked genes between predicted and true bin values.

    Args:
        gepc_scores:   (B, L) inner product scores
        target_values: (B, L) integer bin targets
        is_masked:     (B, L)

    Returns:
        Scalar MSE loss.
    """
    mask = is_masked & (target_values >= 0)
    if mask.sum() == 0:
        return gepc_scores.new_tensor(0.0)

    v_pred = gepc_scores[mask]
    v_tgt = target_values.float()[mask]
    return F.mse_loss(v_pred, v_tgt)


# ---------------------------------------------------------------------------
# ECS Loss  (Section 2.4.2)
# ---------------------------------------------------------------------------

def ecs_loss(
    cell_embeddings: torch.Tensor,
    cell_types: torch.LongTensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Elastic Cell Similarity loss — InfoNCE-style contrastive objective.

    For each anchor cell i, cells of the SAME type are treated as positives
    (uniform weight 1/|P(i)|), and all cells in the minibatch are negatives
    (including self).

    L_ECS,i = - sum_j p_ij * log [ exp(s_ij) / sum_k exp(s_ik) ]
    L_ECS   = (1/N) * sum_i L_ECS,i

    where s_ij = cosine_similarity(e_i, e_j) / temperature

    Args:
        cell_embeddings: (B, D) cell <cls> embeddings
        cell_types:      (B,)   integer cell-type labels (-1 = unknown, skipped)
        temperature:     Softmax temperature τ

    Returns:
        Scalar ECS loss. Returns 0 if fewer than 2 valid cells.
    """
    # Filter out cells with unknown labels
    valid = cell_types >= 0
    if valid.sum() < 2:
        return cell_embeddings.new_tensor(0.0)

    emb = cell_embeddings[valid]           # (N, D)
    labels = cell_types[valid]             # (N,)
    N = emb.shape[0]

    # Normalize embeddings for cosine similarity
    emb_norm = F.normalize(emb, dim=-1)   # (N, D)

    # Pairwise cosine similarity scaled by temperature
    sim = (emb_norm @ emb_norm.T) / temperature   # (N, N)

    # Build soft positive assignment matrix p
    # p[i, j] = 1/|P(i)| if label_j == label_i, else 0
    label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (N, N)
    # Count positives per anchor (including self)
    pos_counts = label_eq.sum(dim=1, keepdim=True)                   # (N, 1)
    pos_counts = pos_counts.clamp(min=1.0)
    p = label_eq / pos_counts                                         # (N, N)

    # Per-anchor cross-entropy: - sum_j p_ij * log softmax(sim)[j]
    log_softmax_sim = F.log_softmax(sim, dim=1)   # (N, N)
    per_anchor = -(p * log_softmax_sim).sum(dim=1) # (N,)
    return per_anchor.mean()


# ---------------------------------------------------------------------------
# Combined Pre-training Objective  (Section 2.3)
# ---------------------------------------------------------------------------

class PretrainingLoss(nn.Module):
    """
    L_pre-train = w_rec * L_rec + w_JEPA * L_JEPA

    From Appendix E.1: w_JEPA = 1000, w_rec = 1.
    """

    def __init__(self, w_rec: float = 1.0, w_jepa: float = 1000.0):
        super().__init__()
        self.w_rec = w_rec
        self.w_jepa = w_jepa

    def forward(
        self,
        model_out: dict,
        target_values: torch.LongTensor,
        is_masked: torch.BoolTensor,
    ) -> dict:
        """
        Args:
            model_out:     Output dict from CellJEPA.forward()
            target_values: (B, L) ground-truth bin indices
            is_masked:     (B, L) mask positions

        Returns:
            Dict with 'loss' (scalar), 'l_jepa', 'l_rec'.
        """
        l_jepa = jepa_loss(model_out["e_tilde"], model_out["e"])
        l_rec = reconstruction_loss(model_out["v_hat"], target_values, is_masked)

        total = self.w_jepa * l_jepa + self.w_rec * l_rec

        return {
            "loss":   total,
            "l_jepa": l_jepa.detach(),
            "l_rec":  l_rec.detach(),
        }


# ---------------------------------------------------------------------------
# Combined Fine-tuning Objective  (Section 2.4.3)
# ---------------------------------------------------------------------------

class FinetuningLoss(nn.Module):
    """
    L_finetune = w_GEP * L_GEP + w_GEPC * L_GEPC + w_ECS * L_ECS + w_JEPA * L_JEPA

    Default weights from Appendix E.2: all = 1, w_JEPA inherited from pre-training.
    """

    def __init__(
        self,
        w_gep: float = 1.0,
        w_gepc: float = 1.0,
        w_ecs: float = 1.0,
        w_jepa: float = 1000.0,
        ecs_temperature: float = 0.1,
        include_jepa: bool = True,
    ):
        super().__init__()
        self.w_gep = w_gep
        self.w_gepc = w_gepc
        self.w_ecs = w_ecs
        self.w_jepa = w_jepa
        self.ecs_temperature = ecs_temperature
        self.include_jepa = include_jepa

    def forward(
        self,
        model_out: dict,
        target_values: torch.LongTensor,
        is_masked: torch.BoolTensor,
        cell_types: Optional[torch.LongTensor] = None,
    ) -> dict:
        l_gep = reconstruction_loss(model_out["v_hat"], target_values, is_masked)
        l_gepc = gepc_loss(model_out["gepc_scores"], target_values, is_masked)

        l_ecs = (
            ecs_loss(model_out["e_hat"], cell_types, self.ecs_temperature)
            if cell_types is not None
            else model_out["e_hat"].new_tensor(0.0)
        )

        l_jepa = (
            jepa_loss(model_out["e_tilde"], model_out["e"])
            if self.include_jepa
            else model_out["e_hat"].new_tensor(0.0)
        )

        total = (
            self.w_gep  * l_gep
            + self.w_gepc * l_gepc
            + self.w_ecs  * l_ecs
            + self.w_jepa * l_jepa
        )

        return {
            "loss":   total,
            "l_gep":  l_gep.detach(),
            "l_gepc": l_gepc.detach(),
            "l_ecs":  l_ecs.detach(),
            "l_jepa": l_jepa.detach(),
        }


# ---------------------------------------------------------------------------
# Perturbation Reconstruction Loss  (Section 2.5)
# ---------------------------------------------------------------------------

def perturbation_rec_loss(
    v_hat_pert: torch.Tensor,
    target_values: torch.LongTensor,
    is_pad: torch.BoolTensor,
) -> torch.Tensor:
    """
    MSE over all (non-padding) genes between predicted and ground-truth
    post-perturbation expression values.

    Unlike the pre-training reconstruction loss, no masking is applied —
    the loss is computed over every gene token (Section 2.5).

    Args:
        v_hat_pert:    (B, L, n_bins+1) logits from perturb_value_head
        target_values: (B, L) ground-truth post-perturbation bin indices
        is_pad:        (B, L) True at padding positions

    Returns:
        Scalar MSE loss.
    """
    probs = F.softmax(v_hat_pert, dim=-1)
    bin_range = torch.arange(
        v_hat_pert.shape[-1], device=v_hat_pert.device, dtype=torch.float32
    )
    v_pred = (probs * bin_range).sum(-1)   # (B, L) expected bin

    mask = ~is_pad & (target_values >= 0)
    if mask.sum() == 0:
        return v_pred.new_tensor(0.0)

    return F.mse_loss(v_pred[mask], target_values.float()[mask])


# ---------------------------------------------------------------------------
# Combined Perturbation Objective  (Section 2.5)
# ---------------------------------------------------------------------------

class PerturbationLoss(nn.Module):
    """
    L_pert = w_pert_rec * L_pert-rec + w_jepa_pert * L_JEPA^pert + w_ecs * L_ECS

    Default weights from Appendix E.3: w_pert_rec=1.0, w_jepa_pert=1.0, w_ecs=0.8.

    When include_jepa=False, the JEPA term is zeroed out (ablation).
    """

    def __init__(
        self,
        w_pert_rec: float = 1.0,
        w_jepa_pert: float = 1.0,
        w_ecs: float = 0.8,
        ecs_temperature: float = 0.1,
        include_jepa: bool = True,
    ):
        super().__init__()
        self.w_pert_rec = w_pert_rec
        self.w_jepa_pert = w_jepa_pert
        self.w_ecs = w_ecs
        self.ecs_temperature = ecs_temperature
        self.include_jepa = include_jepa

    def forward(
        self,
        model_out: dict,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
        cell_types: Optional[torch.LongTensor] = None,
    ) -> dict:
        """
        Args:
            model_out:   Output dict from CellJEPA.forward_perturb()
            pert_values: (B, L) ground-truth post-perturbation bin indices
            is_pad:      (B, L) padding mask
            cell_types:  (B,)  integer cell-type labels for ECS (optional)

        Returns:
            Dict with 'loss' (scalar), 'l_pert_rec', 'l_jepa_pert', 'l_ecs'.
        """
        l_pert_rec = perturbation_rec_loss(
            model_out["v_hat_pert"], pert_values, is_pad
        )

        l_jepa_pert = (
            jepa_loss(model_out["e_tilde_pert"], model_out["e_pert"])
            if self.include_jepa
            else model_out["e_hat_pert"].new_tensor(0.0)
        )

        l_ecs = (
            ecs_loss(model_out["e_hat_pert"], cell_types, self.ecs_temperature)
            if cell_types is not None
            else model_out["e_hat_pert"].new_tensor(0.0)
        )

        total = (
            self.w_pert_rec  * l_pert_rec
            + self.w_jepa_pert * l_jepa_pert
            + self.w_ecs       * l_ecs
        )

        return {
            "loss":        total,
            "l_pert_rec":  l_pert_rec.detach(),
            "l_jepa_pert": l_jepa_pert.detach(),
            "l_ecs":       l_ecs.detach(),
        }


# ---------------------------------------------------------------------------
# Delta Perturbation Loss  (this project — predict perturbed − control delta)
# ---------------------------------------------------------------------------

def perturbation_delta_loss(
    delta_hat: torch.Tensor,
    ctrl_values: torch.LongTensor,
    pert_values: torch.LongTensor,
    is_pad: torch.BoolTensor,
) -> torch.Tensor:
    """
    MSE between predicted delta and true delta over all non-padding genes.

    delta_true = pert_bins − ctrl_bins  (signed difference in bin space)
    delta_hat  = scalar prediction from delta_head (no softmax; direct regression)

    Args:
        delta_hat:   (B, L) scalar predictions from model's delta_head
        ctrl_values: (B, L) control bin indices
        pert_values: (B, L) perturbed bin indices
        is_pad:      (B, L) True at padding positions

    Returns:
        Scalar MSE loss.
    """
    delta_true = (pert_values.float() - ctrl_values.float())  # (B, L)
    mask = ~is_pad & (ctrl_values >= 0) & (pert_values >= 0)
    if mask.sum() == 0:
        return delta_hat.new_tensor(0.0)
    return F.mse_loss(delta_hat[mask], delta_true[mask])


class DeltaPerturbationLoss(nn.Module):
    """
    Like PerturbationLoss but optimises delta prediction instead of absolute
    post-perturbation expression.

    L_delta_pert = w_delta * L_delta + w_jepa_pert * L_JEPA^pert + w_ecs * L_ECS

    The model output dict must contain 'delta_hat' (B, L) — a scalar regression
    head that predicts (pert_bins - ctrl_bins) directly.

    When include_jepa=False, the JEPA term is zeroed out.
    """

    def __init__(
        self,
        w_delta: float = 1.0,
        w_jepa_pert: float = 1.0,
        w_ecs: float = 0.8,
        ecs_temperature: float = 0.1,
        include_jepa: bool = True,
    ):
        super().__init__()
        self.w_delta = w_delta
        self.w_jepa_pert = w_jepa_pert
        self.w_ecs = w_ecs
        self.ecs_temperature = ecs_temperature
        self.include_jepa = include_jepa

    def forward(
        self,
        model_out: dict,
        ctrl_values: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
        cell_types: Optional[torch.LongTensor] = None,
    ) -> dict:
        """
        Args:
            model_out:   Output dict from CellJEPA.forward_perturb()
                         Must include 'delta_hat' (B, L).
            ctrl_values: (B, L) control bin indices
            pert_values: (B, L) ground-truth perturbed bin indices
            is_pad:      (B, L) padding mask
            cell_types:  (B,)  integer cell-type labels for ECS (optional)

        Returns:
            Dict with 'loss', 'l_delta', 'l_jepa_pert', 'l_ecs'.
        """
        l_delta = perturbation_delta_loss(
            model_out["delta_hat"], ctrl_values, pert_values, is_pad
        )

        l_jepa_pert = (
            jepa_loss(model_out["e_tilde_pert"], model_out["e_pert"])
            if self.include_jepa
            else model_out["e_hat_pert"].new_tensor(0.0)
        )

        l_ecs = (
            ecs_loss(model_out["e_hat_pert"], cell_types, self.ecs_temperature)
            if cell_types is not None
            else model_out["e_hat_pert"].new_tensor(0.0)
        )

        total = (
            self.w_delta     * l_delta
            + self.w_jepa_pert * l_jepa_pert
            + self.w_ecs       * l_ecs
        )

        return {
            "loss":        total,
            "l_delta":     l_delta.detach(),
            "l_jepa_pert": l_jepa_pert.detach(),
            "l_ecs":       l_ecs.detach(),
        }


# ---------------------------------------------------------------------------
# SIGReg Perturbation Losses  (no teacher — no l_jepa_pert term)
# ---------------------------------------------------------------------------

class SIGRegPerturbationLoss(nn.Module):
    """
    L = w_pert_rec * L_pert_rec + w_ecs * L_ECS

    Used with CellJEPA_SIGReg.forward_perturb() (absolute prediction).
    No l_jepa_pert term — SIGReg has no EMA teacher.
    """

    def __init__(
        self,
        w_pert_rec: float = 1.0,
        w_ecs: float = 0.8,
        ecs_temperature: float = 0.1,
    ):
        super().__init__()
        self.w_pert_rec = w_pert_rec
        self.w_ecs = w_ecs
        self.ecs_temperature = ecs_temperature

    def forward(
        self,
        model_out: dict,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
        cell_types: Optional[torch.LongTensor] = None,
    ) -> dict:
        l_pert_rec = perturbation_rec_loss(model_out["v_hat_pert"], pert_values, is_pad)
        l_ecs = (
            ecs_loss(model_out["e_hat_pert"], cell_types, self.ecs_temperature)
            if cell_types is not None
            else model_out["e_hat_pert"].new_tensor(0.0)
        )
        total = self.w_pert_rec * l_pert_rec + self.w_ecs * l_ecs
        return {
            "loss":       total,
            "l_pert_rec": l_pert_rec.detach(),
            "l_ecs":      l_ecs.detach(),
        }


class SIGRegDeltaPerturbationLoss(nn.Module):
    """
    L = w_delta * L_delta + w_ecs * L_ECS

    Used with CellJEPA_SIGReg.forward_perturb_delta() (delta prediction).
    No l_jepa_pert term — SIGReg has no EMA teacher.
    """

    def __init__(
        self,
        w_delta: float = 1.0,
        w_ecs: float = 0.8,
        ecs_temperature: float = 0.1,
    ):
        super().__init__()
        self.w_delta = w_delta
        self.w_ecs = w_ecs
        self.ecs_temperature = ecs_temperature

    def forward(
        self,
        model_out: dict,
        ctrl_values: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
        cell_types: Optional[torch.LongTensor] = None,
    ) -> dict:
        l_delta = perturbation_delta_loss(
            model_out["delta_hat"], ctrl_values, pert_values, is_pad
        )
        l_ecs = (
            ecs_loss(model_out["e_hat_pert"], cell_types, self.ecs_temperature)
            if cell_types is not None
            else model_out["e_hat_pert"].new_tensor(0.0)
        )
        total = self.w_delta * l_delta + self.w_ecs * l_ecs
        return {
            "loss":    total,
            "l_delta": l_delta.detach(),
            "l_ecs":   l_ecs.detach(),
        }


# ---------------------------------------------------------------------------
# SIGReg Loss  (Balestriero & LeCun 2025, LeJEPA)
# ---------------------------------------------------------------------------

def _epps_pulley(u: torch.Tensor) -> torch.Tensor:
    """
    Closed-form Epps-Pulley statistic for 1D projections u of shape (N,).

    Measures deviation of the projected distribution from a standard normal
    using the characteristic-function test with Gaussian weighting w(t)=exp(-t²).

    TEP = (1/sqrt(3)) * [
        (2/N) * Σ_{i,j} exp(-(u_i - u_j)²/4)
        - 2*sqrt(2) * Σ_i exp(-u_i²/3)
        + N
    ]  / N

    Normalised by N to be batch-size independent.
    """
    N = u.shape[0]
    diff = u.unsqueeze(0) - u.unsqueeze(1)               # (N, N) pairwise diffs
    pairwise_term = torch.exp(-diff.pow(2) / 4).mean()   # (1/N²) Σ_{i,j}
    marginal_term = torch.exp(-u.pow(2) / 3).mean()      # (1/N) Σ_i
    # Factor N/sqrt(3) * (2*pairwise - 2*sqrt(2)*marginal + 1), then /N
    return (1.0 / math.sqrt(3)) * (
        2.0 * pairwise_term - 2.0 * math.sqrt(2) * marginal_term + 1.0
    )


def sigreg_loss(Z: torch.Tensor, M: int = 256) -> torch.Tensor:
    """
    SIGReg loss: average Epps-Pulley statistic over M random 1D projections.

    Measures how far the distribution of embeddings deviates from an isotropic
    Gaussian. Minimising this encourages the embeddings to fill the space
    uniformly, preventing representational collapse.

    Args:
        Z: (N, K) embedding matrix — all views concatenated along the batch dim
        M: number of random projection directions (default 256)

    Returns:
        Scalar loss (lower = more Gaussian = less collapse).

    ⚠ HYPERPARAMETER NOTE: M=256 is a starting point for K=512-dim embeddings.
      M=512 may give better distributional coverage at modest compute cost.
    """
    N, K = Z.shape
    # Sample M unit-norm random directions on the (K-1)-sphere
    A = F.normalize(torch.randn(K, M, device=Z.device, dtype=Z.dtype), dim=0)
    U = Z @ A                                             # (N, M) projections
    total = torch.stack([_epps_pulley(U[:, m]) for m in range(M)]).mean()
    return total


def sim_loss(
    e_global: torch.Tensor,
    e_views: list[torch.Tensor],
) -> torch.Tensor:
    """
    Similarity loss: mean squared L2 distance between the global CLS embedding
    and each masked-view CLS embedding, computed on L2-normalised vectors.

    L_sim = (1/V) * Σ_v ||norm(e_global) - norm(e_view_v)||²

    This is equivalent to 2*(1 - cosine_similarity) and encourages the masked
    views to produce the same cell-level representation as the global view.

    Args:
        e_global: (B, D) global view CLS embeddings
        e_views:  list of V (B, D) masked view CLS embeddings

    Returns:
        Scalar loss.
    """
    e_g = F.normalize(e_global, dim=-1)   # (B, D)
    total = e_global.new_tensor(0.0)
    for e_v in e_views:
        e_v_norm = F.normalize(e_v, dim=-1)
        total = total + (e_g - e_v_norm).pow(2).sum(-1).mean()
    return total / max(len(e_views), 1)


# ---------------------------------------------------------------------------
# SIGReg Pre-training Loss
# ---------------------------------------------------------------------------

class SIGRegPretrainingLoss(nn.Module):
    """
    L = w_sim * L_sim + w_sigreg * L_SIGReg + w_rec * L_rec

    L_sim    : similarity between global and masked-view CLS embeddings
    L_SIGReg : SIGReg on all CLS embeddings (global + all views concatenated)
    L_rec    : gene-level reconstruction MSE on global view (masked positions)

    ⚠ HYPERPARAMETER NOTE: w_sigreg (λ) and w_rec (γ) are the most sensitive
      parameters. Defaults (0.5, 1.0) are starting points.
      Recommended sweep: λ ∈ {0.1, 0.5, 0.9} × γ ∈ {0.1, 1.0, 10.0}.
    """

    def __init__(
        self,
        w_sim: float = 1.0,
        w_sigreg: float = 0.5,
        w_rec: float = 1.0,
        n_directions: int = 256,
    ):
        super().__init__()
        self.w_sim = w_sim
        self.w_sigreg = w_sigreg
        self.w_rec = w_rec
        self.n_directions = n_directions

    def forward(
        self,
        model_out: dict,
        target_values: torch.LongTensor,
        is_masked: torch.BoolTensor,
    ) -> dict:
        """
        Args:
            model_out:     Output dict from CellJEPA_SIGReg.forward()
            target_values: (B, L) ground-truth bin indices (global view)
            is_masked:     (B, L) True at masked positions in global view
                           (used only for reconstruction loss; SIGReg uses all)

        Returns:
            Dict with 'loss', 'l_sim', 'l_sigreg', 'l_rec'.
        """
        e_global = model_out["e_global"]    # (B, D)
        e_views  = model_out["e_views"]     # list of (B, D)

        l_sim = sim_loss(e_global, e_views)

        # Concatenate all view embeddings for SIGReg
        all_emb = torch.cat([e_global] + e_views, dim=0)   # (B*(V+1), D)
        l_sigreg = sigreg_loss(all_emb, M=self.n_directions)

        l_rec = reconstruction_loss(model_out["v_hat"], target_values, is_masked)

        total = self.w_sim * l_sim + self.w_sigreg * l_sigreg + self.w_rec * l_rec

        return {
            "loss":     total,
            "l_sim":    l_sim.detach(),
            "l_sigreg": l_sigreg.detach(),
            "l_rec":    l_rec.detach(),
        }


# ---------------------------------------------------------------------------
# SIGReg Fine-tuning Loss
# ---------------------------------------------------------------------------

class SIGRegFinetuningLoss(nn.Module):
    """
    L = w_gep*L_gep + w_gepc*L_gepc + w_ecs*L_ecs + w_sim*L_sim + w_sigreg*L_SIGReg

    Analogous to FinetuningLoss but replaces the JEPA cosine objective with
    L_sim + L_SIGReg. ECS uses the global view CLS embedding (e_global).
    """

    def __init__(
        self,
        w_gep: float = 1.0,
        w_gepc: float = 1.0,
        w_ecs: float = 1.0,
        w_sim: float = 1.0,
        w_sigreg: float = 0.5,
        ecs_temperature: float = 0.1,
        n_directions: int = 256,
    ):
        super().__init__()
        self.w_gep = w_gep
        self.w_gepc = w_gepc
        self.w_ecs = w_ecs
        self.w_sim = w_sim
        self.w_sigreg = w_sigreg
        self.ecs_temperature = ecs_temperature
        self.n_directions = n_directions

    def forward(
        self,
        model_out: dict,
        target_values: torch.LongTensor,
        is_masked: torch.BoolTensor,
        cell_types: Optional[torch.LongTensor] = None,
    ) -> dict:
        l_gep  = reconstruction_loss(model_out["v_hat"], target_values, is_masked)
        l_gepc = gepc_loss(model_out["gepc_scores"], target_values, is_masked)

        l_ecs = (
            ecs_loss(model_out["e_global"], cell_types, self.ecs_temperature)
            if cell_types is not None
            else model_out["e_global"].new_tensor(0.0)
        )

        e_views = model_out.get("e_views", [])
        l_sim = sim_loss(model_out["e_global"], e_views) if e_views else model_out["e_global"].new_tensor(0.0)

        all_emb = torch.cat([model_out["e_global"]] + e_views, dim=0)
        l_sigreg = sigreg_loss(all_emb, M=self.n_directions)

        total = (
            self.w_gep     * l_gep
            + self.w_gepc  * l_gepc
            + self.w_ecs   * l_ecs
            + self.w_sim   * l_sim
            + self.w_sigreg * l_sigreg
        )

        return {
            "loss":     total,
            "l_gep":    l_gep.detach(),
            "l_gepc":   l_gepc.detach(),
            "l_ecs":    l_ecs.detach(),
            "l_sim":    l_sim.detach(),
            "l_sigreg": l_sigreg.detach(),
        }

"""
perturb_metrics.py
==================
Evaluation metrics for perturbation response prediction.

Standard metrics used by GEARS, scGPT, and related benchmarks:
  - Mean Pearson correlation (all genes, perturbed expression)
  - Mean Pearson delta (correlation of predicted vs true delta: pert - ctrl)
  - Top-20 DEG Pearson delta (delta correlation on 20 most DE genes per condition)
  - Mean squared error on predicted expression

All functions operate on numpy arrays of raw (log-normalised, continuous)
expression values, not bin indices. A `bins_to_expr` helper converts bin
predictions back to proxy expression values for metric computation.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Bin → expression proxy
# ---------------------------------------------------------------------------

def bins_to_expr(bin_indices: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """
    Convert integer bin indices to a proxy continuous expression value
    by mapping each bin linearly to [0, 1].

    bin 0 (unexpressed) → 0.0
    bin k → k / n_bins  for k in {1, ..., n_bins}

    Args:
        bin_indices: Array of integer bin indices (any shape).
        n_bins:      Number of expression bins (default 50).

    Returns:
        Float array of same shape, values in [0, 1].
    """
    return bin_indices.astype(np.float32) / n_bins


def logits_to_expr(
    logits: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """
    Convert (B, L, n_bins+1) softmax logits to a proxy expression value
    using the expected bin index under the predicted distribution.

    Args:
        logits: (B, L, n_bins+1) float array (raw logits, not softmax).
        n_bins: Number of expression bins.

    Returns:
        (B, L) float array of proxy expression values in [0, 1].
    """
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    bin_range = np.arange(logits.shape[-1], dtype=np.float32) / n_bins
    return (probs * bin_range).sum(axis=-1)


# ---------------------------------------------------------------------------
# Per-condition mean expression helpers
# ---------------------------------------------------------------------------

def mean_expression_by_condition(
    expr_matrix: np.ndarray,
    conditions: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute the mean expression profile for each condition.

    Args:
        expr_matrix: (N, G) float expression matrix.
        conditions:  (N,) string array of condition labels.

    Returns:
        Dict mapping condition label → (G,) mean expression vector.
    """
    means = {}
    for cond in np.unique(conditions):
        mask = conditions == cond
        means[cond] = expr_matrix[mask].mean(axis=0)
    return means


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def pearson_all_genes(
    pred: np.ndarray,
    true: np.ndarray,
) -> float:
    """
    Pearson correlation between predicted and true expression across all genes.

    Args:
        pred: (G,) predicted expression vector for one condition.
        true: (G,) ground-truth expression vector.

    Returns:
        Pearson r (float).
    """
    if pred.std() < 1e-8 or true.std() < 1e-8:
        return 0.0
    r, _ = pearsonr(pred, true)
    return float(r)


def pearson_delta(
    pred_pert: np.ndarray,
    true_pert: np.ndarray,
    ctrl_mean: np.ndarray,
) -> float:
    """
    Pearson correlation of the predicted vs. true *delta* (pert − ctrl).

    PearsonΔ = cor(pred_pert − ctrl_mean, true_pert − ctrl_mean)

    This focuses on the direction of change rather than absolute values,
    and is the primary metric in GEARS and related benchmarks.

    Args:
        pred_pert:  (G,) predicted post-perturbation expression.
        true_pert:  (G,) true post-perturbation expression.
        ctrl_mean:  (G,) mean control expression.

    Returns:
        Pearson r of deltas (float).
    """
    pred_delta = pred_pert - ctrl_mean
    true_delta = true_pert - ctrl_mean
    if pred_delta.std() < 1e-8 or true_delta.std() < 1e-8:
        return 0.0
    r, _ = pearsonr(pred_delta, true_delta)
    return float(r)


def top_k_deg_pearson_delta(
    pred_pert: np.ndarray,
    true_pert: np.ndarray,
    ctrl_mean: np.ndarray,
    k: int = 20,
) -> float:
    """
    Pearson delta computed only on the top-K differentially expressed genes.

    DEGs are selected by |true_pert − ctrl_mean| (largest absolute change),
    which is the standard approach in GEARS evaluations.

    Args:
        pred_pert:  (G,) predicted expression.
        true_pert:  (G,) true expression.
        ctrl_mean:  (G,) mean control expression.
        k:          Number of top DEGs to use (default 20).

    Returns:
        Pearson r on top-k DEGs (float).
    """
    true_delta = true_pert - ctrl_mean
    top_idx = np.argsort(np.abs(true_delta))[-k:]
    return pearson_delta(pred_pert[top_idx], true_pert[top_idx], ctrl_mean[top_idx])


def mse_expr(
    pred: np.ndarray,
    true: np.ndarray,
) -> float:
    """
    Mean squared error between predicted and true expression.

    Args:
        pred: (G,) predicted expression.
        true: (G,) true expression.

    Returns:
        Scalar MSE.
    """
    return float(np.mean((pred - true) ** 2))


# ---------------------------------------------------------------------------
# Full perturbation evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perturbation(
    model,
    dataset,
    conditions: np.ndarray,
    ctrl_label: str = "ctrl",
    n_bins: int = 50,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    predict_delta: bool = False,
) -> dict:
    """
    Evaluate perturbation prediction on a held-out test set.

    For each unique non-control perturbation condition, computes:
      - Mean Pearson r (predicted vs. true post-perturbation expression)
      - Mean Pearson delta (predicted delta vs. true delta)
      - Top-20 DEG Pearson delta
      - MSE

    Args:
        model:        CellJEPA with n_perturbations > 0.
        dataset:      PerturbationDataset (test split).
        conditions:   (N,) string array of condition labels for test cells.
        ctrl_label:   Label string for control cells (default "ctrl").
        n_bins:       Expression bins (default 50).
        batch_size:   Inference batch size.
        device:       Compute device.
        predict_delta: If True, model uses forward_perturb_delta().

    Returns:
        Dict with per-condition results and aggregate means:
        {
          "per_condition": {cond: {"pearson": .., "pearson_delta": .., ...}},
          "mean_pearson": float,
          "mean_pearson_delta": float,
          "mean_top20_deg_pearson_delta": float,
          "mean_mse": float,
        }
    """
    from torch.utils.data import DataLoader

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    n_cells = len(dataset)
    n_genes = dataset.n_genes

    # Ground-truth expression from the full pre-binned matrices — no subsampling
    # artifact. Shape (N, G), values in [0, 1] proxy space.
    true_expr = dataset.pert_binned.astype(np.float32) / n_bins   # (N, G)
    ctrl_expr = dataset.ctrl_binned.astype(np.float32) / n_bins   # (N, G)

    # ctrl_matrix is tiled from a single mean vector, so any row gives ctrl_mean.
    ctrl_mean_full = ctrl_expr[0]  # (G,)

    # Model predictions: start from ctrl_mean as the null hypothesis for every gene.
    # Genes the model actually sees (sampled tokens) override this with real predictions.
    # This means unsampled genes predict "no change" (ctrl), which is conservative and
    # avoids the zero-inflation artifact from initialising to 0.
    pred_gene = np.tile(ctrl_mean_full, (n_cells, 1)).copy()  # (N, G)

    cell_offset = 0
    for batch in loader:
        gene_ids    = batch["gene_ids"].to(device)    # (B, L+1)
        values      = batch["values"].to(device)
        pert_ids    = batch["pert_ids"].to(device)
        pert_values = batch["pert_values"].to(device)
        is_pad      = batch["padding"].to(device)

        if predict_delta:
            out = model.forward_perturb_delta(gene_ids, values, pert_ids, pert_values, is_pad)
            ctrl_proxy = bins_to_expr(values.cpu().numpy(), n_bins)
            delta_hat  = out["delta_hat"].cpu().numpy()
            pred_proxy = ctrl_proxy + delta_hat / n_bins
        else:
            out = model.forward_perturb(gene_ids, values, pert_ids, pert_values, is_pad)
            logits = out["v_hat_pert"].cpu().numpy()
            pred_proxy = logits_to_expr(logits, n_bins)

        pad_mask = is_pad.cpu().numpy()
        gids     = gene_ids.cpu().numpy()
        B        = gids.shape[0]

        # Scatter model predictions into gene space for sampled genes only.
        for b in range(B):
            valid   = ~pad_mask[b]
            tokens  = gids[b, valid]
            seq_pos = np.where(valid)[0]

            is_gene = tokens >= 2
            g_idx   = tokens[is_gene] - 2
            seq_pos = seq_pos[is_gene]

            in_range = (g_idx >= 0) & (g_idx < n_genes)
            pred_gene[cell_offset + b, g_idx[in_range]] = pred_proxy[b, seq_pos[in_range]]

        cell_offset += B

    # Per-condition metrics using full (G,) gene-aligned vectors throughout.
    ctrl_mask_cells = conditions == ctrl_label
    # ctrl_mean_eval from ground-truth ctrl cells; falls back to tiled ctrl_mean_full.
    ctrl_mean_eval = (true_expr[ctrl_mask_cells].mean(axis=0)
                      if ctrl_mask_cells.any() else ctrl_mean_full)

    unique_conds = [c for c in np.unique(conditions) if c != ctrl_label]
    per_condition = {}
    for cond in unique_conds:
        mask = conditions == cond
        if mask.sum() == 0:
            continue
        pred_mean = pred_gene[mask].mean(axis=0)    # (G,) model prediction
        true_mean = true_expr[mask].mean(axis=0)    # (G,) ground truth, no zero inflation

        per_condition[cond] = {
            "pearson":               pearson_all_genes(pred_mean, true_mean),
            "pearson_delta":         pearson_delta(pred_mean, true_mean, ctrl_mean_eval),
            "top20_deg_pearson_delta": top_k_deg_pearson_delta(pred_mean, true_mean, ctrl_mean_eval, k=20),
            "mse":                   mse_expr(pred_mean, true_mean),
            "n_cells":               int(mask.sum()),
        }

    if not per_condition:
        return {"per_condition": {}, "mean_pearson": 0.0,
                "mean_pearson_delta": 0.0, "mean_top20_deg_pearson_delta": 0.0,
                "mean_mse": 0.0}

    mean_pearson       = float(np.mean([v["pearson"]               for v in per_condition.values()]))
    mean_pearson_delta = float(np.mean([v["pearson_delta"]         for v in per_condition.values()]))
    mean_top20         = float(np.mean([v["top20_deg_pearson_delta"] for v in per_condition.values()]))
    mean_mse           = float(np.mean([v["mse"]                   for v in per_condition.values()]))

    return {
        "per_condition":               per_condition,
        "mean_pearson":                mean_pearson,
        "mean_pearson_delta":          mean_pearson_delta,
        "mean_top20_deg_pearson_delta": mean_top20,
        "mean_mse":                    mean_mse,
    }

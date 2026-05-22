"""
compare_perturbation.py
========================
2×2 ablation study on perturbation response prediction using the
Adamson 2016 Perturb-seq dataset (K562, CRISPRi).

Conditions compared:
  A) Absolute rec objective,  JEPA on   (paper default)
  B) Absolute rec objective,  JEPA off
  C) Delta objective,         JEPA on
  D) Delta objective,         JEPA off

For each condition the model is trained from scratch (no prior pretraining)
via PerturbationTrainer, then evaluated on a held-out test split using:
  - Mean Pearson r          (predicted vs. true post-perturbation expression)
  - Mean Pearson delta      (predicted delta vs. true delta)
  - Top-20 DEG Pearson delta
  - MSE

Usage (Colab T4/A100):
    python compare_perturbation.py --device cuda

Smoke test (fast, ~1 min):
    python compare_perturbation.py --smoke_test --device cpu
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import scanpy as sc
import scipy.sparse as sp

from preprocessing import PerturbationDataset
from cell_jepa import CellJEPA
from trainer import PerturbationTrainer, PerturbationConfig
from perturb_metrics import evaluate_perturbation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CellJEPA perturbation 2×2 ablation")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--n_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--l_max", type=int, default=600,
                   help="Max genes per sequence (default 200)")
    p.add_argument("--n_hvg", type=int, default=2000,
                   help="Number of highly variable genes (default 2000)")
    p.add_argument("--test_fraction", type=float, default=0.2,
                   help="Fraction of perturbation conditions held out for test")
    p.add_argument("--results_file", default="results_perturbation.txt")
    p.add_argument("--smoke_test", action="store_true",
                   help="Tiny run: 3 conditions, 1 epoch, 500 cells")
    p.add_argument(
        "--pretrain_checkpoint", default=None,
        help="Path to kidney pre-trained checkpoint (.pt). When provided, the model "
             "backbone is initialised from this checkpoint and the large architecture "
             "(d_model=512, 12 layers) is used to match the pre-training configuration.",
    )
    return p.parse_args()


def get_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading — Adamson 2016
# ---------------------------------------------------------------------------

def _load_adamson_adata():
    """
    Load the Adamson 2016 Perturb-seq AnnData object.

    Tries three methods in order:
      1. pertpy (if importable without JAX conflicts)
      2. Direct .h5ad download from figshare (no JAX dependency)
      3. Local file 'adamson2016.h5ad' if already present
    """
    import anndata as ad

    local_path = "adamson2016.h5ad"

    # Method 1: try pertpy (may fail due to JAX conflicts on Colab)
    try:
        import pertpy as pt
        return pt.dt.adamson_2016_upr_perturb_seq()
    except Exception as e:
        print(f"  pertpy import failed ({type(e).__name__}: {e}) — falling back to direct download")

    # Method 2: load from local cache or download
    if os.path.exists(local_path):
        # Validate the file is a real HDF5 file (not a downloaded HTML error page)
        with open(local_path, "rb") as fh:
            magic = fh.read(8)
        if magic[:4] != b"\x89HDF":
            print(f"  Cached file {local_path} is corrupt — removing and re-downloading")
            os.remove(local_path)

    if not os.path.exists(local_path):
        # Zenodo scPerturb — direct download, no auth required
        # AdamsonWeissman2016_GSM2406681_10X010 is the full UPR Perturb-seq dataset
        url = "https://zenodo.org/records/13350497/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1"
        print(f"  Downloading Adamson 2016 h5ad from Zenodo (~470 MB) …")
        import requests
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(local_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"    {downloaded/1e6:.0f}/{total/1e6:.0f} MB", end="\r")
        print(f"\n  Saved to {local_path}")

    return ad.read_h5ad(local_path)


def load_adamson(n_hvg: int = 2000, smoke_test: bool = False):
    """
    Load the Adamson 2016 Perturb-seq dataset via pertpy.

    Returns:
        ctrl_matrix    : (N, G) float32 — per-cell control (mean of all control cells,
                          broadcast to each perturbed cell's row)
        pert_matrix    : (N, G) float32 — per-cell post-perturbation expression
        pert_ids_cell  : (N,) int32 — perturbation vocab index per cell (0 = control)
        conditions     : (N,) str — condition label per cell
        pert_vocab     : list[str] — perturbation vocab (index 0 = "ctrl")
        gene_names     : list[str]
    """
    print("Loading Adamson 2016 Perturb-seq dataset …")
    adata = _load_adamson_adata()
    print(f"  Raw: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  obs columns: {list(adata.obs.columns)}")

    # Identify the perturbation condition column.
    # pertpy Adamson uses 'perturbation' (newer) or 'condition' (older).
    cond_col = None
    for col in ["perturbation", "condition", "gene_target", "guide_id"]:
        if col in adata.obs.columns:
            cond_col = col
            break
    if cond_col is None:
        raise ValueError(
            f"Cannot find perturbation column. obs columns: {list(adata.obs.columns)}"
        )
    print(f"  Using obs column '{cond_col}' as perturbation identity")

    conditions_raw = adata.obs[cond_col].astype(str).values

    # Drop cells with missing perturbation label
    valid_mask = conditions_raw != "nan"
    if (~valid_mask).sum() > 0:
        print(f"  Dropping {(~valid_mask).sum()} cells with missing perturbation label")
        adata = adata[valid_mask].copy()
        conditions_raw = conditions_raw[valid_mask]

    unique_vals, val_counts = np.unique(conditions_raw, return_counts=True)
    print(f"  Unique condition values (top 10 by count):")
    top10 = np.argsort(val_counts)[::-1][:10]
    for i in top10:
        print(f"    '{unique_vals[i]}': {val_counts[i]} cells")

    # Identify control label.
    # pertpy Adamson typically encodes controls as "ctrl" or the most frequent value.
    # We match case-insensitively and also check for the GEARS-style "+ctrl" suffix pattern.
    ctrl_label = None
    ctrl_candidates = ["ctrl", "control", "Control", "CTRL", "non-targeting", "NT",
                       "non_targeting", "scramble", "Scramble", "AAMP+ctrl"]
    for candidate in ctrl_candidates:
        if candidate in conditions_raw:
            ctrl_label = candidate
            break

    if ctrl_label is None:
        # Check for any value that contains 'ctrl' alone (e.g. exact match only)
        for v in unique_vals:
            if v.lower() == "ctrl":
                ctrl_label = v
                break

    if ctrl_label is None:
        # Fall back: most frequent label
        ctrl_label = unique_vals[np.argmax(val_counts)]
        print(f"  Warning: no standard ctrl label found — using most frequent: '{ctrl_label}'")

    print(f"  Control label: '{ctrl_label}'  ({(conditions_raw == ctrl_label).sum()} cells)")

    # Normalise and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"  After HVG selection: {adata.n_obs} cells × {adata.n_vars} genes")

    # Dense matrix — extract AFTER subsetting columns, keeping all rows
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    gene_names = list(adata.var_names)
    # conditions stays aligned with X rows (obs never changed)
    conditions = conditions_raw

    # Persist gene list so pretrain_kidney.py can align kidney data
    import json
    gene_list_path = "adamson_genes.json"
    with open(gene_list_path, "w") as f:
        json.dump(gene_names, f)
    print(f"  Gene list saved to {gene_list_path}")

    # Sanity check: ctrl cells must exist
    ctrl_mask_check = conditions == ctrl_label
    if ctrl_mask_check.sum() == 0:
        raise RuntimeError(
            f"Control label '{ctrl_label}' matched 0 cells. "
            f"Unique values: {list(unique_vals[:20])}"
        )

    # Perturbation vocabulary: 0 = ctrl, 1..K = unique perturbations
    unique_perts = sorted(set(conditions) - {ctrl_label})
    pert_vocab = [ctrl_label] + unique_perts       # index 0 = ctrl
    pert2idx = {p: i for i, p in enumerate(pert_vocab)}
    pert_ids_cell = np.array([pert2idx[c] for c in conditions], dtype=np.int32)

    if smoke_test:
        # Keep only 3 perturbation conditions + ctrl for speed
        smoke_perts = unique_perts[:3]
        keep_mask = np.isin(conditions, smoke_perts + [ctrl_label])
        X = X[keep_mask]
        conditions = conditions[keep_mask]
        pert_ids_cell = pert_ids_cell[keep_mask]
        # Remap vocab to the subset
        unique_perts = smoke_perts
        pert_vocab = [ctrl_label] + unique_perts
        pert2idx = {p: i for i, p in enumerate(pert_vocab)}
        pert_ids_cell = np.array([pert2idx[c] for c in conditions], dtype=np.int32)
        # Subsample to at most 500 cells total
        if len(X) > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), 500, replace=False)
            X = X[idx]
            conditions = conditions[idx]
            pert_ids_cell = pert_ids_cell[idx]
        print(f"  [smoke test] {len(X)} cells, {len(pert_vocab)} perturbations")

    # Mean control expression — used as the "ctrl_matrix" for every cell
    ctrl_mask = conditions == ctrl_label
    assert ctrl_mask.sum() > 0, f"Bug: ctrl_mask empty after filtering (ctrl_label='{ctrl_label}')"
    ctrl_mean = X[ctrl_mask].mean(axis=0)          # (G,)
    ctrl_matrix = np.tile(ctrl_mean, (len(X), 1))  # (N, G)

    print(f"  {len(X)} cells, {len(pert_vocab)} perturbation vocab entries "
          f"({len(unique_perts)} unique perturbations + ctrl)")
    print(f"  ctrl_mean range: [{ctrl_mean.min():.3f}, {ctrl_mean.max():.3f}]")

    return ctrl_matrix, X, pert_ids_cell, conditions, pert_vocab, gene_names, ctrl_label


# ---------------------------------------------------------------------------
# Train + evaluate one condition
# ---------------------------------------------------------------------------

def run_condition(
    label: str,
    predict_delta: bool,
    include_jepa: bool,
    train_dataset: PerturbationDataset,
    test_dataset: PerturbationDataset,
    test_conditions: np.ndarray,
    vocab_size: int,
    n_perturbations: int,
    args: argparse.Namespace,
    device: torch.device,
    ctrl_label: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Condition: {label}")
    print(f"  predict_delta={predict_delta}  include_jepa={include_jepa}")
    print(f"{'='*60}")

    use_large = bool(args.pretrain_checkpoint) and not args.smoke_test
    d_model   = 512  if use_large else (128 if args.smoke_test else 256)
    n_layers  = 12   if use_large else (4   if args.smoke_test else 6)
    n_heads   = 8    if use_large else 4
    ffn_dim   = 2048 if use_large else (512 if args.smoke_test else 1024)
    p_hidden  = 512  if use_large else (128 if args.smoke_test else 256)

    model = CellJEPA(
        vocab_size=vocab_size,
        n_bins=50,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        dropout=0.1,
        ema_momentum=0.996,
        predictor_hidden=p_hidden,
        n_perturbations=n_perturbations,
        predict_delta=predict_delta,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CellJEPA: {n_params/1e6:.1f}M parameters (d_model={d_model}, n_layers={n_layers})")

    if args.pretrain_checkpoint:
        ckpt = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=False)
        pretrain_state = ckpt["model_state"]
        model_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in pretrain_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f"  Loaded {loaded} layers from pretrained checkpoint ({skipped} skipped — shape mismatch or pert-specific)")

    config = PerturbationConfig(
        lr=1e-4,
        lr_decay=0.9,
        batch_size=args.batch_size,
        num_workers=0,
        n_epochs=1 if args.smoke_test else args.n_epochs,
        grad_clip=1.0,
        log_every=20,
        w_pert_rec=1.0,
        w_jepa_pert=1.0,
        w_ecs=0.8,
        include_jepa=include_jepa,
        predict_delta=predict_delta,
    )

    trainer = PerturbationTrainer(model, train_dataset, config=config, device=device)
    trainer.train()

    print(f"\n  Evaluating {label} …")
    results = evaluate_perturbation(
        model=model,
        dataset=test_dataset,
        conditions=test_conditions,
        ctrl_label=ctrl_label,
        n_bins=50,
        batch_size=args.batch_size,
        device=device,
        predict_delta=predict_delta,
    )

    print(f"  Mean Pearson:            {results['mean_pearson']:.4f}")
    print(f"  Mean Pearson Δ:          {results['mean_pearson_delta']:.4f}")
    print(f"  Top-20 DEG Pearson Δ:    {results['mean_top20_deg_pearson_delta']:.4f}")
    print(f"  Mean MSE:                {results['mean_mse']:.6f}")

    if results['mean_pearson'] > 0.9:
        raise ValueError(
            f"Sanity check failed: mean Pearson = {results['mean_pearson']:.4f} > 0.9. "
            "This likely indicates an evaluation bug (e.g. zero-inflation or misaligned "
            "gene vectors). Check perturb_metrics.py and re-upload before re-running."
        )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save(all_results: dict, results_file: str) -> None:
    col_w = 36
    header = f"{'Condition':<{col_w}} {'Pearson':>8} {'PearsonΔ':>10} {'Top20Δ':>8} {'MSE':>10}"
    sep    = "-" * len(header)
    lines  = [
        "",
        "=" * len(header),
        "  CellJEPA Perturbation Prediction — Adamson 2016",
        "=" * len(header),
        header,
        sep,
    ]
    for label, res in all_results.items():
        lines.append(
            f"{label:<{col_w}} "
            f"{res['mean_pearson']:>8.4f} "
            f"{res['mean_pearson_delta']:>10.4f} "
            f"{res['mean_top20_deg_pearson_delta']:>8.4f} "
            f"{res['mean_mse']:>10.6f}"
        )
    lines.append("=" * len(header))
    output = "\n".join(lines)
    print(output)
    with open(results_file, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {results_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.smoke_test:
        print("\n[SMOKE TEST: 3 perturbations, 1 epoch, 500 cells]\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    ctrl_matrix, pert_matrix, pert_ids_cell, conditions, pert_vocab, _, ctrl_label = \
        load_adamson(n_hvg=args.n_hvg, smoke_test=args.smoke_test)

    n_genes = ctrl_matrix.shape[1]
    n_perturbations = len(pert_vocab)

    # Gene vocab: 0=<cls>, 1=<pad>, 2..n_genes+1=genes
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    vocab_size = n_genes + 2

    # ------------------------------------------------------------------
    # Train / test split by perturbation condition
    # ------------------------------------------------------------------
    unique_perts = [p for p in pert_vocab if p != ctrl_label]
    rng = np.random.default_rng(42)
    n_test = max(1, int(len(unique_perts) * args.test_fraction))
    test_perts  = set(rng.choice(unique_perts, n_test, replace=False).tolist())
    train_perts = set(unique_perts) - test_perts

    # Control cells go into both train and test
    train_mask = np.array(
        [(c in train_perts or c == ctrl_label) for c in conditions]
    )
    test_mask = np.array(
        [(c in test_perts  or c == ctrl_label) for c in conditions]
    )

    print(f"\nTrain: {train_mask.sum()} cells ({len(train_perts)} perturbations + ctrl)")
    print(f"Test:  {test_mask.sum()} cells ({len(test_perts)} perturbations + ctrl)")

    train_ds = PerturbationDataset(
        ctrl_matrix[train_mask],
        pert_matrix[train_mask],
        pert_ids_cell[train_mask],
        gene_vocab=gene_vocab,
        n_bins=50,
        L_max=args.l_max,
        cls_token_id=0,
        pad_token_id=1,
    )
    test_ds = PerturbationDataset(
        ctrl_matrix[test_mask],
        pert_matrix[test_mask],
        pert_ids_cell[test_mask],
        gene_vocab=gene_vocab,
        n_bins=50,
        L_max=args.l_max,
        cls_token_id=0,
        pad_token_id=1,
    )
    test_conditions = conditions[test_mask]

    # ------------------------------------------------------------------
    # 2×2 ablation
    # ------------------------------------------------------------------
    ablations = [
        ("Absolute + JEPA on  (paper default)", False, True),
        ("Absolute + JEPA off",                 False, False),
        ("Delta    + JEPA on",                  True,  True),
        ("Delta    + JEPA off",                 True,  False),
    ]

    all_results = {}
    for label, predict_delta, include_jepa in ablations:
        all_results[label] = run_condition(
            label=label,
            predict_delta=predict_delta,
            include_jepa=include_jepa,
            train_dataset=train_ds,
            test_dataset=test_ds,
            test_conditions=test_conditions,
            vocab_size=vocab_size,
            n_perturbations=n_perturbations,
            args=args,
            device=device,
            ctrl_label=ctrl_label,
        )

    print_and_save(all_results, args.results_file)


if __name__ == "__main__":
    main()

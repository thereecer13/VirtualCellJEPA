"""
compare_traj_modes.py
=====================
3-mode comparison of CellJEPA perturbation strategies on Adamson 2016.

Modes compared:
  A) Absolute   — direct post-perturbation expression prediction (forward_perturb)
  B) Delta      — predict Δ = pert − ctrl per gene (forward_perturb_delta)
  C) Trajectory — decoupled latent trajectory predictor (forward_perturb_traj)

All three modes are fine-tuned from the same pre-trained checkpoint and evaluated
on a held-out 20% perturbation split using Pearson r, Pearson Δ, Top-20 DEG Pearson Δ,
and MSE.

Usage (Colab A100):
    python compare_traj_modes.py --pretrain_checkpoint /path/to/checkpoint.pt --device cuda

Smoke test (CPU, ~2 min):
    python compare_traj_modes.py --smoke_test --device cpu
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
    p = argparse.ArgumentParser(description="CellJEPA 3-mode perturbation comparison")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--n_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--l_max", type=int, default=200)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--results_file", default="results_traj_modes.txt")
    p.add_argument("--modes", default="absolute,delta,traj",
                   help="Comma-separated subset of modes to run: absolute,delta,traj")
    p.add_argument("--smoke_test", action="store_true",
                   help="3 conditions, 1 epoch, 500 cells")
    p.add_argument(
        "--pretrain_checkpoint", default=None,
        help="Path to pre-trained CellJEPA checkpoint (.pt). "
             "Backbone weights are loaded; perturbation-specific heads are left at random init.",
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
# Data loading (reuses compare_perturbation.py logic)
# ---------------------------------------------------------------------------

def _load_adamson_adata():
    import anndata as ad

    local_path = "adamson2016.h5ad"

    try:
        import pertpy as pt
        return pt.dt.adamson_2016_upr_perturb_seq()
    except Exception as e:
        print(f"  pertpy failed ({type(e).__name__}: {e}) — falling back to direct download")

    if os.path.exists(local_path):
        with open(local_path, "rb") as fh:
            magic = fh.read(8)
        if magic[:4] != b"\x89HDF":
            print(f"  Cached file corrupt — removing and re-downloading")
            os.remove(local_path)

    if not os.path.exists(local_path):
        url = "https://zenodo.org/records/13350497/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1"
        print(f"  Downloading Adamson 2016 from Zenodo (~470 MB) …")
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
    print("Loading Adamson 2016 Perturb-seq …")
    adata = _load_adamson_adata()
    print(f"  Raw: {adata.n_obs} cells × {adata.n_vars} genes")

    cond_col = None
    for col in ["perturbation", "condition", "gene_target", "guide_id"]:
        if col in adata.obs.columns:
            cond_col = col
            break
    if cond_col is None:
        raise ValueError(f"Cannot find perturbation column. obs: {list(adata.obs.columns)}")
    print(f"  Using obs column '{cond_col}'")

    conditions_raw = adata.obs[cond_col].astype(str).values
    valid_mask = conditions_raw != "nan"
    if (~valid_mask).sum() > 0:
        adata = adata[valid_mask].copy()
        conditions_raw = conditions_raw[valid_mask]

    unique_vals, val_counts = np.unique(conditions_raw, return_counts=True)
    ctrl_label = None
    for candidate in ["ctrl", "control", "Control", "CTRL", "non-targeting", "NT"]:
        if candidate in conditions_raw:
            ctrl_label = candidate
            break
    if ctrl_label is None:
        for v in unique_vals:
            if v.lower() == "ctrl":
                ctrl_label = v
                break
    if ctrl_label is None:
        ctrl_label = unique_vals[np.argmax(val_counts)]
        print(f"  Warning: using most-frequent as ctrl: '{ctrl_label}'")

    print(f"  Control label: '{ctrl_label}'  ({(conditions_raw == ctrl_label).sum()} cells)")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"  After HVG: {adata.n_obs} cells × {adata.n_vars} genes")

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    conditions = conditions_raw
    gene_names = list(adata.var_names)

    unique_perts = sorted(set(conditions) - {ctrl_label})
    pert_vocab = [ctrl_label] + unique_perts
    pert2idx = {p: i for i, p in enumerate(pert_vocab)}
    pert_ids_cell = np.array([pert2idx[c] for c in conditions], dtype=np.int32)

    if smoke_test:
        smoke_perts = unique_perts[:3]
        keep_mask = np.isin(conditions, smoke_perts + [ctrl_label])
        X, conditions, pert_ids_cell = X[keep_mask], conditions[keep_mask], pert_ids_cell[keep_mask]
        unique_perts = smoke_perts
        pert_vocab = [ctrl_label] + unique_perts
        pert2idx = {p: i for i, p in enumerate(pert_vocab)}
        pert_ids_cell = np.array([pert2idx[c] for c in conditions], dtype=np.int32)
        if len(X) > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), 500, replace=False)
            X, conditions, pert_ids_cell = X[idx], conditions[idx], pert_ids_cell[idx]
        print(f"  [smoke test] {len(X)} cells, {len(pert_vocab)} perturbations")

    ctrl_mask = conditions == ctrl_label
    ctrl_mean = X[ctrl_mask].mean(axis=0)
    ctrl_matrix = np.tile(ctrl_mean, (len(X), 1))

    print(f"  {len(X)} cells, {len(pert_vocab)} pert vocab entries")
    return ctrl_matrix, X, pert_ids_cell, conditions, pert_vocab, gene_names, ctrl_label


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(vocab_size, n_perturbations, args):
    use_large = bool(args.pretrain_checkpoint) and not args.smoke_test
    d_model  = 512  if use_large else (128 if args.smoke_test else 256)
    n_layers = 12   if use_large else (4   if args.smoke_test else 6)
    n_heads  = 8    if use_large else 4
    ffn_dim  = 2048 if use_large else (512 if args.smoke_test else 1024)
    p_hidden = 512  if use_large else (128 if args.smoke_test else 256)

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
        predict_delta=True,  # always build with delta head; unused in non-delta modes
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CellJEPA: {n_params/1e6:.1f}M params (d_model={d_model}, layers={n_layers})")

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
        print(f"  Loaded {loaded} layers from checkpoint ({skipped} skipped)")

    return model


# ---------------------------------------------------------------------------
# Run one mode
# ---------------------------------------------------------------------------

def run_mode(
    label: str,
    mode: str,          # "absolute" | "delta" | "traj"
    train_dataset,
    test_dataset,
    test_conditions,
    vocab_size,
    n_perturbations,
    args,
    device,
    ctrl_label,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {label}")
    print(f"{'='*60}")

    model = build_model(vocab_size, n_perturbations, args)
    model = model.to(device)

    config = PerturbationConfig(
        lr=1e-4,
        lr_decay=0.9,
        batch_size=args.batch_size,
        num_workers=0,
        n_epochs=1 if args.smoke_test else args.n_epochs,
        grad_clip=1.0,
        log_every=20,
        w_pert_rec=1.0,
        w_ecs=0.8,
        include_jepa=(mode != "traj"),   # traj uses delta loss instead of JEPA pert
        predict_delta=(mode == "delta"),
        predict_traj=(mode == "traj"),
        w_delta=1.0,
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
        predict_delta=(mode == "delta"),
        predict_traj=(mode == "traj"),
    )

    print(f"  Pearson:           {results['mean_pearson']:.4f}")
    print(f"  Pearson Δ:         {results['mean_pearson_delta']:.4f}")
    print(f"  Top-20 DEG Δ:      {results['mean_top20_deg_pearson_delta']:.4f}")
    print(f"  MSE:               {results['mean_mse']:.6f}")
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save(all_results: dict, results_file: str) -> None:
    col_w = 28
    header = f"{'Mode':<{col_w}} {'Pearson':>8} {'PearsonΔ':>10} {'Top20Δ':>8} {'MSE':>10}"
    sep    = "-" * len(header)
    lines  = [
        "",
        "=" * len(header),
        "  CellJEPA Perturbation Mode Comparison — Adamson 2016",
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

    ctrl_matrix, pert_matrix, pert_ids_cell, conditions, pert_vocab, _, ctrl_label = \
        load_adamson(n_hvg=args.n_hvg, smoke_test=args.smoke_test)

    n_genes = ctrl_matrix.shape[1]
    n_perturbations = len(pert_vocab)
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    vocab_size = n_genes + 2

    unique_perts = [p for p in pert_vocab if p != ctrl_label]
    rng = np.random.default_rng(42)
    n_test = max(1, int(len(unique_perts) * args.test_fraction))
    test_perts  = set(rng.choice(unique_perts, n_test, replace=False).tolist())
    train_perts = set(unique_perts) - test_perts

    train_mask = np.array([(c in train_perts or c == ctrl_label) for c in conditions])
    test_mask  = np.array([(c in test_perts  or c == ctrl_label) for c in conditions])
    print(f"\nTrain: {train_mask.sum()} cells ({len(train_perts)} perts + ctrl)")
    print(f"Test:  {test_mask.sum()} cells ({len(test_perts)} perts + ctrl)")

    ds_kwargs = dict(
        gene_vocab=gene_vocab, n_bins=50, L_max=args.l_max, cls_token_id=0, pad_token_id=1,
    )
    train_ds = PerturbationDataset(
        ctrl_matrix[train_mask], pert_matrix[train_mask], pert_ids_cell[train_mask], **ds_kwargs
    )
    test_ds = PerturbationDataset(
        ctrl_matrix[test_mask], pert_matrix[test_mask], pert_ids_cell[test_mask], **ds_kwargs
    )
    test_conditions = conditions[test_mask]

    requested = {m.strip().lower() for m in args.modes.split(",")}
    all_modes = [
        ("Absolute",    "absolute"),
        ("Delta",       "delta"),
        ("Trajectory",  "traj"),
    ]
    modes = [(label, mode) for label, mode in all_modes if mode in requested]
    if not modes:
        raise ValueError(f"No valid modes in --modes '{args.modes}'. Choose from: absolute, delta, traj")
    print(f"\nRunning modes: {[label for label, _ in modes]}")

    all_results = {}
    for label, mode in modes:
        all_results[label] = run_mode(
            label=label,
            mode=mode,
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

"""
run_perturbation_sigreg.py — SIGReg vs EMA × Absolute vs Delta Perturbation Ablation
======================================================================================
4-condition ablation on the Adamson 2016 Perturb-seq dataset comparing:

  1. EMA  + Absolute  — Cell-JEPA with absolute reconstruction (paper baseline)
  2. EMA  + Delta     — Cell-JEPA with delta (Δ = pert − ctrl) prediction
  3. SIGReg + Absolute — CellJEPA_SIGReg with absolute reconstruction
  4. SIGReg + Delta   — CellJEPA_SIGReg with delta prediction (primary novel condition)

Hypothesis: predicting in embedding space (JEPA) suppresses small deviations that
delta metrics capture. Predicting Δ directly in expression space should fix this.
SIGReg pre-training may further help by producing a better-organised embedding space.

Usage:
    python run_perturbation_sigreg.py --device cuda
    python run_perturbation_sigreg.py --smoke_test --device cpu
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from cell_jepa import CellJEPA
from cell_sigreg import CellJEPA_SIGReg
from preprocessing import PerturbationDataset
from trainer import (
    PerturbationTrainer, PerturbationConfig,
    SIGRegPerturbationTrainer, SIGRegPerturbationConfig,
)
from perturb_metrics import evaluate_perturbation
from compare_perturbation import load_adamson, print_and_save


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIGReg vs EMA × absolute vs delta perturbation ablation"
    )
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--n_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--l_max", type=int, default=200)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--results_file", default="results_perturbation_sigreg.txt")
    p.add_argument("--smoke_test", action="store_true",
                   help="3 conditions, 1 epoch, 500 cells")
    return p.parse_args()


def get_device(s: str | None) -> torch.device:
    if s:
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# EMA condition runner (reuses CellJEPA + PerturbationTrainer)
# ---------------------------------------------------------------------------

def run_ema_condition(
    label: str,
    predict_delta: bool,
    train_dataset: PerturbationDataset,
    test_dataset: PerturbationDataset,
    test_conditions: np.ndarray,
    vocab_size: int,
    n_perturbations: int,
    args: argparse.Namespace,
    device: torch.device,
    ctrl_label: str,
) -> dict:
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  model=CellJEPA  predict_delta={predict_delta}  include_jepa=True")
    print(f"{'='*64}")

    d_model  = 128 if args.smoke_test else 256
    n_layers = 4   if args.smoke_test else 6
    n_heads  = 4
    ffn_dim  = 512 if args.smoke_test else 1024
    p_hidden = 128 if args.smoke_test else 256

    model = CellJEPA(
        vocab_size=vocab_size, n_bins=50,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        ffn_dim=ffn_dim, dropout=0.1, ema_momentum=0.996,
        predictor_hidden=p_hidden,
        n_perturbations=n_perturbations,
        predict_delta=predict_delta,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CellJEPA: {n_params/1e6:.1f}M params")

    config = PerturbationConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=args.batch_size, num_workers=0,
        n_epochs=1 if args.smoke_test else args.n_epochs,
        grad_clip=1.0, log_every=20,
        w_pert_rec=1.0, w_jepa_pert=1.0, w_ecs=0.8,
        include_jepa=True, predict_delta=predict_delta,
    )
    PerturbationTrainer(model, train_dataset, config=config, device=device).train()

    results = evaluate_perturbation(
        model=model, dataset=test_dataset, conditions=test_conditions,
        ctrl_label=ctrl_label, n_bins=50,
        batch_size=args.batch_size, device=device, predict_delta=predict_delta,
    )
    _print_metrics(results)
    return results


# ---------------------------------------------------------------------------
# SIGReg condition runner (CellJEPA_SIGReg + SIGRegPerturbationTrainer)
# ---------------------------------------------------------------------------

def run_sigreg_condition(
    label: str,
    predict_delta: bool,
    train_dataset: PerturbationDataset,
    test_dataset: PerturbationDataset,
    test_conditions: np.ndarray,
    vocab_size: int,
    n_perturbations: int,
    args: argparse.Namespace,
    device: torch.device,
    ctrl_label: str,
) -> dict:
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  model=CellJEPA_SIGReg  predict_delta={predict_delta}")
    print(f"{'='*64}")

    d_model  = 128 if args.smoke_test else 256
    n_layers = 4   if args.smoke_test else 6
    n_heads  = 4
    ffn_dim  = 512 if args.smoke_test else 1024

    model = CellJEPA_SIGReg(
        vocab_size=vocab_size, n_bins=50,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        ffn_dim=ffn_dim, dropout=0.1,
        n_perturbations=n_perturbations,
        predict_delta=predict_delta,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CellJEPA_SIGReg: {n_params/1e6:.1f}M params")

    config = SIGRegPerturbationConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=args.batch_size, num_workers=0,
        n_epochs=1 if args.smoke_test else args.n_epochs,
        grad_clip=1.0, log_every=20,
        w_pert_rec=1.0, w_ecs=0.8,
        predict_delta=predict_delta,
    )
    SIGRegPerturbationTrainer(model, train_dataset, config=config, device=device).train()

    results = evaluate_perturbation(
        model=model, dataset=test_dataset, conditions=test_conditions,
        ctrl_label=ctrl_label, n_bins=50,
        batch_size=args.batch_size, device=device, predict_delta=predict_delta,
    )
    _print_metrics(results)
    return results


def _print_metrics(results: dict) -> None:
    print(f"  Mean Pearson:            {results['mean_pearson']:.4f}")
    print(f"  Mean Pearson Δ:          {results['mean_pearson_delta']:.4f}")
    print(f"  Top-20 DEG Pearson Δ:    {results['mean_top20_deg_pearson_delta']:.4f}")
    print(f"  Mean MSE:                {results['mean_mse']:.6f}")


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
    print(f"\nTrain: {train_mask.sum()} cells ({len(train_perts)} perturbations + ctrl)")
    print(f"Test:  {test_mask.sum()} cells ({len(test_perts)} perturbations + ctrl)")

    ds_kwargs = dict(gene_vocab=gene_vocab, n_bins=50, L_max=args.l_max,
                     cls_token_id=0, pad_token_id=1)
    train_ds = PerturbationDataset(
        ctrl_matrix[train_mask], pert_matrix[train_mask],
        pert_ids_cell[train_mask], **ds_kwargs,
    )
    test_ds = PerturbationDataset(
        ctrl_matrix[test_mask], pert_matrix[test_mask],
        pert_ids_cell[test_mask], **ds_kwargs,
    )
    test_conditions = conditions[test_mask]

    shared = dict(
        train_dataset=train_ds, test_dataset=test_ds,
        test_conditions=test_conditions,
        vocab_size=vocab_size, n_perturbations=n_perturbations,
        args=args, device=device, ctrl_label=ctrl_label,
    )

    all_results = {}

    all_results["EMA + Absolute  (paper baseline)"] = run_ema_condition(
        "EMA + Absolute  (paper baseline)", predict_delta=False, **shared)

    all_results["EMA + Delta"] = run_ema_condition(
        "EMA + Delta", predict_delta=True, **shared)

    all_results["SIGReg + Absolute"] = run_sigreg_condition(
        "SIGReg + Absolute", predict_delta=False, **shared)

    all_results["SIGReg + Delta    (primary novel)"] = run_sigreg_condition(
        "SIGReg + Delta    (primary novel)", predict_delta=True, **shared)

    print_and_save(all_results, args.results_file)


if __name__ == "__main__":
    main()

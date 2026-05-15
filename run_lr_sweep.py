"""
run_lr_sweep.py — LR Sweep for SIGReg Kidney Fine-Tuning
=========================================================
Tests whether the SIGReg (kidney) fine-tuning collapse (AvgBIO 0.387 → 0.307)
is catastrophic forgetting (fixable with lower LR) or a fundamental geometry
mismatch between kidney and PBMC representations.

Runs SIGReg (kidney) fine-tuning at lr_scale × 1e-4 for each scale in
--lr_scales. The SIGReg (scratch) baseline (lr_scale=1.0) is always included
for reference.

Interpretation guide:
  - If low LR recovers AvgBIO toward scratch fine-tuned (~0.778):
    → Catastrophic forgetting. The kidney representations are useful but
      get overwritten too fast. Fix: lower LR, possibly with warm-up.
  - If low LR still collapses (AvgBIO stays near 0.3):
    → Geometry mismatch. The kidney embedding space is incompatible with
      the PBMC ECS/GEPC objectives regardless of update step size. This is
      a more fundamental and scientifically interesting finding.

Usage:
    python run_lr_sweep.py \
        --sigreg_checkpoint /path/kidney_sigreg_final.pt \
        --sigreg_genes      /path/kidney_sigreg_gene_names.json \
        --device cuda \
        --lr_scales 0.1 0.3 1.0

    # Smoke test:
    python run_lr_sweep.py --smoke_test --device cpu \
        --sigreg_checkpoint /path/... --sigreg_genes /path/...
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from compare_pbmc3k import load_pbmc3k, build_vocab
from run_ablation import _train_test_split
from run_transfer import run_sigreg_scratch, run_sigreg_kidney


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LR sweep for SIGReg kidney fine-tuning collapse diagnosis"
    )
    p.add_argument("--lr_scales", type=float, nargs="+", default=[0.1, 0.3, 1.0],
                   help="Fine-tuning LR multipliers applied to base LR 1e-4 (default: 0.1 0.3 1.0)")
    p.add_argument("--smoke_test", action="store_true",
                   help="200 cells, 1 epoch — quick correctness check")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--results_file", default="results_lr_sweep.txt")
    p.add_argument("--pretrain_epochs", type=int, default=4)
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--l_max", type=int, default=200)
    p.add_argument("--sigreg_checkpoint", default=None, required=False,
                   help="Path to kidney SIGReg checkpoint (.pt)")
    p.add_argument("--sigreg_genes", default=None, required=False,
                   help="Path to kidney SIGReg gene names JSON")
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
# Output
# ---------------------------------------------------------------------------

BASE_LR = 1e-4

def print_and_save_lr_sweep(results: dict, path: str) -> None:
    row = "  {:<36s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}"
    sep = "  " + "-" * 66

    lines = ["\n" + "=" * 70]
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        lines += [
            f"  {phase_label} — SIGReg Kidney Fine-Tuning LR Sweep",
            "=" * 70,
            row.format("Condition", "NMI", "ARI", "ASW", "AvgBIO"),
            sep,
        ]
        for name, cond in results.items():
            res = cond.get(phase_key)
            if not res:
                lines.append(row.format(name, "N/A", "N/A", "N/A", "N/A"))
            else:
                lines.append(row.format(
                    name,
                    f"{res['nmi']:.4f}",
                    f"{res['ari']:.4f}",
                    f"{res['asw']:.4f}",
                    f"{res['avg_bio']:.4f}",
                ))
        lines.append("=" * 70 + "\n")

    output = "\n".join(lines)
    print(output)
    with open(path, "w") as f:
        f.write(output + "\n")
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.smoke_test:
        print("\n[SMOKE TEST: 200 cells, 1 epoch each]\n")
        args.pretrain_epochs = 1
        args.finetune_epochs = 1
        args.l_max = 64

    np.random.seed(42)
    torch.manual_seed(42)

    n_subset = 200 if args.smoke_test else None
    count_matrix, int_labels, label_names, adata_raw, gene_names = load_pbmc3k(n_subset)
    gene_vocab, vocab_size, _, _ = build_vocab(count_matrix.shape[1])

    X_train, y_train, X_test, y_test = _train_test_split(count_matrix, int_labels, seed=42)
    print(f"Train: {X_train.shape[0]} cells  |  Test: {X_test.shape[0]} cells")

    all_results: dict[str, dict] = {}

    # Scratch baseline (single run at lr_scale=1.0 for reference)
    print(f"\n{'='*70}")
    print("  SIGReg (scratch) — baseline")
    print("=" * 70)
    all_results["SIGReg (scratch, lr=1e-4)"] = run_sigreg_scratch(
        X_train, y_train, X_test, y_test,
        gene_vocab, vocab_size, args, device,
        seed=42, lr_scale=1.0,
    )

    # Kidney conditions at each LR scale
    for lr_scale in args.lr_scales:
        lr_actual = BASE_LR * lr_scale
        label = f"SIGReg (kidney, lr={lr_actual:.1e})"
        print(f"\n{'='*70}")
        print(f"  {label}")
        print("=" * 70)
        all_results[label] = run_sigreg_kidney(
            X_train, y_train, X_test, y_test,
            gene_names, args, device,
            seed=42, lr_scale=lr_scale,
        )

    print_and_save_lr_sweep(all_results, args.results_file)


if __name__ == "__main__":
    main()

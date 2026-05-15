"""
run_multiseed.py — Multi-Seed Robustness for SIGReg Conditions
==============================================================
Runs SIGReg (scratch) fine-tuned and SIGReg (kidney) fine-tuned across
multiple random seeds, reporting per-seed metrics and mean ± std.

Addresses the single-seed limitation of the original transfer experiment.
Only the two SIGReg fine-tuned conditions are run (the ones that carry the
main narrative: scratch best fine-tuned overall, kidney collapse).

Usage:
    python run_multiseed.py \
        --sigreg_checkpoint /path/kidney_sigreg_final.pt \
        --sigreg_genes      /path/kidney_sigreg_gene_names.json \
        --device cuda \
        --seeds 42 123 999

    # Smoke test:
    python run_multiseed.py --smoke_test --device cpu \
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
        description="Multi-seed robustness for SIGReg scratch and kidney fine-tuning"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999],
                   help="Random seeds to run (default: 42 123 999)")
    p.add_argument("--smoke_test", action="store_true",
                   help="200 cells, 1 epoch — quick correctness check")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--results_file", default="results_multiseed.txt")
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

def print_and_save_multiseed(
    per_seed: dict[str, dict],   # {label: {zero_shot: {...}, fine_tuned: {...}}}
    seeds: list[int],
    path: str,
) -> None:
    metrics = ["nmi", "ari", "asw", "avg_bio"]
    row  = "  {:<36s}  {:>7s}  {:>7s}  {:>7s}  {:>8s}"
    sep  = "  " + "-" * 70

    lines = ["\n" + "=" * 74]
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        lines += [
            f"  {phase_label} — Multi-Seed Robustness (SIGReg)",
            "=" * 74,
            row.format("Condition", "NMI", "ARI", "ASW", "AvgBIO"),
            sep,
        ]

        # Collect per-condition per-seed values
        conditions = ["SIGReg (scratch)", "SIGReg (kidney)"]
        for cond in conditions:
            seed_vals: dict[str, list[float]] = {m: [] for m in metrics}
            for seed in seeds:
                label = f"{cond} seed={seed}"
                res = per_seed.get(label, {}).get(phase_key)
                if res:
                    for m in metrics:
                        seed_vals[m].append(res[m])
                    lines.append(row.format(
                        label,
                        f"{res['nmi']:.4f}",
                        f"{res['ari']:.4f}",
                        f"{res['asw']:.4f}",
                        f"{res['avg_bio']:.4f}",
                    ))
                else:
                    lines.append(row.format(label, "N/A", "N/A", "N/A", "N/A"))

            # Summary row
            if all(len(v) == len(seeds) for v in seed_vals.values()):
                summary_parts = []
                for m in metrics:
                    mu = np.mean(seed_vals[m])
                    sd = np.std(seed_vals[m])
                    summary_parts.append(f"{mu:.3f}±{sd:.3f}")
                lines.append(row.format(
                    f"  {cond} MEAN±STD",
                    summary_parts[0], summary_parts[1],
                    summary_parts[2], summary_parts[3],
                ))
            lines.append(sep)

        lines.append("=" * 74 + "\n")

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

    n_subset = 200 if args.smoke_test else None
    count_matrix, int_labels, label_names, adata_raw, gene_names = load_pbmc3k(n_subset)
    gene_vocab, vocab_size, _, _ = build_vocab(count_matrix.shape[1])

    per_seed: dict[str, dict] = {}

    for seed in args.seeds:
        print(f"\n{'='*74}")
        print(f"  Seed: {seed}")
        print("=" * 74)

        np.random.seed(seed)
        torch.manual_seed(seed)

        X_train, y_train, X_test, y_test = _train_test_split(
            count_matrix, int_labels, seed=seed
        )
        print(f"  Train: {X_train.shape[0]} cells  |  Test: {X_test.shape[0]} cells")

        # --- SIGReg scratch ---
        print(f"\n  [seed={seed}] SIGReg (scratch)")
        per_seed[f"SIGReg (scratch) seed={seed}"] = run_sigreg_scratch(
            X_train, y_train, X_test, y_test,
            gene_vocab, vocab_size, args, device,
            seed=seed,
        )

        # --- SIGReg kidney ---
        print(f"\n  [seed={seed}] SIGReg (kidney)")
        per_seed[f"SIGReg (kidney) seed={seed}"] = run_sigreg_kidney(
            X_train, y_train, X_test, y_test,
            gene_names, args, device,
            seed=seed,
        )

    print_and_save_multiseed(per_seed, args.seeds, args.results_file)


if __name__ == "__main__":
    main()

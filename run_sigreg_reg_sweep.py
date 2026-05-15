"""
run_sigreg_reg_sweep.py — SIGReg Weight Sweep During Kidney Fine-Tuning
=======================================================================
Tests whether INCREASING the SIGReg regularisation weight during kidney
fine-tuning can counteract the embedding collapse observed at the default
w_sigreg=0.5.

Background: SIGReg is already active during fine-tuning (w_sigreg=0.5 by
default). The LR sweep showed lower LR made collapse WORSE — ruling out
catastrophic forgetting. This experiment tests the complementary axis: if
stronger distributional regularisation forces the embeddings to stay
isotropic, can the model still adapt the PBMC-specific objectives without
collapsing?

Interpretation guide:
  - If high w_sigreg recovers AvgBIO toward scratch fine-tuned (~0.75):
    → The collapse is a regularisation-strength issue. The default w_sigreg
      is too weak relative to the geometry mismatch pulling force from ECS/GEPC.
      Fix: use higher w_sigreg for cross-tissue fine-tuning.
  - If high w_sigreg still collapses OR trades off (e.g. SIGReg improves but
    ECS degrades so AvgBIO stays low):
    → The geometry mismatch is fundamental. The objectives are incompatible
      regardless of regularisation strength, a more interesting finding.

The SIGReg scratch condition is run once (w_sigreg=0.5) as reference.

Usage:
    python run_sigreg_reg_sweep.py \\
        --sigreg_checkpoint /path/kidney_sigreg_final.pt \\
        --sigreg_genes      /path/kidney_sigreg_gene_names.json \\
        --device cuda \\
        --sigreg_weights 0.5 2.0 5.0 10.0

    # Smoke test:
    python run_sigreg_reg_sweep.py --smoke_test --device cpu \\
        --sigreg_checkpoint /path/... --sigreg_genes /path/...
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from compare_pbmc3k import load_pbmc3k, build_vocab
from run_ablation import _train_test_split
from run_transfer import run_sigreg_scratch, run_sigreg_kidney, align_matrix
from preprocessing import SingleCellDataset
from cell_sigreg import CellJEPA_SIGReg
from trainer import SIGRegFinetuner, SIGRegFinetuneConfig
from metrics import extract_embeddings
from compare_pbmc3k import evaluate_embeddings
import json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIGReg weight sweep for kidney fine-tuning collapse diagnosis"
    )
    p.add_argument("--sigreg_weights", type=float, nargs="+",
                   default=[0.5, 2.0, 5.0, 10.0],
                   help="w_sigreg values to sweep (default: 0.5 2.0 5.0 10.0)")
    p.add_argument("--smoke_test", action="store_true",
                   help="200 cells, 1 epoch — quick correctness check")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--results_file", default="results_sigreg_reg_sweep.txt")
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
# Kidney runner with configurable w_sigreg
# ---------------------------------------------------------------------------

def run_sigreg_kidney_reg(
    X_train, y_train, X_test, y_test,
    pbmc_gene_names, args, device,
    w_sigreg: float = 0.5,
    seed: int = 42,
) -> dict:
    """Like run_sigreg_kidney but sweeps w_sigreg instead of lr_scale."""
    if not args.sigreg_checkpoint or not args.sigreg_genes:
        print("  Skipping (no --sigreg_checkpoint / --sigreg_genes provided)")
        return {}

    n_bins = 50
    ft_bs = 8 if args.smoke_test else 64

    with open(args.sigreg_genes) as f:
        kidney_genes = json.load(f)

    n_kidney = len(kidney_genes)
    kidney_vocab = {i: i + 2 for i in range(n_kidney)}
    kidney_vocab_size = n_kidney + 2

    X_train_a, n_mapped, y_train_a = align_matrix(X_train, pbmc_gene_names, kidney_genes, y_train)
    X_test_a,  _,        y_test_a  = align_matrix(X_test,  pbmc_gene_names, kidney_genes, y_test)
    print(f"  Gene alignment: {n_mapped}/{len(pbmc_gene_names)} PBMC genes in kidney vocab ({n_kidney} total)")
    print(f"  After alignment: {X_train_a.shape[0]} train cells, {X_test_a.shape[0]} test cells")

    finetune_ds = SingleCellDataset(
        X_train_a, kidney_vocab, y_train_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test_a, kidney_vocab, y_test_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA_SIGReg(
        vocab_size=kidney_vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, n_views=2,
    )
    ckpt = torch.load(args.sigreg_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded kidney SIGReg checkpoint (w_sigreg={w_sigreg:.1f})")

    bs = 8 if args.smoke_test else 64

    # Zero-shot is the same for all w_sigreg values — only compute once if reusing
    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    SIGRegFinetuner(model, finetune_ds, SIGRegFinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=args.finetune_epochs,
        log_every=20, n_views=2,
        w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_sim=1.0, w_sigreg=w_sigreg, ecs_temperature=0.1,
        n_directions=64 if args.smoke_test else 256,
        num_workers=0, seed=seed,
    ), device).train()

    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save_reg_sweep(results: dict, path: str) -> None:
    row = "  {:<42s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}"
    sep = "  " + "-" * 72

    lines = ["\n" + "=" * 76]
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        lines += [
            f"  {phase_label} — SIGReg Regularisation Weight Sweep (Kidney Fine-Tuning)",
            "=" * 76,
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
        lines.append("=" * 76 + "\n")

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

    # Scratch baseline (w_sigreg=0.5, reference)
    print(f"\n{'='*76}")
    print("  SIGReg (scratch) — baseline")
    print("=" * 76)
    all_results["SIGReg (scratch, w_sigreg=0.5)"] = run_sigreg_scratch(
        X_train, y_train, X_test, y_test,
        gene_vocab, vocab_size, args, device,
        seed=42, lr_scale=1.0,
    )

    # Kidney conditions at each w_sigreg value
    for w_sigreg in args.sigreg_weights:
        label = f"SIGReg (kidney, w_sigreg={w_sigreg:.1f})"
        print(f"\n{'='*76}")
        print(f"  {label}")
        print("=" * 76)
        all_results[label] = run_sigreg_kidney_reg(
            X_train, y_train, X_test, y_test,
            gene_names, args, device,
            w_sigreg=w_sigreg, seed=42,
        )

    print_and_save_reg_sweep(all_results, args.results_file)


if __name__ == "__main__":
    main()

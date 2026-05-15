"""
run_sigreg_w0_test.py — SIGReg w_sigreg=0 Fine-Tuning Test
===========================================================
Tests whether disabling SIGReg loss during fine-tuning (w_sigreg=0) recovers
AvgBIO for pre-trained SIGReg checkpoints that collapse at the default w_sigreg=0.5.

Hypothesis: the SIGReg loss actively conflicts with ECS/GEPC during fine-tuning
by pushing embeddings toward isotropic Gaussian while ECS/GEPC push toward
clustered structure. Setting w_sigreg=0 lets the pre-trained geometry adapt
freely to the fine-tuning objectives.

Four conditions:
  1. SIGReg (kidney,   w_sigreg=0.5) — existing result, reference
  2. SIGReg (kidney,   w_sigreg=0.0) — test: disable SIGReg during fine-tuning
  3. SIGReg (PBMC-68K, w_sigreg=0.5) — existing result, reference
  4. SIGReg (PBMC-68K, w_sigreg=0.0) — test: disable SIGReg during fine-tuning

If w_sigreg=0 improves AvgBIO significantly → SIGReg loss is the culprit.
If w_sigreg=0 still underperforms scratch → geometry mismatch is fundamental.

Usage:
    python run_sigreg_w0_test.py \\
        --kidney_checkpoint  /path/kidney_sigreg_final.pt \\
        --kidney_genes       /path/kidney_sigreg_gene_names.json \\
        --pbmc68k_checkpoint /path/pbmc68k_sigreg_final.pt \\
        --pbmc68k_genes      /path/pbmc68k_gene_names.json \\
        --device cuda

    # Smoke test:
    python run_sigreg_w0_test.py --smoke_test --device cpu \\
        --kidney_checkpoint ... --kidney_genes ... \\
        --pbmc68k_checkpoint ... --pbmc68k_genes ...

    # Skip one source:
    python run_sigreg_w0_test.py --skip_kidney --device cuda ...
    python run_sigreg_w0_test.py --skip_pbmc68k --device cuda ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cell_sigreg import CellJEPA_SIGReg
from compare_pbmc3k import load_pbmc3k, build_vocab, evaluate_embeddings
from metrics import extract_embeddings
from preprocessing import SingleCellDataset
from run_ablation import _train_test_split
from run_transfer import align_matrix
from trainer import SIGRegFinetuner, SIGRegFinetuneConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test w_sigreg=0 during fine-tuning for kidney and PBMC-68K checkpoints"
    )
    p.add_argument("--smoke_test", action="store_true",
                   help="200 cells, 1 epoch — quick correctness check")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--results_file", default="results_sigreg_w0_test.txt")
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--l_max", type=int, default=200)

    p.add_argument("--kidney_checkpoint", default=None)
    p.add_argument("--kidney_genes", default=None)
    p.add_argument("--pbmc68k_checkpoint", default=None)
    p.add_argument("--pbmc68k_genes", default=None)

    p.add_argument("--skip_kidney",  action="store_true")
    p.add_argument("--skip_pbmc68k", action="store_true")
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
# Runner — shared logic for both checkpoint sources
# ---------------------------------------------------------------------------

def run_condition(
    label: str,
    checkpoint: str,
    pretrain_genes: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pbmc3k_gene_names: list[str],
    w_sigreg: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print("=" * 70)

    n_bins = 50
    ft_bs = 8 if args.smoke_test else 64
    bs    = 8 if args.smoke_test else 64

    n_pretrain = len(pretrain_genes)
    vocab = {i: i + 2 for i in range(n_pretrain)}
    vocab_size = n_pretrain + 2

    X_train_a, n_mapped, y_train_a = align_matrix(
        X_train, pbmc3k_gene_names, pretrain_genes, y_train
    )
    X_test_a, _, y_test_a = align_matrix(
        X_test, pbmc3k_gene_names, pretrain_genes, y_test
    )
    print(f"  Gene alignment: {n_mapped}/{len(pbmc3k_gene_names)} PBMC-3K genes "
          f"in pre-train vocab ({n_pretrain} total)")
    print(f"  After alignment: {X_train_a.shape[0]} train, {X_test_a.shape[0]} test cells")

    finetune_ds = SingleCellDataset(
        X_train_a, vocab, y_train_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test_a, vocab, y_test_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA_SIGReg(
        vocab_size=vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, n_views=2,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded checkpoint: {checkpoint}")

    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    SIGRegFinetuner(model, finetune_ds, SIGRegFinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=1 if args.smoke_test else args.finetune_epochs,
        log_every=20, n_views=2,
        w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_sim=1.0, w_sigreg=w_sigreg, ecs_temperature=0.1,
        n_directions=64 if args.smoke_test else 256,
        num_workers=0,
    ), device).train()

    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save(results: dict, path: str) -> None:
    row = "  {:<44s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}"
    sep = "  " + "-" * 74

    lines = []
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        lines += [
            "\n" + "=" * 78,
            f"  {phase_label} — SIGReg w_sigreg=0 Test",
            "=" * 78,
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
        lines.append("=" * 78)

    output = "\n".join(lines)
    print(output)
    with open(path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.smoke_test:
        print("\n[SMOKE TEST: 200 cells, 1 epoch each]\n")
        args.l_max = 64

    n_subset = 200 if args.smoke_test else None
    count_matrix, int_labels, _, _, gene_names = load_pbmc3k(n_subset)
    X_train, y_train, X_test, y_test = _train_test_split(count_matrix, int_labels)
    print(f"Train: {X_train.shape[0]} cells  |  Test: {X_test.shape[0]} cells")

    all_results: dict[str, dict] = {}

    # --- Kidney conditions ---
    if not args.skip_kidney:
        if not args.kidney_checkpoint or not args.kidney_genes:
            print("\nSkipping kidney conditions (no --kidney_checkpoint / --kidney_genes)")
        else:
            with open(args.kidney_genes) as f:
                kidney_genes = json.load(f)

            for w in [0.5, 0.0]:
                label = f"SIGReg (kidney, w_sigreg={w:.1f})"
                all_results[label] = run_condition(
                    label=label,
                    checkpoint=args.kidney_checkpoint,
                    pretrain_genes=kidney_genes,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test,   y_test=y_test,
                    pbmc3k_gene_names=gene_names,
                    w_sigreg=w,
                    args=args, device=device,
                )

    # --- PBMC-68K conditions ---
    if not args.skip_pbmc68k:
        if not args.pbmc68k_checkpoint or not args.pbmc68k_genes:
            print("\nSkipping PBMC-68K conditions (no --pbmc68k_checkpoint / --pbmc68k_genes)")
        else:
            with open(args.pbmc68k_genes) as f:
                pbmc68k_genes = json.load(f)

            for w in [0.5, 0.0]:
                label = f"SIGReg (PBMC-68K, w_sigreg={w:.1f})"
                all_results[label] = run_condition(
                    label=label,
                    checkpoint=args.pbmc68k_checkpoint,
                    pretrain_genes=pbmc68k_genes,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test,   y_test=y_test,
                    pbmc3k_gene_names=gene_names,
                    w_sigreg=w,
                    args=args, device=device,
                )

    print_and_save(all_results, args.results_file)


if __name__ == "__main__":
    main()

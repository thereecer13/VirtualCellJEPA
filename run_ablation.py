"""
run_ablation.py — 2×2 Ablation: Cell-JEPA vs Transformer + SIGReg
==================================================================
Trains and evaluates both variants on PBMC-3K cell-type clustering,
reporting zero-shot and fine-tuned NMI / ARI / ASW / AvgBIO.

Usage:
    python run_ablation.py --variant cell_jepa [--smoke_test] [--device cuda]
    python run_ablation.py --variant sigreg_transformer [--smoke_test] [--device cuda]
    python run_ablation.py --variant all [--smoke_test] [--device cuda]

Results are saved to results_ablation.txt (matches Table 1/2 format of the paper).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from preprocessing import SingleCellDataset
from cell_jepa import CellJEPA
from cell_sigreg import CellJEPA_SIGReg
from trainer import (
    Pretrainer, PretrainConfig,
    Finetuner, FinetuneConfig,
    SIGRegPretrainer, SIGRegPretrainConfig,
    SIGRegFinetuner, SIGRegFinetuneConfig,
)
from metrics import extract_embeddings

# Reuse data loading and evaluation utilities from compare_pbmc3k
from compare_pbmc3k import (
    load_pbmc3k,
    build_vocab,
    evaluate_embeddings,
    print_and_save,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CellJEPA vs SIGReg-Transformer ablation on PBMC-3K"
    )
    p.add_argument(
        "--variant", default="all",
        choices=["cell_jepa", "sigreg_transformer", "all"],
        help="Which variant to run (default: all)",
    )
    p.add_argument("--smoke_test", action="store_true",
                   help="200 cells, 1 epoch each — quick correctness check")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--results_file", default="results_ablation.txt")
    p.add_argument("--pretrain_epochs", type=int, default=4)
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--l_max", type=int, default=200,
                   help="Max genes per cell (200 default; 600 for paper settings)")
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
# Cell-JEPA condition
# ---------------------------------------------------------------------------

def _train_test_split(
    count_matrix: np.ndarray,
    cell_types: np.ndarray,
    test_frac: float = 0.2,
    seed: int = 42,
):
    """80/20 stratified split — returns (X_train, y_train, X_test, y_test)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(count_matrix.shape[0])
    n_test = max(1, int(len(idx) * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (
        count_matrix[train_idx], cell_types[train_idx],
        count_matrix[test_idx],  cell_types[test_idx],
    )


def run_cell_jepa(
    count_matrix: np.ndarray,
    cell_types: np.ndarray,
    gene_vocab: dict,
    vocab_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    n_bins = 50
    bs     = 8  if args.smoke_test else 32
    ft_bs  = 8  if args.smoke_test else 16

    X_train, y_train, X_test, y_test = _train_test_split(count_matrix, cell_types)
    print(f"  Train: {X_train.shape[0]} cells  |  Test (held-out): {X_test.shape[0]} cells")

    pretrain_ds = SingleCellDataset(
        X_train, gene_vocab, np.zeros(X_train.shape[0], dtype=np.int32),
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )
    finetune_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test, gene_vocab, y_test,
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA(
        vocab_size=vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CellJEPA: {n_params/1e6:.1f}M trainable parameters")

    print(f"\n  Pre-training ({args.pretrain_epochs} epochs) …")
    Pretrainer(
        model, pretrain_ds,
        PretrainConfig(
            lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
            batch_size=bs, n_epochs=args.pretrain_epochs,
            log_every=20, w_rec=1.0, w_jepa=1000.0, num_workers=0,
        ),
        device,
    ).train()

    # Zero-shot evaluation (before fine-tuning)
    print("  Evaluating zero-shot …")
    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs_results = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs_results['avg_bio']:.4f}")

    print(f"\n  Fine-tuning ({args.finetune_epochs} epochs) …")
    Finetuner(
        model, finetune_ds,
        FinetuneConfig(
            lr=1e-4, lr_decay=0.9,
            batch_size=ft_bs, n_epochs=args.finetune_epochs,
            log_every=20, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
            w_jepa=1000.0, ecs_temperature=0.1, include_jepa=True, num_workers=0,
        ),
        device,
    ).train()

    print("  Evaluating fine-tuned …")
    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft_results = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft_results['avg_bio']:.4f}")

    return {"zero_shot": zs_results, "fine_tuned": ft_results}


# ---------------------------------------------------------------------------
# Transformer + SIGReg condition
# ---------------------------------------------------------------------------

def run_sigreg_transformer(
    count_matrix: np.ndarray,
    cell_types: np.ndarray,
    gene_vocab: dict,
    vocab_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    n_bins = 50
    bs     = 8   if args.smoke_test else 128
    ft_bs  = 8   if args.smoke_test else 64

    X_train, y_train, X_test, y_test = _train_test_split(count_matrix, cell_types)
    print(f"  Train: {X_train.shape[0]} cells  |  Test (held-out): {X_test.shape[0]} cells")

    pretrain_ds = SingleCellDataset(
        X_train, gene_vocab, np.zeros(X_train.shape[0], dtype=np.int32),
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )
    finetune_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test, gene_vocab, y_test,
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA_SIGReg(
        vocab_size=vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, n_views=2,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CellJEPA_SIGReg: {n_params/1e6:.1f}M trainable parameters")

    print(f"\n  Pre-training ({args.pretrain_epochs} epochs) …")
    SIGRegPretrainer(
        model, pretrain_ds,
        SIGRegPretrainConfig(
            lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
            warmup_steps=100 if args.smoke_test else 1000,
            batch_size=bs, n_epochs=args.pretrain_epochs,
            log_every=20, n_views=2,
            w_sim=1.0, w_sigreg=0.5, w_rec=1.0,
            n_directions=64 if args.smoke_test else 256,
            num_workers=0,
        ),
        device,
    ).train()

    # Zero-shot evaluation (before fine-tuning)
    print("  Evaluating zero-shot …")
    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs_results = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs_results['avg_bio']:.4f}")

    print(f"\n  Fine-tuning ({args.finetune_epochs} epochs) …")
    SIGRegFinetuner(
        model, finetune_ds,
        SIGRegFinetuneConfig(
            lr=1e-4, lr_decay=0.9,
            batch_size=ft_bs, n_epochs=args.finetune_epochs,
            log_every=20, n_views=2,
            w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
            w_sim=1.0, w_sigreg=0.5, ecs_temperature=0.1,
            n_directions=64 if args.smoke_test else 256,
            num_workers=0,
        ),
        device,
    ).train()

    print("  Evaluating fine-tuned …")
    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft_results = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft_results['avg_bio']:.4f}")

    return {"zero_shot": zs_results, "fine_tuned": ft_results}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save_ablation(results: dict, path: str) -> None:
    """
    Print a two-section table (zero-shot / fine-tuned) and save to file.
    Matches the column format of Table 1 / Table 2 of the paper.
    """
    row = "  {:<28s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}"
    sep = "  " + "-" * 58

    lines = ["\n" + "=" * 62]
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        lines += [
            f"  {phase_label} — PBMC-3K Cell Clustering",
            "=" * 62,
            row.format("Model", "NMI", "ARI", "ASW", "AvgBIO"),
            sep,
        ]
        for name, cond in results.items():
            res = cond.get(phase_key)
            if res is None:
                lines.append(row.format(name, "N/A", "N/A", "N/A", "N/A"))
            else:
                lines.append(row.format(
                    name,
                    f"{res['nmi']:.4f}",
                    f"{res['ari']:.4f}",
                    f"{res['asw']:.4f}",
                    f"{res['avg_bio']:.4f}",
                ))
        lines.append("=" * 62 + "\n")

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

    variants_to_run = (
        ["cell_jepa", "sigreg_transformer"] if args.variant == "all"
        else [args.variant]
    )

    all_results: dict[str, dict] = {}

    for variant in variants_to_run:
        print(f"\n{'='*62}")
        print(f"  Variant: {variant}")
        print("=" * 62)

        if variant == "cell_jepa":
            all_results["Cell-JEPA"] = run_cell_jepa(
                count_matrix, int_labels, gene_vocab, vocab_size, args, device
            )
        elif variant == "sigreg_transformer":
            all_results["Transformer + SIGReg"] = run_sigreg_transformer(
                count_matrix, int_labels, gene_vocab, vocab_size, args, device
            )

    print_and_save_ablation(all_results, args.results_file)


if __name__ == "__main__":
    main()

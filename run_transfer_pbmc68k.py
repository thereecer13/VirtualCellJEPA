"""
run_transfer_pbmc68k.py — PBMC-68K → PBMC-3K Transfer Experiment
=================================================================
Tests whether pre-training on PBMC-68K (same tissue, ~25× more cells) transfers
to PBMC-3K cell-type clustering compared to training from scratch on PBMC-3K.

Because both datasets are peripheral blood, any transfer failure cannot be
attributed to tissue/domain gap — it isolates whether SIGReg embeddings are
generally non-transferable or whether the kidney collapse was domain-specific.

Four conditions:
  1. Cell-JEPA    — scratch  (PBMC-3K pre-train + fine-tune)
  2. Cell-JEPA    — PBMC-68K (pre-train on 68K, fine-tune on 3K)
  3. SIGReg       — scratch  (PBMC-3K pre-train + fine-tune)
  4. SIGReg       — PBMC-68K (pre-train on 68K, fine-tune on 3K)

Each condition reports zero-shot AvgBIO (after pre-training only) and
fine-tuned AvgBIO (after fine-tuning on PBMC-3K labels).

Usage:
    python run_transfer_pbmc68k.py \
        --jepa_checkpoint  /path/pbmc68k_jepa_final.pt \
        --sigreg_checkpoint /path/pbmc68k_sigreg_final.pt \
        --pbmc68k_genes    /path/pbmc68k_gene_names.json \
        --device cuda

    # Smoke test (200 cells, 1 epoch, CPU):
    python run_transfer_pbmc68k.py --smoke_test --device cpu

    # Skip conditions:
    python run_transfer_pbmc68k.py --skip_jepa_scratch --skip_sigreg_scratch \
        --jepa_checkpoint ... --sigreg_checkpoint ... --pbmc68k_genes ...
"""

from __future__ import annotations

import argparse
import json

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
from compare_pbmc3k import load_pbmc3k, build_vocab, evaluate_embeddings
from run_ablation import _train_test_split
from run_transfer import align_matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PBMC-68K → PBMC-3K transfer experiment"
    )
    p.add_argument("--smoke_test", action="store_true",
                   help="200 cells, 1 epoch — quick correctness check")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--results_file", default="results_transfer_pbmc68k.txt")

    p.add_argument("--pretrain_epochs", type=int, default=4,
                   help="Epochs for scratch pre-training on PBMC-3K")
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--l_max", type=int, default=200)

    # PBMC-68K checkpoints (both models share the same gene names file)
    p.add_argument("--jepa_checkpoint", default=None,
                   help="Path to PBMC-68K-pretrained Cell-JEPA checkpoint (.pt)")
    p.add_argument("--sigreg_checkpoint", default=None,
                   help="Path to PBMC-68K-pretrained SIGReg checkpoint (.pt)")
    p.add_argument("--pbmc68k_genes", default=None,
                   help="Path to PBMC-68K gene names JSON (shared by both models)")

    # Skip flags
    p.add_argument("--skip_jepa_scratch",   action="store_true")
    p.add_argument("--skip_jepa_pbmc68k",   action="store_true")
    p.add_argument("--skip_sigreg_scratch", action="store_true")
    p.add_argument("--skip_sigreg_pbmc68k", action="store_true")

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
# Condition runners
# ---------------------------------------------------------------------------

def run_jepa_scratch(
    X_train, y_train, X_test, y_test,
    gene_vocab, vocab_size, args, device,
) -> dict:
    n_bins = 50
    bs    = 8  if args.smoke_test else 32
    ft_bs = 8  if args.smoke_test else 16

    pretrain_ds = SingleCellDataset(
        X_train, gene_vocab, np.zeros(len(X_train), dtype=np.int32),
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )
    finetune_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test, gene_vocab, y_test,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA(
        vocab_size=vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    Pretrainer(model, pretrain_ds, PretrainConfig(
        lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
        batch_size=bs, n_epochs=args.pretrain_epochs,
        log_every=20, w_rec=1.0, w_jepa=1000.0, num_workers=0,
    ), device).train()

    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    Finetuner(model, finetune_ds, FinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=args.finetune_epochs,
        log_every=20, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_jepa=1000.0, ecs_temperature=0.1, include_jepa=True, num_workers=0,
    ), device).train()

    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft}


def run_jepa_pbmc68k(
    X_train, y_train, X_test, y_test,
    pbmc3k_gene_names, args, device,
) -> dict:
    if not args.jepa_checkpoint or not args.pbmc68k_genes:
        print("  Skipping (no --jepa_checkpoint / --pbmc68k_genes provided)")
        return {}

    n_bins = 50
    ft_bs = 8 if args.smoke_test else 16
    bs    = 8 if args.smoke_test else 32

    with open(args.pbmc68k_genes) as f:
        pbmc68k_genes = json.load(f)

    n_68k = len(pbmc68k_genes)
    vocab_68k = {i: i + 2 for i in range(n_68k)}
    vocab_size_68k = n_68k + 2

    X_train_a, n_mapped, y_train_a = align_matrix(X_train, pbmc3k_gene_names, pbmc68k_genes, y_train)
    X_test_a,  _,        y_test_a  = align_matrix(X_test,  pbmc3k_gene_names, pbmc68k_genes, y_test)
    print(f"  Gene alignment: {n_mapped}/{len(pbmc3k_gene_names)} PBMC-3K genes in PBMC-68K vocab ({n_68k} total)")
    print(f"  After alignment: {X_train_a.shape[0]} train cells, {X_test_a.shape[0]} test cells")

    finetune_ds = SingleCellDataset(
        X_train_a, vocab_68k, y_train_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test_a, vocab_68k, y_test_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA(
        vocab_size=vocab_size_68k, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
    )
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded PBMC-68K Cell-JEPA checkpoint from {args.jepa_checkpoint}")

    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    Finetuner(model, finetune_ds, FinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=args.finetune_epochs,
        log_every=20, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_jepa=1000.0, ecs_temperature=0.1, include_jepa=True, num_workers=0,
    ), device).train()

    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft}


def run_sigreg_scratch(
    X_train, y_train, X_test, y_test,
    gene_vocab, vocab_size, args, device,
) -> dict:
    n_bins = 50
    bs    = 8  if args.smoke_test else 128
    ft_bs = 8  if args.smoke_test else 64

    pretrain_ds = SingleCellDataset(
        X_train, gene_vocab, np.zeros(len(X_train), dtype=np.int32),
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )
    finetune_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test, gene_vocab, y_test,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA_SIGReg(
        vocab_size=vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, n_views=2,
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    SIGRegPretrainer(model, pretrain_ds, SIGRegPretrainConfig(
        lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
        warmup_steps=100 if args.smoke_test else 1000,
        batch_size=bs, n_epochs=args.pretrain_epochs,
        log_every=20, n_views=2,
        w_sim=1.0, w_sigreg=0.5, w_rec=1.0,
        n_directions=64 if args.smoke_test else 256,
        num_workers=0,
    ), device).train()

    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    SIGRegFinetuner(model, finetune_ds, SIGRegFinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=args.finetune_epochs,
        log_every=20, n_views=2,
        w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_sim=1.0, w_sigreg=0.5, ecs_temperature=0.1,
        n_directions=64 if args.smoke_test else 256,
        num_workers=0,
    ), device).train()

    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft}


def run_sigreg_pbmc68k(
    X_train, y_train, X_test, y_test,
    pbmc3k_gene_names, args, device,
) -> dict:
    if not args.sigreg_checkpoint or not args.pbmc68k_genes:
        print("  Skipping (no --sigreg_checkpoint / --pbmc68k_genes provided)")
        return {}

    n_bins = 50
    ft_bs = 8 if args.smoke_test else 64
    bs    = 8 if args.smoke_test else 64

    with open(args.pbmc68k_genes) as f:
        pbmc68k_genes = json.load(f)

    n_68k = len(pbmc68k_genes)
    vocab_68k = {i: i + 2 for i in range(n_68k)}
    vocab_size_68k = n_68k + 2

    X_train_a, n_mapped, y_train_a = align_matrix(X_train, pbmc3k_gene_names, pbmc68k_genes, y_train)
    X_test_a,  _,        y_test_a  = align_matrix(X_test,  pbmc3k_gene_names, pbmc68k_genes, y_test)
    print(f"  Gene alignment: {n_mapped}/{len(pbmc3k_gene_names)} PBMC-3K genes in PBMC-68K vocab ({n_68k} total)")
    print(f"  After alignment: {X_train_a.shape[0]} train cells, {X_test_a.shape[0]} test cells")

    finetune_ds = SingleCellDataset(
        X_train_a, vocab_68k, y_train_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test_a, vocab_68k, y_test_a,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = CellJEPA_SIGReg(
        vocab_size=vocab_size_68k, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, n_views=2,
    )
    ckpt = torch.load(args.sigreg_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded PBMC-68K SIGReg checkpoint from {args.sigreg_checkpoint}")

    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    SIGRegFinetuner(model, finetune_ds, SIGRegFinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=args.finetune_epochs,
        log_every=20, n_views=2,
        w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_sim=1.0, w_sigreg=0.5, ecs_temperature=0.1,
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

def print_and_save_transfer(results: dict, path: str) -> None:
    row = "  {:<36s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}"
    sep = "  " + "-" * 66

    lines = []
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        lines += [
            "\n" + "=" * 70,
            f"  {phase_label} — PBMC-68K → PBMC-3K Transfer",
            "=" * 70,
            row.format("Model", "NMI", "ARI", "ASW", "AvgBIO"),
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
        lines.append("=" * 70)

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
        args.pretrain_epochs = 1
        args.finetune_epochs = 1
        args.l_max = 64

    n_subset = 200 if args.smoke_test else None
    count_matrix, int_labels, label_names, adata_raw, gene_names = load_pbmc3k(n_subset)
    gene_vocab, vocab_size, _, _ = build_vocab(count_matrix.shape[1])

    X_train, y_train, X_test, y_test = _train_test_split(count_matrix, int_labels)
    print(f"\nTrain: {X_train.shape[0]} cells  |  Test: {X_test.shape[0]} cells")

    all_results: dict[str, dict] = {}

    if not args.skip_jepa_scratch:
        print(f"\n{'='*70}")
        print("  Cell-JEPA (scratch, PBMC-3K)")
        print("=" * 70)
        all_results["Cell-JEPA (scratch)"] = run_jepa_scratch(
            X_train, y_train, X_test, y_test, gene_vocab, vocab_size, args, device,
        )

    if not args.skip_jepa_pbmc68k:
        print(f"\n{'='*70}")
        print("  Cell-JEPA (PBMC-68K pre-train → PBMC-3K fine-tune)")
        print("=" * 70)
        all_results["Cell-JEPA (PBMC-68K)"] = run_jepa_pbmc68k(
            X_train, y_train, X_test, y_test, gene_names, args, device,
        )

    if not args.skip_sigreg_scratch:
        print(f"\n{'='*70}")
        print("  SIGReg (scratch, PBMC-3K)")
        print("=" * 70)
        all_results["SIGReg (scratch)"] = run_sigreg_scratch(
            X_train, y_train, X_test, y_test, gene_vocab, vocab_size, args, device,
        )

    if not args.skip_sigreg_pbmc68k:
        print(f"\n{'='*70}")
        print("  SIGReg (PBMC-68K pre-train → PBMC-3K fine-tune)")
        print("=" * 70)
        all_results["SIGReg (PBMC-68K)"] = run_sigreg_pbmc68k(
            X_train, y_train, X_test, y_test, gene_names, args, device,
        )

    print_and_save_transfer(all_results, args.results_file)


if __name__ == "__main__":
    main()

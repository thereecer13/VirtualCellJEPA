"""
run_transfer_universal.py — Universal Vocab Transfer Experiment
================================================================
Tests whether the universal gene vocabulary (built by build_universal_vocab.py
and used in pretrain_universal.py) fixes the gene-overlap collapse seen in the
original gene-aligned transfer experiments.

Background:
  - Gene-aligned kidney→PBMC-3K: only 139/2000 HVGs overlap (7%)
    → SIGReg fine-tuned AvgBIO collapses to 0.307
  - Gene-aligned PBMC-68K→PBMC-3K: 352/2000 HVGs overlap (18%)
    → SIGReg fine-tuned AvgBIO 0.453 (still below scratch 0.753)
  - Universal vocab: PBMC-3K uses the same token IDs as the pre-training data
    → No alignment needed; model sees the same gene representations it learned

Four conditions (2 sources × 2 models):
  1. Cell-JEPA  (kidney,   universal vocab, pre-train → fine-tune on PBMC-3K)
  2. SIGReg     (kidney,   universal vocab, pre-train → fine-tune on PBMC-3K)
  3. Cell-JEPA  (PBMC-68K, universal vocab, pre-train → fine-tune on PBMC-3K)
  4. SIGReg     (PBMC-68K, universal vocab, pre-train → fine-tune on PBMC-3K)

Each condition includes zero-shot and fine-tuned AvgBIO. Scratch baselines
(train from scratch on PBMC-3K with the same universal vocab) are also run
for direct comparison.

Usage:
    python run_transfer_universal.py \\
        --vocab_file /path/universal_gene_names.json \\
        --kidney_jepa_checkpoint   /path/kidney_universal_jepa_final.pt \\
        --kidney_sigreg_checkpoint /path/kidney_universal_sigreg_final.pt \\
        --pbmc68k_jepa_checkpoint   /path/pbmc68k_universal_jepa_final.pt \\
        --pbmc68k_sigreg_checkpoint /path/pbmc68k_universal_sigreg_final.pt \\
        --device cuda

    # Smoke test (no checkpoints needed for scratch baselines):
    python run_transfer_universal.py --smoke_test --device cpu \\
        --vocab_file /path/universal_gene_names.json

    # Skip conditions:
    python run_transfer_universal.py --skip_kidney --skip_pbmc68k \\
        --vocab_file /path/universal_gene_names.json --device cuda
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cell_jepa import CellJEPA
from cell_sigreg import CellJEPA_SIGReg
from compare_pbmc3k import load_pbmc3k_universal, evaluate_embeddings
from metrics import extract_embeddings
from preprocessing import SingleCellDataset
from run_ablation import _train_test_split
from trainer import (
    Pretrainer, PretrainConfig,
    Finetuner, FinetuneConfig,
    SIGRegPretrainer, SIGRegPretrainConfig,
    SIGRegFinetuner, SIGRegFinetuneConfig,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Universal vocab transfer: kidney / PBMC-68K → PBMC-3K"
    )
    p.add_argument("--vocab_file", required=True,
                   help="Path to universal_gene_names.json")
    p.add_argument("--smoke_test",  action="store_true",
                   help="200 cells, 1 epoch — quick correctness check")
    p.add_argument("--device",      default=None)
    p.add_argument("--results_file", default="results_transfer_universal.txt")
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--pretrain_epochs", type=int, default=4,
                   help="Epochs for scratch baselines only")
    p.add_argument("--l_max",       type=int, default=600)

    # Kidney checkpoints
    p.add_argument("--kidney_jepa_checkpoint",   default=None)
    p.add_argument("--kidney_sigreg_checkpoint", default=None)

    # PBMC-68K checkpoints
    p.add_argument("--pbmc68k_jepa_checkpoint",   default=None)
    p.add_argument("--pbmc68k_sigreg_checkpoint", default=None)

    # Skip flags
    p.add_argument("--skip_scratch",  action="store_true",
                   help="Skip scratch baselines (Cell-JEPA and SIGReg from scratch on PBMC-3K)")
    p.add_argument("--skip_kidney",   action="store_true")
    p.add_argument("--skip_pbmc68k",  action="store_true")

    # Label override — controls how the pre-trained conditions are named in output.
    # Default "kidney" preserves backward compatibility; pass "multitissue" when
    # --kidney_jepa_checkpoint / --kidney_sigreg_checkpoint point to multi-tissue
    # checkpoints.
    p.add_argument("--pretrain_source", default="kidney",
                   help="Label used for the pre-trained conditions in results output "
                        "(e.g. 'kidney', 'multitissue'). Does not affect which "
                        "checkpoint is loaded.")

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
    bs     = 8 if args.smoke_test else 32
    ft_bs  = 8 if args.smoke_test else 16

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


def run_sigreg_scratch(
    X_train, y_train, X_test, y_test,
    gene_vocab, vocab_size, args, device,
) -> dict:
    n_bins = 50
    bs     = 8 if args.smoke_test else 16
    ft_bs  = 8 if args.smoke_test else 32

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


def run_pretrained(
    label: str,
    checkpoint: str,
    model_class,          # CellJEPA or CellJEPA_SIGReg
    finetuner_class,      # Finetuner or SIGRegFinetuner
    finetune_config_class,
    finetune_config_kwargs: dict,
    X_train, y_train, X_test, y_test,
    gene_vocab, vocab_size, args, device,
) -> dict:
    """Generic runner for pre-trained checkpoints (JEPA or SIGReg)."""
    n_bins = 50
    bs     = 8 if args.smoke_test else 64

    finetune_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test, gene_vocab, y_test,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )

    model = model_class(
        vocab_size=vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2,
        **({"ema_momentum": 0.996, "predictor_hidden": 512}
           if model_class is CellJEPA else {"n_views": 2}),
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded checkpoint: {checkpoint}")

    zs_emb, zs_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    zs = evaluate_embeddings(zs_emb, zs_labels)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    finetuner_class(
        model, finetune_ds,
        finetune_config_class(
            n_epochs=1 if args.smoke_test else args.finetune_epochs,
            **finetune_config_kwargs,
        ),
        device,
    ).train()

    ft_emb, ft_labels = extract_embeddings(model, eval_ds, batch_size=bs, device=device)
    ft = evaluate_embeddings(ft_emb, ft_labels)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save(results: dict, path: str) -> None:
    row = "  {:<48s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}"
    sep = "  " + "-" * 78

    lines = []
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        lines += [
            "\n" + "=" * 82,
            f"  {phase_label} — Universal Vocab Transfer (→ PBMC-3K)",
            "=" * 82,
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
        lines.append("=" * 82)

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
        args.pretrain_epochs = 1
        args.finetune_epochs = 1

    # Load universal vocabulary
    with open(args.vocab_file) as f:
        universal_gene_names: list[str] = json.load(f)
    n_genes    = len(universal_gene_names)
    vocab_size = n_genes + 2
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    print(f"Universal vocab: {n_genes:,} genes  (vocab_size={vocab_size:,})")

    # Load PBMC-3K into universal vocab space
    n_subset = 200 if args.smoke_test else None
    X_pbmc3k, int_labels, label_names, _, _ = load_pbmc3k_universal(
        universal_gene_names, n_cells_subset=n_subset,
    )
    X_train, y_train, X_test, y_test = _train_test_split(X_pbmc3k, int_labels)
    print(f"\nTrain: {X_train.shape[0]} cells  |  Test: {X_test.shape[0]} cells")

    all_results: dict[str, dict] = {}

    shared_scratch = dict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gene_vocab=gene_vocab, vocab_size=vocab_size, args=args, device=device,
    )

    # ---- Scratch baselines ----
    if not args.skip_scratch:
        print(f"\n{'='*82}")
        print("  Cell-JEPA (scratch, universal vocab)")
        print("=" * 82)
        all_results["Cell-JEPA scratch (universal)"] = run_jepa_scratch(**shared_scratch)
        torch.cuda.empty_cache()

        print(f"\n{'='*82}")
        print("  SIGReg (scratch, universal vocab)")
        print("=" * 82)
        all_results["SIGReg scratch (universal)"] = run_sigreg_scratch(**shared_scratch)
        torch.cuda.empty_cache()

    shared_pretrained = dict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gene_vocab=gene_vocab, vocab_size=vocab_size, args=args, device=device,
    )

    jepa_ft_kwargs = dict(
        lr=1e-4, lr_decay=0.9,
        batch_size=8 if args.smoke_test else 16,
        log_every=20, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_jepa=1000.0, ecs_temperature=0.1, include_jepa=True, num_workers=0,
    )
    sigreg_ft_kwargs = dict(
        lr=1e-4, lr_decay=0.9,
        batch_size=8 if args.smoke_test else 32,
        log_every=20, n_views=2,
        w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_sim=1.0, w_sigreg=0.0, ecs_temperature=0.1,
        n_directions=64 if args.smoke_test else 256,
        num_workers=0,
    )

    # ---- Pre-trained conditions (labelled by --pretrain_source) ----
    src = args.pretrain_source
    if not args.skip_kidney:
        if args.kidney_jepa_checkpoint:
            label_jepa = f"Cell-JEPA ({src}, universal)"
            print(f"\n{'='*82}")
            print(f"  Cell-JEPA ({src}, universal vocab → PBMC-3K fine-tune)")
            print("=" * 82)
            all_results[label_jepa] = run_pretrained(
                label=label_jepa,
                checkpoint=args.kidney_jepa_checkpoint,
                model_class=CellJEPA, finetuner_class=Finetuner,
                finetune_config_class=FinetuneConfig,
                finetune_config_kwargs=jepa_ft_kwargs,
                **shared_pretrained,
            )
            torch.cuda.empty_cache()
        else:
            print(f"\nSkipping Cell-JEPA ({src}) — no --kidney_jepa_checkpoint")

        if args.kidney_sigreg_checkpoint:
            label_sigreg = f"SIGReg ({src}, universal)"
            print(f"\n{'='*82}")
            print(f"  SIGReg ({src}, universal vocab → PBMC-3K fine-tune)")
            print("=" * 82)
            all_results[label_sigreg] = run_pretrained(
                label=label_sigreg,
                checkpoint=args.kidney_sigreg_checkpoint,
                model_class=CellJEPA_SIGReg, finetuner_class=SIGRegFinetuner,
                finetune_config_class=SIGRegFinetuneConfig,
                finetune_config_kwargs=sigreg_ft_kwargs,
                **shared_pretrained,
            )
            torch.cuda.empty_cache()
        else:
            print(f"\nSkipping SIGReg ({src}) — no --kidney_sigreg_checkpoint")

    # ---- PBMC-68K conditions ----
    if not args.skip_pbmc68k:
        if args.pbmc68k_jepa_checkpoint:
            print(f"\n{'='*82}")
            print("  Cell-JEPA (PBMC-68K, universal vocab → PBMC-3K fine-tune)")
            print("=" * 82)
            all_results["Cell-JEPA (PBMC-68K, universal)"] = run_pretrained(
                label="Cell-JEPA (PBMC-68K, universal)",
                checkpoint=args.pbmc68k_jepa_checkpoint,
                model_class=CellJEPA, finetuner_class=Finetuner,
                finetune_config_class=FinetuneConfig,
                finetune_config_kwargs=jepa_ft_kwargs,
                **shared_pretrained,
            )
            torch.cuda.empty_cache()
        else:
            print("\nSkipping Cell-JEPA (PBMC-68K) — no --pbmc68k_jepa_checkpoint")

        if args.pbmc68k_sigreg_checkpoint:
            print(f"\n{'='*82}")
            print("  SIGReg (PBMC-68K, universal vocab → PBMC-3K fine-tune)")
            print("=" * 82)
            all_results["SIGReg (PBMC-68K, universal)"] = run_pretrained(
                label="SIGReg (PBMC-68K, universal)",
                checkpoint=args.pbmc68k_sigreg_checkpoint,
                model_class=CellJEPA_SIGReg, finetuner_class=SIGRegFinetuner,
                finetune_config_class=SIGRegFinetuneConfig,
                finetune_config_kwargs=sigreg_ft_kwargs,
                **shared_pretrained,
            )
        else:
            print("\nSkipping SIGReg (PBMC-68K) — no --pbmc68k_sigreg_checkpoint")

    print_and_save(all_results, args.results_file)


if __name__ == "__main__":
    main()

"""
pretrain_kidney_sigreg.py
=========================
Pre-train CellJEPA_SIGReg on human kidney scRNA-seq data from CELLxGENE Census
(~200k cells), then save the checkpoint so run_transfer.py can load it for
gene-aligned fine-tuning / zero-shot evaluation on PBMC-3K.

Usage (Colab A100):
    python pretrain_kidney_sigreg.py --device cuda \
        --drive_dir /content/drive/MyDrive/CellJEPA_results/kidney_sigreg/

Smoke test (500 cells, 1 epoch):
    python pretrain_kidney_sigreg.py --smoke_test --device cpu
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from cell_sigreg import CellJEPA_SIGReg
from preprocessing import SingleCellDataset
from trainer import SIGRegPretrainer, SIGRegPretrainConfig

# Reuse the Census data loader from the CellJEPA kidney script
from pretrain_kidney import load_kidney_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CellJEPA_SIGReg kidney pre-training")
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size (default 32 — same as Cell-JEPA kidney; "
                        "SIGReg runs 3 encoder passes per step so needs lower batch than PBMC ablation)")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--drive_dir",
        default="/content/drive/MyDrive/CellJEPA_results/kidney_sigreg/",
        help="Directory for checkpoints and gene name file",
    )
    p.add_argument("--smoke_test", action="store_true",
                   help="500 cells, 1 epoch — quick correctness check")
    p.add_argument(
        "--w_sigreg", type=float, default=0.5,
        help="SIGReg loss weight (default 0.5)",
    )
    p.add_argument(
        "--n_directions", type=int, default=128,
        help="Number of random projections M for SIGReg (default 128 for kidney; use 256 for PBMC ablation)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.drive_dir, exist_ok=True)

    if args.smoke_test:
        args.n_epochs = 1
        print("[smoke test] 1 epoch, 500 cells")

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X, gene_names = load_kidney_data(smoke_test=args.smoke_test)

    gene_path = os.path.join(args.drive_dir, "kidney_sigreg_gene_names.json")
    with open(gene_path, "w") as f:
        json.dump(gene_names, f)
    print(f"Gene names saved to {gene_path}")

    n_genes = X.shape[1]
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    vocab_size = n_genes + 2
    labels = np.zeros(X.shape[0], dtype=np.int32)

    l_max = 64 if args.smoke_test else 600
    dataset = SingleCellDataset(
        X, gene_vocab, labels,
        n_bins=50, L_max=l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = CellJEPA_SIGReg(
        vocab_size=vocab_size, n_bins=50,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, n_views=2,
        grad_checkpoint=True,  # 3 encoder passes/step — checkpoint layers to stay within 40GB
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CellJEPA_SIGReg: {n_params / 1e6:.1f}M trainable parameters")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    warmup = 100 if args.smoke_test else 1000
    n_dir  = 64  if args.smoke_test else args.n_directions
    bs     = 8   if args.smoke_test else args.batch_size

    config = SIGRegPretrainConfig(
        lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
        warmup_steps=warmup,
        batch_size=bs,
        n_epochs=args.n_epochs,
        log_every=50,
        n_views=2,
        w_sim=1.0,
        w_sigreg=args.w_sigreg,
        w_rec=1.0,
        n_directions=n_dir,
        num_workers=0,
    )

    trainer = SIGRegPretrainer(model, dataset, config=config, device=device)

    def save_epoch_ckpt(epoch: int):
        path = os.path.join(args.drive_dir, f"kidney_sigreg_epoch{epoch}.pt")
        trainer.save(path)

    print(f"\n--- Pre-training ({args.n_epochs} epochs) ---")
    trainer.train(epoch_callback=save_epoch_ckpt)

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    final_path = os.path.join(args.drive_dir, "kidney_sigreg_final.pt")
    trainer.save(final_path)
    print(f"\nPre-training complete. Final checkpoint: {final_path}")
    print(f"Gene names: {gene_path}")
    print("\nNext step: run_transfer.py with:")
    print(f"  --sigreg_checkpoint {final_path}")
    print(f"  --sigreg_genes {gene_path}")


if __name__ == "__main__":
    main()

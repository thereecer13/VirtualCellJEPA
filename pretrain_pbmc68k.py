"""
pretrain_pbmc68k.py
===================
Pre-train CellJEPA and CellJEPA_SIGReg on the canonical PBMC-68K dataset
(Zheng et al. 2017, "Fresh 68k PBMCs", 10x Genomics), then save checkpoints
for fine-tuning on PBMC-3K.

This is the same-domain transfer experiment: PBMC-68K and PBMC-3K are both
peripheral blood from the same protocol, so any transfer failure cannot be
attributed to tissue/domain gap.

Usage (Colab A100):
    python pretrain_pbmc68k.py --device cuda \
        --drive_dir /content/drive/MyDrive/CellJEPA_results/pbmc68k_pretrain/

Smoke test (500 cells, 1 epoch):
    python pretrain_pbmc68k.py --smoke_test --device cpu

Skip one model:
    python pretrain_pbmc68k.py --skip_jepa --device cuda
    python pretrain_pbmc68k.py --skip_sigreg --device cuda
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scanpy as sc
import torch

from cell_jepa import CellJEPA
from cell_sigreg import CellJEPA_SIGReg
from preprocessing import SingleCellDataset
from trainer import (
    Pretrainer, PretrainConfig,
    SIGRegPretrainer, SIGRegPretrainConfig,
)

N_HVG = 2000

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pbmc68k_data(
    smoke_test: bool = False,
    tar_path: str | None = None,
    cache_dir: str = "/content",
):
    """
    Load the Zheng et al. 2017 PBMC-68K dataset.

    Smoke test: uses scanpy's built-in pbmc68k_reduced() (instant, no download).
    Full run:   extracts the filtered_gene_bc_matrices tarball supplied via tar_path.

    Args:
        smoke_test: Subsample to 500 cells using the reduced dataset.
        tar_path:   Path to fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz.
                    Required for the full run; ignored in smoke test mode.
        cache_dir:  Directory to extract the tarball into (default /content).

    Returns:
        X          : (N, N_HVG) float32 log-normalised expression matrix
        gene_names : list of gene symbol strings (length N_HVG)
    """
    import os, tarfile, scipy.sparse as sp

    if smoke_test:
        print("Loading PBMC-68K (reduced) via scanpy built-in...")
        adata = sc.datasets.pbmc68k_reduced()
        if adata.raw is not None:
            adata = adata.raw.to_adata()
        adata.var_names_make_unique()
        print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], min(500, X.shape[0]), replace=False)
        X = X[idx]
        print(f"  [smoke test] subsampled to {X.shape[0]} cells")
        gene_names = list(adata.var_names)
        print(f"  Final: {X.shape[0]:,} cells × {X.shape[1]:,} genes")
        return X, gene_names

    # Full run — extract from user-supplied tarball
    if tar_path is None:
        raise ValueError(
            "tar_path is required for the full run. "
            "Pass --tar_path /path/to/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
        )

    out_dir    = os.path.join(cache_dir, "pbmc68k")
    matrix_dir = os.path.join(out_dir, "filtered_matrices_mex", "hg19")

    if not os.path.exists(matrix_dir):
        print(f"Extracting {tar_path} to {out_dir}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(out_dir)
        print("  Done.")
    else:
        print(f"  Found cached PBMC-68K at {matrix_dir}")

    print("Loading from 10x matrix market files...")
    adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    print(f"  Raw: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After QC: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    expressed = (X > 0).any(axis=1)
    n_dropped = (~expressed).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} cells with zero expressed HVGs")
        X = X[expressed]

    gene_names = list(adata.var_names)
    print(f"  Final: {X.shape[0]:,} cells × {X.shape[1]:,} HVGs")
    print(f"  Expression range: [{X.min():.3f}, {X.max():.3f}]")
    return X, gene_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CellJEPA + SIGReg PBMC-68K pre-training")
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--drive_dir",
        default="/content/drive/MyDrive/CellJEPA_results/pbmc68k_pretrain/",
        help="Directory for checkpoints and gene name files",
    )
    p.add_argument(
        "--tar_path", default=None,
        help="Path to fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz "
             "(required for full run; ignored in smoke test mode)",
    )
    p.add_argument("--cache_dir", default="/content",
                   help="Directory to extract the tarball into")
    p.add_argument("--smoke_test", action="store_true",
                   help="500 cells, 1 epoch — quick correctness check")
    p.add_argument("--skip_jepa", action="store_true",
                   help="Skip Cell-JEPA pre-training")
    p.add_argument("--skip_sigreg", action="store_true",
                   help="Skip SIGReg pre-training")
    p.add_argument("--w_sigreg", type=float, default=0.5)
    p.add_argument("--n_directions", type=int, default=256)
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

    X, gene_names = load_pbmc68k_data(
        smoke_test=args.smoke_test,
        tar_path=args.tar_path,
        cache_dir=args.cache_dir,
    )

    gene_path = os.path.join(args.drive_dir, "pbmc68k_gene_names.json")
    with open(gene_path, "w") as f:
        json.dump(gene_names, f)
    print(f"Gene names saved to {gene_path}")

    n_genes = X.shape[1]
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    vocab_size = n_genes + 2
    labels = np.zeros(X.shape[0], dtype=np.int32)

    l_max = 64 if args.smoke_test else 600
    bs     = 8  if args.smoke_test else args.batch_size
    n_dir  = 64 if args.smoke_test else args.n_directions
    warmup = 100 if args.smoke_test else 1000

    dataset = SingleCellDataset(
        X, gene_vocab, labels,
        n_bins=50, L_max=l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )

    # ------------------------------------------------------------------
    # Cell-JEPA pre-training
    # ------------------------------------------------------------------
    if not args.skip_jepa:
        print(f"\n{'='*60}")
        print("  Cell-JEPA pre-training on PBMC-68K")
        print("=" * 60)

        jepa_model = CellJEPA(
            vocab_size=vocab_size, n_bins=50,
            d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
            dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
        )
        n_params = sum(p.numel() for p in jepa_model.parameters() if p.requires_grad)
        print(f"  CellJEPA: {n_params/1e6:.1f}M params")

        jepa_trainer = Pretrainer(jepa_model, dataset, PretrainConfig(
            lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
            batch_size=bs, n_epochs=args.n_epochs,
            w_rec=1.0, w_jepa=1000.0,
            log_every=50, num_workers=0,
        ), device)

        def save_jepa_epoch(epoch: int):
            path = os.path.join(args.drive_dir, f"pbmc68k_jepa_epoch{epoch}.pt")
            jepa_trainer.save(path)

        jepa_trainer.train(epoch_callback=save_jepa_epoch)

        jepa_final = os.path.join(args.drive_dir, "pbmc68k_jepa_final.pt")
        jepa_trainer.save(jepa_final)
        print(f"  Cell-JEPA checkpoint saved to {jepa_final}")

    # ------------------------------------------------------------------
    # SIGReg pre-training
    # ------------------------------------------------------------------
    if not args.skip_sigreg:
        print(f"\n{'='*60}")
        print("  CellJEPA_SIGReg pre-training on PBMC-68K")
        print("=" * 60)

        sigreg_model = CellJEPA_SIGReg(
            vocab_size=vocab_size, n_bins=50,
            d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
            dropout=0.2, n_views=2,
            grad_checkpoint=True,
        )
        n_params = sum(p.numel() for p in sigreg_model.parameters() if p.requires_grad)
        print(f"  CellJEPA_SIGReg: {n_params/1e6:.1f}M params")

        sigreg_trainer = SIGRegPretrainer(sigreg_model, dataset, SIGRegPretrainConfig(
            lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
            warmup_steps=warmup,
            batch_size=bs, n_epochs=args.n_epochs,
            log_every=50, n_views=2,
            w_sim=1.0, w_sigreg=args.w_sigreg, w_rec=1.0,
            n_directions=n_dir,
            num_workers=0,
        ), device)

        def save_sigreg_epoch(epoch: int):
            path = os.path.join(args.drive_dir, f"pbmc68k_sigreg_epoch{epoch}.pt")
            sigreg_trainer.save(path)

        sigreg_trainer.train(epoch_callback=save_sigreg_epoch)

        sigreg_final = os.path.join(args.drive_dir, "pbmc68k_sigreg_final.pt")
        sigreg_trainer.save(sigreg_final)
        print(f"  SIGReg checkpoint saved to {sigreg_final}")

    print(f"\nPre-training complete.")
    print(f"Gene names: {gene_path}")
    print(f"\nNext step: run_transfer_pbmc68k.py with:")
    if not args.skip_jepa:
        print(f"  --jepa_checkpoint {os.path.join(args.drive_dir, 'pbmc68k_jepa_final.pt')}")
    if not args.skip_sigreg:
        print(f"  --sigreg_checkpoint {os.path.join(args.drive_dir, 'pbmc68k_sigreg_final.pt')}")
    print(f"  --pbmc68k_genes {gene_path}")


if __name__ == "__main__":
    main()

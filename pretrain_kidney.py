"""
pretrain_kidney.py
==================
Pre-train CellJEPA on human kidney scRNA-seq data from CELLxGENE Census
(~700k–800k cells spanning multiple studies), then save the checkpoint so
it can be loaded by compare_pbmc3k.py for fine-tuning/evaluation on PBMC 3k.

Usage (Colab A100):
    python pretrain_kidney.py --device cuda --drive_dir /content/drive/MyDrive/CellJEPA_results/kidney_pretrain/

Resume after session expiry:
    python pretrain_kidney.py --device cuda --drive_dir ... --resume /content/drive/.../kidney_pretrain_epoch2.pt

Smoke test (500 cells, 1 epoch):
    python pretrain_kidney.py --smoke_test --device cuda
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scanpy as sc
import torch

from cell_jepa import CellJEPA
from preprocessing import SingleCellDataset
from trainer import Pretrainer, PretrainConfig

N_HVG = 2000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kidney_data(smoke_test: bool = False, gene_list: list[str] | None = None):
    """
    Download human kidney cells from CELLxGENE Census, apply QC filtering,
    and select 2000 highly variable genes (or align to a pre-specified gene list).

    Args:
        smoke_test: Subsample to 500 cells for quick testing.
        gene_list:  If provided, subset kidney data to exactly these genes
                    (must match var_names after QC). Used to align with
                    Adamson gene vocabulary for perturbation fine-tuning.

    Returns:
        X          : (N, N_HVG) float32 log-normalised expression matrix
        gene_names : list of gene symbol strings (length N_HVG)
    """
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError(
            "cellxgene-census is required: pip install cellxgene-census"
        )

    print("Querying CELLxGENE Census for human kidney cells...")
    census = cellxgene_census.open_soma()
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=(
            "tissue_general == 'kidney' "
            "and is_primary_data == True"
        ),
    )
    census.close()
    print(f"  Downloaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    if smoke_test:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, min(500, adata.n_obs), replace=False)
        adata = adata[idx].copy()
        print(f"  [smoke test] Subsampled to {adata.n_obs} cells")

    # Basic QC
    sc.pp.filter_cells(adata, min_genes=200)
    print(f"  After filter_cells: {adata.n_obs:,} cells")
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After filter_genes: {adata.n_vars:,} genes")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"  Normalised and log1p transformed")

    # Gene selection: use pre-specified list or select HVGs
    if gene_list is not None:
        available = set(adata.var_names)
        common = [g for g in gene_list if g in available]
        missing = len(gene_list) - len(common)
        if missing > 0:
            print(f"  Warning: {missing}/{len(gene_list)} genes from gene_list not in Census data — using {len(common)} common genes")
        adata = adata[:, common].copy()
    else:
        # Select 2000 HVGs so the vocabulary aligns with our PBMC 3k experiments
        sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()

    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Drop cells with no expressed genes after HVG filtering
    expressed = (X > 0).any(axis=1)
    n_dropped = (~expressed).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} cells with zero expressed HVGs")
        X = X[expressed]

    gene_names = list(adata.var_names)
    print(f"  Final: {X.shape[0]:,} cells × {X.shape[1]:,} HVGs")
    if X.shape[0] == 0:
        raise RuntimeError("No cells remaining after QC and HVG filtering — check Census data format.")
    if X.shape[1] == 0:
        raise RuntimeError("No genes remaining after HVG filtering.")
    print(f"  Expression range: [{X.min():.3f}, {X.max():.3f}]  (should be log-normalised, ~0–10)")
    return X, gene_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CellJEPA kidney pre-training")
    p.add_argument("--n_epochs", type=int, default=4,
                   help="Number of pre-training epochs (default: 4)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--drive_dir",
        default="/content/drive/MyDrive/CellJEPA_results/kidney_pretrain/",
        help="Google Drive directory for checkpoints and gene name file",
    )
    p.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint (.pt) to resume pre-training from",
    )
    p.add_argument(
        "--gene_list", default=None,
        help="Path to JSON file with a list of gene names to use as vocabulary "
             "(e.g. adamson_genes.json). If provided, skips HVG selection and "
             "aligns kidney data to the supplied gene set.",
    )
    p.add_argument(
        "--smoke_test", action="store_true",
        help="Quick test: 500 cells, 1 epoch",
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
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load kidney data
    # ------------------------------------------------------------------
    gene_list = None
    if args.gene_list:
        with open(args.gene_list) as f:
            gene_list = json.load(f)
        print(f"Using pre-specified gene list: {len(gene_list)} genes from {args.gene_list}")

    X, gene_names = load_kidney_data(smoke_test=args.smoke_test, gene_list=gene_list)

    # Persist gene names so compare_pbmc3k.py can align PBMC 3k genes later
    gene_path = os.path.join(args.drive_dir, "kidney_gene_names.json")
    with open(gene_path, "w") as f:
        json.dump(gene_names, f)
    print(f"Gene names saved to {gene_path}")

    # ------------------------------------------------------------------
    # Build dataset and model
    # ------------------------------------------------------------------
    n_genes = X.shape[1]
    # Token IDs: 0=<cls>, 1=<pad>, 2..n_genes+1=genes
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    vocab_size = n_genes + 2

    # Dummy labels (pre-training is unsupervised)
    labels = np.zeros(X.shape[0], dtype=np.int32)

    dataset = SingleCellDataset(
        X, gene_vocab, labels,
        n_bins=50, L_max=600,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )

    model = CellJEPA(
        vocab_size=vocab_size, n_bins=50,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CellJEPA: {n_params / 1e6:.1f}M trainable parameters")

    # ------------------------------------------------------------------
    # Build or resume trainer
    # ------------------------------------------------------------------
    config = PretrainConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        weight_decay=2e-4,
        lr_decay=0.9,
        mask_ratio=0.15,
        w_rec=1.0,
        w_jepa=1000.0,
        num_workers=2,
        log_every=50,
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer = Pretrainer.load_checkpoint(
            args.resume, model, dataset, device=device
        )
        # Honour --n_epochs even when resuming (allows extending training)
        trainer.config.n_epochs = args.n_epochs
    else:
        trainer = Pretrainer(model, dataset, config=config, device=device)

    # ------------------------------------------------------------------
    # Train with per-epoch checkpoint saves to Drive
    # ------------------------------------------------------------------
    def save_epoch_ckpt(epoch: int):
        path = os.path.join(args.drive_dir, f"kidney_pretrain_epoch{epoch}.pt")
        trainer.save(path)

    print(f"\n--- Pre-training ({args.n_epochs} epochs) ---")
    trainer.train(epoch_callback=save_epoch_ckpt)

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    final_path = os.path.join(args.drive_dir, "kidney_pretrain_final.pt")
    trainer.save(final_path)
    print(f"\nPre-training complete. Final checkpoint: {final_path}")
    print(f"Gene names: {gene_path}")
    print("\nNext step: run compare_pbmc3k.py with:")
    print(f"  --kidney_checkpoint {final_path}")
    print(f"  --kidney_genes {gene_path}")


if __name__ == "__main__":
    main()

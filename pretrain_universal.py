"""
pretrain_universal.py — Pre-train with Universal Gene Vocabulary
================================================================
Pre-trains CellJEPA and CellJEPA_SIGReg on kidney or PBMC-68K data using
a shared universal gene vocabulary (built by build_universal_vocab.py).

The universal vocab maps every gene symbol to a fixed token ID regardless of
which dataset it came from. This eliminates the HVG-overlap problem seen in
the gene-aligned transfer experiments (7% overlap for kidney→PBMC-3K,
18% for PBMC-68K→PBMC-3K).

At training time, each cell randomly samples up to L_max=600 of its expressed
genes from the full universal gene set — exactly as described in the Cell-JEPA
paper (Section 2.1). Since PBMC-3K uses the same token IDs at fine-tuning time,
no alignment step is needed.

Usage (Colab A100):
    # Build vocab first (once):
    python build_universal_vocab.py \\
        --tar_path /path/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz \\
        --drive_dir /content/drive/MyDrive/CellJEPA_results/universal_vocab/

    # Pre-train on kidney:
    python pretrain_universal.py \\
        --source kidney \\
        --vocab_file /content/drive/MyDrive/CellJEPA_results/universal_vocab/universal_gene_names.json \\
        --drive_dir /content/drive/MyDrive/CellJEPA_results/universal_kidney/ \\
        --device cuda

    # Pre-train on PBMC-68K:
    python pretrain_universal.py \\
        --source pbmc68k \\
        --vocab_file /content/drive/MyDrive/CellJEPA_results/universal_vocab/universal_gene_names.json \\
        --tar_path /path/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz \\
        --drive_dir /content/drive/MyDrive/CellJEPA_results/universal_pbmc68k/ \\
        --device cuda

    # Smoke test:
    python pretrain_universal.py --smoke_test --source kidney --device cpu \\
        --vocab_file /path/universal_gene_names.json

    # Skip one model:
    python pretrain_universal.py --source kidney --skip_jepa --device cuda ...
    python pretrain_universal.py --source kidney --skip_sigreg --device cuda ...
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from cell_jepa import CellJEPA
from cell_sigreg import CellJEPA_SIGReg
from preprocessing import SingleCellDataset
from trainer import (
    Pretrainer, PretrainConfig,
    SIGRegPretrainer, SIGRegPretrainConfig,
)

N_HVG_FALLBACK = 2000  # used only if Census returns fewer genes than expected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CellJEPA / SIGReg pre-training with universal gene vocabulary"
    )
    p.add_argument("--source", choices=["kidney", "pbmc68k", "multitissue"], required=True,
                   help="Pre-training data source. 'multitissue' samples uniformly across "
                        "6 non-blood tissues (kidney, lung, liver, brain, heart, intestine).")
    p.add_argument("--vocab_file", required=True,
                   help="Path to universal_gene_names.json from build_universal_vocab.py")
    p.add_argument("--n_epochs",   type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--drive_dir",
                   default="/content/drive/MyDrive/CellJEPA_results/universal_pretrain/",
                   help="Output directory for checkpoints")
    p.add_argument("--tar_path",   default=None,
                   help="PBMC-68K tarball path (required when --source pbmc68k, "
                        "ignored for --source kidney)")
    p.add_argument("--cache_dir",  default="/content",
                   help="Directory to extract PBMC-68K tarball")
    p.add_argument("--smoke_test", action="store_true",
                   help="500 cells, 1 epoch — quick correctness check")
    p.add_argument("--skip_jepa",   action="store_true")
    p.add_argument("--skip_sigreg", action="store_true")
    p.add_argument("--w_sigreg",    type=float, default=0.5)
    p.add_argument("--n_directions", type=int,  default=256)
    p.add_argument("--n_cells_per_tissue", type=int, default=None,
                   help="Cells per tissue for --source multitissue. "
                        "Default: 500 (smoke) or 8333 (full, giving ~50k across 6 tissues).")
    p.add_argument("--resume_jepa",   default=None,
                   help="Resume Cell-JEPA from this checkpoint (.pt)")
    p.add_argument("--resume_sigreg", default=None,
                   help="Resume SIGReg from this checkpoint (.pt)")
    p.add_argument("--skip_epoch_saves", action="store_true",
                   help="Skip per-epoch checkpoint saves (saves Drive space); "
                        "only the final checkpoint is written")
    p.add_argument("--data_cache_dir", default=None,
                   help="Directory to cache downloaded tissue arrays as .npy files. "
                        "Use a Drive path (e.g. /content/drive/MyDrive/CellJEPA_results/data_cache/) "
                        "to avoid re-downloading on subsequent runs.")
    return p.parse_args()


def get_device(s: str) -> torch.device:
    if s == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(s)
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading — project into universal vocab space
# ---------------------------------------------------------------------------

def _project_to_universal(
    X_raw: np.ndarray,
    source_gene_names: list[str],
    universal_gene_names: list[str],
) -> tuple[np.ndarray, int]:
    """
    Re-index a (cells × source_genes) matrix into (cells × universal_genes).
    Missing universal genes are left as zero. Returns (X_universal, n_mapped).
    """
    uni_idx = {g: i for i, g in enumerate(universal_gene_names)}
    n_uni   = len(universal_gene_names)
    n_cells = X_raw.shape[0]

    X_uni = np.zeros((n_cells, n_uni), dtype=np.float32)
    n_mapped = 0
    for src_idx, gene in enumerate(source_gene_names):
        if gene in uni_idx:
            X_uni[:, uni_idx[gene]] = X_raw[:, src_idx]
            n_mapped += 1

    return X_uni, n_mapped


def load_kidney_universal(
    universal_gene_names: list[str],
    smoke_test: bool = False,
) -> np.ndarray:
    """
    Download kidney cells from CELLxGENE Census, apply QC, then project
    into the universal gene vocabulary space.
    """
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError("pip install cellxgene-census")

    n_target = 500 if smoke_test else 50_000
    print(f"Querying CELLxGENE Census for kidney cells (target {n_target:,}) …")

    census = cellxgene_census.open_soma()
    obs_df = census["census_data"]["homo_sapiens"]["obs"].read(
        value_filter="tissue_general == 'kidney' and is_primary_data == True",
        column_names=["soma_joinid"],
    ).concat().to_pandas()
    census.close()

    n_sample = min(n_target, len(obs_df))
    sampled  = obs_df["soma_joinid"].sample(n=n_sample, random_state=42).tolist()
    print(f"  Downloading {n_sample:,} kidney cells …")

    census = cellxgene_census.open_soma()
    adata  = cellxgene_census.get_anndata(
        census=census, organism="Homo sapiens", obs_coords=sampled,
    )
    census.close()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if "feature_name" in adata.var.columns:
        src_genes = list(adata.var["feature_name"].astype(str))
    else:
        src_genes = list(adata.var_names)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    X_uni, n_mapped = _project_to_universal(X, src_genes, universal_gene_names)
    print(f"  Mapped {n_mapped}/{len(universal_gene_names)} universal genes from kidney "
          f"({n_mapped/len(universal_gene_names)*100:.1f}%)")
    print(f"  Final: {X_uni.shape[0]:,} cells × {X_uni.shape[1]:,} genes (universal)")

    expressed = (X_uni > 0).any(axis=1)
    if (~expressed).sum():
        print(f"  Dropping {(~expressed).sum()} cells with no expressed universal genes")
        X_uni = X_uni[expressed]
    return X_uni


# Tissues used for multi-tissue pre-training.
# Blood / PBMC deliberately excluded so PBMC-3K remains a held-out transfer target.
MULTITISSUE_TISSUES = [
    "kidney",
    "lung",
    "liver",
    "brain",
    "heart",
    "intestine",
]


def _load_tissue_cells(
    tissue: str,
    n_cells: int,
    universal_gene_names: list[str],
    seed: int,
    data_cache_dir: str | None = None,
) -> np.ndarray:
    """
    Download `n_cells` cells from a single tissue via CELLxGENE Census,
    apply QC, and project into the universal vocab space.
    Returns (n_cells_after_qc, n_universal) float32 array.

    If data_cache_dir is set, the processed array is saved/loaded as a .npy
    file named {tissue}_{n_cells}cells_universal.npy to avoid re-downloading.
    """
    if data_cache_dir is not None:
        os.makedirs(data_cache_dir, exist_ok=True)
        cache_path = os.path.join(
            data_cache_dir, f"{tissue}_{n_cells}cells_universal.npy"
        )
        if os.path.exists(cache_path):
            print(f"  [{tissue}] loading from cache: {cache_path}")
            X_uni = np.load(cache_path)
            print(f"  [{tissue}] {X_uni.shape[0]:,} cells (cached)")
            return X_uni
    else:
        cache_path = None

    import cellxgene_census

    print(f"  [{tissue}] querying Census …")
    census = cellxgene_census.open_soma()
    obs_df = census["census_data"]["homo_sapiens"]["obs"].read(
        value_filter=(
            f"tissue_general == '{tissue}' and is_primary_data == True"
        ),
        column_names=["soma_joinid"],
    ).concat().to_pandas()
    census.close()

    n_available = len(obs_df)
    n_sample = min(n_cells, n_available)
    if n_sample == 0:
        print(f"  [{tissue}] WARNING: no cells found, skipping")
        return np.zeros((0, len(universal_gene_names)), dtype=np.float32)

    sampled = obs_df["soma_joinid"].sample(n=n_sample, random_state=seed).tolist()
    print(f"  [{tissue}] downloading {n_sample:,} / {n_available:,} available cells …")

    census = cellxgene_census.open_soma()
    adata = cellxgene_census.get_anndata(
        census=census, organism="Homo sapiens", obs_coords=sampled,
    )
    census.close()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if "feature_name" in adata.var.columns:
        src_genes = list(adata.var["feature_name"].astype(str))
    else:
        src_genes = list(adata.var_names)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    X_uni, n_mapped = _project_to_universal(X, src_genes, universal_gene_names)

    expressed = (X_uni > 0).any(axis=1)
    X_uni = X_uni[expressed]
    print(f"  [{tissue}] {X_uni.shape[0]:,} cells retained  "
          f"({n_mapped}/{len(universal_gene_names)} genes mapped, "
          f"{n_mapped/len(universal_gene_names)*100:.1f}%)")

    if cache_path is not None:
        np.save(cache_path, X_uni)
        size_mb = os.path.getsize(cache_path) / 1e6
        print(f"  [{tissue}] cached to {cache_path} ({size_mb:.0f} MB)")

    return X_uni


def load_multitissue_universal(
    universal_gene_names: list[str],
    n_cells_per_tissue: int = 8333,
    smoke_test: bool = False,
    data_cache_dir: str | None = None,
) -> np.ndarray:
    """
    Download cells uniformly from MULTITISSUE_TISSUES (excluding blood/PBMC),
    project into the universal vocab, and return a single concatenated matrix.

    With default n_cells_per_tissue=8333 and 6 tissues the total is ~50k cells.
    Blood and PBMC are intentionally excluded so PBMC-3K stays a held-out
    cross-tissue transfer target.

    Args:
        universal_gene_names: ordered list from universal_gene_names.json
        n_cells_per_tissue:   cells to sample per tissue (after QC some will drop)
        smoke_test:           use 50 cells per tissue for a quick sanity check
        data_cache_dir:       if set, cache each tissue array as a .npy file here
    """
    try:
        import cellxgene_census  # noqa: F401
    except ImportError:
        raise ImportError("pip install cellxgene-census")

    if smoke_test:
        n_cells_per_tissue = 50

    print(f"\nMulti-tissue pre-training: {len(MULTITISSUE_TISSUES)} tissues, "
          f"target {n_cells_per_tissue:,} cells each")
    print(f"Tissues: {', '.join(MULTITISSUE_TISSUES)}")
    print("(blood/PBMC excluded — PBMC-3K is the held-out transfer target)\n")
    if data_cache_dir:
        print(f"Data cache: {data_cache_dir}\n")

    chunks = []
    for i, tissue in enumerate(MULTITISSUE_TISSUES):
        # Use different seeds per tissue so sampling is independent
        X_tissue = _load_tissue_cells(
            tissue, n_cells_per_tissue, universal_gene_names,
            seed=42 + i, data_cache_dir=data_cache_dir,
        )
        if X_tissue.shape[0] > 0:
            chunks.append(X_tissue)

    X_all = np.concatenate(chunks, axis=0)

    # Shuffle so batches see mixed tissues rather than one tissue at a time
    rng = np.random.default_rng(42)
    perm = rng.permutation(X_all.shape[0])
    X_all = X_all[perm]

    print(f"\nMulti-tissue dataset: {X_all.shape[0]:,} cells × "
          f"{X_all.shape[1]:,} genes  "
          f"({X_all.shape[0]*X_all.shape[1]*4/1e9:.1f} GB)\n")
    return X_all


def load_pbmc68k_universal(
    universal_gene_names: list[str],
    smoke_test: bool = False,
    tar_path: str | None = None,
    cache_dir: str = "/content",
) -> np.ndarray:
    """
    Load PBMC-68K (full tarball or smoke-test reduced), apply QC, then project
    into the universal gene vocabulary space.
    """
    if smoke_test:
        print("Loading PBMC-68K (reduced built-in) …")
        adata = sc.datasets.pbmc68k_reduced()
        if adata.raw is not None:
            adata = adata.raw.to_adata()
        adata.var_names_make_unique()
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, min(500, adata.n_obs), replace=False)
        adata = adata[idx].copy()
        print(f"  [smoke test] {adata.n_obs} cells")
        src_genes = list(adata.var_names)

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

    else:
        if tar_path is None:
            raise ValueError(
                "--tar_path is required for full PBMC-68K run. "
                "Pass --tar_path /path/to/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
            )
        out_dir    = os.path.join(cache_dir, "pbmc68k")
        matrix_dir = os.path.join(out_dir, "filtered_matrices_mex", "hg19")
        if not os.path.exists(matrix_dir):
            print(f"Extracting {tar_path} …")
            with tarfile.open(tar_path) as tar:
                tar.extractall(out_dir)

        print("Loading PBMC-68K from 10x matrix market files …")
        adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", cache=False)
        adata.var_names_make_unique()
        print(f"  Raw: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        src_genes = list(adata.var_names)

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        expressed = (X > 0).any(axis=1)
        if (~expressed).sum():
            print(f"  Dropping {(~expressed).sum()} all-zero cells")
            X = X[expressed]

    X_uni, n_mapped = _project_to_universal(X, src_genes, universal_gene_names)
    print(f"  Mapped {n_mapped}/{len(universal_gene_names)} universal genes from PBMC-68K "
          f"({n_mapped/len(universal_gene_names)*100:.1f}%)")
    print(f"  Final: {X_uni.shape[0]:,} cells × {X_uni.shape[1]:,} genes (universal)")

    expressed = (X_uni > 0).any(axis=1)
    if (~expressed).sum():
        print(f"  Dropping {(~expressed).sum()} cells with no expressed universal genes")
        X_uni = X_uni[expressed]
    return X_uni


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.drive_dir, exist_ok=True)

    if args.smoke_test:
        args.n_epochs = 1
        print("[smoke test] 1 epoch, 500 cells")

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load universal vocabulary
    with open(args.vocab_file) as f:
        universal_gene_names: list[str] = json.load(f)
    n_genes    = len(universal_gene_names)
    vocab_size = n_genes + 2  # 0=<cls>, 1=<pad>
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    print(f"Universal vocab: {n_genes:,} genes  →  vocab_size={vocab_size:,}")

    # Load pre-training data projected into universal space
    if args.source == "kidney":
        X = load_kidney_universal(universal_gene_names, smoke_test=args.smoke_test)
        prefix = "kidney_universal"
    elif args.source == "multitissue":
        n_per_tissue = args.n_cells_per_tissue
        if n_per_tissue is None:
            n_per_tissue = 50 if args.smoke_test else 8333
        X = load_multitissue_universal(
            universal_gene_names,
            n_cells_per_tissue=n_per_tissue,
            smoke_test=args.smoke_test,
            data_cache_dir=args.data_cache_dir,
        )
        prefix = "multitissue_universal"
    else:
        X = load_pbmc68k_universal(
            universal_gene_names,
            smoke_test=args.smoke_test,
            tar_path=args.tar_path,
            cache_dir=args.cache_dir,
        )
        prefix = "pbmc68k_universal"

    labels  = np.zeros(X.shape[0], dtype=np.int32)
    l_max   = 64  if args.smoke_test else 600
    bs      = 8   if args.smoke_test else args.batch_size
    n_dir   = 64  if args.smoke_test else args.n_directions
    warmup  = 100 if args.smoke_test else 1000

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
        print(f"  Cell-JEPA pre-training ({args.source}, universal vocab)")
        print("=" * 60)

        jepa_model = CellJEPA(
            vocab_size=vocab_size, n_bins=50,
            d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
            dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
        )
        n_params = sum(p.numel() for p in jepa_model.parameters() if p.requires_grad)
        print(f"  CellJEPA: {n_params/1e6:.1f}M params  "
              f"(gene embedding table: {n_genes}×512 = {n_genes*512/1e6:.1f}M of those)")

        jepa_config = PretrainConfig(
            lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
            batch_size=bs, n_epochs=args.n_epochs,
            w_rec=1.0, w_jepa=1000.0,
            log_every=50, num_workers=0,
        )
        if args.resume_jepa:
            jepa_trainer = Pretrainer.load_checkpoint(
                args.resume_jepa, jepa_model, dataset, device=device)
            jepa_trainer.config.n_epochs = args.n_epochs
        else:
            jepa_trainer = Pretrainer(jepa_model, dataset, jepa_config, device)

        def save_jepa_epoch(epoch: int):
            path = os.path.join(args.drive_dir, f"{prefix}_jepa_epoch{epoch}.pt")
            jepa_trainer.save(path)

        jepa_trainer.train(epoch_callback=save_jepa_epoch)

        jepa_final = os.path.join(args.drive_dir, f"{prefix}_jepa_final.pt")
        jepa_trainer.save(jepa_final)
        print(f"  Cell-JEPA checkpoint saved to {jepa_final}")

    # ------------------------------------------------------------------
    # SIGReg pre-training
    # ------------------------------------------------------------------
    if not args.skip_sigreg:
        print(f"\n{'='*60}")
        print(f"  CellJEPA_SIGReg pre-training ({args.source}, universal vocab)")
        print("=" * 60)

        sigreg_model = CellJEPA_SIGReg(
            vocab_size=vocab_size, n_bins=50,
            d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
            dropout=0.2, n_views=2,
            grad_checkpoint=True,
        )
        n_params = sum(p.numel() for p in sigreg_model.parameters() if p.requires_grad)
        print(f"  CellJEPA_SIGReg: {n_params/1e6:.1f}M params")

        sigreg_config = SIGRegPretrainConfig(
            lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
            warmup_steps=warmup,
            batch_size=bs, n_epochs=args.n_epochs,
            log_every=50, n_views=2,
            w_sim=1.0, w_sigreg=args.w_sigreg, w_rec=1.0,
            n_directions=n_dir,
            num_workers=0,
        )
        if args.resume_sigreg:
            sigreg_trainer = SIGRegPretrainer.load_checkpoint(
                args.resume_sigreg, sigreg_model, dataset, device=device)
            sigreg_trainer.config.n_epochs = args.n_epochs
        else:
            sigreg_trainer = SIGRegPretrainer(sigreg_model, dataset, sigreg_config, device)

        def save_sigreg_epoch(epoch: int):
            if args.skip_epoch_saves:
                return
            path = os.path.join(args.drive_dir, f"{prefix}_sigreg_epoch{epoch}.pt")
            sigreg_trainer.save(path, epoch=epoch)

        sigreg_trainer.train(epoch_callback=save_sigreg_epoch)

        sigreg_final = os.path.join(args.drive_dir, f"{prefix}_sigreg_final.pt")
        sigreg_trainer.save(sigreg_final)
        print(f"  SIGReg checkpoint saved to {sigreg_final}")

    print(f"\nPre-training complete. Checkpoints in {args.drive_dir}")
    print(f"\nNext step: run_transfer_universal.py with:")
    print(f"  --vocab_file {args.vocab_file}")
    if not args.skip_jepa:
        print(f"  --jepa_checkpoint {os.path.join(args.drive_dir, f'{prefix}_jepa_final.pt')}")
    if not args.skip_sigreg:
        print(f"  --sigreg_checkpoint {os.path.join(args.drive_dir, f'{prefix}_sigreg_final.pt')}")
    print(f"  --source {args.source}")


if __name__ == "__main__":
    main()

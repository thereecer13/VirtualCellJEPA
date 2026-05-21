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

import scipy.sparse as sp
import scanpy as sc

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

    # Swap fine-tuning / eval dataset
    p.add_argument("--eval_pbmc10k", action="store_true",
                   help="Fine-tune and evaluate on PBMC-10k (~11k cells) instead of "
                        "PBMC-3K. Provides ~4× more fine-tuning cells.")
    p.add_argument("--eval_heart", action="store_true",
                   help="Also evaluate on Heart Cell Atlas (non-blood, held-out domain).")

    # Robustness
    p.add_argument("--n_seeds", type=int, default=1,
                   help="Number of random seeds to average over. Reports mean ± std "
                        "when > 1. Each seed re-initialises the model and re-runs "
                        "pre-training + fine-tuning.")

    # Baselines
    p.add_argument("--skip_baselines", action="store_true",
                   help="Skip PCA and scVI baselines")
    p.add_argument("--scvi_epochs", type=int, default=100,
                   help="Training epochs for scVI baseline (default 100)")
    p.add_argument("--pca_components", type=int, default=50,
                   help="Number of PCA components (default 50)")
    p.add_argument("--w_jepa_finetune", type=float, default=1.0,
                   help="JEPA loss weight during Cell-JEPA fine-tuning (default 1.0). "
                        "Set to 0.0 to disable JEPA entirely during fine-tuning.")
    p.add_argument("--n_hvg", type=int, default=2000,
                   help="Number of highly variable genes to select from the eval dataset "
                        "before fine-tuning (default 2000, matching the Cell-JEPA paper). "
                        "Set to 0 to disable HVG selection and use all expressed genes.")

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
# Linear probe
# ---------------------------------------------------------------------------

def linear_probe_accuracy(
    train_emb: np.ndarray, train_labels: np.ndarray,
    test_emb:  np.ndarray, test_labels:  np.ndarray,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_emb)
    X_te = scaler.transform(test_emb)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, train_labels)
    return float(clf.score(X_te, test_labels))


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_pbmc10k_universal(
    universal_gene_names: list[str],
    n_cells_subset: int | None = None,
    n_hvg: int = 2000,
):
    """
    Load PBMC-10k (Zheng 2017, ~11k cells) via scvi-tools and project into
    the universal vocab space, matching the same format as load_pbmc3k_universal().
    Selects the top n_hvg HVGs before projecting (set n_hvg=0 to disable).
    Returns (X, int_labels, label_names, gene_coverage_n, gene_coverage_pct).
    """
    print("Loading PBMC-10k (Zheng 2017) via scvi-tools …")
    try:
        import scvi
    except ImportError:
        raise ImportError("pip install scvi-tools")

    adata = scvi.data.pbmc_dataset()
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Remap var_names to gene symbols if needed
    universal_set = set(universal_gene_names)
    if len(universal_set & set(adata.var_names)) < 10:
        for col in ["gene_symbols", "gene_names", "Symbol", "gene_name", "name"]:
            if col in adata.var.columns:
                adata.var_names = adata.var[col].astype(str).values
                adata.var_names_make_unique()
                print(f"  Remapped var_names via '{col}' column")
                break

    # HVG selection — compute on all cells before subsetting
    if n_hvg and n_hvg > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
        hvg_set = set(adata.var_names[adata.var["highly_variable"]])
        print(f"  Selected {len(hvg_set)} HVGs from PBMC-10k")
    else:
        hvg_set = None

    if n_cells_subset and adata.n_obs > n_cells_subset:
        import random; random.seed(42)
        idx = np.random.choice(adata.n_obs, n_cells_subset, replace=False)
        adata = adata[idx].copy()

    # Project into universal vocab space (HVGs only if selected)
    gene_to_col = {g: i for i, g in enumerate(universal_gene_names)}
    n_universal = len(universal_gene_names)
    X_full = np.zeros((adata.n_obs, n_universal), dtype=np.float32)
    present = [g for g in adata.var_names
               if g in gene_to_col and (hvg_set is None or g in hvg_set)]
    for g in present:
        col_src = adata.var_names.get_loc(g)
        col_dst = gene_to_col[g]
        x_col = adata.X[:, col_src]
        if sp.issparse(x_col):
            x_col = np.asarray(x_col.todense()).squeeze()
        X_full[:, col_dst] = x_col

    coverage_n   = len(present)
    coverage_pct = coverage_n / n_universal * 100
    hvg_note = f" ({n_hvg} HVGs)" if hvg_set is not None else ""
    print(f"  Gene coverage: {coverage_n}/{n_universal} universal genes mapped from PBMC-10k{hvg_note} ({coverage_pct:.1f}%)")

    # Cell type labels
    if "cell_types" in adata.obs.columns:
        label_col = "cell_types"
    elif "str_labels" in adata.obs.columns:
        label_col = "str_labels"
    else:
        label_col = adata.obs.columns[0]
    raw_labels  = adata.obs[label_col].astype(str).tolist()
    label_names = sorted(set(raw_labels))
    label_map   = {l: i for i, l in enumerate(label_names)}
    int_labels  = np.array([label_map[l] for l in raw_labels], dtype=np.int32)

    print(f"  {adata.n_obs} cells × {n_universal} genes (universal vocab)")
    print(f"  {len(label_names)} cell types: {label_names}")
    return X_full, int_labels, label_names, coverage_n, coverage_pct


def load_heart_universal(universal_gene_names: list[str], n_cells_subset: int | None = None):
    """
    Load Heart Cell Atlas subsampled dataset via scvi-tools and project into
    universal vocab space. Heart is a non-blood tissue not seen during pre-training,
    making it a held-out domain transfer test.
    """
    print("Loading Heart Cell Atlas (subsampled) via scvi-tools …")
    try:
        import scvi
    except ImportError:
        raise ImportError("pip install scvi-tools")

    adata = scvi.data.heart_cell_atlas_subsampled()
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Remap var_names to gene symbols if needed
    universal_set = set(universal_gene_names)
    if len(universal_set & set(adata.var_names)) < 10:
        for col in ["gene_symbols", "gene_names", "Symbol", "gene_name", "name"]:
            if col in adata.var.columns:
                adata.var_names = adata.var[col].astype(str).values
                adata.var_names_make_unique()
                print(f"  Remapped var_names via '{col}' column")
                break

    if n_cells_subset and adata.n_obs > n_cells_subset:
        idx = np.random.choice(adata.n_obs, n_cells_subset, replace=False)
        adata = adata[idx].copy()

    # Project into universal vocab space
    gene_to_col = {g: i for i, g in enumerate(universal_gene_names)}
    n_universal = len(universal_gene_names)
    X_full = np.zeros((adata.n_obs, n_universal), dtype=np.float32)
    present = [g for g in adata.var_names if g in gene_to_col]
    for g in present:
        col_src = adata.var_names.get_loc(g)
        col_dst = gene_to_col[g]
        x_col = adata.X[:, col_src]
        if sp.issparse(x_col):
            x_col = np.asarray(x_col.todense()).squeeze()
        X_full[:, col_dst] = x_col

    coverage_n   = len(present)
    coverage_pct = coverage_n / n_universal * 100
    print(f"  Gene coverage: {coverage_n}/{n_universal} universal genes present in Heart ({coverage_pct:.1f}%)")

    # Cell type labels — Heart Cell Atlas uses 'cell_type' column
    label_col = None
    for col in ["cell_type", "str_labels", "celltype", "cell_types"]:
        if col in adata.obs.columns:
            label_col = col
            break
    if label_col is None:
        label_col = adata.obs.columns[0]
    raw_labels  = adata.obs[label_col].astype(str).tolist()
    label_names = sorted(set(raw_labels))
    label_map   = {l: i for i, l in enumerate(label_names)}
    int_labels  = np.array([label_map[l] for l in raw_labels], dtype=np.int32)

    print(f"  {adata.n_obs} cells × {n_universal} genes (universal vocab)")
    print(f"  {len(label_names)} cell types: {label_names}")
    return X_full, int_labels, label_names, coverage_n, coverage_pct


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_full(
    emb: np.ndarray, labels: np.ndarray,
    train_emb: np.ndarray = None, train_labels: np.ndarray = None,
) -> dict:
    """Clustering metrics + linear probe accuracy."""
    result = evaluate_embeddings(emb, labels)
    if train_emb is not None:
        acc = linear_probe_accuracy(train_emb, train_labels, emb, labels)
        result["linear_probe"] = acc
        print(f"  Linear probe accuracy: {acc:.4f}")
    return result


def merge_seed_results(seed_results: list[dict]) -> dict:
    """Average a list of result dicts across seeds, adding mean ± std."""
    if len(seed_results) == 1:
        return seed_results[0]
    merged = {}
    for key in seed_results[0]:
        if key == "ft_history":
            merged[key] = seed_results[0][key]
            continue
        for phase in ("zero_shot", "fine_tuned"):
            if key not in (phase,):
                continue
            vals = {m: [r[key][m] for r in seed_results if m in r.get(key, {})]
                    for m in ["nmi", "ari", "asw", "avg_bio", "linear_probe"]}
            merged[key] = {m: float(np.mean(v)) for m, v in vals.items() if v}
            merged[f"{key}_std"] = {m: float(np.std(v)) for m, v in vals.items() if v}
    return merged


# ---------------------------------------------------------------------------
# Classical / scVI baselines
# ---------------------------------------------------------------------------

def run_pca_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    n_components: int = 50,
) -> dict:
    """PCA on log-normalised counts (already normalised in the loader)."""
    from sklearn.decomposition import PCA
    print(f"  Fitting PCA ({n_components} PCs) on {X_train.shape[0]} train cells …")
    pca = PCA(n_components=n_components, random_state=42)
    train_emb = pca.fit_transform(X_train)
    test_emb  = pca.transform(X_test)
    var_exp   = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {var_exp*100:.1f}%")
    result = evaluate_full(test_emb, y_test, train_emb, y_train)
    print(f"  PCA AvgBIO: {result['avg_bio']:.4f}")
    return {"zero_shot": result}


def run_scvi_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    n_latent: int = 30,
    n_epochs: int = 100,
    smoke_test: bool = False,
) -> dict:
    """
    scVI VAE baseline. Fits on train cells, embeds train+test via get_latent_representation().
    X_train/X_test must be log-normalised (as produced by the loaders).
    scVI internally treats counts, so we pass raw counts by reversing log1p → expm1.
    """
    import anndata
    try:
        import scvi as _scvi
    except ImportError:
        raise ImportError("pip install scvi-tools")

    actual_epochs = 3 if smoke_test else n_epochs
    print(f"  Fitting scVI (n_latent={n_latent}, {actual_epochs} epochs) "
          f"on {X_train.shape[0]} train cells …")

    # Reverse log1p to approximate raw counts (scVI works on count-scale data)
    X_tr_counts = np.expm1(X_train).astype(np.float32)
    X_te_counts = np.expm1(X_test).astype(np.float32)

    # Build AnnData objects
    adata_train = anndata.AnnData(X=X_tr_counts)
    adata_all   = anndata.AnnData(
        X=np.concatenate([X_tr_counts, X_te_counts], axis=0)
    )
    adata_all.obs["split"] = (
        ["train"] * len(X_tr_counts) + ["test"] * len(X_te_counts)
    )

    _scvi.settings.progress_bar_style = "tqdm"
    _scvi.model.SCVI.setup_anndata(adata_all)
    model = _scvi.model.SCVI(adata_all, n_latent=n_latent, n_layers=2)
    model.train(max_epochs=actual_epochs)

    latent = model.get_latent_representation(adata_all)
    train_emb = latent[: len(X_tr_counts)]
    test_emb  = latent[len(X_tr_counts):]

    result = evaluate_full(test_emb, y_test, train_emb, y_train)
    print(f"  scVI AvgBIO: {result['avg_bio']:.4f}")
    return {"zero_shot": result}


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def run_jepa_scratch(
    X_train, y_train, X_test, y_test,
    gene_vocab, vocab_size, args, device, seed: int = 0,
) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    n_bins = 50
    bs     = 8 if args.smoke_test else 128   # pre-train: single forward pass
    ft_bs  = 8 if args.smoke_test else 128   # fine-tune: student+teacher, ~2× memory

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
    train_eval_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
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
        log_every=20, w_rec=1.0, w_jepa=1000.0, num_workers=2,
    ), device).train()

    zs_emb, zs_labels   = extract_embeddings(model, eval_ds,       batch_size=bs, device=device)
    zs_tr_emb, zs_tr_lb = extract_embeddings(model, train_eval_ds, batch_size=bs, device=device)
    zs = evaluate_full(zs_emb, zs_labels, zs_tr_emb, zs_tr_lb)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    ft_history = Finetuner(model, finetune_ds, FinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=args.finetune_epochs,
        log_every=20, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_jepa=args.w_jepa_finetune, ecs_temperature=0.1,
        include_jepa=args.w_jepa_finetune > 0.0, num_workers=2,
    ), device).train()

    ft_emb, ft_labels   = extract_embeddings(model, eval_ds,       batch_size=bs, device=device)
    ft_tr_emb, ft_tr_lb = extract_embeddings(model, train_eval_ds, batch_size=bs, device=device)
    ft = evaluate_full(ft_emb, ft_labels, ft_tr_emb, ft_tr_lb)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft, "ft_history": ft_history}


def run_sigreg_scratch(
    X_train, y_train, X_test, y_test,
    gene_vocab, vocab_size, args, device, seed: int = 0,
) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    n_bins = 50
    bs     = 8 if args.smoke_test else 64   # pre-train: 2 views × forward pass
    ft_bs  = 8 if args.smoke_test else 64   # fine-tune: same 2-view budget

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
    train_eval_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
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
        num_workers=2,
    ), device).train()

    zs_emb, zs_labels   = extract_embeddings(model, eval_ds,       batch_size=bs, device=device)
    zs_tr_emb, zs_tr_lb = extract_embeddings(model, train_eval_ds, batch_size=bs, device=device)
    zs = evaluate_full(zs_emb, zs_labels, zs_tr_emb, zs_tr_lb)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    ft_history = SIGRegFinetuner(model, finetune_ds, SIGRegFinetuneConfig(
        lr=1e-4, lr_decay=0.9,
        batch_size=ft_bs, n_epochs=args.finetune_epochs,
        log_every=20, n_views=2,
        w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_sim=1.0, w_sigreg=0.5, ecs_temperature=0.1,
        n_directions=64 if args.smoke_test else 256,
        num_workers=2,
    ), device).train()

    ft_emb, ft_labels   = extract_embeddings(model, eval_ds,       batch_size=bs, device=device)
    ft_tr_emb, ft_tr_lb = extract_embeddings(model, train_eval_ds, batch_size=bs, device=device)
    ft = evaluate_full(ft_emb, ft_labels, ft_tr_emb, ft_tr_lb)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft, "ft_history": ft_history}


def run_pretrained(
    label: str,
    checkpoint: str,
    model_class,          # CellJEPA or CellJEPA_SIGReg
    finetuner_class,      # Finetuner or SIGRegFinetuner
    finetune_config_class,
    finetune_config_kwargs: dict,
    X_train, y_train, X_test, y_test,
    gene_vocab, vocab_size, args, device, seed: int = 0,
) -> dict:
    """Generic runner for pre-trained checkpoints (JEPA or SIGReg)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n_bins = 50
    # SIGReg processes 2 views per forward pass so needs a smaller batch
    is_sigreg = finetuner_class is SIGRegFinetuner
    bs = 8 if args.smoke_test else (32 if is_sigreg else 128)

    finetune_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )
    eval_ds = SingleCellDataset(
        X_test, gene_vocab, y_test,
        n_bins=n_bins, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
    )
    train_eval_ds = SingleCellDataset(
        X_train, gene_vocab, y_train,
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

    zs_emb, zs_labels   = extract_embeddings(model, eval_ds,       batch_size=bs, device=device)
    zs_tr_emb, zs_tr_lb = extract_embeddings(model, train_eval_ds, batch_size=bs, device=device)
    zs = evaluate_full(zs_emb, zs_labels, zs_tr_emb, zs_tr_lb)
    print(f"  Zero-shot AvgBIO: {zs['avg_bio']:.4f}")

    ft_history = finetuner_class(
        model, finetune_ds,
        finetune_config_class(
            n_epochs=1 if args.smoke_test else args.finetune_epochs,
            **finetune_config_kwargs,
        ),
        device,
    ).train()

    ft_emb, ft_labels   = extract_embeddings(model, eval_ds,       batch_size=bs, device=device)
    ft_tr_emb, ft_tr_lb = extract_embeddings(model, train_eval_ds, batch_size=bs, device=device)
    ft = evaluate_full(ft_emb, ft_labels, ft_tr_emb, ft_tr_lb)
    print(f"  Fine-tuned AvgBIO: {ft['avg_bio']:.4f}")
    return {"zero_shot": zs, "fine_tuned": ft, "ft_history": ft_history}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_loss_curves(results: dict, save_path: str = "loss_curves.png") -> None:
    import matplotlib.pyplot as plt

    conditions = {k: v for k, v in results.items() if v.get("ft_history")}
    if not conditions:
        print("No fine-tuning history to plot.")
        return

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (name, cond) in zip(axes[0], conditions.items()):
        history = cond["ft_history"]
        epochs     = list(range(1, len(history) + 1))
        train_loss = [h["train_loss"] for h in history]
        val_loss   = [h["val_loss"]   for h in history]

        ax.plot(epochs, train_loss, label="train", color="#4C72B0")
        ax.plot(epochs, val_loss,   label="val",   color="#DD8452", linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("Fine-tuning Loss Curves", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")

def print_and_save(results: dict, path: str, eval_dataset_label: str = "PBMC-3K") -> None:
    row = "  {:<48s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}  {:>8s}"
    sep = "  " + "-" * 90

    lines = []
    for phase_label, phase_key in [("Zero-shot", "zero_shot"), ("Fine-tuned", "fine_tuned")]:
        std_key = f"{phase_key}_std"
        lines += [
            "\n" + "=" * 94,
            f"  {phase_label} — Universal Vocab Transfer (→ {eval_dataset_label})",
            "=" * 94,
            row.format("Condition", "NMI", "ARI", "ASW", "AvgBIO", "LinProbe"),
            sep,
        ]
        for name, cond in results.items():
            res = cond.get(phase_key)
            std = cond.get(std_key, {})
            if not res:
                lines.append(row.format(name, "N/A", "N/A", "N/A", "N/A", "N/A"))
            else:
                def fmt(key):
                    v = res.get(key)
                    s = std.get(key)
                    if v is None:
                        return "N/A"
                    return f"{v:.4f}" if s is None else f"{v:.3f}±{s:.3f}"
                lines.append(row.format(
                    name, fmt("nmi"), fmt("ari"), fmt("asw"),
                    fmt("avg_bio"), fmt("linear_probe"),
                ))
        lines.append("=" * 94)

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

    # Load fine-tuning / eval dataset
    n_subset = 200 if args.smoke_test else None
    n_hvg = args.n_hvg if args.n_hvg > 0 else None
    if args.eval_pbmc10k:
        X_eval, int_labels, label_names, _, _ = load_pbmc10k_universal(
            universal_gene_names, n_cells_subset=n_subset, n_hvg=n_hvg,
        )
        eval_dataset_label = "PBMC-10k"
    else:
        X_eval, int_labels, label_names, _, _ = load_pbmc3k_universal(
            universal_gene_names, n_cells_subset=n_subset, n_hvg=n_hvg,
        )
        eval_dataset_label = "PBMC-3K"
    X_train, y_train, X_test, y_test = _train_test_split(X_eval, int_labels)
    print(f"\nTrain: {X_train.shape[0]} cells  |  Test: {X_test.shape[0]} cells")

    all_results: dict[str, dict] = {}

    shared_scratch = dict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gene_vocab=gene_vocab, vocab_size=vocab_size, args=args, device=device,
    )

    seeds = list(range(args.n_seeds))

    jepa_ft_kwargs = dict(
        lr=1e-4, lr_decay=0.9,
        batch_size=8 if args.smoke_test else 128,
        log_every=20, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_jepa=args.w_jepa_finetune, ecs_temperature=0.1,
        include_jepa=args.w_jepa_finetune > 0.0, num_workers=2,
    )
    sigreg_ft_kwargs = dict(
        lr=1e-4, lr_decay=0.9,
        batch_size=8 if args.smoke_test else 32,
        log_every=20, n_views=2,
        w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
        w_sim=1.0, w_sigreg=0.0, ecs_temperature=0.1,
        n_directions=64 if args.smoke_test else 256,
        num_workers=2,
    )

    shared = dict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gene_vocab=gene_vocab, vocab_size=vocab_size, args=args, device=device,
    )

    def run_seeds(result_key, fn, **kwargs):
        seed_res = []
        for s in seeds:
            if len(seeds) > 1:
                print(f"  [seed {s+1}/{len(seeds)}]")
            seed_res.append(fn(**kwargs, seed=s))
            torch.cuda.empty_cache()
        all_results[result_key] = merge_seed_results(seed_res)

    # ---- Classical / scVI baselines ----
    if not args.skip_baselines:
        print(f"\n{'='*94}\n  PCA baseline ({args.pca_components} PCs)\n{'='*94}")
        all_results["PCA (50 PCs)"] = run_pca_baseline(
            X_train, y_train, X_test, y_test,
            n_components=args.pca_components,
        )

        print(f"\n{'='*94}\n  scVI baseline\n{'='*94}")
        all_results["scVI"] = run_scvi_baseline(
            X_train, y_train, X_test, y_test,
            n_latent=30, n_epochs=args.scvi_epochs,
            smoke_test=args.smoke_test,
        )

    # ---- Scratch baselines ----
    if not args.skip_scratch:
        print(f"\n{'='*94}\n  Cell-JEPA (scratch, universal vocab)\n{'='*94}")
        run_seeds("Cell-JEPA scratch (universal)", run_jepa_scratch, **shared)

        print(f"\n{'='*94}\n  SIGReg (scratch, universal vocab)\n{'='*94}")
        run_seeds("SIGReg scratch (universal)", run_sigreg_scratch, **shared)

    # ---- Pre-trained conditions ----
    src = args.pretrain_source
    if not args.skip_kidney:
        if args.kidney_jepa_checkpoint:
            label = f"Cell-JEPA ({src}, universal)"
            print(f"\n{'='*94}\n  {label}\n{'='*94}")
            run_seeds(label, run_pretrained,
                      label=label, checkpoint=args.kidney_jepa_checkpoint,
                      model_class=CellJEPA, finetuner_class=Finetuner,
                      finetune_config_class=FinetuneConfig,
                      finetune_config_kwargs=jepa_ft_kwargs, **shared)
        else:
            print(f"\nSkipping Cell-JEPA ({src}) — no --kidney_jepa_checkpoint")

        if args.kidney_sigreg_checkpoint:
            label = f"SIGReg ({src}, universal)"
            print(f"\n{'='*94}\n  {label}\n{'='*94}")
            run_seeds(label, run_pretrained,
                      label=label, checkpoint=args.kidney_sigreg_checkpoint,
                      model_class=CellJEPA_SIGReg, finetuner_class=SIGRegFinetuner,
                      finetune_config_class=SIGRegFinetuneConfig,
                      finetune_config_kwargs=sigreg_ft_kwargs, **shared)
        else:
            print(f"\nSkipping SIGReg ({src}) — no --kidney_sigreg_checkpoint")

    # ---- PBMC-68K conditions ----
    if not args.skip_pbmc68k:
        if args.pbmc68k_jepa_checkpoint:
            label = "Cell-JEPA (PBMC-68K, universal)"
            print(f"\n{'='*94}\n  {label}\n{'='*94}")
            run_seeds(label, run_pretrained,
                      label=label, checkpoint=args.pbmc68k_jepa_checkpoint,
                      model_class=CellJEPA, finetuner_class=Finetuner,
                      finetune_config_class=FinetuneConfig,
                      finetune_config_kwargs=jepa_ft_kwargs, **shared)
        else:
            print("\nSkipping Cell-JEPA (PBMC-68K) — no --pbmc68k_jepa_checkpoint")

        if args.pbmc68k_sigreg_checkpoint:
            label = "SIGReg (PBMC-68K, universal)"
            print(f"\n{'='*94}\n  {label}\n{'='*94}")
            run_seeds(label, run_pretrained,
                      label=label, checkpoint=args.pbmc68k_sigreg_checkpoint,
                      model_class=CellJEPA_SIGReg, finetuner_class=SIGRegFinetuner,
                      finetune_config_class=SIGRegFinetuneConfig,
                      finetune_config_kwargs=sigreg_ft_kwargs, **shared)
        else:
            print("\nSkipping SIGReg (PBMC-68K) — no --pbmc68k_sigreg_checkpoint")

    print_and_save(all_results, args.results_file, eval_dataset_label)

    # ---- Heart Cell Atlas (held-out domain) ----
    if args.eval_heart:
        print("\n\n" + "=" * 94)
        print("  Heart Cell Atlas — Zero-shot Evaluation (held-out domain)")
        print("=" * 94)
        n_subset_heart = 200 if args.smoke_test else None
        X_heart, y_heart, _, _, _ = load_heart_universal(
            universal_gene_names, n_cells_subset=n_subset_heart,
        )
        X_h_tr, y_h_tr, X_h_te, y_h_te = _train_test_split(X_heart, y_heart)
        heart_results: dict[str, dict] = {}
        heart_eval_ds = SingleCellDataset(
            X_h_te, gene_vocab, y_h_te,
            n_bins=50, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
        )
        heart_train_ds = SingleCellDataset(
            X_h_tr, gene_vocab, y_h_tr,
            n_bins=50, L_max=args.l_max, cls_token_id=0, pad_token_id=1, mask_ratio=0.0,
        )
        bs = 8 if args.smoke_test else 64
        for label, cond in all_results.items():
            # Re-load checkpoint for pre-trained conditions — not available for scratch
            ckpt_key = None
            if "multitissue" in label or src in label:
                if "Cell-JEPA" in label and args.kidney_jepa_checkpoint:
                    ckpt_key = args.kidney_jepa_checkpoint
                elif "SIGReg" in label and args.kidney_sigreg_checkpoint:
                    ckpt_key = args.kidney_sigreg_checkpoint
            if ckpt_key is None:
                print(f"  Skipping {label} on Heart (no checkpoint — scratch models not re-evaluated)")
                continue
            model_class = CellJEPA if "Cell-JEPA" in label else CellJEPA_SIGReg
            model = model_class(
                vocab_size=vocab_size, n_bins=50,
                d_model=512, n_layers=12, n_heads=8, ffn_dim=2048, dropout=0.2,
                **({"ema_momentum": 0.996, "predictor_hidden": 512}
                   if model_class is CellJEPA else {"n_views": 2}),
            )
            ckpt = torch.load(ckpt_key, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
            h_emb, h_labels   = extract_embeddings(model, heart_eval_ds,   batch_size=bs, device=device)
            h_tr_emb, h_tr_lb = extract_embeddings(model, heart_train_ds,  batch_size=bs, device=device)
            h_res = evaluate_full(h_emb, h_labels, h_tr_emb, h_tr_lb)
            heart_results[label] = {"zero_shot": h_res}
            print(f"  {label}: AvgBIO={h_res['avg_bio']:.4f}  LinProbe={h_res.get('linear_probe', float('nan')):.4f}")
            torch.cuda.empty_cache()

        heart_file = args.results_file.replace(".txt", "_heart.txt")
        print_and_save(heart_results, heart_file, "Heart Cell Atlas (zero-shot)")


if __name__ == "__main__":
    main()

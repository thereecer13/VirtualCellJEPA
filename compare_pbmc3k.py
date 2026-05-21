"""
CellJEPA vs scGPT: Cell Clustering Comparison on PBMC3k
=========================================================
Evaluates both models on the canonical PBMC3k dataset using
NMI / ARI / ASW / AvgBIO (same suite as the CellJEPA paper).

Phase 1 — local smoke test (verify code is error-free):
    python compare_pbmc3k.py --smoke_test

Phase 2 — full comparison on EC2 (scGPT downloaded automatically):
    python compare_pbmc3k.py

CellJEPA is trained from scratch on PBMC3k (pre-train + fine-tune).
scGPT is used in zero-shot mode with the pre-trained whole-human checkpoint.
Both models are evaluated with an identical Louvain-sweep pipeline.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import scanpy as sc

from preprocessing import SingleCellDataset
from cell_jepa import CellJEPA
from trainer import Pretrainer, Finetuner, PretrainConfig, FinetuneConfig
from metrics import (
    extract_embeddings,
    louvain_cluster,
    compute_nmi,
    compute_ari,
    compute_asw,
    avg_bio,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CellJEPA vs scGPT on PBMC3k")
    p.add_argument(
        "--smoke_test", action="store_true",
        help="Run a tiny smoke test (200 cells, 1 epoch each, no scGPT).",
    )
    p.add_argument(
        "--scgpt_model_dir", default=None,
        help="Path to local scGPT checkpoint dir. Auto-downloaded if omitted.",
    )
    p.add_argument("--l_max", type=int, default=200,
                   help="Max genes per cell (200 = demo; 600 = paper).")
    p.add_argument("--pretrain_epochs", type=int, default=4)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--results_file", default="results.txt")
    p.add_argument("--no_jepa", action="store_true",
                   help="Disable JEPA objective (w_jepa=0) for ablation.")
    p.add_argument("--pretrain_10k", action="store_true",
                   help="Pre-train on PBMC 10k, fine-tune/eval on PBMC 3k.")
    p.add_argument("--w_jepa", type=float, default=None,
                   help="Override w_jepa weight directly. Ignores --no_jepa if set.")
    p.add_argument("--kidney_checkpoint", default=None,
                   help="Path to kidney pre-trained checkpoint (.pt) from pretrain_kidney.py.")
    p.add_argument("--kidney_genes", default=None,
                   help="Path to kidney gene names JSON saved by pretrain_kidney.py.")
    return p.parse_args()


def get_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pbmc3k(
    n_cells_subset: int | None = None,
    seed: int = 42,
):
    """
    Load PBMC3k raw counts with cell-type labels from the processed version.

    Returns:
        count_matrix : (N, G) float32 raw counts
        int_labels   : (N,)  int32 cell-type indices
        label_names  : list of label strings (length = n_types)
        adata_aligned: AnnData aligned to the same N cells (for scGPT)
    """
    print("Loading PBMC3k …")
    adata_raw  = sc.datasets.pbmc3k()
    adata_proc = sc.datasets.pbmc3k_processed()

    # Intersect barcodes: processed has 2638 cells after QC filtering
    common = adata_raw.obs_names.intersection(adata_proc.obs_names)
    adata_raw   = adata_raw[common].copy()
    adata_proc  = adata_proc[common]

    # Encode louvain string labels → integers
    labels_str  = adata_proc.obs["louvain"].values
    label_names = sorted(set(labels_str))
    lbl2int     = {l: i for i, l in enumerate(label_names)}
    int_labels  = np.array([lbl2int[l] for l in labels_str], dtype=np.int32)

    # Select top 2000 highly variable genes to keep memory manageable
    # (scGPT also uses HVG selection — 1200 HVGs — so this is fair)
    import scipy.sparse as sp
    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)
    sc.pp.highly_variable_genes(adata_raw, n_top_genes=2000, flavor="seurat")
    adata_raw = adata_raw[:, adata_raw.var["highly_variable"]].copy()
    print(f"  Reduced to {adata_raw.n_vars} highly variable genes")

    # Dense count matrix (HVG subset)
    X = adata_raw.X
    if sp.issparse(X):
        X = X.toarray()
    count_matrix = X.astype(np.float32)

    # Optional subsample for smoke test
    if n_cells_subset is not None and n_cells_subset < count_matrix.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(count_matrix.shape[0], n_cells_subset, replace=False)
        count_matrix = count_matrix[idx]
        int_labels   = int_labels[idx]
        adata_raw    = adata_raw[idx].copy()

    print(f"  {count_matrix.shape[0]} cells × {count_matrix.shape[1]} genes")
    print(f"  {len(label_names)} cell types: {label_names}")
    return count_matrix, int_labels, label_names, adata_raw, list(adata_raw.var_names)


def load_pbmc3k_universal(
    universal_gene_names: list[str],
    n_cells_subset: int | None = None,
    seed: int = 42,
    n_hvg: int = 2000,
):
    """
    Load PBMC3k aligned to a pre-built universal gene vocabulary.

    Selects the top n_hvg highly variable genes from PBMC-3K (matching the
    Cell-JEPA paper) and projects only those into the universal vocab space.
    Non-HVG columns are left as zero so SingleCellDataset never samples them.
    Universal token IDs are preserved — no alignment problem.

    Set n_hvg=0 to disable HVG selection and use all expressed genes.

    Args:
        universal_gene_names: Ordered list of gene symbols in the universal vocab.
        n_cells_subset: Optional subsample for smoke test.
        seed: RNG seed for subsample.
        n_hvg: Number of highly variable genes to select (default 2000, paper value).

    Returns:
        count_matrix : (N, len(universal_gene_names)) float32
        int_labels   : (N,)  int32 cell-type indices
        label_names  : list of label strings
        adata_raw    : AnnData (for optional scGPT evaluation)
        universal_gene_names : the same list passed in (for convenience)
    """
    import scipy.sparse as sp

    print("Loading PBMC3k (universal vocab) …")
    adata_raw  = sc.datasets.pbmc3k()
    adata_proc = sc.datasets.pbmc3k_processed()

    common = adata_raw.obs_names.intersection(adata_proc.obs_names)
    adata_raw  = adata_raw[common].copy()
    adata_proc = adata_proc[common]

    labels_str  = adata_proc.obs["louvain"].values
    label_names = sorted(set(labels_str))
    lbl2int     = {l: i for i, l in enumerate(label_names)}
    int_labels  = np.array([lbl2int[l] for l in labels_str], dtype=np.int32)

    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)

    # HVG selection — compute on all cells before any subsetting
    if n_hvg and n_hvg > 0:
        sc.pp.highly_variable_genes(adata_raw, n_top_genes=n_hvg, flavor="seurat")
        hvg_set = set(adata_raw.var_names[adata_raw.var["highly_variable"]])
        print(f"  Selected {len(hvg_set)} HVGs from PBMC-3K")
    else:
        hvg_set = None

    # Project PBMC-3K genes into universal vocab space (HVGs only if selected)
    pbmc_gene_set = {g: i for i, g in enumerate(adata_raw.var_names)}
    n_universal = len(universal_gene_names)
    n_cells = adata_raw.n_obs

    X = adata_raw.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    X_universal = np.zeros((n_cells, n_universal), dtype=np.float32)
    n_mapped = 0
    for uni_idx, gene in enumerate(universal_gene_names):
        if gene in pbmc_gene_set and (hvg_set is None or gene in hvg_set):
            X_universal[:, uni_idx] = X[:, pbmc_gene_set[gene]]
            n_mapped += 1

    hvg_note = f" ({n_hvg} HVGs)" if hvg_set is not None else ""
    print(f"  Gene coverage: {n_mapped}/{n_universal} universal genes mapped from PBMC-3K{hvg_note} "
          f"({n_mapped/n_universal*100:.1f}%)")

    if n_cells_subset is not None and n_cells_subset < n_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_cells, n_cells_subset, replace=False)
        X_universal = X_universal[idx]
        int_labels  = int_labels[idx]
        adata_raw   = adata_raw[idx].copy()

    print(f"  {X_universal.shape[0]} cells × {n_universal} genes (universal vocab)")
    print(f"  {len(label_names)} cell types: {label_names}")
    return X_universal, int_labels, label_names, adata_raw, universal_gene_names


# ---------------------------------------------------------------------------
# PBMC 10k loader (for transfer learning)
# ---------------------------------------------------------------------------

def load_pbmc10k_aligned(gene_names_3k: list):
    """
    Download the Zheng 2017 PBMC dataset (~11k cells) via scvi-tools and
    restrict to PBMC 3k's HVGs.  Returns (count_matrix, shared_gene_names).
    """
    import scipy.sparse as sp

    print("Loading PBMC dataset via scvi-tools (Zheng 2017, ~11k cells) …")
    try:
        import scvi
    except ImportError:
        raise ImportError("scvi-tools is required: pip install scvi-tools")

    adata = scvi.data.pbmc_dataset()
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"  Sample var_names: {list(adata.var_names[:5])}")
    print(f"  var columns: {list(adata.var.columns)}")

    # If var_names don't overlap with PBMC 3k gene symbols, try remapping
    # from a gene_symbols / gene_names column in adata.var
    gene_names_3k_set = set(gene_names_3k)
    if len(gene_names_3k_set & set(adata.var_names)) < 10:
        remapped = False
        for col in ["gene_symbols", "gene_names", "Symbol", "gene_name", "name"]:
            if col in adata.var.columns:
                adata.var_names = adata.var[col].astype(str).values
                adata.var_names_make_unique()
                print(f"  Remapped var_names via '{col}' column")
                remapped = True
                break
        if not remapped:
            print("  WARNING: could not remap var_names — overlap may be low")

    common = [g for g in gene_names_3k if g in adata.var_names]
    print(f"  {adata.n_obs} cells, {len(common)}/{len(gene_names_3k)} shared HVGs with PBMC 3k")
    if len(common) == 0:
        raise RuntimeError(
            "No shared genes between PBMC 10k and PBMC 3k HVGs. "
            f"Sample PBMC 10k var_names: {list(adata.var_names[:10])}"
        )
    adata = adata[:, common].copy()

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Drop cells with no expressed genes in the shared HVG set
    expressed = (X > 0).any(axis=1)
    n_dropped = (~expressed).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} cells with no expressed shared HVGs")
        X = X[expressed]
    print(f"  Using {X.shape[0]} cells for pre-training")
    return X, common


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(n_genes: int):
    """Token IDs: 0=<cls>, 1=<pad>, 2..n_genes+1=genes (same as example.py)."""
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    vocab_size  = n_genes + 2
    cls_id, pad_id = 0, 1
    return gene_vocab, vocab_size, cls_id, pad_id


# ---------------------------------------------------------------------------
# CellJEPA training
# ---------------------------------------------------------------------------

def train_celljepа(
    count_matrix: np.ndarray,
    cell_types: np.ndarray,
    gene_vocab: dict,
    vocab_size: int,
    args: argparse.Namespace,
    device: torch.device,
    pretrain_data: np.ndarray | None = None,
) -> CellJEPA:
    n_bins = 50
    bs     = 8 if args.smoke_test else 32
    ft_bs  = 8 if args.smoke_test else 16
    if args.w_jepa is not None:
        w_jepa = args.w_jepa
    else:
        w_jepa = 0.0 if args.no_jepa else 1000.0
    include_jepa = w_jepa > 0

    # ------------------------------------------------------------------
    # Kidney checkpoint path: skip pre-training, load weights, align genes
    # ------------------------------------------------------------------
    if args.kidney_checkpoint:
        import json
        if not args.kidney_genes:
            raise ValueError(
                "--kidney_genes must be provided alongside --kidney_checkpoint"
            )
        with open(args.kidney_genes) as f:
            kidney_genes = json.load(f)

        # Map PBMC 3k gene names to kidney gene indices
        pbmc_gene_list = list(args._pbmc_gene_names)  # set in main() below
        kidney_gene_map = {g: i for i, g in enumerate(kidney_genes)}
        n_kidney_genes = len(kidney_genes)
        kidney_vocab_size = n_kidney_genes + 2
        kidney_gene_vocab = {i: i + 2 for i in range(n_kidney_genes)}

        # Build aligned PBMC 3k matrix in the kidney gene space
        X_aligned = np.zeros(
            (count_matrix.shape[0], n_kidney_genes), dtype=np.float32
        )
        n_mapped = 0
        for pbmc_idx, g in enumerate(pbmc_gene_list):
            if g in kidney_gene_map:
                X_aligned[:, kidney_gene_map[g]] = count_matrix[:, pbmc_idx]
                n_mapped += 1
        print(f"  Gene alignment: {n_mapped}/{len(pbmc_gene_list)} PBMC 3k genes "
              f"found in kidney vocabulary ({n_kidney_genes} total)")

        model = CellJEPA(
            vocab_size=kidney_vocab_size, n_bins=n_bins,
            d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
            dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
        )
        ckpt = torch.load(args.kidney_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"CellJEPA (kidney pretrain): {n_params / 1e6:.1f}M parameters loaded")

        finetune_ds = SingleCellDataset(
            X_aligned, kidney_gene_vocab, cell_types,
            n_bins=n_bins, L_max=args.l_max,
            cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
        )
        print(f"\n--- Fine-tuning on PBMC 3k ({args.finetune_epochs} epochs) ---")
        Finetuner(
            model, finetune_ds,
            FinetuneConfig(
                lr=1e-4, lr_decay=0.9,
                batch_size=ft_bs, n_epochs=args.finetune_epochs,
                log_every=5, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
                w_jepa=1000.0, ecs_temperature=0.1,
                include_jepa=True, num_workers=0,
            ),
            device,
        ).train()

        # Store aligned matrix and vocab on args so evaluate step can use them
        args._kidney_X_aligned = X_aligned
        args._kidney_gene_vocab = kidney_gene_vocab
        args._kidney_vocab_size = kidney_vocab_size
        return model

    # ------------------------------------------------------------------
    # Standard path: pre-train (optionally on external data) + fine-tune
    # ------------------------------------------------------------------
    pretrain_matrix = pretrain_data if pretrain_data is not None else count_matrix
    # cell_types not needed for pre-training (unsupervised), pass zeros as placeholder
    pretrain_labels = np.zeros(pretrain_matrix.shape[0], dtype=np.int32)

    pretrain_ds = SingleCellDataset(
        pretrain_matrix, gene_vocab, pretrain_labels,
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.15,
    )
    finetune_ds = SingleCellDataset(
        count_matrix, gene_vocab, cell_types,
        n_bins=n_bins, L_max=args.l_max,
        cls_token_id=0, pad_token_id=1, mask_ratio=0.40,
    )

    model = CellJEPA(
        vocab_size=vocab_size, n_bins=n_bins,
        d_model=512, n_layers=12, n_heads=8, ffn_dim=2048,
        dropout=0.2, ema_momentum=0.996, predictor_hidden=512,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CellJEPA: {n_params / 1e6:.1f}M trainable parameters")

    print(f"\n--- Pre-training ({args.pretrain_epochs} epochs) ---")
    Pretrainer(
        model, pretrain_ds,
        PretrainConfig(
            lr=1e-4, weight_decay=2e-4, lr_decay=0.9,
            batch_size=bs, n_epochs=args.pretrain_epochs,
            log_every=5, w_rec=1.0, w_jepa=w_jepa, num_workers=0,
        ),
        device,
    ).train()

    print(f"\n--- Fine-tuning ({args.finetune_epochs} epochs) ---")
    Finetuner(
        model, finetune_ds,
        FinetuneConfig(
            lr=1e-4, lr_decay=0.9,
            batch_size=ft_bs, n_epochs=args.finetune_epochs,
            log_every=5, w_gep=1.0, w_gepc=1.0, w_ecs=1.0,
            w_jepa=w_jepa, ecs_temperature=0.1,
            include_jepa=include_jepa, num_workers=0,
        ),
        device,
    ).train()

    return model


# ---------------------------------------------------------------------------
# Shared evaluation (applies to both models)
# ---------------------------------------------------------------------------

def evaluate_embeddings(
    embeddings: np.ndarray,
    int_labels: np.ndarray,
    n_neighbors: int = 15,
) -> dict:
    """
    Louvain sweep (0.1–2.0) → best NMI resolution → NMI / ARI / ASW / AvgBIO.
    Mirrors the pipeline in metrics.py so results are directly comparable.
    """
    valid   = int_labels >= 0
    emb_v   = embeddings[valid]
    lab_v   = int_labels[valid]

    print("  Running Louvain clustering sweep …")
    cluster_results = louvain_cluster(emb_v, n_neighbors=n_neighbors)

    nmi, best_idx = compute_nmi(lab_v, cluster_results)
    ari = compute_ari(lab_v, cluster_results[best_idx])
    asw = compute_asw(emb_v, lab_v)
    ab  = avg_bio(nmi, ari, asw)

    return {
        "nmi": nmi, "ari": ari, "asw": asw, "avg_bio": ab,
        "best_resolution": round(best_idx * 0.1 + 0.1, 1),
    }


# ---------------------------------------------------------------------------
# scGPT evaluation
# ---------------------------------------------------------------------------

def run_scgpt(
    adata_raw,
    int_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
):
    """
    Extract cell embeddings using scGPT's pre-trained whole-human model,
    then evaluate with the same pipeline as CellJEPA.
    Returns (embeddings, results_dict) or (None, None) on failure.
    """
    try:
        from scgpt.tasks import embed_data
    except ImportError:
        print("scgpt not installed. Install with: pip install scgpt")
        return None, None

    model_dir = args.scgpt_model_dir
    if model_dir is None:
        try:
            from huggingface_hub import snapshot_download
            print("Downloading scGPT whole-human model from Hugging Face …")
            model_dir = snapshot_download("bowang-lab/scGPT_human")
            print(f"  Downloaded to: {model_dir}")
        except Exception as e:
            print(f"Could not auto-download scGPT model: {e}")
            print("Re-run with:  --scgpt_model_dir /path/to/model")
            return None, None

    print("Extracting scGPT embeddings …")
    try:
        adata_sg = adata_raw.copy()
        # embed_data expects gene names in adata.var; var_names are gene symbols
        adata_sg.var["gene_name"] = adata_sg.var_names
        adata_sg = embed_data(
            adata_sg,
            model_dir=model_dir,
            gene_col="gene_name",
            max_length=1200,
            batch_size=32,
            device=str(device),
            return_new_adata=False,
        )
        scgpt_emb = adata_sg.obsm["X_scGPT"]
        print(f"  scGPT embedding shape: {scgpt_emb.shape}")
        results = evaluate_embeddings(scgpt_emb, int_labels)
        return scgpt_emb, results
    except Exception as e:
        print(f"scGPT evaluation failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save(results_dict: dict, results_file: str) -> None:
    row = "  {:<12s}  {:>6s}  {:>6s}  {:>6s}  {:>7s}"
    sep = "  " + "-" * 51
    lines = [
        "",
        "=" * 55,
        "  PBMC3k Cell Clustering Comparison",
        "=" * 55,
        row.format("Model", "NMI", "ARI", "ASW", "AvgBIO"),
        sep,
    ]
    for name, res in results_dict.items():
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
    lines.append("=" * 55)

    output = "\n".join(lines)
    print(output)
    with open(results_file, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {results_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.w_jepa is not None and args.results_file == "results.txt":
        args.results_file = f"results_wjepa{int(args.w_jepa)}.txt"
    elif args.no_jepa and args.results_file == "results.txt":
        args.results_file = "results_no_jepa.txt"
    elif args.pretrain_10k and args.results_file == "results.txt":
        args.results_file = "results_pretrain10k.txt"
    elif args.kidney_checkpoint and args.results_file == "results.txt":
        args.results_file = "results_kidney_pretrain.txt"

    if args.smoke_test:
        print("\n[SMOKE TEST: 200 cells, 1 epoch each, no scGPT]\n")
        args.pretrain_epochs = 1
        args.finetune_epochs = 1
        args.l_max = 64

    n_cells_subset = 200 if args.smoke_test else None
    count_matrix, int_labels, label_names, adata_raw, gene_names = load_pbmc3k(n_cells_subset)

    gene_vocab, vocab_size, cls_id, pad_id = build_vocab(count_matrix.shape[1])

    # ------------------------------------------------------------------ #
    #  Optional: pre-train on PBMC 10k                                    #
    # ------------------------------------------------------------------ #
    pretrain_data = None
    if args.pretrain_10k:
        pretrain_X, shared_genes = load_pbmc10k_aligned(gene_names)
        # Rebuild vocab and restrict PBMC 3k to the shared gene set
        gene_vocab, vocab_size, cls_id, pad_id = build_vocab(len(shared_genes))
        shared_idx = [gene_names.index(g) for g in shared_genes]
        count_matrix = count_matrix[:, shared_idx]
        pretrain_data = pretrain_X

    # ------------------------------------------------------------------ #
    #  CellJEPA                                                            #
    # ------------------------------------------------------------------ #
    if args.w_jepa is not None:
        model_name = f"CellJEPA (w_jepa={int(args.w_jepa)})"
    elif args.kidney_checkpoint:
        model_name = "CellJEPA (kidney pretrain)"
    elif args.pretrain_10k and args.no_jepa:
        model_name = "CellJEPA (pretrain 10k, no JEPA)"
    elif args.pretrain_10k:
        model_name = "CellJEPA (pretrain 10k)"
    elif args.no_jepa:
        model_name = "CellJEPA (no JEPA)"
    else:
        model_name = "CellJEPA"
    print("\n" + "=" * 55)
    print(f"  {model_name}")
    print("=" * 55)

    # Pass gene names to args so train_celljepа can align the kidney vocab
    args._pbmc_gene_names = gene_names

    model = train_celljepа(count_matrix, int_labels, gene_vocab, vocab_size, args, device,
                            pretrain_data=pretrain_data)

    # Use kidney-aligned matrix and vocab for eval if kidney checkpoint was used
    if args.kidney_checkpoint:
        eval_matrix = args._kidney_X_aligned
        eval_vocab = args._kidney_gene_vocab
        eval_cls_id, eval_pad_id = 0, 1
    else:
        eval_matrix = count_matrix
        eval_vocab = gene_vocab
        eval_cls_id, eval_pad_id = cls_id, pad_id

    eval_ds = SingleCellDataset(
        eval_matrix, eval_vocab, int_labels,
        n_bins=50, L_max=args.l_max,
        cls_token_id=eval_cls_id, pad_token_id=eval_pad_id,
        mask_ratio=0.0,
    )
    eval_bs = 8 if args.smoke_test else 32
    print(f"\nEvaluating {model_name} …")
    cj_emb, cj_labels = extract_embeddings(
        model, eval_ds, batch_size=eval_bs, device=device, use_teacher=True,
    )
    cj_results = evaluate_embeddings(cj_emb, cj_labels)
    print(f"  {model_name} AvgBIO: {cj_results['avg_bio']:.4f}")

    # ------------------------------------------------------------------ #
    #  scGPT (zero-shot; skipped in smoke test and ablation runs)         #
    # ------------------------------------------------------------------ #
    scgpt_results = None
    if not args.smoke_test and not args.no_jepa and not args.pretrain_10k and not args.kidney_checkpoint:
        print("\n" + "=" * 55)
        print("  scGPT (zero-shot, whole-human checkpoint)")
        print("=" * 55)
        _, scgpt_results = run_scgpt(adata_raw, int_labels, args, device)

    # ------------------------------------------------------------------ #
    #  Comparison table                                                    #
    # ------------------------------------------------------------------ #
    results = {model_name: cj_results}
    if scgpt_results is not None:
        results["scGPT"] = scgpt_results

    print_and_save(results, args.results_file)


if __name__ == "__main__":
    main()

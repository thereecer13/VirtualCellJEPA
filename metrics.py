"""
Cell-JEPA Evaluation
=====================
Implements the evaluation metrics from Section 3 / Appendix D.1:

    AvgBIO = (NMI_cell + ARI_cell + ASW_cell) / 3

Steps:
  1. Extract cell embeddings from the (teacher) encoder.
  2. Run Louvain clustering at multiple resolutions.
  3. Compute NMI (max over resolutions), ARI, ASW against ground-truth labels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional

try:
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not found — clustering metrics unavailable.")

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False
    print("scanpy/anndata not found — Louvain clustering unavailable.")

from cell_jepa import CellJEPA


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: CellJEPA,
    dataset,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    use_teacher: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract cell-level <cls> embeddings for an entire dataset.

    Args:
        model:       CellJEPA (eval mode).
        dataset:     SingleCellDataset (mask_ratio should be 0 for evaluation).
        batch_size:  Inference batch size.
        device:      Target device.
        use_teacher: Use teacher encoder (recommended, as in zero-shot eval).

    Returns:
        embeddings: (N, D) float32 array.
        labels:     (N,)   int32 cell-type labels (-1 if unavailable).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_emb, all_labels = [], []
    for batch in loader:
        gene_ids  = batch["gene_ids"].to(device)
        values    = batch["values"].to(device)
        is_pad    = batch["padding"].to(device)

        emb = model.encode(gene_ids, values, is_pad, use_teacher=use_teacher)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(batch["cell_type"].numpy())

    return np.concatenate(all_emb, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# Louvain clustering helper
# ---------------------------------------------------------------------------

def louvain_cluster(
    embeddings: np.ndarray,
    resolutions: np.ndarray | list = None,
    n_neighbors: int = 15,
) -> list[np.ndarray]:
    """
    Run Louvain clustering over a grid of resolutions using scanpy.

    Args:
        embeddings:  (N, D) cell embeddings.
        resolutions: List of resolution values (default 0.1 to 2.0 step 0.1).
        n_neighbors: k-NN graph neighbours.

    Returns:
        List of (N,) integer cluster-label arrays, one per resolution.
    """
    if not HAS_SCANPY:
        raise ImportError("scanpy required for Louvain clustering.")

    if resolutions is None:
        resolutions = np.arange(0.1, 2.1, 0.1)

    adata = ad.AnnData(X=embeddings.astype(np.float32))
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors)

    cluster_results = []
    for res in resolutions:
        sc.tl.leiden(adata, resolution=float(res), key_added="leiden_tmp",
                     flavor="igraph", n_iterations=2, directed=False)
        labels = adata.obs["leiden_tmp"].astype(int).values
        cluster_results.append(labels)

    return cluster_results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_nmi(
    true_labels: np.ndarray,
    cluster_labels_list: list[np.ndarray],
) -> float:
    """
    NMI_cell: maximum NMI across all Louvain resolutions.
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for NMI.")
    scores = [
        normalized_mutual_info_score(true_labels, cl, average_method="arithmetic")
        for cl in cluster_labels_list
    ]
    return float(np.max(scores)), int(np.argmax(scores))


def compute_ari(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """ARI_cell: ARI at the resolution that maximises NMI."""
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for ARI.")
    return float(adjusted_rand_score(true_labels, cluster_labels))


def compute_asw(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    ASW_cell: average silhouette width with respect to ground-truth cell types.
    Rescaled to [0, 1] via (ASW + 1) / 2 following Luecken et al. 2022.
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for ASW.")
    if len(np.unique(true_labels)) < 2:
        return 0.0
    raw_asw = silhouette_score(embeddings, true_labels, metric="euclidean")
    return float((raw_asw + 1.0) / 2.0)


def avg_bio(nmi: float, ari: float, asw: float) -> float:
    """AvgBIO = (NMI + ARI + ASW) / 3  (Appendix D.1)."""
    return (nmi + ari + asw) / 3.0


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model: CellJEPA,
    dataset,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    use_teacher: bool = True,
    louvain_resolutions: Optional[list] = None,
    n_neighbors: int = 15,
) -> dict:
    """
    Run the full AvgBIO evaluation pipeline.

    Args:
        model:                CellJEPA model.
        dataset:              SingleCellDataset with cell_types populated.
        batch_size:           Inference batch size.
        device:               Compute device.
        use_teacher:          Use teacher encoder for embeddings.
        louvain_resolutions:  List of Louvain resolutions to sweep.
        n_neighbors:          k-NN neighbours for graph construction.

    Returns:
        Dict with keys: embeddings, labels, nmi, ari, asw, avg_bio,
                        best_resolution, cluster_labels.
    """
    print("Extracting embeddings …")
    embeddings, labels = extract_embeddings(
        model, dataset, batch_size=batch_size,
        device=device, use_teacher=use_teacher,
    )

    # Filter cells with valid labels
    valid = labels >= 0
    emb_valid = embeddings[valid]
    lab_valid = labels[valid]

    if len(np.unique(lab_valid)) < 2:
        print("Warning: fewer than 2 cell types found — metrics undefined.")
        return {"embeddings": embeddings, "labels": labels}

    print("Running Louvain clustering …")
    cluster_results = louvain_cluster(
        emb_valid,
        resolutions=louvain_resolutions,
        n_neighbors=n_neighbors,
    )

    nmi_score, best_res_idx = compute_nmi(lab_valid, cluster_results)
    best_clusters = cluster_results[best_res_idx]

    ari_score = compute_ari(lab_valid, best_clusters)
    asw_score = compute_asw(emb_valid, lab_valid)
    avg_bio_score = avg_bio(nmi_score, ari_score, asw_score)

    results = {
        "embeddings":      embeddings,
        "labels":          labels,
        "cluster_labels":  best_clusters,
        "best_resolution": best_res_idx * 0.1 + 0.1,
        "nmi":             nmi_score,
        "ari":             ari_score,
        "asw":             asw_score,
        "avg_bio":         avg_bio_score,
    }

    print(
        f"\n{'='*40}\n"
        f"  NMI_cell  : {nmi_score:.4f}\n"
        f"  ARI_cell  : {ari_score:.4f}\n"
        f"  ASW_cell  : {asw_score:.4f}\n"
        f"  AvgBIO    : {avg_bio_score:.4f}\n"
        f"  Best res  : {results['best_resolution']:.1f}\n"
        f"{'='*40}"
    )

    return results

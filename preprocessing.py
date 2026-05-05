"""
Cell-JEPA Data Preprocessing
=============================
Implements Section 2.1 of the paper:
  - Per-cell quantile binning of expression values into B=50 bins
  - Gene subsampling to L_max=600 expressed genes per cell
  - Dataset wrapper for sparse count matrices

Also provides PerturbationDataset for Perturb-seq experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Quantile Binning
# ---------------------------------------------------------------------------

def quantile_bin_cell(
    expression_values: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """
    Discretize non-zero expression values for a single cell via per-cell
    quantile binning (Section 2.1).

    Zero values are left as 0 (unexpressed / dropped-out genes).
    Non-zero values are mapped to bins {1, ..., n_bins}.

    Args:
        expression_values: 1-D array of raw counts for one cell.
        n_bins: Number of quantile bins (B in the paper, default 50).

    Returns:
        Integer bin indices of the same shape. Zeros stay zero.
    """
    binned = np.zeros_like(expression_values, dtype=np.int32)
    nonzero_mask = expression_values > 0
    nonzero_vals = expression_values[nonzero_mask]

    if nonzero_vals.size == 0:
        return binned

    # Compute quantile bin edges over the non-zero values of this cell
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(nonzero_vals, quantiles)
    # Avoid duplicate edges at extremes
    bin_edges = np.unique(bin_edges)

    # np.digitize returns 1-indexed; clip to [1, n_bins]
    indices = np.digitize(nonzero_vals, bin_edges[1:], right=False) + 1
    indices = np.clip(indices, 1, n_bins)
    binned[nonzero_mask] = indices
    return binned


def quantile_bin_matrix(
    count_matrix: np.ndarray | sp.spmatrix,
    n_bins: int = 50,
) -> np.ndarray:
    """
    Apply per-cell quantile binning to a full cell-by-gene count matrix.

    Args:
        count_matrix: (n_cells, n_genes) dense or sparse matrix.
        n_bins: Number of quantile bins.

    Returns:
        Integer array of shape (n_cells, n_genes).
    """
    if sp.issparse(count_matrix):
        count_matrix = count_matrix.toarray()

    binned = np.zeros_like(count_matrix, dtype=np.int32)
    for i in range(count_matrix.shape[0]):
        binned[i] = quantile_bin_cell(count_matrix[i], n_bins=n_bins)
    return binned


# ---------------------------------------------------------------------------
# Single-Cell Dataset
# ---------------------------------------------------------------------------

class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for scRNA-seq data.

    Stores cells as sparse aligned (gene_id, bin_value) sequences and
    applies stochastic gene subsampling at __getitem__ time (Section 2.1),
    which means each epoch sees a different partial view of every cell.

    Args:
        count_matrix: (n_cells, n_genes) raw count matrix (dense or sparse).
        gene_vocab:   Mapping from gene index -> vocabulary index. If None,
                      gene indices are used directly.
        cell_types:   Optional (n_cells,) integer cell-type label array for
                      use during fine-tuning (ECS loss).
        n_bins:       Quantile bins (B, default 50).
        L_max:        Maximum genes per cell sequence (default 600).
        cls_token_id: Vocabulary ID for the <cls> token.
        pad_token_id: Vocabulary ID for the <pad> token.
        mask_ratio:   Fraction of expressed genes to mask during training.
    """

    def __init__(
        self,
        count_matrix: np.ndarray | sp.spmatrix,
        gene_vocab: Optional[dict] = None,
        cell_types: Optional[np.ndarray] = None,
        n_bins: int = 50,
        L_max: int = 600,
        cls_token_id: int = 0,
        pad_token_id: int = 1,
        mask_ratio: float = 0.15,
    ):
        if sp.issparse(count_matrix):
            count_matrix = count_matrix.toarray()
        self.count_matrix = count_matrix.astype(np.float32)
        self.gene_vocab = gene_vocab
        self.cell_types = cell_types
        self.n_bins = n_bins
        self.L_max = L_max
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.mask_ratio = mask_ratio
        self.n_cells, self.n_genes = count_matrix.shape

        # Pre-compute binned matrix once (binning is deterministic)
        print("Pre-computing quantile bins …")
        self.binned = quantile_bin_matrix(count_matrix, n_bins=n_bins)
        print("Done.")

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with keys:
          gene_ids   : (L_max+1,) int tensor  — gene vocab IDs (incl. <cls>)
          values     : (L_max+1,) int tensor  — bin indices (0 for <cls>)
          masked_vals: (L_max+1,) int tensor  — values with masked positions = -1
          mask       : (L_max+1,) bool tensor — True where gene is masked
          padding    : (L_max+1,) bool tensor — True where position is <pad>
          cell_type  : int scalar (-1 if unavailable)
        """
        binned_cell = self.binned[idx]  # (n_genes,)
        expressed_mask = binned_cell > 0
        expressed_gene_ids = np.where(expressed_mask)[0]
        expressed_values = binned_cell[expressed_mask]

        # --- Gene Subsampling (stochastic, Section 2.1) ---
        n_expressed = len(expressed_gene_ids)
        if n_expressed > self.L_max:
            chosen = np.random.choice(n_expressed, self.L_max, replace=False)
            chosen = np.sort(chosen)
            expressed_gene_ids = expressed_gene_ids[chosen]
            expressed_values = expressed_values[chosen]
            seq_len = self.L_max
        else:
            seq_len = n_expressed

        # Map gene indices through vocab if provided
        if self.gene_vocab is not None:
            gene_ids = np.array(
                [self.gene_vocab.get(g, self.pad_token_id) for g in expressed_gene_ids],
                dtype=np.int64,
            )
        else:
            gene_ids = expressed_gene_ids.astype(np.int64)

        # --- Masking (Section 2.2.1) ---
        n_mask = max(1, int(self.mask_ratio * seq_len))
        mask_positions = np.sort(
            np.random.choice(seq_len, n_mask, replace=False)
        )
        is_masked = np.zeros(seq_len, dtype=bool)
        is_masked[mask_positions] = True

        masked_values = expressed_values.copy().astype(np.int64)
        masked_values[is_masked] = -1  # sentinel vmask

        # --- Prepend <cls> token ---
        cls_gene = np.array([self.cls_token_id], dtype=np.int64)
        cls_val = np.array([0], dtype=np.int64)
        cls_mask = np.array([False])

        gene_ids_full = np.concatenate([cls_gene, gene_ids])
        values_full = np.concatenate([cls_val, expressed_values.astype(np.int64)])
        masked_vals_full = np.concatenate([cls_val, masked_values])
        is_masked_full = np.concatenate([cls_mask, is_masked])

        # --- Padding to L_max + 1 ---
        total_len = self.L_max + 1
        pad_len = total_len - len(gene_ids_full)
        is_pad = np.zeros(total_len, dtype=bool)

        if pad_len > 0:
            gene_ids_full = np.pad(gene_ids_full, (0, pad_len),
                                   constant_values=self.pad_token_id)
            values_full = np.pad(values_full, (0, pad_len), constant_values=0)
            masked_vals_full = np.pad(masked_vals_full, (0, pad_len),
                                      constant_values=0)
            is_masked_full = np.pad(is_masked_full, (0, pad_len),
                                    constant_values=False)
            is_pad[seq_len + 1:] = True  # +1 for <cls>

        cell_type = int(self.cell_types[idx]) if self.cell_types is not None else -1

        return {
            "gene_ids":    torch.tensor(gene_ids_full,    dtype=torch.long),
            "values":      torch.tensor(values_full,      dtype=torch.long),
            "masked_vals": torch.tensor(masked_vals_full, dtype=torch.long),
            "mask":        torch.tensor(is_masked_full,   dtype=torch.bool),
            "padding":     torch.tensor(is_pad,           dtype=torch.bool),
            "cell_type":   torch.tensor(cell_type,        dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Perturbation Dataset  (for Perturb-seq experiments)
# ---------------------------------------------------------------------------

class PerturbationDataset(Dataset):
    """
    PyTorch Dataset for Perturb-seq data.

    Each item pairs a perturbed cell with its matched control mean, enabling
    supervised training of perturbation response prediction.

    The dataset expects pre-aligned arrays:
        control_matrix  : (n_cells, n_genes) float32 — per-cell control (or
                          mean-of-controls for that perturbation condition)
        perturbed_matrix: (n_cells, n_genes) float32 — post-perturbation expression
        pert_ids_per_cell: (n_cells,) int32 — index into perturbation vocabulary
                           (0 = control / no-op perturbation)

    The perturbation vocabulary index is broadcast to ALL genes of a cell as
    the `pert_id` token value — matching the convention in CellJEPA.forward_perturb().
    (Unperturbed genes and the <cls> token receive pert_id = 0.)

    Args:
        control_matrix:    (n_cells, n_genes) baseline expression
        perturbed_matrix:  (n_cells, n_genes) post-perturbation expression
        pert_ids_per_cell: (n_cells,) perturbation vocab index per cell
        gene_vocab:        Mapping gene_index -> vocab_id (None = identity)
        cell_types:        Optional (n_cells,) integer cell-type labels
        n_bins:            Quantile bins (default 50)
        L_max:             Max genes per sequence (default 200)
        cls_token_id:      Vocab ID for <cls>
        pad_token_id:      Vocab ID for <pad>
    """

    def __init__(
        self,
        control_matrix: np.ndarray,
        perturbed_matrix: np.ndarray,
        pert_ids_per_cell: np.ndarray,
        gene_vocab: Optional[dict] = None,
        cell_types: Optional[np.ndarray] = None,
        n_bins: int = 50,
        L_max: int = 200,
        cls_token_id: int = 0,
        pad_token_id: int = 1,
    ):
        if sp.issparse(control_matrix):
            control_matrix = control_matrix.toarray()
        if sp.issparse(perturbed_matrix):
            perturbed_matrix = perturbed_matrix.toarray()

        self.control_matrix = control_matrix.astype(np.float32)
        self.perturbed_matrix = perturbed_matrix.astype(np.float32)
        self.pert_ids_per_cell = pert_ids_per_cell.astype(np.int32)
        self.gene_vocab = gene_vocab
        self.cell_types = cell_types
        self.n_bins = n_bins
        self.L_max = L_max
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.n_cells, self.n_genes = control_matrix.shape

        print("Pre-computing quantile bins …")
        self.ctrl_binned = quantile_bin_matrix(control_matrix, n_bins=n_bins)
        self.pert_binned = quantile_bin_matrix(perturbed_matrix, n_bins=n_bins)
        print("Done.")

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with keys:
          gene_ids   : (L_max+1,) int  — gene vocab IDs (incl. <cls>)
          values     : (L_max+1,) int  — control bin indices (student input)
          pert_ids   : (L_max+1,) int  — perturbation vocab ID broadcast to all positions
          pert_values: (L_max+1,) int  — perturbed bin indices (teacher / target)
          padding    : (L_max+1,) bool — True at padding positions
          cell_type  : int scalar (-1 if unavailable)
        """
        ctrl_cell = self.ctrl_binned[idx]    # (n_genes,)
        pert_cell = self.pert_binned[idx]    # (n_genes,)

        # Use union of expressed genes in either condition as the token sequence
        expressed_mask = (ctrl_cell > 0) | (pert_cell > 0)
        expressed_idx = np.where(expressed_mask)[0]
        ctrl_vals = ctrl_cell[expressed_idx]
        pert_vals = pert_cell[expressed_idx]

        # Stochastic subsampling to L_max genes
        n_expressed = len(expressed_idx)
        if n_expressed > self.L_max:
            chosen = np.sort(np.random.choice(n_expressed, self.L_max, replace=False))
            expressed_idx = expressed_idx[chosen]
            ctrl_vals = ctrl_vals[chosen]
            pert_vals = pert_vals[chosen]
            seq_len = self.L_max
        else:
            seq_len = n_expressed

        # Map gene indices through vocab
        if self.gene_vocab is not None:
            gene_ids = np.array(
                [self.gene_vocab.get(int(g), self.pad_token_id) for g in expressed_idx],
                dtype=np.int64,
            )
        else:
            gene_ids = expressed_idx.astype(np.int64)

        # Perturbation ID is the same for all genes in the cell
        pert_id = int(self.pert_ids_per_cell[idx])
        pert_ids_seq = np.full(seq_len, pert_id, dtype=np.int64)

        # Prepend <cls> token
        gene_ids_full = np.concatenate([[self.cls_token_id], gene_ids])
        ctrl_vals_full = np.concatenate([[0], ctrl_vals.astype(np.int64)])
        pert_vals_full = np.concatenate([[0], pert_vals.astype(np.int64)])
        pert_ids_full  = np.concatenate([[0], pert_ids_seq])  # <cls> gets pert_id=0

        # Pad to L_max + 1
        total_len = self.L_max + 1
        pad_len = total_len - len(gene_ids_full)
        is_pad = np.zeros(total_len, dtype=bool)

        if pad_len > 0:
            gene_ids_full = np.pad(gene_ids_full, (0, pad_len),
                                   constant_values=self.pad_token_id)
            ctrl_vals_full = np.pad(ctrl_vals_full, (0, pad_len), constant_values=0)
            pert_vals_full = np.pad(pert_vals_full, (0, pad_len), constant_values=0)
            pert_ids_full  = np.pad(pert_ids_full,  (0, pad_len), constant_values=0)
            is_pad[seq_len + 1:] = True  # +1 for <cls>

        cell_type = int(self.cell_types[idx]) if self.cell_types is not None else -1

        return {
            "gene_ids":    torch.tensor(gene_ids_full,  dtype=torch.long),
            "values":      torch.tensor(ctrl_vals_full, dtype=torch.long),
            "pert_ids":    torch.tensor(pert_ids_full,  dtype=torch.long),
            "pert_values": torch.tensor(pert_vals_full, dtype=torch.long),
            "padding":     torch.tensor(is_pad,         dtype=torch.bool),
            "cell_type":   torch.tensor(cell_type,      dtype=torch.long),
        }

"""
CellJEPA_SIGReg — Transformer + SIGReg variant
================================================
Replaces the EMA student-teacher collapse-prevention mechanism of Cell-JEPA with
SIGReg (Sketched Isotropic Gaussian Regularization, Balestriero & LeCun 2025).

Key differences from CellJEPA (cell_jepa.py):
  - Single shared encoder instead of student + teacher (EMA copy)
  - No predictor MLP (the JEPA cosine objective is replaced by L_sim + L_SIGReg)
  - Forward pass accepts V masked views and one global unmasked view
  - encode() keeps the same signature as CellJEPA.encode() for eval compatibility
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_jepa import (
    ValueEmbedding,
    TransformerEncoder,
    build_attention_mask,
)


class CellJEPA_SIGReg(nn.Module):
    """
    Cell-JEPA variant using SIGReg instead of EMA + stop-gradient.

    Architecture
    ------------
    f_gene  : Gene embedding lookup  (vocab_size, d_model)
    f_val   : Value embedding MLP    scalar -> d_model
    encoder : Single shared TransformerEncoder (12 layers, same as Cell-JEPA student)
    r(·)    : Value head MLP         d_model -> n_bins+1  (reconstruction)
    gepc    : GEPC projection + weight matrix (fine-tuning)

    Removed vs. CellJEPA
    --------------------
    - teacher encoder (no EMA copy)
    - predictor MLP p(·)
    - update_teacher() method

    Args:
        vocab_size:  Total gene vocabulary size (incl. special tokens).
        n_bins:      Expression quantile bins (default 50).
        d_model:     Hidden dimension (default 512).
        n_layers:    Transformer depth (default 12).
        n_heads:     Attention heads (default 8).
        ffn_dim:     FFN hidden dim (default 2048).
        dropout:     Dropout probability (default 0.2).
        n_views:     Number of masked views per forward call (default 2).
    """

    def __init__(
        self,
        vocab_size: int,
        n_bins: int = 50,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.2,
        n_views: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_bins = n_bins
        self.n_views = n_views

        # --- Tokenization & Embeddings ---
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = ValueEmbedding(d_model, dropout=dropout)

        # --- Single shared encoder (no teacher copy) ---
        self.encoder = TransformerEncoder(d_model, n_layers, n_heads, ffn_dim, dropout)

        # --- Reconstruction value head r(·) ---
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_bins + 1),
        )

        # --- GEPC heads (used during fine-tuning) ---
        self.gepc_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.gepc_W = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def embed(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
    ) -> torch.Tensor:
        """z_i = f_gene(y_i) + f_val(v_i)  →  (B, L, d_model)"""
        return self.gene_embedding(gene_ids) + self.value_embedding(values)

    def forward(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        masked_vals_list: list[torch.LongTensor],
        is_masked_list: list[torch.BoolTensor],
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Forward pass for SIGReg pre-training.

        Args:
            gene_ids:          (B, L) gene vocabulary IDs (incl. <cls> at pos 0)
            values:            (B, L) unmasked bin indices — global view
            masked_vals_list:  list of n_views (B, L) tensors — masked bin indices
            is_masked_list:    list of n_views (B, L) bool tensors — mask positions
            is_pad:            (B, L) True at padding positions

        Returns dict with keys:
            e_global   : (B, D) CLS embedding of global (unmasked) view
            e_views    : list of n_views (B, D) CLS embeddings of masked views
            v_hat      : (B, L, n_bins+1) reconstruction logits from global view
            gepc_scores: (B, L) GEPC inner-product scores
            H_global   : (B, L, D) global view hidden states (for reconstruction loss)
        """
        # --- Global (unmasked) view ---
        Z_global = self.embed(gene_ids, values)
        H_global = self.encoder(Z_global, key_padding_mask=is_pad)
        e_global = H_global[:, 0, :]       # (B, D) CLS

        # --- Masked views (shared encoder, no stop-grad) ---
        e_views = []
        for masked_vals, is_masked in zip(masked_vals_list, is_masked_list):
            attn_mask = build_attention_mask(is_masked, is_pad)
            Z_view = self.embed(gene_ids, masked_vals)
            H_view = self.encoder(
                Z_view, attn_mask=attn_mask, key_padding_mask=is_pad
            )
            e_views.append(H_view[:, 0, :])   # (B, D) CLS per view

        # --- Reconstruction head (from global view) ---
        v_hat = self.value_head(H_global)      # (B, L, n_bins+1)

        # --- GEPC head ---
        gene_query = self.gepc_proj(self.gene_embedding(gene_ids))  # (B, L, D)
        cell_proj = e_global @ self.gepc_W                          # (B, D)
        gepc_scores = (gene_query * cell_proj.unsqueeze(1)).sum(-1) # (B, L)

        return {
            "e_global":    e_global,
            "e_views":     e_views,
            "v_hat":       v_hat,
            "gepc_scores": gepc_scores,
            "H_global":    H_global,
        }

    @torch.no_grad()
    def encode(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        is_pad: torch.BoolTensor,
        use_teacher: bool = True,
    ) -> torch.Tensor:
        """
        Extract cell-level <cls> embeddings for evaluation.

        API-compatible with CellJEPA.encode() — use_teacher kwarg is accepted
        but ignored (single shared encoder).

        Args:
            gene_ids: (B, L)
            values:   (B, L) unmasked bin indices
            is_pad:   (B, L)
            use_teacher: ignored, kept for API compatibility

        Returns:
            (B, d_model) cell embeddings
        """
        Z = self.embed(gene_ids, values)
        H = self.encoder(Z, key_padding_mask=is_pad)
        return H[:, 0, :]

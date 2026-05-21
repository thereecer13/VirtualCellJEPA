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
from torch.utils.checkpoint import checkpoint

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
        vocab_size:       Total gene vocabulary size (incl. special tokens).
        n_bins:           Expression quantile bins (default 50).
        d_model:          Hidden dimension (default 512).
        n_layers:         Transformer depth (default 12).
        n_heads:          Attention heads (default 8).
        ffn_dim:          FFN hidden dim (default 2048).
        dropout:          Dropout probability (default 0.2).
        n_views:          Number of masked views per forward call (default 2).
        n_perturbations:  Perturbation vocab size; 0 disables perturbation heads.
        predict_delta:    If True, adds a delta regression head for Δ prediction.
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
        grad_checkpoint: bool = False,
        n_perturbations: int = 0,
        predict_delta: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_bins = n_bins
        self.n_views = n_views
        self.grad_checkpoint = grad_checkpoint
        self.n_perturbations = n_perturbations
        self.predict_delta = predict_delta

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

        # --- Perturbation heads (optional) ---
        if n_perturbations > 0:
            self.perturb_embedding = nn.Embedding(n_perturbations, d_model)
            self.perturb_value_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, n_bins + 1),
            )
            # Trajectory predictor p_traj: (e_ctrl ‖ p_global) → Δe_pred
            self.p_traj = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            if predict_delta:
                self.delta_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, 1),
                )

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

    def _encode(self, Z, attn_mask, key_padding_mask):
        """Encoder call, optionally with activation checkpointing."""
        if self.grad_checkpoint and self.training:
            # Checkpoint each transformer layer individually to minimise peak memory.
            x = Z
            for layer in self.encoder.layers:
                x = checkpoint(layer, x, attn_mask, key_padding_mask, use_reentrant=False)
            return self.encoder.final_norm(x)
        return self.encoder(Z, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

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
        H_global = self._encode(Z_global, None, is_pad)
        e_global = H_global[:, 0, :]       # (B, D) CLS

        # --- Masked views (shared encoder, no stop-grad) ---
        e_views = []
        for masked_vals, is_masked in zip(masked_vals_list, is_masked_list):
            attn_mask = build_attention_mask(is_masked, is_pad)
            Z_view = self.embed(gene_ids, masked_vals)
            H_view = self._encode(Z_view, attn_mask, is_pad)
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

    def forward_perturb(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        pert_ids: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Forward pass for perturbation fine-tuning (single encoder, no teacher).

        The encoder sees control expression + perturbation embedding and predicts
        the post-perturbation state. Unlike CellJEPA, there is no EMA teacher,
        so no e_pert / e_tilde_pert targets are produced.

        Args:
            gene_ids:    (B, L) gene vocabulary IDs
            values:      (B, L) baseline (control) bin indices
            pert_ids:    (B, L) perturbation vocabulary IDs (0 = no perturbation)
            pert_values: (B, L) ground-truth post-perturbation bin indices (unused
                         in forward; kept for API compatibility with CellJEPA)
            is_pad:      (B, L) True at padding positions

        Returns dict with keys:
            e_hat_pert  : (B, D) CLS embedding
            v_hat_pert  : (B, L, n_bins+1) reconstruction logits
            H_hat_pert  : (B, L, D) hidden states
        """
        if self.n_perturbations == 0:
            raise RuntimeError(
                "forward_perturb requires n_perturbations > 0 at model init."
            )
        p = self.perturb_embedding(pert_ids)              # (B, L, D)
        Z = self.embed(gene_ids, values) + p              # ctrl + pert embedding
        H = self._encode(Z, attn_mask=None, key_padding_mask=is_pad)
        e_hat = H[:, 0, :]                                # (B, D) CLS
        v_hat_pert = self.perturb_value_head(H)           # (B, L, n_bins+1)
        return {"e_hat_pert": e_hat, "v_hat_pert": v_hat_pert, "H_hat_pert": H}

    def forward_perturb_delta(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        pert_ids: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Like forward_perturb() but also predicts Δ = pert_bins − ctrl_bins
        as a direct scalar regression per gene token.

        Requires predict_delta=True at model init.

        Returns same keys as forward_perturb() plus:
            delta_hat: (B, L) scalar delta predictions
        """
        if not self.predict_delta:
            raise RuntimeError(
                "forward_perturb_delta requires predict_delta=True at model init."
            )
        out = self.forward_perturb(gene_ids, values, pert_ids, pert_values, is_pad)
        delta_hat = self.delta_head(out["H_hat_pert"]).squeeze(-1)  # (B, L)
        return {**out, "delta_hat": delta_hat}

    def forward_perturb_traj(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        pert_ids: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Decoupled perturbation forward pass for SIGReg (no EMA teacher).

        The encoder is kept blind to the perturbation identity and produces a
        clean control embedding e_ctrl.  The trajectory predictor p_traj maps
        (e_ctrl ‖ p_global) → Δe_pred, and e_pert_pred = e_ctrl + Δe_pred.

        Since SIGReg has no EMA teacher, stop-gradient targets are computed by
        running the same encoder under torch.no_grad():
            e_ctrl_sg  = e_ctrl.detach()
            e_pert_sg  = encoder(no_grad, pert_values)[:, 0, :]
            delta_target = e_pert_sg − e_ctrl_sg

        Returns dict with keys:
            e_ctrl       : (B, D) control CLS embedding
            delta_pred   : (B, D) predicted latent trajectory Δe_pred
            delta_target : (B, D) stop-gradient true trajectory
            e_pert_pred  : (B, D) predicted perturbed embedding
            v_hat_pert   : (B, L, n_bins+1) reconstruction logits
        """
        if self.n_perturbations == 0:
            raise RuntimeError(
                "forward_perturb_traj requires n_perturbations > 0 at model init."
            )

        # Step 1: Encoder sees ONLY the control state (no perturbation embedding)
        Z_ctrl = self.embed(gene_ids, values)                          # (B, L, D)
        H_ctrl = self._encode(Z_ctrl, attn_mask=None,
                              key_padding_mask=is_pad)                 # (B, L, D)
        e_ctrl = H_ctrl[:, 0, :]                                       # (B, D)

        # Step 2: Global perturbation embedding — one vector per cell
        p_global = self.perturb_embedding(pert_ids[:, 0])              # (B, D)

        # Step 3: Trajectory predictor
        traj_input  = torch.cat([e_ctrl, p_global], dim=-1)            # (B, 2D)
        delta_pred  = self.p_traj(traj_input)                          # (B, D)
        e_pert_pred = e_ctrl + delta_pred                              # (B, D)

        # Step 4: Stop-gradient targets (no EMA teacher — same encoder, no_grad)
        e_ctrl_sg = e_ctrl.detach()                                    # (B, D)
        with torch.no_grad():
            Z_pert_sg = self.embed(gene_ids, pert_values)              # (B, L, D)
            H_pert_sg = self._encode(Z_pert_sg, attn_mask=None,
                                     key_padding_mask=is_pad)
            e_pert_sg = H_pert_sg[:, 0, :]                             # (B, D)
        delta_target = e_pert_sg - e_ctrl_sg                           # (B, D)

        # Step 5: Reconstruct gene expression — per-token hidden states shifted by
        # the global perturbation prediction (H_ctrl gives local gene context;
        # e_pert_pred.unsqueeze(1) broadcasts the latent delta across all positions).
        v_hat_pert = self.perturb_value_head(H_ctrl + e_pert_pred.unsqueeze(1))  # (B, L, n_bins+1)

        return {
            "e_ctrl":       e_ctrl,
            "delta_pred":   delta_pred,
            "delta_target": delta_target,
            "e_pert_pred":  e_pert_pred,
            "v_hat_pert":   v_hat_pert,
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

"""
Cell-JEPA Model Architecture
==============================
Implements Section 2.2 of the paper:

  2.2.1  Tokenization & Input Embeddings
           - Gene embedding lookup  : f_gene(y_i)
           - Value embedding MLP    : f_val(v_i)
           - z_i = y_i + v_i  (element-wise sum)

  2.2.2  Student-Teacher Encoders
           - Both share the scGPT-style bidirectional Transformer backbone
           - Teacher updated via EMA of student weights

  2.2.3  Transformer Block & Attention Masking
           - Structured attention mask: masked tokens attend only to
             unmasked tokens (and themselves), not to other masked tokens

  Additional:
           - MLP Predictor head p(·) for JEPA objective  (Section 2.3)
           - MLP Value head r(·) for reconstruction loss  (Section 2.3)
           - GEPC head for fine-tuning                    (Section 2.4)
"""

from __future__ import annotations

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Value Embedding MLP  — f_val(v_i)
# ---------------------------------------------------------------------------

class ValueEmbedding(nn.Module):
    """
    Maps a scalar bin index (or the sentinel -1) to the model's hidden dim.
    Input is a float scalar; the MLP is applied element-wise across sequence.

    Architecture: Linear -> GELU -> Linear
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (B (batch size), L (sequence length)) integer bin indices (may contain -1 for masked tokens)
        Returns:
            (B, L, d_model) value embeddings
        """
        # Cast to float and add feature dim for the MLP, because MLP expects (B, L, 1) input
        return self.net(v.float().unsqueeze(-1))


# ---------------------------------------------------------------------------
# Structured Attention Mask Builder
# ---------------------------------------------------------------------------

def build_attention_mask(
    is_masked: torch.BoolTensor,
    is_pad: torch.BoolTensor,
) -> torch.Tensor:
    """
    Build the structured attention mask described in Section 2.2.3.

    Rules for each (query i, key j) pair:
      - If j is a padding token          → -inf  (never attend to pad)
      - If j is masked AND i != j        → -inf  (no cross-masked attention)
      - Otherwise                        → 0

    This means:
      • Unmasked tokens attend to all non-pad tokens.
      • Masked tokens attend to all unmasked tokens + themselves.
      • No masked token attends to any OTHER masked token.

    Args:
        is_masked: (B, L) — True where gene values are masked
        is_pad:    (B, L) — True where position is padding

    Returns:
        (B, L, L) additive attention bias (0 or -inf)
    """
    B, L = is_masked.shape
    device = is_masked.device

    # Start with a zeros matrix; shape (B, L, L)
    attn_mask = torch.zeros(B, L, L, device=device)

    # Positions that are padding — block entire column
    # Shape: (B, 1, L) broadcast over rows
    pad_cols = is_pad.unsqueeze(1).expand(B, L, L)
    attn_mask = attn_mask.masked_fill(pad_cols, float("-inf"))

    # Masked tokens as keys — build (B, L, L) grid
    # masked_as_key[b, i, j] = True  iff j is masked
    masked_as_key = is_masked.unsqueeze(1).expand(B, L, L)
    # Self-attention exemption: (b, i, i) where i is masked is allowed
    eye = torch.eye(L, device=device).bool().unsqueeze(0)  # (1, L, L)
    cross_masked = masked_as_key & ~eye  # block cross-masked, keep diagonal

    attn_mask = attn_mask.masked_fill(cross_masked, float("-inf"))

    return attn_mask


# ---------------------------------------------------------------------------
# Transformer Block (scGPT-style bidirectional)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Standard pre-LN Transformer block:
        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Uses nn.MultiheadAttention which accepts an additive attn_mask.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, L, D)
            attn_mask:        (B*n_heads, L, L) or (B, L, L) additive mask
            key_padding_mask: (B, L) — True where key is padding
        """
        # Pre-LN attention
        normed = self.norm1(x)

        # nn.MultiheadAttention requires 3D attn_mask of shape (B*n_heads, L, L).
        # Our mask is (B, L, L), so expand it here before the call.
        expanded_mask = attn_mask
        if attn_mask is not None and attn_mask.dim() == 3:
            expanded_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)

        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=expanded_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder (scGPT backbone, shared by student & teacher)
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Stack of TransformerBlocks forming the scGPT-style bidirectional encoder.

    Hyperparameters from Appendix E.1:
        d_model = 512, n_layers = 12, n_heads = 8, ffn_dim = 512*4 = 2048
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, L, D)
            attn_mask:        (B, L, L) additive attention bias
            key_padding_mask: (B, L) — True at padding positions
        Returns:
            H: (B, L, D) hidden states
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Full Cell-JEPA Model
# ---------------------------------------------------------------------------

class CellJEPA(nn.Module):
    """
    Cell-JEPA: Joint Embedding Predictive Architecture for scRNA-seq.

    Components
    ----------
    f_gene  : Gene embedding lookup table  (vocab_size, d_model)
    f_val   : Value embedding MLP          scalar -> d_model
    g_S     : Student Transformer encoder
    g_T     : Teacher Transformer encoder  (EMA copy of g_S)
    p(·)    : MLP Predictor head           d_model -> d_model  (JEPA)
    r(·)    : MLP Value head               d_model -> n_bins   (reconstruction)

    
    Fine-tuning extras
    ------------------
    gepc_proj : MLP for GEPC query projection  d_model -> d_model
    gepc_W    : Weight matrix for GEPC inner product

    Args:
        vocab_size: Total gene vocabulary size (incl. special tokens).
        n_bins:     Number of expression quantile bins (B, default 50).
        d_model:    Hidden dimension (default 512).
        n_layers:   Transformer depth (default 12).
        n_heads:    Attention heads (default 8).
        ffn_dim:    FFN hidden dim (default 2048).
        dropout:    Dropout probability (default 0.2).
        ema_momentum: EMA coefficient m for teacher update (default 0.996).
        predictor_hidden: Hidden dim of the MLP predictor head.
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
        ema_momentum: float = 0.996,
        predictor_hidden: int = 512,
        n_perturbations: int = 0,
        predict_delta: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_bins = n_bins
        self.ema_momentum = ema_momentum
        self.n_perturbations = n_perturbations
        self.predict_delta = predict_delta

        # --- Tokenization & Embeddings (Section 2.2.1) ---
        # Gene embedding lookup: f_gene
        # vocab includes regular genes + <cls> + <pad> + any other special tokens
        self.gene_embedding = nn.Embedding(vocab_size, d_model)

        # Value embedding MLP: f_val
        # Accepts bin indices in {-1, 0, 1, ..., n_bins}
        # -1 is the mask sentinel vmask
        self.value_embedding = ValueEmbedding(d_model, dropout=dropout)

        # --- Student Encoder g_S ---
        self.student = TransformerEncoder(d_model, n_layers, n_heads, ffn_dim, dropout)

        # --- Teacher Encoder g_T (EMA, no gradients) ---
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # --- JEPA Predictor head p(·) (Section 2.3) ---
        # 2-layer MLP with GELU (matches "embedding alignment MLP" in Appendix E.1)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, d_model),
        )

        # --- Reconstruction value head r(·) (Section 2.3) ---
        # Applied row-wise to student hidden states to predict bin indices
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_bins + 1),  # bins 0..n_bins; 0 = unexpressed
        )

        # --- GEPC heads (Section 2.4.2) ---
        self.gepc_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.gepc_W = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        # --- Perturbation heads (Section 2.5) ---
        # f_perturb: perturbation ID embedding (only created when n_perturbations > 0)
        if n_perturbations > 0:
            self.perturb_embedding = nn.Embedding(n_perturbations, d_model)
            # Separate predictor head p^pert(·) for perturbed JEPA
            self.perturb_predictor = nn.Sequential(
                nn.Linear(d_model, predictor_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(predictor_hidden, d_model),
            )
            # Separate value head r^pert(·) for perturbation reconstruction
            self.perturb_value_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_bins + 1),
            )
            # Trajectory predictor p_traj: (e_ctrl ‖ p_global) → Δe_pred
            # Input is 2×d_model (concat of cell embedding and perturbation embedding).
            self.p_traj = nn.Sequential(
                nn.Linear(2 * d_model, predictor_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(predictor_hidden, d_model),
            )
            if predict_delta:
                # Delta head: predicts (pert_bins - ctrl_bins) as a scalar per gene
                self.delta_head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),  # scalar per token
                )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # EMA Teacher Update  (Section 2.2.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self):
        """
        θ_T ← m * θ_T + (1 - m) * θ_S
        Call once per training step AFTER the backward pass.
        """
        m = self.ema_momentum
        for s_param, t_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

    # ------------------------------------------------------------------
    # Input embedding  (Eq. 2.1)
    # ------------------------------------------------------------------

    def embed(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
    ) -> torch.Tensor:
        """
        z_i = f_gene(y_i) + f_val(v_i)

        Args:
            gene_ids: (B, L) gene vocabulary IDs
            values:   (B, L) bin indices (may be -1 for masked tokens)
        Returns:
            (B, L, d_model)
        """
        y = self.gene_embedding(gene_ids)    # (B, L, D)
        v = self.value_embedding(values)     # (B, L, D)
        return y + v

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        masked_vals: torch.LongTensor,
        is_masked: torch.BoolTensor,
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Full forward pass for pre-training.

        Args:
            gene_ids:    (B, L) — gene vocabulary IDs (incl. <cls> at pos 0)
            values:      (B, L) — unmasked bin indices (for teacher)
            masked_vals: (B, L) — masked bin indices  (for student; -1 at masked pos)
            is_masked:   (B, L) — True at masked gene positions
            is_pad:      (B, L) — True at padding positions

        Returns dict with keys:
            e_hat       : (B, D) student <cls> embedding
            e           : (B, D) teacher <cls> embedding  (stop-grad during JEPA loss)
            e_tilde     : (B, D) predictor output  p(e_hat)
            v_hat       : (B, L, n_bins+1) value head logits (for reconstruction)
            gepc_scores : (B, L) GEPC inner-product scores
        """
        B, L = gene_ids.shape

        # --- Build attention mask ---
        # Don't include <cls> (pos 0) or padding in "is_masked"
        attn_mask = build_attention_mask(is_masked, is_pad)  # (B, L, L)

        # nn.MultiheadAttention with batch_first=True and 3D attn_mask
        # needs shape (B, L, L) — PyTorch >=2.0 supports this natively.
        key_padding_mask = is_pad  # (B, L)

        # --- Student: processes MASKED input Z_tilde ---
        Z_tilde = self.embed(gene_ids, masked_vals)   # (B, L, D)
        H_hat = self.student(Z_tilde, attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask)  # (B, L, D)
        e_hat = H_hat[:, 0, :]   # <cls> embedding  (B, D)

        # --- Teacher: processes UNMASKED input Z ---
        with torch.no_grad():
            Z = self.embed(gene_ids, values)           # (B, L, D)
            H = self.teacher(Z, key_padding_mask=key_padding_mask)  # (B, L, D)
            e = H[:, 0, :]        # teacher <cls>     (B, D)

        # --- JEPA Predictor head ---
        e_tilde = self.predictor(e_hat)               # (B, D)

        # --- Gene-level Reconstruction head ---
        v_hat = self.value_head(H_hat)                # (B, L, n_bins+1)

        # --- GEPC head ---
        # v_tilde_i = f(y_i)^T W e_hat
        gene_query = self.gepc_proj(self.gene_embedding(gene_ids))  # (B, L, D)
        # Projected cell embedding: e_hat W   shape (B, D)
        cell_proj = e_hat @ self.gepc_W               # (B, D)
        # Inner product: (B, L, D) · (B, D, 1) -> (B, L)
        gepc_scores = (gene_query * cell_proj.unsqueeze(1)).sum(-1)  # (B, L)

        return {
            "e_hat":        e_hat,
            "e":            e,
            "e_tilde":      e_tilde,
            "v_hat":        v_hat,
            "gepc_scores":  gepc_scores,
            "H_hat":        H_hat,
        }

    # ------------------------------------------------------------------
    # Encode (inference / zero-shot embedding extraction)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        is_pad: torch.BoolTensor,
        use_teacher: bool = True,
    ) -> torch.Tensor:
        """
        Extract cell-level <cls> embeddings for downstream evaluation.

        Args:
            gene_ids: (B, L)
            values:   (B, L) — unmasked bin indices
            is_pad:   (B, L)
            use_teacher: Use teacher encoder (recommended for evaluation).

        Returns:
            (B, d_model) cell embeddings
        """
        encoder = self.teacher if use_teacher else self.student
        Z = self.embed(gene_ids, values)
        H = encoder(Z, key_padding_mask=is_pad)
        return H[:, 0, :]  # <cls> embedding

    # ------------------------------------------------------------------
    # Perturbation Forward Pass  (Section 2.5)
    # ------------------------------------------------------------------

    def forward_perturb(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        pert_ids: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Forward pass for perturbation finetuning (Section 2.5).

        No input masking is applied; the student sees the perturbed gene
        expression directly. The teacher processes the ground-truth
        perturbed expression as the target.

        Token construction (Eq. in Section 2.5):
            student input:  z̃_i^pert = y_i + p_i + v_i  (current / baseline values)
            teacher target: z_i^pert  = y_i + p_i + v_i^pert (ground-truth perturbed)

        Args:
            gene_ids:   (B, L) gene vocabulary IDs (incl. <cls> at pos 0)
            values:     (B, L) baseline (unperturbed) bin indices for student input
            pert_ids:   (B, L) perturbation vocabulary IDs (0 for unperturbed genes)
            pert_values:(B, L) ground-truth post-perturbation bin indices for teacher
            is_pad:     (B, L) True at padding positions

        Returns dict with keys:
            e_hat_pert  : (B, D) student perturbed <cls> embedding
            e_pert      : (B, D) teacher perturbed <cls> embedding (stop-grad)
            e_tilde_pert: (B, D) predictor output p^pert(e_hat_pert)
            v_hat_pert  : (B, L, n_bins+1) value head logits for perturbation reconstruction
        """
        if self.n_perturbations == 0:
            raise RuntimeError(
                "forward_perturb requires n_perturbations > 0 at model init."
            )

        # Perturbation embeddings: p_i = f_perturb(pert_ids)
        p = self.perturb_embedding(pert_ids)   # (B, L, D)

        # --- Student: z̃_i^pert = y_i + p_i + v_i (baseline values) ---
        Z_pert_student = self.embed(gene_ids, values) + p   # (B, L, D)
        H_hat_pert = self.student(Z_pert_student, key_padding_mask=is_pad)
        e_hat_pert = H_hat_pert[:, 0, :]                    # (B, D)

        # --- Teacher: z_i^pert = y_i + p_i + v_i^pert (ground-truth perturbed) ---
        with torch.no_grad():
            Z_pert_teacher = self.embed(gene_ids, pert_values) + p   # (B, L, D)
            H_pert = self.teacher(Z_pert_teacher, key_padding_mask=is_pad)
            e_pert = H_pert[:, 0, :]                                  # (B, D)

        # --- Perturbed JEPA predictor ---
        e_tilde_pert = self.perturb_predictor(e_hat_pert)   # (B, D)

        # --- Perturbation reconstruction head (over all genes, no masking) ---
        v_hat_pert = self.perturb_value_head(H_hat_pert)    # (B, L, n_bins+1)

        return {
            "e_hat_pert":   e_hat_pert,
            "e_pert":       e_pert,
            "e_tilde_pert": e_tilde_pert,
            "v_hat_pert":   v_hat_pert,
            "H_hat_pert":   H_hat_pert,
        }

    # ------------------------------------------------------------------
    # Delta Perturbation Forward Pass  (predict pert − ctrl delta)
    # ------------------------------------------------------------------

    def forward_perturb_delta(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        pert_ids: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Like forward_perturb() but also returns delta predictions from delta_head.

        The model predicts (pert_bins − ctrl_bins) directly as a scalar per gene,
        rather than predicting absolute post-perturbation expression.

        Requires predict_delta=True at model init (creates self.delta_head).

        Returns dict with same keys as forward_perturb() plus:
            delta_hat: (B, L) scalar delta predictions per gene token
        """
        if not self.predict_delta:
            raise RuntimeError(
                "forward_perturb_delta requires predict_delta=True at model init."
            )

        out = self.forward_perturb(gene_ids, values, pert_ids, pert_values, is_pad)

        # delta_head applied to student hidden states: (B, L, D) -> (B, L, 1) -> (B, L)
        delta_hat = self.delta_head(out["H_hat_pert"]).squeeze(-1)  # (B, L)

        return {**out, "delta_hat": delta_hat}

    # ------------------------------------------------------------------
    # Trajectory Predictor Forward Pass  (decoupled cell-level strategy)
    # ------------------------------------------------------------------

    def forward_perturb_traj(
        self,
        gene_ids: torch.LongTensor,
        values: torch.LongTensor,
        pert_ids: torch.LongTensor,
        pert_values: torch.LongTensor,
        is_pad: torch.BoolTensor,
    ) -> dict:
        """
        Decoupled perturbation forward pass (trajectory predictor strategy).

        The Student Encoder is kept blind to the perturbation identity —
        it receives only the baseline control expression and produces a
        clean cell embedding e_ctrl.  A separate trajectory predictor head
        p_traj then maps (e_ctrl ‖ p_global) to a latent delta Δe_pred,
        and the predicted perturbed embedding is e_pert_pred = e_ctrl + Δe_pred.

        Token construction:
            student input : y_i + v_i  (control values, NO perturbation embedding)
            teacher ctrl  : y_i + v_i  (same as student, for e_ctrl_target)
            teacher pert  : y_i + v_i^pert (ground-truth perturbed, for e_pert_target)

        Args:
            gene_ids:    (B, L) gene vocabulary IDs (incl. <cls> at pos 0)
            values:      (B, L) baseline (control) bin indices
            pert_ids:    (B, L) perturbation vocab IDs — same value broadcast per cell
            pert_values: (B, L) ground-truth post-perturbation bin indices
            is_pad:      (B, L) True at padding positions

        Returns dict with keys:
            e_ctrl       : (B, D) student control <cls> embedding
            delta_pred   : (B, D) predicted latent trajectory Δe_pred from p_traj
            delta_target : (B, D) true latent trajectory Δe_target (stop-grad)
            e_pert_pred  : (B, D) predicted perturbed embedding = e_ctrl + delta_pred
            v_hat_pert   : (B, L, n_bins+1) reconstruction logits from e_pert_pred
        """
        if self.n_perturbations == 0:
            raise RuntimeError(
                "forward_perturb_traj requires n_perturbations > 0 at model init."
            )

        # Step 1: Student sees ONLY the control state (no perturbation embedding)
        Z_ctrl = self.embed(gene_ids, values)                      # (B, L, D)
        H_ctrl = self.student(Z_ctrl, key_padding_mask=is_pad)     # (B, L, D)
        e_ctrl = H_ctrl[:, 0, :]                                   # (B, D)

        # Step 2: Global perturbation embedding — one vector per cell.
        # pert_ids[:, 0] is safe because PerturbationDataset broadcasts the same
        # pert_id to every token position in the sequence.
        p_global = self.perturb_embedding(pert_ids[:, 0])          # (B, D)

        # Step 3: Trajectory predictor
        traj_input = torch.cat([e_ctrl, p_global], dim=-1)         # (B, 2D)
        delta_pred = self.p_traj(traj_input)                       # (B, D)
        e_pert_pred = e_ctrl + delta_pred                          # (B, D)

        # Step 4: Teacher targets — both ctrl and pert, no perturbation embedding
        with torch.no_grad():
            H_ctrl_t     = self.teacher(Z_ctrl, key_padding_mask=is_pad)
            e_ctrl_target = H_ctrl_t[:, 0, :]                     # (B, D)

            Z_pert_t      = self.embed(gene_ids, pert_values)      # (B, L, D)
            H_pert_t      = self.teacher(Z_pert_t, key_padding_mask=is_pad)
            e_pert_target = H_pert_t[:, 0, :]                     # (B, D)

            delta_target  = e_pert_target - e_ctrl_target          # (B, D)

        # Step 5: Reconstruct gene expression — per-token hidden states shifted by
        # the global perturbation prediction (H_ctrl gives local gene context;
        # e_pert_pred.unsqueeze(1) broadcasts the latent delta across all positions).
        v_hat_pert = self.perturb_value_head(H_ctrl + e_pert_pred.unsqueeze(1))  # (B, L, n_bins+1)

        return {
            "e_ctrl":        e_ctrl,
            "delta_pred":    delta_pred,
            "delta_target":  delta_target,
            "e_pert_pred":   e_pert_pred,
            "v_hat_pert":    v_hat_pert,
        }

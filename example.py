"""
Cell-JEPA: End-to-End Example
================================
Demonstrates the full pipeline on synthetic data:

  1. Data preprocessing (quantile binning + subsampling)
  2. Model instantiation (student-teacher CellJEPA)
  3. Pre-training (L_rec + L_JEPA)
  4. Fine-tuning (GEP + GEPC + ECS + JEPA)
  5. Evaluation (AvgBIO — requires scanpy + sklearn)

To run on real data, replace `make_synthetic_data()` with your AnnData
count matrix and cell-type annotations.
"""

import numpy as np
import torch

from preprocessing import SingleCellDataset
from cell_jepa import CellJEPA
from trainer import Pretrainer, Finetuner, PretrainConfig, FinetuneConfig


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_cells: int = 500,
    n_genes: int = 3000,
    n_cell_types: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple synthetic scRNA-seq count matrix with ~90% sparsity.
    Returns (count_matrix, cell_types).
    """
    rng = np.random.default_rng(seed)
    cell_types = rng.integers(0, n_cell_types, size=n_cells)

    # Cell-type specific mean expression profiles
    type_means = rng.exponential(scale=2.0, size=(n_cell_types, n_genes))

    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        ct = cell_types[i]
        # Poisson counts from cell-type mean
        raw = rng.poisson(type_means[ct])
        # Simulate ~85% dropout
        dropout_mask = rng.random(n_genes) < 0.85
        raw[dropout_mask] = 0
        counts[i] = raw

    return counts, cell_types


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocab(n_genes: int) -> tuple[dict, int, int, int]:
    """
    Build a simple gene vocabulary.
    Token IDs: 0=<cls>, 1=<pad>, 2..n_genes+1=genes

    Returns:
        gene_vocab:    dict mapping gene_index -> token_id
        vocab_size:    total vocab size
        cls_token_id:  0
        pad_token_id:  1
    """
    cls_token_id = 0
    pad_token_id = 1
    # gene i -> token id i+2
    gene_vocab = {i: i + 2 for i in range(n_genes)}
    vocab_size = n_genes + 2
    return gene_vocab, vocab_size, cls_token_id, pad_token_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # 1. Data
    # -----------------------------------------------------------------------
    print("\n--- Generating synthetic scRNA-seq data ---")
    n_cells, n_genes = 500, 3000
    counts, cell_types = make_synthetic_data(n_cells=n_cells, n_genes=n_genes)
    print(f"Count matrix shape: {counts.shape}")
    print(f"Sparsity: {(counts == 0).mean():.1%}")

    gene_vocab, vocab_size, cls_id, pad_id = build_vocab(n_genes)

    # Pre-training dataset (mask_ratio=0.15, as in Appendix E.1)
    pretrain_dataset = SingleCellDataset(
        count_matrix=counts,
        gene_vocab=gene_vocab,
        cell_types=cell_types,
        n_bins=50,
        L_max=200,  # 600 for full training; 200 keeps attention masks small on local GPU/MPS
        cls_token_id=cls_id,
        pad_token_id=pad_id,
        mask_ratio=0.15,
    )

    # Fine-tuning dataset (mask_ratio=0.40, as in Appendix E.2)
    finetune_dataset = SingleCellDataset(
        count_matrix=counts,
        gene_vocab=gene_vocab,
        cell_types=cell_types,
        n_bins=50,
        L_max=200,  # 600 for full training; 200 keeps attention masks small on local GPU/MPS
        cls_token_id=cls_id,
        pad_token_id=pad_id,
        mask_ratio=0.40,
    )

    # -----------------------------------------------------------------------
    # 2. Model
    # -----------------------------------------------------------------------
    print("\n--- Instantiating CellJEPA model ---")
    model = CellJEPA(
        vocab_size=vocab_size,
        n_bins=50,
        d_model=512,
        n_layers=12,
        n_heads=8,
        ffn_dim=2048,
        dropout=0.2,
        ema_momentum=0.996,
        predictor_hidden=512,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.1f}M")

    # -----------------------------------------------------------------------
    # 3. Pre-training
    # -----------------------------------------------------------------------
    print("\n--- Pre-training (demo: 1 epoch on small dataset) ---")
    pretrain_cfg = PretrainConfig(
        lr=1e-4,
        weight_decay=2e-4,
        lr_decay=0.9,
        batch_size=16,       # Reduced from 32 to stay within t3.medium RAM
        n_epochs=1,          # Use 4 in full training (Appendix E.1)
        log_every=5,
        w_rec=1.0,
        w_jepa=1000.0,
        num_workers=0,       # Set to 4 for real training
    )
    pretrainer = Pretrainer(model, pretrain_dataset, pretrain_cfg, device)
    pretrain_history = pretrainer.train()
    pretrainer.save("/tmp/cell_jepa_pretrain.pt")

    # -----------------------------------------------------------------------
    # 4. Fine-tuning
    # -----------------------------------------------------------------------
    print("\n--- Fine-tuning (demo: 2 epochs) ---")
    finetune_cfg = FinetuneConfig(
        lr=1e-4,
        lr_decay=0.9,
        batch_size=16,       # Reduced from 64 to stay within t3.medium RAM
        n_epochs=2,          # Use 30 in full training (Appendix E.2)
        log_every=5,
        w_gep=1.0,
        w_gepc=1.0,
        w_ecs=1.0,
        w_jepa=1000.0,
        ecs_temperature=0.1,
        include_jepa=True,
        num_workers=0,
    )
    finetuner = Finetuner(model, finetune_dataset, finetune_cfg, device)
    finetune_history = finetuner.train()
    finetuner.save("/tmp/cell_jepa_finetune.pt")

    # -----------------------------------------------------------------------
    # 5. Embedding extraction (zero-shot style)
    # -----------------------------------------------------------------------
    print("\n--- Extracting cell embeddings ---")
    from metrics import extract_embeddings

    eval_dataset = SingleCellDataset(
        count_matrix=counts,
        gene_vocab=gene_vocab,
        cell_types=cell_types,
        n_bins=50,
        L_max=200,  # 600 for full training; 200 keeps attention masks small on local GPU/MPS
        cls_token_id=cls_id,
        pad_token_id=pad_id,
        mask_ratio=0.0,   # No masking during evaluation
    )

    embeddings, labels = extract_embeddings(
        model, eval_dataset, batch_size=32, device=device, use_teacher=True  # Reduced from 128
    )
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding mean norm: {np.linalg.norm(embeddings, axis=1).mean():.3f}")

    # -----------------------------------------------------------------------
    # 6. Evaluation (requires scanpy + sklearn)
    # -----------------------------------------------------------------------
    try:
        from metrics import evaluate
        print("\n--- Running AvgBIO evaluation ---")
        results = evaluate(
            model, eval_dataset,
            batch_size=32, device=device, use_teacher=True,  # Reduced from 128
        )
        print(f"AvgBIO: {results['avg_bio']:.4f}")
    except ImportError as e:
        print(f"\nSkipping full evaluation ({e}).")
        print("Install scanpy + sklearn to enable AvgBIO metrics.")

    print("\nDone!")


if __name__ == "__main__":
    main()

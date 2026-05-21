# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

Key experimental results and insights are stored in `results.md`. Read this file when asked about prior findings or when designing follow-up experiments.

## Project Overview

An independent PyTorch re-implementation of **Cell-JEPA** (ElSheikh et al., arXiv:2602.02093), a joint-embedding predictive architecture for single-cell RNA-seq representation learning. The repo also includes **CellJEPA_SIGReg**, an EMA-free variant that replaces the student-teacher JEPA objective with SIGReg (Sketched Isotropic Gaussian Regularization).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ required. Uses `.venv/` or `.venv-1/` virtual environments locally.

## Common Commands

### Run end-to-end on synthetic data (sanity check, no GPU needed)
```bash
python example.py
```

### Smoke test for perturbation ablation (~1 min, CPU)
```bash
python compare_perturbation.py --smoke_test --device cpu
```

### Pre-train on kidney (~800k cells from CELLxGENE Census)
```bash
python pretrain_kidney.py --device cuda --drive_dir ./checkpoints/
# Resume after interruption:
python pretrain_kidney.py --device cuda --drive_dir ./checkpoints/ --resume ./checkpoints/kidney_pretrain_epoch2.pt
```

### Pre-train with universal gene vocabulary
```bash
# Build vocab first (once, merges kidney+PBMC-68K gene names):
python build_universal_vocab.py --drive_dir ./checkpoints/universal_vocab/

# Pre-train:
python pretrain_universal.py --source kidney --vocab_file ./checkpoints/universal_vocab/universal_gene_names.json --drive_dir ./checkpoints/ --device cuda
# Skip one model variant:
python pretrain_universal.py --source kidney --skip_sigreg --device cuda ...
```

### Evaluate cell-type clustering on PBMC 3k
```bash
python compare_pbmc3k.py --device cuda
```

### Transfer experiment with universal vocab
```bash
python run_transfer_universal.py --vocab_file /path/universal_gene_names.json \
    --kidney_jepa_checkpoint /path/kidney_jepa_final.pt \
    --kidney_sigreg_checkpoint /path/kidney_sigreg_final.pt --device cuda
```

### Run 2×2 perturbation ablation (absolute vs. delta × JEPA on/off)
```bash
python compare_perturbation.py --device cuda
# With pre-trained backbone:
python compare_perturbation.py --device cuda --pretrain_checkpoint ./checkpoints/kidney_pretrain_final.pt
```

## Architecture

### Core Model — `cell_jepa.py`

`CellJEPA` implements the student-teacher Transformer for scRNA-seq:
- **Tokenization**: Gene embedding lookup `f_gene` (vocab lookup) + value embedding MLP `f_val` (scalar bin → d_model). Sum to form `z_i = f_gene(y_i) + f_val(v_i)`.
- **Student/Teacher encoders**: Both are 12-layer bidirectional Transformer (`TransformerEncoder`). Teacher is a frozen EMA copy updated each step via `update_teacher()`.
- **Structured attention mask** (`build_attention_mask`): Masked tokens attend only to unmasked tokens and themselves — prevents cross-masked attention.
- **JEPA predictor head** `p(·)`: 2-layer MLP. Predicts teacher `<cls>` embedding from masked student `<cls>` embedding.
- **Reconstruction head** `r(·)`: Per-token MLP → softmax → expected bin value → MSE loss.
- **GEPC head**: Inner-product scoring for fine-tuning cell-type clustering.
- **Perturbation heads**: Optional embedding + separate predictor/value heads for perturbation fine-tuning. Activated when `n_perturbations > 0`. Delta mode (`predict_delta=True`) adds a regression head that predicts `pert_bins − ctrl_bins`.

### Alternate Model — `cell_sigreg.py`

`CellJEPA_SIGReg` is the EMA-free variant:
- Single encoder (no teacher, no predictor MLP, no `update_teacher()`)
- Forward pass accepts `V` masked views + 1 global unmasked view, returning `e_global` and `e_views`
- `encode()` has the same signature as `CellJEPA.encode()` for eval compatibility
- Replaces JEPA cosine loss with `L_sim` (view alignment) + `L_SIGReg` (Epps-Pulley-based Gaussian regularization over random projections)

### Data Pipeline — `preprocessing.py`

- `quantile_bin_cell`: Per-cell discretization of raw counts into B=50 bins; zeros stay 0.
- `SingleCellDataset`: Stochastic gene subsampling at `__getitem__` time (up to `L_max=600` expressed genes), masking, and padding. Prepends `<cls>` token at position 0.
- `PerturbationDataset`: Pairs perturbed cells with matched control means. The perturbation ID is broadcast to all gene tokens in a cell.

Vocab convention: `<cls>=0`, `<pad>=1`, genes start at index 2.

### Loss Functions — `losses.py`

| Loss class | Used for |
|---|---|
| `PretrainingLoss` | `w_rec * L_rec + w_jepa * L_JEPA` (default w_jepa=1000) |
| `FinetuningLoss` | `L_GEP + L_GEPC + L_ECS + L_JEPA` |
| `PerturbationLoss` | `L_pert_rec + L_JEPA_pert + L_ECS` |
| `DeltaPerturbationLoss` | Same but predicts Δ = pert−ctrl instead of absolute expression |
| `SIGRegPretrainingLoss` | `L_sim + L_SIGReg + L_rec` |
| `SIGRegFinetuningLoss` | `L_GEP + L_GEPC + L_ECS + L_sim + L_SIGReg` |
| `SIGRegPerturbationLoss` / `SIGRegDeltaPerturbationLoss` | SIGReg variants (no l_jepa_pert) |

`reconstruction_loss` uses expected bin value (softmax-weighted argmax) for MSE, not argmax or cross-entropy.

ECS loss is an InfoNCE contrastive objective over cell-type labels; cells with `cell_type=-1` are skipped.

### Training — `trainer.py`

Config dataclasses set all hyperparameters. Trainers:
- `Pretrainer` / `SIGRegPretrainer`: Pre-training with `save()` / `load_checkpoint()` for resume. SIGReg version uses linear LR warmup + exponential decay; standard version uses exponential decay only.
- `Finetuner` / `SIGRegFinetuner`: Fine-tuning with 90/10 train/val split.
- `PerturbationTrainer` / `SIGRegPerturbationTrainer`: Perturbation fine-tuning. SIGReg version does not call `update_teacher()`.

All trainers use AMP on CUDA. The `_sample_mask` helper in `trainer.py` generates fresh random masks per step for SIGReg (multi-view training), without requiring dataset changes.

### Evaluation — `metrics.py`, `perturb_metrics.py`

- `metrics.py`: `extract_embeddings` (runs `model.encode()` over a dataset), `evaluate` (NMI, ARI, ASW, AvgBIO via Louvain/UMAP on the `<cls>` embeddings).
- `perturb_metrics.py`: Pearson correlation, PearsonΔ, Top-20 DEG PearsonΔ, MSE for perturbation response.

### Experiment Scripts

| Script | Purpose |
|---|---|
| `pretrain_kidney.py` | Pre-train on ~800k kidney cells from CELLxGENE Census |
| `pretrain_pbmc68k.py` | Pre-train on PBMC-68K |
| `pretrain_universal.py` | Pre-train both CellJEPA and SIGReg with shared universal gene vocab |
| `build_universal_vocab.py` | Merge gene name sets from multiple datasets into one JSON vocab file |
| `compare_pbmc3k.py` | PBMC-3K clustering benchmark |
| `compare_perturbation.py` | 2×2 ablation on Adamson 2016 (absolute/delta × JEPA on/off) |
| `run_transfer_universal.py` | Universal vocab transfer: kidney/PBMC-68K → PBMC-3K |
| `run_ablation.py` | General ablation runner |
| `run_transfer.py` / `run_transfer_pbmc68k.py` | Gene-aligned transfer experiments |

Colab notebooks (`.ipynb` files) are the primary way experiments are run at scale on A100 GPUs, with Google Drive for checkpointing. The `upload_to_colab.sh` / `deploy_ec2_*.sh` scripts handle environment setup.

## Key Design Decisions

**EMA weight** (default 0.996): Teacher is updated after every gradient step, not at the end of each epoch.

**Reconstruction loss**: Computed as MSE over the **expected bin value** from softmax distribution, not cross-entropy. This follows the paper's "MSE on masked gene token" description.

**SIGReg instability**: Without EMA, `l_sim` can become erratic in the first ~5K steps. If this happens, increase `SIGRegPretrainConfig.warmup_steps` from 1000 to 5000.

**Universal vocab**: Solves the gene-overlap problem in transfer experiments (only 7% of HVGs overlap between kidney and PBMC-3K with gene-aligned approach). All datasets share the same token IDs.

**SIGReg hyperparameters**: `w_sigreg` (λ) and `w_rec` (γ) are the most sensitive SIGReg parameters. Recommended sweep: λ ∈ {0.1, 0.5, 0.9} × γ ∈ {0.1, 1.0, 10.0}.

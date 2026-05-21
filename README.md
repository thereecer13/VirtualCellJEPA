# VirtualCellJEPA

> **Disclaimer:** This is an independent re-implementation of the Cell-JEPA architecture described in ElSheikh et al. (2026). It is not affiliated with or endorsed by the original authors.

A PyTorch implementation of **Cell-JEPA**, a joint-embedding predictive architecture for single-cell RNA-seq representation learning, based on the paper:

> *Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics*
> ElSheikh et al., arXiv:2602.02093 (2026)

This repository also includes **CellJEPA_SIGReg**, an EMA-free variant that replaces the student-teacher JEPA objective with SIGReg (Sketched Isotropic Gaussian Regularization, Balestriero & LeCun 2025).

---

## Models

### Cell-JEPA (`cell_jepa.py`)

Student-teacher Transformer for scRNA-seq. The student encoder processes masked gene expression; an EMA teacher encoder processes the unmasked global view. A predictor MLP is trained to predict the teacher's CLS embedding from the student's masked CLS embedding (JEPA cosine loss). An EMA momentum of 0.996 provides stable targets without collapse.

### CellJEPA_SIGReg (`cell_sigreg.py`)

EMA-free variant with a single shared encoder. Processes one global unmasked view and V masked views. Replaces the JEPA objective with:
- **L_sim**: cosine alignment between global and masked view CLS embeddings
- **L_SIGReg**: Epps-Pulley-based Gaussian regularization over random projections — prevents dimensional collapse without a teacher network

`encode()` has the same signature as `CellJEPA.encode()` for eval compatibility.

---

## Perturbation Prediction

Both models support three perturbation prediction strategies (activated when `n_perturbations > 0`):

| Mode | Method | Key idea |
|---|---|---|
| **Absolute** | `forward_perturb` | Inject perturbation embedding into tokens; predict post-pert expression directly |
| **Delta** | `forward_perturb_delta` | Same injection; add regression head for Δ = pert − ctrl per gene |
| **Trajectory** | `forward_perturb_traj` | Encoder sees only control state; decoupled `p_traj` MLP predicts latent shift Δe; reconstruction from `H_ctrl + e_pert_pred` |

The trajectory mode's delta loss `1 − cos_sim(Δe_pred, Δe_target)` supervises the direction of change in embedding space, which empirically improves Top-20 DEG Pearson Δ — the most clinically relevant perturbation metric.

---

## Repository Structure

### Core

| File | Description |
|---|---|
| `cell_jepa.py` | Cell-JEPA model — student/teacher transformer, EMA update, all perturbation heads |
| `cell_sigreg.py` | CellJEPA_SIGReg — single encoder, multi-view SIGReg, same perturbation heads |
| `losses.py` | All loss functions: pre-training, fine-tuning, perturbation (including SIGReg and trajectory variants) |
| `trainer.py` | Config dataclasses and trainers for all modes (`Pretrainer`, `Finetuner`, `PerturbationTrainer`, SIGReg equivalents) |
| `preprocessing.py` | Per-cell quantile binning, `SingleCellDataset`, `PerturbationDataset` |
| `metrics.py` | Clustering evaluation: NMI, ARI, ASW, AvgBIO |
| `perturb_metrics.py` | Perturbation evaluation: Pearson, PearsonΔ, Top-20 DEG PearsonΔ, MSE |

### Experiment Scripts

| File | Description |
|---|---|
| `pretrain_kidney.py` | Pre-train on ~800k human kidney cells from CELLxGENE Census |
| `pretrain_pbmc68k.py` | Pre-train on PBMC-68K |
| `pretrain_universal.py` | Pre-train both CellJEPA and SIGReg with a shared universal gene vocabulary |
| `build_universal_vocab.py` | Merge gene name sets from multiple datasets into one JSON vocab file |
| `compare_pbmc3k.py` | Cell-type clustering benchmark on PBMC 3k (NMI, ARI, ASW, AvgBIO) |
| `compare_perturbation.py` | 2×2 ablation: absolute vs. delta × JEPA on/off (Adamson 2016) |
| `compare_traj_modes.py` | 3-mode comparison: absolute vs. delta vs. trajectory (Adamson 2016) |
| `run_ablation.py` | CellJEPA vs. SIGReg zero-shot and fine-tuned clustering ablation |
| `run_transfer_universal.py` | Universal vocab transfer: kidney/PBMC-68K pre-training → PBMC-3K evaluation |
| `run_transfer.py` / `run_transfer_pbmc68k.py` | Gene-aligned transfer experiments |
| `example.py` | Minimal end-to-end usage example (no GPU needed) |

### Notebooks (Colab)

| Notebook | Description |
|---|---|
| `JEPA_Colab.ipynb` | Kidney pre-training + perturbation ablation on A100 |
| `SIGReg_Ablation_Colab.ipynb` | CellJEPA vs. SIGReg 2×2 clustering ablation |
| `UniversalVocab_Colab.ipynb` | Universal vocab pre-training + transfer experiment |
| `TrajModes_Colab.ipynb` | 3-mode perturbation comparison from pre-trained checkpoint |
| `DataDiagnostics_Colab.ipynb` | Dataset statistics, HVG overlap heatmaps, UMAP, transfer signal analysis |

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Sanity check (no GPU)

```bash
python example.py
```

### Kidney pre-training

```bash
python pretrain_kidney.py --device cuda --drive_dir ./checkpoints/
# Resume after interruption:
python pretrain_kidney.py --device cuda --drive_dir ./checkpoints/ --resume ./checkpoints/kidney_pretrain_epoch2.pt
```

### Universal vocabulary pre-training

Solves the gene-overlap problem for cross-dataset transfer (only ~7% HVG overlap between kidney and PBMC-3K with gene-aligned approach).

```bash
# Build shared vocab once:
python build_universal_vocab.py --drive_dir ./checkpoints/universal_vocab/

# Pre-train both variants:
python pretrain_universal.py \
    --source kidney \
    --vocab_file ./checkpoints/universal_vocab/universal_gene_names.json \
    --drive_dir ./checkpoints/ \
    --device cuda
```

### Perturbation mode comparison (Adamson 2016)

Compares absolute, delta, and trajectory prediction modes from a pre-trained backbone:

```bash
python compare_traj_modes.py \
    --pretrain_checkpoint ./checkpoints/jepa_final.pt \
    --device cuda

# Smoke test (~2 min, CPU):
python compare_traj_modes.py --smoke_test --device cpu
```

### 2×2 perturbation ablation

```bash
python compare_perturbation.py --device cuda --pretrain_checkpoint ./checkpoints/jepa_final.pt
```

### PBMC clustering benchmark

```bash
python compare_pbmc3k.py --device cuda
```

### CellJEPA vs. SIGReg ablation

```bash
python run_ablation.py --variant all --device cuda
```

---

## Pre-training Configuration

Default hyperparameters match Appendix E.1 of the paper:

| Hyperparameter | Value |
|---|---|
| Architecture | d_model=512, 12 layers, 8 heads, ffn_dim=2048 |
| Optimizer | AdamW, lr=1e-4, weight_decay=2e-4 |
| LR schedule | Exponential decay, γ=0.9 per epoch |
| Epochs | 4 |
| Mask ratio | 0.15 |
| L_max | 600 genes per cell |
| EMA momentum | 0.996 |
| JEPA loss weight | 1000 |
| Reconstruction loss weight | 1 |

SIGReg uses linear warmup (1000 steps default) + exponential decay. If `l_sim` is erratic in early training, increase `warmup_steps` to 5000.

---

## Citation

```bibtex
@article{elsheikh2026celljepa,
  title   = {Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics},
  author  = {ElSheikh, Ali and Wang, Rui-Xi and Wu, Weimin and Wen, Yibo and
             Dibaeinia, Payam and Zhang, Jennifer Yuntong and Hu, Jerry Yao-Chieh and
             Knudson, Mei and Babu, Sudarshan and Sun, Shao-Hua and Khan, Aly A. and Liu, Han},
  journal = {arXiv preprint arXiv:2602.02093},
  year    = {2026}
}
```

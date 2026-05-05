# VirtualCellJEPA

A PyTorch implementation of **Cell-JEPA**, a joint-embedding predictive architecture for single-cell RNA-seq representation learning, based on the paper:

> *Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics*
> ElSheikh et al., arXiv:2602.02093 (2026)

Cell-JEPA augments a masked gene-expression reconstruction objective (following scGPT) with a latent-space JEPA loss: a student encoder processes masked inputs and is trained to predict the cell-level embedding produced by an EMA teacher encoder from unmasked inputs. This encourages the model to learn dropout-robust representations of cellular state rather than memorising measurement noise.

---

## Repository Structure

| File | Description |
|---|---|
| `cell_jepa.py` | Core model — student/teacher transformer, EMA update, perturbation heads |
| `losses.py` | Pre-training, fine-tuning, and perturbation loss functions |
| `trainer.py` | `Pretrainer`, `Finetuner`, and `PerturbationTrainer` with checkpoint save/resume |
| `preprocessing.py` | Per-cell quantile binning, `SingleCellDataset`, `PerturbationDataset` |
| `perturb_metrics.py` | Perturbation evaluation: Pearson, PearsonΔ, Top-20 DEG PearsonΔ, MSE |
| `metrics.py` | Clustering evaluation: NMI, ARI, ASW, AvgBIO |
| `pretrain_kidney.py` | Pre-train on ~800k human kidney cells from CELLxGENE Census |
| `compare_perturbation.py` | 2×2 ablation: absolute vs. delta objective × JEPA on/off (Adamson 2016) |
| `compare_pbmc3k.py` | Cell-type clustering evaluation on PBMC 3k (NMI, ARI, ASW, AvgBIO) |
| `example.py` | Minimal end-to-end usage example |
| `JEPA_Colab.ipynb` | Colab notebook: kidney pre-training + perturbation ablation on A100 |
| `requirements.txt` | Python dependencies |

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Kidney pre-training (replicates paper Section 3.1)

Downloads ~800k human kidney cells from CELLxGENE Census, trains Cell-JEPA for 4 epochs, and saves per-epoch checkpoints.

```bash
python pretrain_kidney.py --device cuda --drive_dir ./checkpoints/
```

Resume after interruption:

```bash
python pretrain_kidney.py --device cuda --drive_dir ./checkpoints/ --resume ./checkpoints/kidney_pretrain_epoch2.pt
```

To align the kidney gene vocabulary with a downstream task's gene set (recommended before perturbation fine-tuning):

```bash
python pretrain_kidney.py --device cuda --drive_dir ./checkpoints/ --gene_list adamson_genes.json
```

### 2. Perturbation ablation (replicates paper Section 3.4)

Runs a 2×2 ablation on the Adamson 2016 Perturb-seq dataset comparing absolute vs. delta reconstruction objectives with and without the JEPA loss. Downloads the dataset automatically (~470 MB).

Without pre-training:
```bash
python compare_perturbation.py --device cuda
```

With kidney pre-trained backbone:
```bash
python compare_perturbation.py --device cuda --pretrain_checkpoint ./checkpoints/kidney_pretrain_final.pt
```

Smoke test (~1 min, CPU):
```bash
python compare_perturbation.py --smoke_test --device cpu
```

### 3. PBMC clustering (replicates paper Section 3.2–3.3)

Evaluates cell-type clustering on PBMC 3k after fine-tuning from a pre-trained checkpoint.

```bash
python compare_pbmc3k.py --device cuda
```

### 4. Colab notebook

`JEPA_Colab.ipynb` provides a self-contained 7-cell workflow for running kidney pre-training and the perturbation ablation end-to-end on a Colab A100, with automatic Drive checkpointing and resume.

---

## Pre-training Configuration

The default hyperparameters match Appendix E.1 of the paper:

| Hyperparameter | Value |
|---|---|
| Architecture | d_model=512, 12 layers, 8 heads, ffn_dim=2048 |
| Optimizer | AdamW, lr=1e-4, weight_decay=2e-4 |
| LR schedule | Exponential decay, γ=0.9 per epoch |
| Epochs | 4 |
| Mask ratio | 0.15 |
| L_max | 600 genes per cell |
| JEPA loss weight | 1000 |
| Reconstruction loss weight | 1 |

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

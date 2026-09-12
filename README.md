# scMRDR Reproduction

[![GitHub stars](https://img.shields.io/github/stars/zhangye-zoe/scMRDR_RE?style=flat&logo=github)](https://github.com/zhangye-zoe/scMRDR_RE/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/zhangye-zoe/scMRDR_RE?style=flat&logo=github)](https://github.com/zhangye-zoe/scMRDR_RE/network/members)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)

This repository contains adapted code for reproducing [**scMRDR**](https://github.com/sjl-sjtu/scMRDR) on the **PBMC** and **BMMC** multi-omics datasets, covering data preparation, model training, cross-omics prediction, and latent-space visualization.

---

## 1. 🚀 Installation

Activate the environment and install the repository from the project root:

```bash
conda activate scmrdr
pip install -e .
```

For the original scMRDR environment and API examples, please refer to the [official repository](https://github.com/sjl-sjtu/scMRDR).

---

## 2. 📊 Data Preparation

The reproduction contains three dataset/task settings:

| Dataset | Modalities | Prediction task | Data preparation |
| --- | --- | --- | --- |
| **PBMC** | RNA + ATAC | ATAC → RNA | [`scripts/PBMC_train/data.ipynb`](scripts/PBMC_train/data.ipynb) |
| **BMMC** | RNA + ATAC | ATAC → RNA | [`scripts/BMMC_train/data_rna_atac.ipynb`](scripts/BMMC_train/data_rna_atac.ipynb) |
| **BMMC** | RNA + Protein | RNA → Protein | [`scripts/BMMC_train/data_rna_protein.ipynb`](scripts/BMMC_train/data_rna_protein.ipynb) |

PBMC cell-type annotations used for UMAP visualization were generated with [**Azimuth**](https://github.com/zhangye-zoe/Azimuth). BMMC cell-type annotations are read directly from the dataset.

---

## 3. 🧠 Model Training

Run the corresponding script from the repository root:

```bash
# PBMC RNA-ATAC: ATAC -> RNA
python scripts/PBMC_train/train.py

# BMMC RNA-ATAC: ATAC -> RNA
python scripts/BMMC_train/train_atac_rna.py

# BMMC RNA-Protein: RNA -> Protein
python scripts/BMMC_train/train_rna_protein.py
```

Each script evaluates all configured partial-pairing ratios and saves the learned latent representations, cross-omics predictions, and evaluation metrics.

> **Note:** Update `INPUT_DIR`, `SPLIT_ROOT`, and `OUTPUT_DIR` / `OUT_ROOT` in the training scripts if your local directory structure is different.

---

## 4. 🎨 Visualization

Aligned UMAPs of `adata.obsm["latent_shared"]` can be reproduced with:

| Dataset / Task | Visualization |
| --- | --- |
| PBMC RNA–ATAC | [`notebooks/pbmc_atac_rna_umap.ipynb`](notebooks/pbmc_atac_rna_umap.ipynb) |
| BMMC RNA–ATAC | [`notebooks/visualization/bmmc_rna_atac_umap.ipynb`](notebooks/visualization/bmmc_rna_atac_umap.ipynb) |
| BMMC RNA–Protein | [`notebooks/visualization/bmmc_rna_protein_umap.ipynb`](notebooks/visualization/bmmc_rna_protein_umap.ipynb) |

The UMAPs are colored by **modality** and **cell type**.

---

## 5. 📈 Reproduction Results

We report the first two tasks used in the scMRDR comparison:

- **T1 — Cross-modal Alignment:** FOSCTTM (**FOS**) ↓ and paired embedding distance (**PED**) ↓
- **T2 — Cross-omics Prediction:** Pearson correlation (**Pear**) ↑ and **RMSE** ↓

FOS and Pearson are reported as percentages. The tables use the paper's **paired-data ratio** convention:

```text
Pairing Ratio = 100% - Single-modality Ratio
```

Thus, `single_100` corresponds to **0% paired data**, while `single_000` corresponds to **100% paired data**.

### 5.1 PBMC — RNA–ATAC

**Dataset:** PBMC  
**Modalities:** RNA and ATAC  
**Prediction direction:** **ATAC → RNA**

| Pairing Ratio | FOS ↓ (%) | PED ↓ | Pear ↑ (%) | RMSE ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 42.46 | 72.77 | 54.19 | 0.680 |
| 20% | 44.69 | 82.97 | 52.87 | 0.763 |
| 40% | 43.45 | 78.78 | 52.73 | 0.723 |
| 60% | 42.47 | 62.28 | 52.90 | 0.620 |
| 80% | 45.03 | 69.70 | 50.90 | 0.685 |
| 100% | 44.19 | 77.30 | 53.02 | 0.739 |

### 5.2 BMMC — RNA–ATAC

**Dataset:** BMMC  
**Modalities:** RNA and ATAC  
**Prediction direction:** **ATAC → RNA**

| Pairing Ratio | FOS ↓ (%) | PED ↓ | Pear ↑ (%) | RMSE ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 44.55 | 45.43 | 44.65 | 0.615 |
| 20% | 45.15 | 52.76 | 43.99 | 0.772 |
| 40% | 45.71 | 54.03 | 43.43 | 0.731 |
| 60% | 45.25 | 48.81 | 43.97 | 0.646 |
| 80% | 45.33 | 49.33 | 43.32 | 0.625 |
| 100% | 45.53 | 53.37 | 43.16 | 0.719 |

### 5.3 BMMC — RNA–Protein

**Dataset:** BMMC  
**Modalities:** RNA and Protein  
**Prediction direction:** **RNA → Protein**

| Pairing Ratio | FOS ↓ (%) | PED ↓ | Pear ↑ (%) | RMSE ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 49.79 | 27.81 | 15.13 | 2.804 |
| 20% | 48.06 | 95.70 | 42.02 | 21.560 |
| 40% | 49.90 | 246.42 | 20.98 | 28.391 |
| 60% | 47.36 | 24.66 | 24.98 | 2.658 |
| 80% | 44.81 | 22.09 | 29.08 | 2.244 |
| 100% | 47.90 | 87.86 | 46.27 | 14.909 |

> **Note:** Metric magnitudes should primarily be compared within the same dataset/task setting because RNA–ATAC and RNA–Protein use different feature spaces and value scales.

---

## 6. 📖 Citation

If this reproduction is useful for your work, please cite the original scMRDR paper:

```bibtex
@inproceedings{sun2025scmrdr,
  title={scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration},
  author={Sun, Jianle and Liang, Chaoqi and Wei, Ran and Zheng, Peng and Bai, Lei and Ouyang, Wanli and Yan, Hongliang and Ye, Peng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

This repository is an adapted reproduction based on the official [scMRDR](https://github.com/sjl-sjtu/scMRDR) implementation.

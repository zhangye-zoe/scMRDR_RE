# scMRDR Reproduction

[![GitHub stars](https://img.shields.io/github/stars/zhangye-zoe/scMRDR_RE?style=flat&logo=github)](https://github.com/zhangye-zoe/scMRDR_RE/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/zhangye-zoe/scMRDR_RE?style=flat&logo=github)](https://github.com/zhangye-zoe/scMRDR_RE/network/members)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)

This repository contains adapted code for reproducing [**scMRDR**](https://github.com/sjl-sjtu/scMRDR) on the **PBMC** and **BMMC** multi-omics datasets, including data preparation, model training, cross-omics prediction, and latent-space visualization.


## 1. 🚀 Installation

Activate the environment and install the repository from the project root:

```bash
conda activate scmrdr
pip install -e .
```

For the original scMRDR environment and API examples, please refer to the [official repository](https://github.com/sjl-sjtu/scMRDR).


## 2. 📊 Data Preparation

The training splits are generated with the following notebooks:

| Dataset / Task | Data Preparation |
| --- | --- |
| PBMC RNA–ATAC | [`scripts/PBMC_train/data.ipynb`](scripts/PBMC_train/data.ipynb) |
| BMMC RNA–ATAC | [`scripts/BMMC_train/data_rna_atac.ipynb`](scripts/BMMC_train/data_atac_rna.ipynb) |
| BMMC RNA–Protein | [`scripts/BMMC_train/data_rna_protein.ipynb`](scripts/BMMC_train/data_rna_protein.ipynb) |

PBMC cell-type annotations used for UMAP visualization were generated with [**Azimuth**](https://github.com/zhangye-zoe/Azimuth).



## 3. 🧠 Model Training

Run the corresponding script from the repository root:

```bash
# PBMC: ATAC -> RNA
python scripts/PBMC_train/train.py
```

```bash
# BMMC: ATAC -> RNA
python scripts/BMMC_train/train_atac_rna.py
```

```bash
# BMMC: RNA -> Protein
python scripts/BMMC_train/train_rna_protein.py
```

Each script automatically evaluates the configured partial-pairing ratios and saves the learned latent representations, cross-omics predictions, and evaluation metrics.

> **Note:** Update `INPUT_DIR`, `SPLIT_ROOT`, and `OUTPUT_DIR` / `OUT_ROOT` in the training scripts if your local directory structure is different.


## 4. 🎨 Visualization

Aligned UMAPs of the learned shared latent space can be reproduced with:

```text
notebooks/aligned_umap.ipynb
```

The notebook visualizes `adata.obsm["latent_shared"]` by **modality** and **cell type**, and supports the external PBMC Azimuth annotations.


## 5. 📈 PBMC Reproduction Results

We report the first two tasks used in the scMRDR comparison:

- **T1 — Cross-modal Alignment:** FOSCTTM (**FOS**) ↓ and paired embedding distance (**PED**) ↓
- **T2 — Cross-omics Prediction:** Pearson correlation (**Pear**) ↑ and **RMSE** ↓

FOS and Pearson are reported as percentages. The table follows the paper's **paired-data ratio** convention:

```text
Pairing Ratio = 100% - Single-modality Ratio
```

Therefore, for example, `single_100` corresponds to **0% paired data**, while `single_000` corresponds to **100% paired data**.

| Pairing Ratio | FOS ↓ (%) | PED ↓ | Pear ↑ (%) | RMSE ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 42.46 | 72.77 | 54.19 | 0.680 |
| 20% | 44.69 | 82.97 | 52.87 | 0.763 |
| 40% | 43.45 | 78.78 | 52.73 | 0.723 |
| 60% | 42.47 | 62.28 | 52.90 | 0.620 |
| 80% | 45.03 | 69.70 | 50.90 | 0.685 |
| 100% | 44.19 | 77.30 | 53.02 | 0.739 |


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

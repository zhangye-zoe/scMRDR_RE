# scMRDR Reproduction on PBMC and BMMC Datasets

This repository is used to reproduce **scMRDR** on our own **PBMC** and **BMMC** single-cell multi-omics datasets, and to evaluate its performance under our experimental pipeline.

The original scMRDR repository is:
**[sjl-sjtu/scMRDR](https://github.com/sjl-sjtu/scMRDR)**

---

## 1. Project goal

This project aims to:

1. Reproduce the **scMRDR** framework in a stable and runnable environment.
2. Adapt the original implementation to our own datasets and preprocessing pipeline.
3. Train and evaluate scMRDR on **PBMC** and **BMMC** data.
4. Benchmark model performance on downstream tasks such as:
   - **T1: Cross-modal Alignment**
   - **T2: Cross-omics Prediction**

This repository mainly focuses on reproduction, data adaptation, training, evaluation, and result organization for our experiments.

---

## 2. Datasets

The experiments are based on two datasets:

- **PBMC**: Peripheral Blood Mononuclear Cells
- **BMMC**: Bone Marrow Mononuclear Cells

These datasets are processed under our own preprocessing pipeline before being used for scMRDR training and evaluation.

### Input modalities

Depending on the experiment setting, the input may include paired or partially paired multi-omics data, such as:

- RNA
- ATAC / ATAC gene activity
- other aligned modality matrices used in our pipeline

### Input files

Typical processed input files may include:

- `train_rna_ref.h5ad`
- `train_atac_activity.h5ad`
- `val_true_rna.h5ad`
- `val_atac_activity.h5ad`
- `split_info.json`

The exact filenames may vary depending on the dataset version and preprocessing script.

---

## 3. Experimental setup

We evaluate scMRDR under our own benchmark protocol.

### Main tasks

#### T1: Cross-modal Alignment

This task measures whether the model can align cells from different modalities into a shared representation space.

Typical metrics may include:

- FOSCTTM
- matching accuracy
- Top-k retrieval accuracy
- paired embedding distance (PED)

#### T2: Cross-omics Prediction

This task measures whether the model can predict one modality from another.

Typical metrics may include:

- cell-wise Pearson correlation
- gene-wise Pearson correlation
- MSE / RMSE

### Dataset split

The training and validation sets are generated from our preprocessing pipeline. Depending on the experiment, data may include:

- fully paired cells
- partially paired cells
- modality-specific cells

Please check the corresponding preprocessing scripts for exact split details.

---

## 4. Repository purpose

This repository is mainly used for:

- reproducing the official scMRDR code
- adapting the model to our PBMC and BMMC datasets
- debugging environment and dependency issues
- running training experiments
- generating benchmark results and visualizations

It is intended as an experimental reproduction and extension project rather than an official reimplementation.

---

## 5. Environment

A clean and stable environment is recommended before running scMRDR.

### Suggested setup

- Python `3.9` or `3.10`
- PyTorch version compatible with your CUDA driver
- `scanpy`
- `anndata`
- `numpy`
- `pandas`
- other dependencies required by the original scMRDR codebase

Because package compatibility can vary across machines, it is recommended to create a separate conda environment.

### Example

```bash
conda create -n scmrdr python=3.10 -y
conda activate scmrdr
pip install -r requirements.txt
```

---

## 6. Installation

Clone this repository:

```bash
git clone https://github.com/zhangye-zoe/scMRDR_RE.git
cd scMRDR_RE
```

You may also want to refer to the original scMRDR repository:

```bash
git clone https://github.com/sjl-sjtu/scMRDR.git
```

Then install dependencies in your environment.

---

## 7. Data preparation

Before training, the datasets should be preprocessed into the format expected by scMRDR.

Typical preprocessing steps may include:

1. quality control
2. feature filtering
3. normalization
4. modality alignment
5. generation of train / validation splits
6. saving processed matrices as `.h5ad` or other required formats

For ATAC-related experiments, gene activity matrices may be used instead of raw peak matrices, depending on the pipeline.

Please make sure that:

- feature names are correctly aligned across modalities
- cell barcodes are consistent where pairing is required
- input dimensions match the model assumptions

---

## 8. Training

After data preparation, run the training script for the target dataset.

Example:

```bash
python train.py
```

Or, depending on your file structure:

```bash
python train_pbmc.py
python train_bmmc.py
```

Please check the actual script names in this repository.

The training script usually performs the following steps:

1. load processed training and validation data
2. construct model inputs for multiple modalities
3. train scMRDR
4. generate latent representations
5. perform cross-modal prediction
6. evaluate T1 and T2 metrics
7. save outputs and summary files

---

## 9. Output files

Typical output files may include:

- `T1_metrics.csv`
- `T2_metrics.csv`
- `summary_results.csv`
- `pred_rna_val.h5ad`
- `true_rna_val.h5ad`
- latent embedding files
- visualization files

The exact outputs depend on the current script version.

---

## 10. Evaluation

Model performance is evaluated under our benchmark protocol.

### T1: Cross-modal Alignment

Typical evaluation includes:

- FOSCTTM
- Top1 / Top5 / Top10 retrieval accuracy
- paired distance in latent or PCA space

### T2: Cross-omics Prediction

Typical evaluation includes comparison between predicted and true expression profiles:

- mean cell-wise Pearson correlation
- mean gene-wise Pearson correlation
- MSE / RMSE

Please make sure the metric definitions are consistent when comparing scMRDR with other baselines.

---

## 11. Comparison with other methods

In our experiments, scMRDR may be compared with other integration or prediction baselines such as:

- GLUE
- Seurat
- Harmony
- other methods in our benchmarking pipeline

For fair comparison, all methods should use:

- the same train / validation split
- the same input features
- the same metric definitions
- the same evaluation protocol

---

## 12. Common issues

### 12.1 Dependency conflicts

Package version mismatches may cause import or runtime errors, especially for:

- PyTorch
- NumPy
- scanpy
- anndata

Using a clean conda environment is strongly recommended.

### 12.2 Feature mismatch

If the model fails during training, check whether feature names are aligned correctly across modalities.

This is especially important when using:

- RNA and ATAC gene activity
- RNA and protein
- different processed versions of the same dataset

### 12.3 Shape mismatch

Please verify that:

- cell numbers are correct
- feature dimensions match expectations
- train / validation files correspond to the same preprocessing version

### 12.4 GPU / CUDA issues

If GPU initialization fails, please check:

- CUDA version
- PyTorch version
- NVIDIA driver compatibility

---

## 13. Notes

- This repository is based on reproduction of the original scMRDR method.
- The code has been adapted for our own PBMC and BMMC datasets.
- Some scripts may differ from the official version in order to support our preprocessing pipeline, feature alignment strategy, or evaluation protocol.

Therefore, when interpreting results, please distinguish between:

- the official scMRDR implementation
- our reproduction / adapted experimental version

---

## 14. Acknowledgement

This project is based on the official scMRDR repository:

**[sjl-sjtu/scMRDR](https://github.com/sjl-sjtu/scMRDR)**

We thank the original authors for making their code publicly available.

---

## 15. Contact

This repository is maintained for reproduction and experimental research purposes.

If you use this repository, please also cite and acknowledge the original scMRDR work.

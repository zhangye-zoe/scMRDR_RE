# scMRDR Reproduction

This repository contains the adapted code used to reproduce [scMRDR](https://github.com/sjl-sjtu/scMRDR) on the PBMC and BMMC datasets.

## Run

Activate the environment first:

```bash
conda activate scmrdr
```

Run from the repository root.

### PBMC: ATAC → RNA

```bash
python scripts/PBMC_train/train.py
```

### BMMC: ATAC → RNA

```bash
python scripts/BMMC_train/train_atac_rna.py
```

### BMMC: RNA → Protein

```bash
python scripts/BMMC_train/train_rna_protein.py
```

Each script automatically runs the configured partial-pairing ratios and saves the corresponding training, prediction, and evaluation results.

> **Note:** Input and output paths are currently defined inside each training script. Update `INPUT_DIR`, `SPLIT_ROOT`, and `OUTPUT_DIR`/`OUT_ROOT` before running if your local directory structure is different.

## Acknowledgement

This reproduction is based on the official [scMRDR](https://github.com/sjl-sjtu/scMRDR) implementation.

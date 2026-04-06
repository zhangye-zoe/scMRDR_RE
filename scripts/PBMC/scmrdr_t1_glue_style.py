#!/usr/bin/env python3
"""
Looped scMRDR training/evaluation across ratios.

This version modifies T1 to match the GLUE-style cross-modal alignment
interpretation:
- T1 is computed from validation RNA latent embeddings vs validation ATAC latent
  embeddings (same cells, two modalities), not from true-RNA vs predicted-RNA.
- T2 remains cross-omics prediction: predicted RNA vs true RNA.

Important practical note
------------------------
Because the original scMRDR script only included validation ATAC query cells in
`adata_model`, it did not have latent embeddings for validation RNA cells.
To support GLUE-style T1, this script also concatenates validation true RNA into
`adata_model` so that scMRDR can infer embeddings for both modalities on the
same validation cells.

The script still supports auto-detection of:
- predicted RNA source (layers/obsm)
- latent embedding source (obsm)
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from scMRDR.module import Integration


# =========================================================
# 0. Parameters
# =========================================================
BASE_DIR = "/data5/zhangye/scMRDR/scMRDR_zy/PBMC/output_pbmc_multiome"
SPLIT_ROOT = os.path.join(BASE_DIR, "results_ratio_loop")
ATAC_GAS_PATH = os.path.join(BASE_DIR, "ATAC_gas.h5ad")
OUT_ROOT = os.path.join(BASE_DIR, "results_scMRDR_loop_glue_t1")

RATIO_LABELS = [f"single_{x:03d}" for x in [0, 20, 40, 60, 80, 100]]
SEED = 1234

# scMRDR model/training settings
HIDDEN_LAYERS = [512, 512]
LATENT_DIM_SHARED = 20
LATENT_DIM_SPECIFIC = 20
BETA = 2
GAMMA = 5
LAMBDA_ADV = 5
DROPOUT_RATE = 0.2

EPOCH_NUM = 200
BATCH_SIZE = 128
LR = 1e-3
ADAPTLR = False
NUM_WARMUP = 0
EARLY_STOPPING = True
VALID_PROP = 0.1
WEIGHTED = False
PATIENCE = 10

# Evaluation / PCA settings
N_PCS_EVAL = 30
MIN_COMMON_FEATURES = 50

# Predicted RNA source in adata_post; None -> auto detect
PRED_RNA_SOURCE = None

# Latent embedding source in adata_post; None -> auto detect from obsm
LATENT_SOURCE = None

# Include validation ATAC and validation RNA in adata passed to scMRDR so we can
# obtain latent embeddings for both modalities during inference.
INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA = True
INCLUDE_VAL_TRUE_RNA_IN_MODEL_ADATA = True


# =========================================================
# 1. Helper functions
# =========================================================
def ensure_dir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_cor(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1 or y.size <= 1:
        return np.nan
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def hit_at_k(cross_dist: np.ndarray, k: int = 5) -> float:
    nq = cross_dist.shape[0]
    hits = []
    for i in range(nq):
        ord_idx = np.argsort(cross_dist[i, :])[: min(k, cross_dist.shape[1])]
        hits.append(i in ord_idx)
    return float(np.mean(hits)) if len(hits) > 0 else np.nan


def sanitize_adata_for_write(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    if isinstance(adata.X, np.matrix):
        adata.X = np.asarray(adata.X)
    for k in list(adata.layers.keys()):
        if isinstance(adata.layers[k], np.matrix):
            adata.layers[k] = np.asarray(adata.layers[k])
    return adata


def save_h5ad_safe(adata: ad.AnnData, path: os.PathLike) -> None:
    adata = sanitize_adata_for_write(adata)
    adata.write(str(path))


def subset_adata_by_cells(adata: ad.AnnData, cells: Sequence[str]) -> ad.AnnData:
    cells = [c for c in cells if c in adata.obs_names]
    return adata[cells].copy()


def get_common_names(*arrays: Sequence[str]) -> List[str]:
    if len(arrays) == 0:
        return []
    common = set(arrays[0])
    for arr in arrays[1:]:
        common &= set(arr)
    return sorted(common)


def inspect_prediction_keys(adata: ad.AnnData) -> None:
    print("adata.layers keys:", list(adata.layers.keys()))
    print("adata.obsm keys:", list(adata.obsm.keys()))
    print("adata.uns keys:", list(adata.uns.keys()))


def extract_matrix_from_source(adata: ad.AnnData, source: str) -> np.ndarray:
    """
    source examples:
      layer:pred_rna
      obsm:X_rna_pred
      obsm:latent_shared
    """
    if ":" not in source:
        raise ValueError(f"Invalid source format: {source}")
    kind, key = source.split(":", 1)

    if kind == "layer":
        if key not in adata.layers:
            raise KeyError(f"{key} not in adata.layers")
        return to_dense(adata.layers[key])

    elif kind == "obsm":
        if key not in adata.obsm:
            raise KeyError(f"{key} not in adata.obsm")
        return np.asarray(adata.obsm[key])

    else:
        raise ValueError(f"Unsupported source kind: {kind}")


def auto_extract_predicted_rna_matrix(adata: ad.AnnData) -> Tuple[np.ndarray, str]:
    candidates = [
        ("layer", "pred_rna"),
        ("layer", "rna_pred"),
        ("layer", "imputed_rna"),
        ("layer", "reconstructed_rna"),
        ("layer", "recon_rna"),
        ("layer", "impute"),
        ("layer", "recon"),
        ("obsm", "X_rna_pred"),
        ("obsm", "rna_pred"),
        ("obsm", "pred_rna"),
        ("obsm", "imputed_rna"),
    ]

    for kind, key in candidates:
        try:
            mat = extract_matrix_from_source(adata, f"{kind}:{key}")
        except Exception:
            continue
        if mat.shape[0] == adata.n_obs:
            return np.asarray(mat), f"{kind}:{key}"

    inspect_prediction_keys(adata)
    raise RuntimeError(
        "Could not auto-detect predicted RNA in scMRDR output. "
        "Please inspect printed keys and set PRED_RNA_SOURCE manually."
    )


def auto_extract_latent_matrix(adata: ad.AnnData) -> Tuple[np.ndarray, str]:
    candidates = [
        ("obsm", "latent_shared"),
        ("obsm", "X_latent_shared"),
        ("obsm", "X_shared"),
        ("obsm", "z_shared"),
        ("obsm", "shared_latent"),
        ("obsm", "X_emb"),
        ("obsm", "embedding"),
    ]

    for kind, key in candidates:
        try:
            mat = extract_matrix_from_source(adata, f"{kind}:{key}")
        except Exception:
            continue
        if mat.shape[0] == adata.n_obs:
            return np.asarray(mat), f"{kind}:{key}"

    inspect_prediction_keys(adata)
    raise RuntimeError(
        "Could not auto-detect latent embedding in scMRDR output. "
        "Please inspect printed keys and set LATENT_SOURCE manually, e.g. 'obsm:latent_shared'."
    )


def get_predicted_rna_adata(
    adata_post: ad.AnnData,
    query_cells: Sequence[str],
    true_rna_val: ad.AnnData,
    pred_source: Optional[str] = None,
) -> Tuple[ad.AnnData, str]:
    if pred_source is None:
        pred_mat_all, used_source = auto_extract_predicted_rna_matrix(adata_post)
    else:
        pred_mat_all = extract_matrix_from_source(adata_post, pred_source)
        used_source = pred_source

    if pred_mat_all.shape[0] != adata_post.n_obs:
        raise ValueError(
            f"Predicted matrix row count {pred_mat_all.shape[0]} != adata_post.n_obs {adata_post.n_obs}"
        )

    query_cells = [c for c in query_cells if c in adata_post.obs_names]
    if len(query_cells) == 0:
        raise ValueError("No validation query cells found in scMRDR output adata.")

    row_idx = [adata_post.obs_names.get_loc(c) for c in query_cells]
    pred_query = pred_mat_all[row_idx, :]

    pred_var_names = list(adata_post.var_names)
    if pred_query.shape[1] != len(pred_var_names):
        raise ValueError(
            f"Predicted matrix columns ({pred_query.shape[1]}) do not match adata_post.var_names ({len(pred_var_names)})."
        )

    common_features = get_common_names(pred_var_names, true_rna_val.var_names.tolist())
    if len(common_features) == 0:
        raise ValueError("No common features between predicted RNA and true RNA.")

    pred_df = pd.DataFrame(pred_query, index=query_cells, columns=pred_var_names)
    pred_df = pred_df.loc[:, common_features]

    pred_adata = ad.AnnData(
        X=np.asarray(pred_df.values, dtype=float),
        obs=pd.DataFrame(index=pred_df.index),
        var=pd.DataFrame(index=pred_df.columns),
    )
    pred_adata.obs["modality"] = "pred_from_atac"
    pred_adata.layers["data"] = pred_adata.X.copy()
    return pred_adata, used_source


def get_latent_embeddings_for_cells(
    adata_post: ad.AnnData,
    cells: Sequence[str],
    modality: str,
    latent_source: Optional[str] = None,
) -> Tuple[np.ndarray, str, List[str]]:
    if latent_source is None:
        latent_all, used_source = auto_extract_latent_matrix(adata_post)
    else:
        latent_all = extract_matrix_from_source(adata_post, latent_source)
        used_source = latent_source

    if latent_all.shape[0] != adata_post.n_obs:
        raise ValueError(
            f"Latent matrix row count {latent_all.shape[0]} != adata_post.n_obs {adata_post.n_obs}"
        )

    kept_cells = []
    row_idx = []
    obs_names = adata_post.obs_names.astype(str).tolist()
    modalities = adata_post.obs["modality"].astype(str).tolist() if "modality" in adata_post.obs.columns else [""] * adata_post.n_obs

    for cell in cells:
        matches = [i for i, (name, mod) in enumerate(zip(obs_names, modalities)) if name == str(cell) and mod == modality]
        if len(matches) > 0:
            row_idx.append(matches[0])
            kept_cells.append(str(cell))

    if len(kept_cells) == 0:
        raise ValueError(f"Requested cells for latent extraction with modality={modality} were not found in adata_post.")

    latent = np.asarray(latent_all[row_idx, :], dtype=float)
    return latent, used_source, kept_cells


# =========================================================
# 2. Build scMRDR input for one ratio
# =========================================================
def build_model_input_for_ratio(split_dir: os.PathLike, atac_gas_global: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData, dict]:
    split_dir = Path(split_dir)

    train_rna_ref = sc.read_h5ad(str(split_dir / "train_rna_ref.h5ad"))
    val_true_rna = sc.read_h5ad(str(split_dir / "val_true_rna.h5ad"))
    val_atac_activity = sc.read_h5ad(str(split_dir / "val_atac_activity.h5ad"))

    with open(split_dir / "split_info.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_cells = split_info["train_cells"]
    val_query_atac_cells = split_info["val_query_atac_cells"]

    train_atac_activity = subset_adata_by_cells(atac_gas_global, train_cells)

    common_features = get_common_names(
        train_rna_ref.var_names.tolist(),
        train_atac_activity.var_names.tolist(),
        val_atac_activity.var_names.tolist(),
        val_true_rna.var_names.tolist(),
    )
    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(f"Too few common features in {split_dir.name}: {len(common_features)}")

    train_rna_ref = train_rna_ref[:, common_features].copy()
    train_atac_activity = train_atac_activity[:, common_features].copy()
    val_atac_activity = val_atac_activity[:, common_features].copy()
    val_true_rna = val_true_rna[:, common_features].copy()

    train_rna_ref.obs["modality"] = "rna"
    train_atac_activity.obs["modality"] = "atac"
    val_atac_activity.obs["modality"] = "atac"
    val_true_rna.obs["modality"] = "rna"

    if "batch" not in train_rna_ref.obs.columns:
        train_rna_ref.obs["batch"] = "batch0"
    if "batch" not in train_atac_activity.obs.columns:
        train_atac_activity.obs["batch"] = "batch0"
    if "batch" not in val_atac_activity.obs.columns:
        val_atac_activity.obs["batch"] = "batch0"
    if "batch" not in val_true_rna.obs.columns:
        val_true_rna.obs["batch"] = "batch0"

    train_rna_ref.layers["count"] = train_rna_ref.layers["counts"] if "counts" in train_rna_ref.layers else train_rna_ref.X.copy()
    train_atac_activity.layers["count"] = train_atac_activity.layers["counts"] if "counts" in train_atac_activity.layers else train_atac_activity.X.copy()
    val_atac_activity.layers["count"] = val_atac_activity.layers["counts"] if "counts" in val_atac_activity.layers else val_atac_activity.X.copy()
    val_true_rna.layers["count"] = val_true_rna.layers["counts"] if "counts" in val_true_rna.layers else val_true_rna.X.copy()

    concat_dict = {
        "train_rna": train_rna_ref.copy(),
        "train_atac": train_atac_activity.copy(),
    }
    if INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA:
        concat_dict["val_atac"] = val_atac_activity.copy()
    if INCLUDE_VAL_TRUE_RNA_IN_MODEL_ADATA:
        concat_dict["val_rna"] = val_true_rna.copy()

    adata_model = ad.concat(
        concat_dict,
        axis=0,
        join="inner",
        label="dataset_block",
        index_unique=None,
    )

    if "count" not in adata_model.layers:
        raise ValueError("Concatenated scMRDR input is missing layer 'count'.")

    adata_model.obs["is_val_query_atac"] = adata_model.obs_names.isin(val_query_atac_cells)
    adata_model.obs["is_val_rna"] = adata_model.obs_names.isin(val_true_rna.obs_names)
    return adata_model, val_true_rna, split_info


# =========================================================
# 3. Training one ratio
# =========================================================
def train_scmrdr_for_ratio(adata_model: ad.AnnData) -> Tuple[Integration, ad.AnnData]:
    model = Integration(
        data=adata_model,
        modality_key="modality",
        layer="count",
        batch_key="batch",
        feature_list=None,
        distribution="ZINB",
    )

    model.setup(
        hidden_layers=HIDDEN_LAYERS,
        latent_dim_shared=LATENT_DIM_SHARED,
        latent_dim_specific=LATENT_DIM_SPECIFIC,
        beta=BETA,
        gamma=GAMMA,
        lambda_adv=LAMBDA_ADV,
        dropout_rate=DROPOUT_RATE,
    )

    model.train(
        epoch_num=EPOCH_NUM,
        batch_size=BATCH_SIZE,
        lr=LR,
        adaptlr=ADAPTLR,
        num_warmup=NUM_WARMUP,
        early_stopping=EARLY_STOPPING,
        valid_prop=VALID_PROP,
        weighted=WEIGHTED,
        patience=PATIENCE,
    )

    model.inference(n_samples=1, update=True, returns=False)
    adata_post = model.get_adata()
    return model, adata_post


# =========================================================
# 4. Evaluation
# =========================================================
def evaluate_t2(pred_rna_val: ad.AnnData, true_rna_val: ad.AnnData, outdir: os.PathLike) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    common_features = get_common_names(pred_rna_val.var_names.tolist(), true_rna_val.var_names.tolist())
    common_cells = get_common_names(pred_rna_val.obs_names.tolist(), true_rna_val.obs_names.tolist())

    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(f"Too few common features for T2 evaluation: {len(common_features)}")
    if len(common_cells) == 0:
        raise ValueError("No common cells for T2 evaluation.")

    pred = pred_rna_val[common_cells, common_features].copy()
    true = true_rna_val[common_cells, common_features].copy()

    pred_mat = to_dense(pred.X).T
    true_mat = to_dense(true.X).T

    cell_cor = np.array([safe_cor(pred_mat[:, i], true_mat[:, i]) for i in range(pred_mat.shape[1])], dtype=float)
    gene_cor = np.array([safe_cor(pred_mat[i, :], true_mat[i, :]) for i in range(pred_mat.shape[0])], dtype=float)

    mse = float(np.nanmean((pred_mat - true_mat) ** 2))
    rmse = float(np.sqrt(mse))

    t2_metrics = pd.DataFrame(
        {
            "metric": ["pearson_cell_mean", "pearson_gene_mean", "rmse"],
            "value": [float(np.nanmean(cell_cor)), float(np.nanmean(gene_cor)), rmse],
        }
    )
    t2_metrics.to_csv(Path(outdir) / "T2_metrics.csv", index=False)

    pd.DataFrame({"cell": common_cells, "cellwise_pearson": cell_cor}).to_csv(
        Path(outdir) / "T2_cellwise_pearson.csv", index=False
    )
    pd.DataFrame({"gene": common_features, "genewise_pearson": gene_cor}).to_csv(
        Path(outdir) / "T2_genewise_pearson.csv", index=False
    )

    return t2_metrics, pred_mat, true_mat, common_cells, common_features


def evaluate_t1_alignment_from_latent(
    rna_latent: np.ndarray,
    atac_latent: np.ndarray,
    common_cells: Sequence[str],
    outdir: os.PathLike,
    metric: str = "euclidean",
) -> pd.DataFrame:
    if rna_latent.shape[0] == 0 or atac_latent.shape[0] == 0:
        raise ValueError("Empty latent matrix for T1 evaluation.")
    if rna_latent.shape[0] != atac_latent.shape[0]:
        raise ValueError("RNA and ATAC latent matrices must have same number of rows for paired T1.")

    cross_dist = pairwise_distances(rna_latent, atac_latent, metric=metric)
    paired_dist = np.diag(cross_dist).astype(float)
    n_q = cross_dist.shape[0]
    ped = float(np.nanmean(paired_dist))

    foscttm_each = np.array([np.mean(cross_dist[i, :] < paired_dist[i]) for i in range(n_q)], dtype=float)
    top1_acc = hit_at_k(cross_dist, k=1)
    top5_acc = hit_at_k(cross_dist, k=5)
    top10_acc = hit_at_k(cross_dist, k=10)

    nn_idx = np.argmin(cross_dist, axis=1)
    nn_match = (nn_idx == np.arange(n_q))

    t1_metrics = pd.DataFrame(
        {
            "metric": ["FOSCTTM", "PED", "Top1_ACC", "Top5_ACC", "Top10_ACC"],
            "value": [float(np.nanmean(foscttm_each)), ped, top1_acc, top5_acc, top10_acc],
        }
    )
    t1_metrics.to_csv(Path(outdir) / "T1_metrics.csv", index=False)

    pd.DataFrame(
        {
            "cell": list(common_cells),
            "FOSCTTM": foscttm_each,
            "paired_dist": paired_dist,
            "Top1_match": nn_match.astype(bool),
        }
    ).to_csv(Path(outdir) / "T1_per_cell_metrics.csv", index=False)

    return t1_metrics


def save_true_pred_pca(pred_mat: np.ndarray, true_mat: np.ndarray, common_cells: Sequence[str], outdir: os.PathLike) -> None:
    true_cells_by_features = true_mat.T
    pred_cells_by_features = pred_mat.T
    mix = np.vstack([true_cells_by_features, pred_cells_by_features])

    n_components = min(N_PCS_EVAL, mix.shape[0] - 1, mix.shape[1])
    if n_components < 2:
        return

    pca = PCA(n_components=n_components, random_state=SEED)
    emb = pca.fit_transform(mix)
    n_q = true_cells_by_features.shape[0]

    pca_df = pd.DataFrame(emb[:, : min(5, emb.shape[1])], columns=[f"PC{i+1}" for i in range(min(5, emb.shape[1]))])
    pca_df["group"] = ["True_RNA"] * n_q + ["Pred_from_ATAC"] * n_q
    pca_df["cell"] = [f"true_{c}" for c in common_cells] + [f"pred_{c}" for c in common_cells]
    pca_df.to_csv(Path(outdir) / "pca_true_pred_coords.csv", index=False)


# =========================================================
# 5. Main loop
# =========================================================
def main():
    ensure_dir(OUT_ROOT)
    atac_gas_global = sc.read_h5ad(ATAC_GAS_PATH)

    all_summary = []

    for ratio_label in RATIO_LABELS:
        split_dir = Path(SPLIT_ROOT) / ratio_label
        outdir = Path(OUT_ROOT) / ratio_label
        ensure_dir(outdir)

        if not split_dir.exists():
            warnings.warn(f"Missing split directory: {split_dir}. Skipping.")
            continue

        print("\n============================")
        print("Running ratio:", ratio_label)
        print("Split dir:", split_dir)
        print("Out dir:", outdir)
        print("============================")

        try:
            adata_model, true_rna_val, split_info = build_model_input_for_ratio(split_dir, atac_gas_global)

            save_h5ad_safe(adata_model, outdir / "training_adata_input.h5ad")

            model, adata_post = train_scmrdr_for_ratio(adata_model)
            save_h5ad_safe(adata_post, outdir / "training_adata_post.h5ad")

            pred_rna_val, used_pred_source = get_predicted_rna_adata(
                adata_post=adata_post,
                query_cells=split_info["val_query_atac_cells"],
                true_rna_val=true_rna_val,
                pred_source=PRED_RNA_SOURCE,
            )
            print("Predicted RNA source used:", used_pred_source)

            # T2 outputs
            save_h5ad_safe(pred_rna_val, outdir / "pred_rna_val.h5ad")
            save_h5ad_safe(true_rna_val, outdir / "true_rna_val.h5ad")

            t2_metrics, pred_mat, true_mat, common_cells_t2, common_features = evaluate_t2(
                pred_rna_val=pred_rna_val,
                true_rna_val=true_rna_val,
                outdir=outdir,
            )
            save_true_pred_pca(pred_mat, true_mat, common_cells_t2, outdir)

            # T1: GLUE-style latent alignment on validation RNA vs validation ATAC
            val_paired_cells = split_info.get("val_paired_cells", [])
            rna_latent, used_latent_source, rna_kept = get_latent_embeddings_for_cells(
                adata_post=adata_post,
                cells=val_paired_cells,
                modality="rna",
                latent_source=LATENT_SOURCE,
            )
            atac_latent, _, atac_kept = get_latent_embeddings_for_cells(
                adata_post=adata_post,
                cells=val_paired_cells,
                modality="atac",
                latent_source=used_latent_source,
            )

            # Because row selection is by cell names and the same list is used above,
            # just make the cell set explicit again for safety.
            common_cells_t1 = [c for c in val_paired_cells if c in rna_kept and c in atac_kept]
            if len(common_cells_t1) == 0:
                raise ValueError("No common validation paired cells available for T1 latent alignment.")

            # Re-extract in the exact common order to avoid any row mismatch.
            rna_latent, _, _ = get_latent_embeddings_for_cells(adata_post, common_cells_t1, "rna", used_latent_source)
            atac_latent, _, _ = get_latent_embeddings_for_cells(adata_post, common_cells_t1, "atac", used_latent_source)

            t1_metrics = evaluate_t1_alignment_from_latent(
                rna_latent=rna_latent,
                atac_latent=atac_latent,
                common_cells=common_cells_t1,
                outdir=outdir,
                metric="euclidean",
            )

            with open(outdir / "prediction_source.txt", "w", encoding="utf-8") as f:
                f.write(used_pred_source + "\n")
            with open(outdir / "latent_source.txt", "w", encoding="utf-8") as f:
                f.write(used_latent_source + "\n")
            with open(outdir / "split_info_used.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)

            t2_map = dict(zip(t2_metrics["metric"], t2_metrics["value"]))
            t1_map = dict(zip(t1_metrics["metric"], t1_metrics["value"]))

            all_summary.append(
                {
                    "ratio_label": ratio_label,
                    "single_frac": split_info["single_frac"],
                    "train_paired": len(split_info["train_paired_cells"]),
                    "train_rna_only": len(split_info["train_rna_only_cells"]),
                    "train_atac_only": len(split_info["train_atac_only_cells"]),
                    "val_paired": len(split_info["val_paired_cells"]),
                    "val_rna_only": len(split_info["val_rna_only_cells"]),
                    "val_atac_only": len(split_info["val_atac_only_cells"]),
                    "query_cells_t2": len(common_cells_t2),
                    "paired_cells_t1": len(common_cells_t1),
                    "common_features": len(common_features),
                    "cellwise_pearson_mean": t2_map["pearson_cell_mean"],
                    "genewise_pearson_mean": t2_map["pearson_gene_mean"],
                    "rmse": t2_map["rmse"],
                    "foscttm": t1_map["FOSCTTM"],
                    "ped": t1_map["PED"],
                    "top1_acc": t1_map["Top1_ACC"],
                    "top5_acc": t1_map["Top5_ACC"],
                    "top10_acc": t1_map["Top10_ACC"],
                    "pred_source": used_pred_source,
                    "latent_source": used_latent_source,
                }
            )

            print("Finished:", ratio_label)
            print(pd.DataFrame(all_summary).tail(1))

        except Exception as e:
            warnings.warn(f"Failed on {ratio_label}: {repr(e)}")
            with open(outdir / "error.txt", "w", encoding="utf-8") as f:
                f.write(repr(e) + "\n")
            continue

    if len(all_summary) > 0:
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(Path(OUT_ROOT) / "summary_all_ratios_scMRDR.csv", index=False)
        print("\nAll finished. Summary saved to:")
        print(Path(OUT_ROOT) / "summary_all_ratios_scMRDR.csv")
        print(summary_df)
    else:
        print("No successful ratio runs.")


if __name__ == "__main__":
    main()

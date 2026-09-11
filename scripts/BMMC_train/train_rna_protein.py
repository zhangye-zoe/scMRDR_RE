#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from scMRDR.module2_latent_patched import Integration

# =========================================================
# RNA -> Protein scMRDR training / evaluation script
# ---------------------------------------------------------
# Expected split files from the RNA->Protein preparation step:
#   results_ratio_loop_rna_to_protein/single_xxx/
#       train_rna_paired.h5ad
#       train_protein_paired.h5ad
#       train_rna_only.h5ad                (optional, can be empty / missing)
#       train_protein_only.h5ad            (optional, can be empty / missing)
#       val_rna_query.h5ad
#       val_true_protein.h5ad
#       split_info.json
#
# Important:
# - This script works on the overlap between RNA genes and protein markers
#   after protein markers have been mapped to gene symbols.
# - Therefore, prediction targets are the overlapping mapped protein features.
# =========================================================

INPUT_DIR = "/data5/zhangye/scMRDR/input/BMMC/preprocessed_input/RNA_Protein"
OUTPUT_DIR = "/data5/zhangye/scMRDR/output/BMMC"
SPLIT_ROOT = os.path.join(INPUT_DIR, "results_ratio_loop_rna_to_protein")
OUT_ROOT = os.path.join(OUTPUT_DIR, "scMRDR_results_rna_to_protein")

RATIO_LABELS = [f"single_{x:03d}" for x in [0, 20, 40, 60, 80, 100]]
SEED = 1234

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

PREDICT_MODALITY = "protein"
PREDICT_STRATEGY = "latent"
PREDICT_METHOD = "knn"
PREDICT_K = 10

N_PCS_EVAL = 30
MIN_COMMON_FEATURES = 10

# Include validation RNA query cells inside model AnnData so the model can infer
# missing protein for these RNA cells during prediction.
INCLUDE_VAL_QUERY_RNA_IN_MODEL_ADATA = True


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_cor(x, y):
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


def sanitize_adata_for_write(adata):
    adata = adata.copy()
    if isinstance(adata.X, np.matrix):
        adata.X = np.asarray(adata.X)
    for k in list(adata.layers.keys()):
        if isinstance(adata.layers[k], np.matrix):
            adata.layers[k] = np.asarray(adata.layers[k])
    return adata


def save_h5ad_safe(adata, path):
    sanitize_adata_for_write(adata).write(str(path))


def get_common_names(*arrays):
    if len(arrays) == 0:
        return []
    common = set(arrays[0])
    for arr in arrays[1:]:
        common &= set(arr)
    return sorted(common)


def maybe_read_h5ad(path):
    path = Path(path)
    if not path.exists():
        return None
    return sc.read_h5ad(str(path))


def prepare_count_layer(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    if "count" not in adata.layers:
        if "counts" in adata.layers:
            adata.layers["count"] = adata.layers["counts"].copy()
        else:
            adata.layers["count"] = adata.X.copy()
    return adata


def ensure_batch_column(adata: ad.AnnData, value: str = "batch0") -> ad.AnnData:
    adata = adata.copy()
    if "batch" not in adata.obs.columns:
        adata.obs["batch"] = value
    else:
        adata.obs["batch"] = adata.obs["batch"].astype(str)
    return adata


def build_model_input_for_ratio(split_dir):
    split_dir = Path(split_dir)

    train_rna_paired = sc.read_h5ad(str(split_dir / "train_rna_paired.h5ad"))
    train_protein_paired = sc.read_h5ad(str(split_dir / "train_protein_paired.h5ad"))
    val_rna_query = sc.read_h5ad(str(split_dir / "val_rna_query.h5ad"))
    val_true_protein = sc.read_h5ad(str(split_dir / "val_true_protein.h5ad"))

    train_rna_only = maybe_read_h5ad(split_dir / "train_rna_only.h5ad")
    train_protein_only = maybe_read_h5ad(split_dir / "train_protein_only.h5ad")

    with open(split_dir / "split_info.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)

    common_feature_inputs = [
        train_rna_paired.var_names.tolist(),
        train_protein_paired.var_names.tolist(),
        val_rna_query.var_names.tolist(),
        val_true_protein.var_names.tolist(),
    ]
    if train_rna_only is not None and train_rna_only.n_obs > 0:
        common_feature_inputs.append(train_rna_only.var_names.tolist())
    if train_protein_only is not None and train_protein_only.n_obs > 0:
        common_feature_inputs.append(train_protein_only.var_names.tolist())

    common_features = get_common_names(*common_feature_inputs)
    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(
            f"Too few common RNA/protein features in {split_dir.name}: {len(common_features)}. "
            "Check protein gene-symbol mapping and overlap."
        )

    train_rna_paired = train_rna_paired[:, common_features].copy()
    train_protein_paired = train_protein_paired[:, common_features].copy()
    val_rna_query = val_rna_query[:, common_features].copy()
    val_true_protein = val_true_protein[:, common_features].copy()

    if train_rna_only is not None and train_rna_only.n_obs > 0:
        train_rna_only = train_rna_only[:, common_features].copy()
    if train_protein_only is not None and train_protein_only.n_obs > 0:
        train_protein_only = train_protein_only[:, common_features].copy()

    train_rna_paired.obs["modality"] = "rna"
    train_protein_paired.obs["modality"] = "protein"
    val_rna_query.obs["modality"] = "rna"

    train_rna_paired = ensure_batch_column(prepare_count_layer(train_rna_paired))
    train_protein_paired = ensure_batch_column(prepare_count_layer(train_protein_paired))
    val_rna_query = ensure_batch_column(prepare_count_layer(val_rna_query))
    val_true_protein = ensure_batch_column(val_true_protein)

    blocks = {
        "train_rna_paired": train_rna_paired.copy(),
        "train_protein_paired": train_protein_paired.copy(),
    }

    if train_rna_only is not None and train_rna_only.n_obs > 0:
        train_rna_only.obs["modality"] = "rna"
        train_rna_only = ensure_batch_column(prepare_count_layer(train_rna_only))
        blocks["train_rna_only"] = train_rna_only.copy()

    if train_protein_only is not None and train_protein_only.n_obs > 0:
        train_protein_only.obs["modality"] = "protein"
        train_protein_only = ensure_batch_column(prepare_count_layer(train_protein_only))
        blocks["train_protein_only"] = train_protein_only.copy()

    if INCLUDE_VAL_QUERY_RNA_IN_MODEL_ADATA:
        blocks["val_rna_query"] = val_rna_query.copy()

    adata_model = ad.concat(
        blocks,
        axis=0,
        join="inner",
        label="dataset_block",
        index_unique=None,
    )

    if "count" not in adata_model.layers:
        raise ValueError("Concatenated scMRDR input is missing layer 'count'.")

    adata_model.obs["is_val_query"] = adata_model.obs_names.isin(val_rna_query.obs_names)

    return adata_model, val_true_protein, split_info, common_features


def train_scmrdr_for_ratio(adata_model):
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

    infer_ret = model.inference(
        n_samples=1,
        update=True,
        returns=True,
        predict_modalities=[PREDICT_MODALITY],
        predict_strategy=PREDICT_STRATEGY,
        predict_method=PREDICT_METHOD,
        predict_k=PREDICT_K,
    )
    adata_post = model.get_adata()
    pred_adata_all = model.get_prediction_adata(PREDICT_MODALITY, infer_ret["predictions"][PREDICT_MODALITY])
    return model, adata_post, pred_adata_all


def evaluate_t2(pred_protein_val, true_protein_val, outdir):
    common_features = get_common_names(pred_protein_val.var_names.tolist(), true_protein_val.var_names.tolist())
    common_cells = get_common_names(pred_protein_val.obs_names.tolist(), true_protein_val.obs_names.tolist())

    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(f"Too few common protein features for evaluation: {len(common_features)}")
    if len(common_cells) == 0:
        raise ValueError("No common cells for protein evaluation.")

    pred = pred_protein_val[common_cells, common_features].copy()
    true = true_protein_val[common_cells, common_features].copy()

    pred_mat = to_dense(pred.X).T
    true_mat = to_dense(true.X).T

    cell_cor = np.array([safe_cor(pred_mat[:, i], true_mat[:, i]) for i in range(pred_mat.shape[1])], dtype=float)
    feature_cor = np.array([safe_cor(pred_mat[i, :], true_mat[i, :]) for i in range(pred_mat.shape[0])], dtype=float)

    mse = float(np.nanmean((pred_mat - true_mat) ** 2))
    rmse = float(np.sqrt(mse))

    t2_metrics = pd.DataFrame({
        "metric": ["pearson_cell_mean", "pearson_feature_mean", "rmse"],
        "value": [float(np.nanmean(cell_cor)), float(np.nanmean(feature_cor)), rmse],
    })
    t2_metrics.to_csv(Path(outdir) / "T2_metrics_protein.csv", index=False)

    pd.DataFrame({"cell": common_cells, "cellwise_pearson": cell_cor}).to_csv(
        Path(outdir) / "T2_cellwise_pearson_protein.csv", index=False
    )
    pd.DataFrame({"feature": common_features, "featurewise_pearson": feature_cor}).to_csv(
        Path(outdir) / "T2_featurewise_pearson_protein.csv", index=False
    )

    return t2_metrics, pred_mat, true_mat, common_cells, common_features


def evaluate_t1_from_true_pred(pred_mat, true_mat, common_cells, common_features, outdir):
    true_cells_by_features = true_mat.T
    pred_cells_by_features = pred_mat.T
    mix = np.vstack([true_cells_by_features, pred_cells_by_features])

    n_components = min(N_PCS_EVAL, mix.shape[0] - 1, mix.shape[1])
    if n_components < 2:
        raise ValueError("PCA components < 2, cannot evaluate T1-style matching.")

    pca = PCA(n_components=n_components, random_state=SEED)
    emb = pca.fit_transform(mix)

    n_q = true_cells_by_features.shape[0]
    emb_true = emb[:n_q, :]
    emb_pred = emb[n_q:, :]

    dist_mat = pairwise_distances(np.vstack([emb_true, emb_pred]), metric="euclidean")
    cross_dist = dist_mat[:n_q, n_q : (2 * n_q)]

    paired_dist = np.diag(cross_dist).astype(float)
    foscttm_each = np.array([np.mean(cross_dist[i, :] < paired_dist[i]) for i in range(n_q)], dtype=float)
    top1_acc = hit_at_k(cross_dist, k=1)
    top5_acc = hit_at_k(cross_dist, k=5)
    top10_acc = hit_at_k(cross_dist, k=10)

    nn_idx = np.argmin(cross_dist, axis=1)
    nn_match = (nn_idx == np.arange(n_q))

    t1_metrics = pd.DataFrame({
        "metric": [
            "paired_embedding_distance_mean",
            "paired_embedding_distance_median",
            "FOSCTTM",
            "Top1_ACC",
            "Top5_ACC",
            "Top10_ACC",
        ],
        "value": [
            float(np.nanmean(paired_dist)),
            float(np.nanmedian(paired_dist)),
            float(np.nanmean(foscttm_each)),
            top1_acc,
            top5_acc,
            top10_acc,
        ],
    })
    t1_metrics.to_csv(Path(outdir) / "T1_metrics_protein_space.csv", index=False)

    pd.DataFrame({
        "cell": list(common_cells),
        "FOSCTTM": foscttm_each,
        "paired_dist": paired_dist,
        "Top1_match": nn_match.astype(bool),
    }).to_csv(Path(outdir) / "T1_per_cell_metrics_protein_space.csv", index=False)

    pca_df = pd.DataFrame(
        emb[:, : min(5, emb.shape[1])],
        columns=[f"PC{i + 1}" for i in range(min(5, emb.shape[1]))]
    )
    pca_df["group"] = ["True_Protein"] * n_q + ["Pred_from_RNA"] * n_q
    pca_df["cell"] = [f"true_{c}" for c in common_cells] + [f"pred_{c}" for c in common_cells]
    pca_df.to_csv(Path(outdir) / "pca_true_pred_protein_coords.csv", index=False)

    return t1_metrics


def main():
    ensure_dir(OUT_ROOT)
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
        print("============================")

        try:
            adata_model, true_protein_val, split_info, common_features = build_model_input_for_ratio(split_dir)
            save_h5ad_safe(adata_model, outdir / "training_adata_input.h5ad")

            model, adata_post, pred_protein_all = train_scmrdr_for_ratio(adata_model)
            save_h5ad_safe(adata_post, outdir / "training_adata_post.h5ad")
            save_h5ad_safe(pred_protein_all, outdir / "pred_protein_all_nonprotein.h5ad")

            val_cells = split_info.get("val_query_cells", split_info.get("val_rna_query_cells", []))
            pred_protein_val = pred_protein_all[[c for c in val_cells if c in pred_protein_all.obs_names]].copy()

            save_h5ad_safe(pred_protein_val, outdir / "pred_protein_val.h5ad")
            save_h5ad_safe(true_protein_val, outdir / "true_protein_val.h5ad")

            t2_metrics, pred_mat, true_mat, common_cells, common_features_eval = evaluate_t2(
                pred_protein_val=pred_protein_val,
                true_protein_val=true_protein_val,
                outdir=outdir,
            )

            t1_metrics = evaluate_t1_from_true_pred(
                pred_mat=pred_mat,
                true_mat=true_mat,
                common_cells=common_cells,
                common_features=common_features_eval,
                outdir=outdir,
            )

            with open(outdir / "split_info_used.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)

            t2_map = dict(zip(t2_metrics["metric"], t2_metrics["value"]))
            t1_map = dict(zip(t1_metrics["metric"], t1_metrics["value"]))

            all_summary.append({
                "ratio_label": ratio_label,
                "single_frac": split_info["single_frac"],
                "train_paired": len(split_info.get("train_paired_cells", [])),
                "train_rna_only": len(split_info.get("train_rna_only_cells", [])),
                "train_protein_only": len(split_info.get("train_protein_only_cells", [])),
                "val_paired": len(split_info.get("val_paired_cells", [])),
                "val_rna_only": len(split_info.get("val_rna_only_cells", [])),
                "val_protein_only": len(split_info.get("val_protein_only_cells", [])),
                "query_cells": len(common_cells),
                "common_features": len(common_features_eval),
                "cellwise_pearson_mean": t2_map["pearson_cell_mean"],
                "featurewise_pearson_mean": t2_map["pearson_feature_mean"],
                "rmse": t2_map["rmse"],
                "paired_embedding_distance_mean": t1_map["paired_embedding_distance_mean"],
                "paired_embedding_distance_median": t1_map["paired_embedding_distance_median"],
                "foscttm": t1_map["FOSCTTM"],
                "top1_acc": t1_map["Top1_ACC"],
                "top5_acc": t1_map["Top5_ACC"],
                "top10_acc": t1_map["Top10_ACC"],
                "pred_strategy": PREDICT_STRATEGY,
                "pred_method": PREDICT_METHOD,
                "pred_k": PREDICT_K,
            })

            print("Finished:", ratio_label)
            print(pd.DataFrame(all_summary).tail(1))

        except Exception as e:
            warnings.warn(f"Failed on {ratio_label}: {repr(e)}")
            with open(outdir / "error.txt", "w", encoding="utf-8") as f:
                f.write(repr(e) + "\n")
            continue

    if len(all_summary) > 0:
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(Path(OUT_ROOT) / "summary_all_ratios_scMRDR_rna_to_protein.csv", index=False)
        print("\nAll finished. Summary saved to:")
        print(Path(OUT_ROOT) / "summary_all_ratios_scMRDR_rna_to_protein.csv")
        print(summary_df)
    else:
        print("No successful ratio runs.")


if __name__ == "__main__":
    main()

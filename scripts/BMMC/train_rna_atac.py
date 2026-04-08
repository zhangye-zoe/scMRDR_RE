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

INPUT_DIR = "/data5/zhangye/scMRDR/input/BMMC/preprocessed_input/RNA_ATAC"
OUTPUT_DIR = "/data5/zhangye/scMRDR/output/BMMC"
SPLIT_ROOT = os.path.join(INPUT_DIR, "results_ratio_loop_rna_atac")   # produced earlier
ATAC_GAS_PATH = os.path.join(INPUT_DIR, "ATAC_gas.h5ad")
OUT_ROOT = os.path.join(OUTPUT_DIR, "scMRDR_results")

RATIO_LABELS = [f"single_{x:03d}" for x in [40, 60, 80, 100, 0, 20]]
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

PREDICT_MODALITY = "rna"
PREDICT_STRATEGY = "latent"
PREDICT_METHOD = "knn"
PREDICT_K = 10

N_PCS_EVAL = 30
MIN_COMMON_FEATURES = 50
INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA = True

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

def subset_adata_by_cells(adata, cells):
    cells = [c for c in cells if c in adata.obs_names]
    return adata[cells].copy()

def get_common_names(*arrays):
    if len(arrays) == 0:
        return []
    common = set(arrays[0])
    for arr in arrays[1:]:
        common &= set(arr)
    return sorted(common)

def build_model_input_for_ratio(split_dir, atac_gas_global):
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

    if INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA:
        adata_model = ad.concat(
            {
                "train_rna": train_rna_ref.copy(),
                "train_atac": train_atac_activity.copy(),
                "val_atac": val_atac_activity.copy(),
            },
            axis=0,
            join="inner",
            label="dataset_block",
            index_unique=None,
        )
    else:
        adata_model = ad.concat(
            {
                "train_rna": train_rna_ref.copy(),
                "train_atac": train_atac_activity.copy(),
            },
            axis=0,
            join="inner",
            label="dataset_block",
            index_unique=None,
        )

    if "count" not in adata_model.layers:
        raise ValueError("Concatenated scMRDR input is missing layer 'count'.")

    adata_model.obs["is_val_query"] = adata_model.obs_names.isin(val_query_atac_cells)
    return adata_model, val_true_rna, split_info

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

def evaluate_t2(pred_rna_val, true_rna_val, outdir):
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

    t2_metrics = pd.DataFrame({
        "metric": ["pearson_cell_mean", "pearson_gene_mean", "rmse"],
        "value": [float(np.nanmean(cell_cor)), float(np.nanmean(gene_cor)), rmse],
    })
    t2_metrics.to_csv(Path(outdir) / "T2_metrics.csv", index=False)

    pd.DataFrame({"cell": common_cells, "cellwise_pearson": cell_cor}).to_csv(
        Path(outdir) / "T2_cellwise_pearson.csv", index=False
    )
    pd.DataFrame({"gene": common_features, "genewise_pearson": gene_cor}).to_csv(
        Path(outdir) / "T2_genewise_pearson.csv", index=False
    )

    return t2_metrics, pred_mat, true_mat, common_cells, common_features

def evaluate_t1_from_true_pred(pred_mat, true_mat, common_cells, common_features, outdir):
    true_cells_by_features = true_mat.T
    pred_cells_by_features = pred_mat.T
    mix = np.vstack([true_cells_by_features, pred_cells_by_features])

    n_components = min(N_PCS_EVAL, mix.shape[0] - 1, mix.shape[1])
    if n_components < 2:
        raise ValueError("PCA components < 2, cannot evaluate T1.")

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
    t1_metrics.to_csv(Path(outdir) / "T1_metrics.csv", index=False)

    pd.DataFrame({
        "cell": list(common_cells),
        "FOSCTTM": foscttm_each,
        "paired_dist": paired_dist,
        "Top1_match": nn_match.astype(bool),
    }).to_csv(Path(outdir) / "T1_per_cell_metrics.csv", index=False)

    pca_df = pd.DataFrame(emb[:, : min(5, emb.shape[1])], columns=[f"PC{i+1}" for i in range(min(5, emb.shape[1]))])
    pca_df["group"] = ["True_RNA"] * n_q + ["Pred_from_ATAC"] * n_q
    pca_df["cell"] = [f"true_{c}" for c in common_cells] + [f"pred_{c}" for c in common_cells]
    pca_df.to_csv(Path(outdir) / "pca_true_pred_coords.csv", index=False)

    return t1_metrics

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
        print("============================")

        try:
            adata_model, true_rna_val, split_info = build_model_input_for_ratio(split_dir, atac_gas_global)
            save_h5ad_safe(adata_model, outdir / "training_adata_input.h5ad")

            model, adata_post, pred_rna_all = train_scmrdr_for_ratio(adata_model)
            save_h5ad_safe(adata_post, outdir / "training_adata_post.h5ad")
            save_h5ad_safe(pred_rna_all, outdir / "pred_rna_all_nonrna.h5ad")

            val_cells = split_info["val_query_atac_cells"]
            pred_rna_val = pred_rna_all[[c for c in val_cells if c in pred_rna_all.obs_names]].copy()

            save_h5ad_safe(pred_rna_val, outdir / "pred_rna_val.h5ad")
            save_h5ad_safe(true_rna_val, outdir / "true_rna_val.h5ad")

            t2_metrics, pred_mat, true_mat, common_cells, common_features = evaluate_t2(
                pred_rna_val=pred_rna_val,
                true_rna_val=true_rna_val,
                outdir=outdir,
            )

            t1_metrics = evaluate_t1_from_true_pred(
                pred_mat=pred_mat,
                true_mat=true_mat,
                common_cells=common_cells,
                common_features=common_features,
                outdir=outdir,
            )

            with open(outdir / "split_info_used.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)

            t2_map = dict(zip(t2_metrics["metric"], t2_metrics["value"]))
            t1_map = dict(zip(t1_metrics["metric"], t1_metrics["value"]))

            all_summary.append({
                "ratio_label": ratio_label,
                "single_frac": split_info["single_frac"],
                "train_paired": len(split_info["train_paired_cells"]),
                "train_rna_only": len(split_info["train_rna_only_cells"]),
                "train_atac_only": len(split_info["train_atac_only_cells"]),
                "val_paired": len(split_info["val_paired_cells"]),
                "val_rna_only": len(split_info["val_rna_only_cells"]),
                "val_atac_only": len(split_info["val_atac_only_cells"]),
                "query_cells": len(common_cells),
                "common_features": len(common_features),
                "cellwise_pearson_mean": t2_map["pearson_cell_mean"],
                "genewise_pearson_mean": t2_map["pearson_gene_mean"],
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
        summary_df.to_csv(Path(OUT_ROOT) / "summary_all_ratios_scMRDR.csv", index=False)
        print("\nAll finished. Summary saved to:")
        print(Path(OUT_ROOT) / "summary_all_ratios_scMRDR.csv")
        print(summary_df)
    else:
        print("No successful ratio runs.")

if __name__ == "__main__":
    main()

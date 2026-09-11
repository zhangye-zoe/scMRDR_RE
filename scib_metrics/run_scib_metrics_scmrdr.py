#!/usr/bin/env python3
"""
Compute a comprehensive scib-metrics evaluation for scMRDR BMMC RNA-ATAC outputs.

Source layout used by the downloaded scMRDR code:
  <scmrdr-root>/<ratio>/training_adata_post.h5ad
  adata.obsm['latent_shared']

The training script builds training_adata_post.h5ad from train RNA, train ATAC,
and (optionally) validation-query ATAC observations. Cell-type labels are recovered
from the split h5ad files when they are absent from training_adata_post.h5ad.

Example:
python run_scib_metrics_scmrdr.py \
  --scmrdr-root /data5/zhangye/scMRDR/output/BMMC/scMRDR_results_rna_to_atac \
  --split-root /data5/zhangye/scMRDR/input/BMMC/preprocessed_input/RNA_ATAC/results_ratio_loop_rna_atac \
  --ratios single_000 \
  --scope all \
  --out-root /data5/zhangye/scMRDR/output/BMMC/scMRDR_results_rna_to_atac/scib_full_test \
  --n-jobs 8
"""

from __future__ import annotations

import argparse
import inspect
import json
import warnings
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from scib_metrics import silhouette_batch
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation


CELLTYPE_CANDIDATES = [
    "cell_type",
    "celltype",
    "celltypes",
    "CellType",
    "cell_type_l1",
    "cell_type_l2",
    "predicted.celltype.l1",
    "predicted.celltype.l2",
    "annotation",
    "label",
    "labels",
]

SPLIT_LABEL_FILES = [
    "train_rna_ref.h5ad",
    "train_atac_activity.h5ad",
    "train_atac_full.h5ad",
    "val_true_rna.h5ad",
    "val_atac_activity.h5ad",
    "val_query_atac.h5ad",
]


def to_numpy(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def normalize_modality(value: object) -> str:
    text = str(value).strip().lower()
    if "rna" in text or text in {"gex", "transcriptome"}:
        return "RNA"
    if "atac" in text or "chrom" in text or "activity" in text:
        return "ATAC"
    return str(value).strip().upper()


def choose_label_key(columns: Iterable[str], requested: Optional[str]) -> Optional[str]:
    columns = list(columns)
    if requested is not None:
        return requested if requested in columns else None
    for key in CELLTYPE_CANDIDATES:
        if key in columns:
            return key
    return None


def read_obs_only(path: Path) -> pd.DataFrame:
    """Read only .obs where possible, without materializing the expression matrix."""
    a = sc.read_h5ad(path, backed="r")
    obs = a.obs.copy()
    obs.index = obs.index.astype(str)
    if getattr(a, "file", None) is not None:
        try:
            a.file.close()
        except Exception:
            pass
    return obs


def build_label_map(
    output_obs: pd.DataFrame,
    split_dir: Path,
    requested_key: Optional[str],
    external_label_h5ad: Optional[Path],
) -> tuple[pd.Series, pd.DataFrame]:
    """Build cell_id -> cell_type mapping and an audit table."""
    audit_rows: list[dict] = []
    sources: list[tuple[str, pd.DataFrame]] = []

    output_key = choose_label_key(output_obs.columns, requested_key)
    if output_key is not None:
        tmp = output_obs[[output_key]].copy()
        sources.append(("training_adata_post.h5ad", tmp))
        audit_rows.append(
            {
                "source": "training_adata_post.h5ad",
                "label_key": output_key,
                "n_rows": len(tmp),
                "n_nonmissing": int(tmp[output_key].notna().sum()),
            }
        )

    for name in SPLIT_LABEL_FILES:
        path = split_dir / name
        if not path.exists():
            continue
        obs = read_obs_only(path)
        key = choose_label_key(obs.columns, requested_key)
        audit_rows.append(
            {
                "source": str(path),
                "label_key": key or "<not found>",
                "n_rows": len(obs),
                "n_nonmissing": int(obs[key].notna().sum()) if key else 0,
            }
        )
        if key is not None:
            sources.append((str(path), obs[[key]].rename(columns={key: "_label"})))

    if external_label_h5ad is not None:
        if not external_label_h5ad.exists():
            raise FileNotFoundError(f"External label h5ad does not exist: {external_label_h5ad}")
        obs = read_obs_only(external_label_h5ad)
        key = choose_label_key(obs.columns, requested_key)
        audit_rows.append(
            {
                "source": str(external_label_h5ad),
                "label_key": key or "<not found>",
                "n_rows": len(obs),
                "n_nonmissing": int(obs[key].notna().sum()) if key else 0,
            }
        )
        if key is not None:
            sources.append((str(external_label_h5ad), obs[[key]].rename(columns={key: "_label"})))

    if not sources:
        raise KeyError(
            "No cell-type annotation was found in training_adata_post.h5ad, the split h5ad files, "
            "or the optional external label h5ad. Run the inspection command shown in the response "
            "and pass --celltype-key with the correct column name."
        )

    records: list[pd.DataFrame] = []
    for source_name, frame in sources:
        frame = frame.copy()
        if "_label" not in frame.columns:
            only_col = frame.columns[0]
            frame = frame.rename(columns={only_col: "_label"})
        frame["cell_id"] = frame.index.astype(str)
        frame["_source"] = source_name
        records.append(frame[["cell_id", "_label", "_source"]])

    all_labels = pd.concat(records, ignore_index=True)
    all_labels = all_labels.dropna(subset=["_label"])
    all_labels["_label"] = all_labels["_label"].astype(str)
    all_labels = all_labels[~all_labels["_label"].str.lower().isin({"nan", "none", "unknown", ""})]

    # Detect contradictory labels for the same cell before choosing the first source.
    conflict_counts = all_labels.groupby("cell_id")["_label"].nunique()
    conflicts = conflict_counts[conflict_counts > 1]
    if len(conflicts) > 0:
        examples = conflicts.index[:10].tolist()
        warnings.warn(
            f"Found {len(conflicts)} cells with conflicting labels across sources. "
            f"The first available label is used. Examples: {examples}"
        )

    label_map = all_labels.drop_duplicates("cell_id", keep="first").set_index("cell_id")["_label"]
    audit = pd.DataFrame(audit_rows)
    return label_map, audit


def canonical_id(modality: str, cell_id: str, occurrence: int = 0) -> str:
    base = f"{modality}::{cell_id}"
    return base if occurrence == 0 else f"{base}::dup{occurrence}"


def apply_scope(obs: pd.DataFrame, X: np.ndarray, scope: str) -> tuple[pd.DataFrame, np.ndarray]:
    keep = np.ones(len(obs), dtype=bool)

    if scope in {"train", "paired-train"}:
        if "dataset_block" in obs.columns:
            keep &= ~obs["dataset_block"].astype(str).str.lower().str.startswith("val").to_numpy()
        elif "is_val_query" in obs.columns:
            keep &= ~obs["is_val_query"].fillna(False).astype(bool).to_numpy()
        else:
            warnings.warn(
                "Neither dataset_block nor is_val_query exists; --scope train cannot explicitly "
                "remove validation-query observations. Keeping all observations."
            )

    obs = obs.loc[keep].copy()
    X = np.asarray(X[keep], dtype=np.float32)

    if scope == "paired-train":
        modality_sets = {
            m: set(obs.loc[obs["modality"] == m, "cell_id"].astype(str))
            for m in ["RNA", "ATAC"]
        }
        paired_ids = modality_sets["RNA"] & modality_sets["ATAC"]
        keep2 = obs["cell_id"].astype(str).isin(paired_ids).to_numpy()
        obs = obs.loc[keep2].copy()
        X = X[keep2]

    return obs, X


def stratified_downsample(
    obs: pd.DataFrame,
    X: np.ndarray,
    max_cells: Optional[int],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_cells is None or len(obs) <= max_cells:
        return obs, X

    rng = np.random.default_rng(seed)
    strata = obs["modality"].astype(str) + "||" + obs["cell_type"].astype(str)
    selected: list[int] = []

    for _, idx_values in pd.Series(np.arange(len(obs))).groupby(strata.to_numpy()):
        idx = idx_values.to_numpy(dtype=int)
        quota = max(1, int(round(max_cells * len(idx) / len(obs))))
        selected.extend(rng.choice(idx, size=min(quota, len(idx)), replace=False).tolist())

    selected = sorted(set(selected))
    if len(selected) > max_cells:
        selected = sorted(rng.choice(selected, size=max_cells, replace=False).tolist())
    elif len(selected) < max_cells:
        remaining = np.setdiff1d(np.arange(len(obs)), np.asarray(selected, dtype=int))
        extra_n = min(max_cells - len(selected), len(remaining))
        if extra_n > 0:
            selected.extend(rng.choice(remaining, size=extra_n, replace=False).tolist())
            selected = sorted(selected)

    return obs.iloc[selected].copy(), X[selected]


def load_scmrdr_ratio(
    ratio_dir: Path,
    split_dir: Path,
    latent_key: str,
    modality_key: str,
    celltype_key: Optional[str],
    external_label_h5ad: Optional[Path],
    scope: str,
    max_cells: Optional[int],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    post_path = ratio_dir / "training_adata_post.h5ad"
    if not post_path.exists():
        raise FileNotFoundError(f"Missing scMRDR output: {post_path}")

    a = sc.read_h5ad(post_path)
    if latent_key not in a.obsm:
        raise KeyError(
            f"{latent_key!r} not found in {post_path}. Available obsm keys: {list(a.obsm.keys())}"
        )
    if modality_key not in a.obs.columns:
        raise KeyError(
            f"{modality_key!r} not found in {post_path}. Available obs columns: {list(a.obs.columns)}"
        )

    X = np.asarray(a.obsm[latent_key], dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != a.n_obs:
        raise ValueError(f"Invalid latent shape {X.shape} for {a.n_obs} observations")
    if not np.isfinite(X).all():
        raise ValueError("latent_shared contains NaN or Inf")

    obs = a.obs.copy()
    obs["cell_id"] = a.obs_names.astype(str)
    obs["modality"] = obs[modality_key].map(normalize_modality)
    keep_modality = obs["modality"].isin(["RNA", "ATAC"]).to_numpy()
    obs = obs.loc[keep_modality].copy()
    X = X[keep_modality]

    label_map, label_audit = build_label_map(
        output_obs=a.obs.copy(),
        split_dir=split_dir,
        requested_key=celltype_key,
        external_label_h5ad=external_label_h5ad,
    )
    obs["cell_type"] = obs["cell_id"].map(label_map)

    missing_before = int(obs["cell_type"].isna().sum())
    valid_label = obs["cell_type"].notna() & ~obs["cell_type"].astype(str).str.lower().isin(
        {"nan", "none", "unknown", ""}
    )
    obs = obs.loc[valid_label].copy()
    X = X[valid_label.to_numpy()]

    obs, X = apply_scope(obs, X, scope)
    obs, X = stratified_downsample(obs, X, max_cells=max_cells, seed=seed)

    if obs["modality"].nunique() < 2:
        raise ValueError(f"Only one modality remains after filtering: {obs['modality'].value_counts().to_dict()}")
    if obs["cell_type"].nunique() < 2:
        raise ValueError(f"Fewer than two cell types remain after filtering")

    # Build unique observation IDs while preserving modality-cell identity.
    counts: dict[str, int] = {}
    new_index: list[str] = []
    for modality, cell_id in zip(obs["modality"].astype(str), obs["cell_id"].astype(str)):
        base = f"{modality}::{cell_id}"
        occurrence = counts.get(base, 0)
        new_index.append(canonical_id(modality, cell_id, occurrence))
        counts[base] = occurrence + 1
    obs.index = new_index

    obs["modality"] = obs["modality"].astype("category")
    obs["cell_type"] = obs["cell_type"].astype(str).astype("category")

    audit = {
        "post_h5ad": str(post_path),
        "latent_key": latent_key,
        "scope": scope,
        "n_output_rows_before_modality_filter": int(a.n_obs),
        "n_missing_labels_before_drop": missing_before,
        "n_evaluated_observations": int(len(obs)),
        "embedding_dim": int(X.shape[1]),
        "n_cell_types": int(obs["cell_type"].nunique()),
        "modality_counts": {str(k): int(v) for k, v in obs["modality"].value_counts().items()},
        "cell_type_counts": {str(k): int(v) for k, v in obs["cell_type"].value_counts().items()},
        "dataset_block_counts": (
            {str(k): int(v) for k, v in obs["dataset_block"].astype(str).value_counts().items()}
            if "dataset_block" in obs.columns
            else None
        ),
    }
    return obs, X, label_audit, audit


def make_batch_metrics() -> BatchCorrection:
    kwargs = {
        "bras": True,
        "ilisi_knn": True,
        "kbet_per_label": True,
        "graph_connectivity": True,
        "pcr_comparison": False,
    }
    if "sbee" in inspect.signature(BatchCorrection).parameters:
        kwargs["sbee"] = True
    return BatchCorrection(**kwargs)


def run_one_ratio(
    ratio: str,
    scmrdr_root: Path,
    split_root: Path,
    out_root: Path,
    latent_key: str,
    modality_key: str,
    celltype_key: Optional[str],
    external_label_h5ad: Optional[Path],
    scope: str,
    max_cells: Optional[int],
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    ratio_dir = scmrdr_root / ratio
    split_dir = split_root / ratio
    ratio_out = out_root / ratio
    ratio_out.mkdir(parents=True, exist_ok=True)

    obs, X, label_audit, input_audit = load_scmrdr_ratio(
        ratio_dir=ratio_dir,
        split_dir=split_dir,
        latent_key=latent_key,
        modality_key=modality_key,
        celltype_key=celltype_key,
        external_label_h5ad=external_label_h5ad,
        scope=scope,
        max_cells=max_cells,
        seed=seed,
    )

    label_audit.to_csv(ratio_out / "label_source_audit.csv", index=False)
    obs.to_csv(ratio_out / "evaluated_observations.csv")
    with open(ratio_out / "input_audit.json", "w", encoding="utf-8") as f:
        json.dump(input_audit, f, indent=2, ensure_ascii=False)

    print("Input audit:")
    print(json.dumps(input_audit, indent=2, ensure_ascii=False))

    # Benchmarker consumes embeddings from .obsm. X is supplied only as a valid placeholder;
    # PCR comparison is intentionally disabled because no common pre-integration representation is used.
    a = ad.AnnData(X=X.copy(), obs=obs.copy())
    a.obsm["scMRDR"] = X

    bio = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_kmeans=True,
        nmi_ari_cluster_labels_leiden=True,
        silhouette_label=True,
        clisi_knn=True,
    )
    batch = make_batch_metrics()

    bm_kwargs = {
        "adata": a,
        "batch_key": "modality",
        "label_key": "cell_type",
        "embedding_obsm_keys": ["scMRDR"],
        "bio_conservation_metrics": bio,
        "batch_correction_metrics": batch,
        "pre_integrated_embedding_obsm_key": None,
        "n_jobs": n_jobs,
        "progress_bar": True,
    }
    if "solver" in inspect.signature(Benchmarker).parameters:
        bm_kwargs["solver"] = "arpack"

    bm = Benchmarker(**bm_kwargs)
    if hasattr(bm, "prepare"):
        bm.prepare()
    bm.benchmark()
    results = bm.get_results(min_max_scale=False, clean_names=True)

    label_codes = obs["cell_type"].cat.codes.to_numpy(dtype=np.int32)
    batch_codes = obs["modality"].cat.codes.to_numpy(dtype=np.int32)
    results.loc["scMRDR", "Batch ASW"] = float(
        silhouette_batch(
            X=X,
            labels=label_codes,
            batch=batch_codes,
            rescale=True,
            chunk_size=256,
            metric="euclidean",
            between_cluster_distances="nearest",
        )
    )
    results.loc["Metric Type", "Batch ASW"] = "Batch correction"

    results.to_csv(ratio_out / "scib_results_native.csv")

    if "Metric Type" not in results.index:
        raise ValueError(f"Unexpected Benchmarker output; rows are {list(results.index)}")
    metric_types = results.loc["Metric Type"].copy()
    numeric = results.drop(index="Metric Type").apply(pd.to_numeric, errors="coerce")

    wide = numeric.T
    wide.insert(0, "Category", metric_types.reindex(wide.index))
    wide.index.name = "Metric"
    wide.to_csv(ratio_out / "scib_results_wide.csv", float_format="%.8f")

    long = numeric.reset_index(names="Method").melt(
        id_vars="Method", var_name="Metric", value_name="Score"
    )
    long.insert(0, "Ratio", ratio)
    long["SingleModalityPercent"] = int(ratio.split("_")[-1])
    long["PairedPercent"] = 100 - long["SingleModalityPercent"]
    long["Scope"] = scope
    long["Category"] = long["Metric"].map(metric_types)
    long.to_csv(ratio_out / "scib_results_long.csv", index=False, float_format="%.8f")

    print("\nResults:")
    print(wide.to_string(float_format=lambda value: f"{value:.4f}"))
    return long


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scmrdr-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["single_000", "single_020", "single_040", "single_060", "single_080", "single_100"],
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--latent-key", type=str, default="latent_shared")
    parser.add_argument("--modality-key", type=str, default="modality")
    parser.add_argument(
        "--celltype-key",
        type=str,
        default=None,
        help="Exact cell-type column. Omit to auto-detect common names.",
    )
    parser.add_argument(
        "--external-label-h5ad",
        type=Path,
        default=None,
        help="Optional raw/reference h5ad used only to recover cell-type labels by obs_names.",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "train", "paired-train"],
        default="all",
        help=(
            "all: exact observations in training_adata_post.h5ad; "
            "train: remove val-query observations when identifiable; "
            "paired-train: retain only train cell IDs present in both modalities."
        ),
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional stratified downsampling for a quick test. Omit for final full-data results.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_long: list[pd.DataFrame] = []

    for ratio in args.ratios:
        print(f"\n{'=' * 80}\nEvaluating {ratio}\n{'=' * 80}")
        try:
            long = run_one_ratio(
                ratio=ratio,
                scmrdr_root=args.scmrdr_root,
                split_root=args.split_root,
                out_root=args.out_root,
                latent_key=args.latent_key,
                modality_key=args.modality_key,
                celltype_key=args.celltype_key,
                external_label_h5ad=args.external_label_h5ad,
                scope=args.scope,
                max_cells=args.max_cells,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            all_long.append(long)
        except Exception as exc:
            error_dir = args.out_root / ratio
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
            raise

    combined = pd.concat(all_long, ignore_index=True)
    combined.to_csv(args.out_root / "scib_all_ratios_long.csv", index=False, float_format="%.8f")

    summary = (
        combined.groupby(["Method", "Metric", "Category", "Scope"], dropna=False)["Score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(args.out_root / "scib_all_ratios_mean_std.csv", index=False, float_format="%.8f")

    print("\nSaved results under:", args.out_root)
    print("Main summary:", args.out_root / "scib_all_ratios_mean_std.csv")


if __name__ == "__main__":
    main()

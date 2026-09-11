#!/usr/bin/env python3
"""
TRACE: Temporal Relationship-Adaptive Cross-omics Engine

A practical prototype for:
1) cell-level cross-modal alignment under partial / full unpairing
2) directional cross-omics prediction (T2)
3) adaptive lag-aware dynamics for T3 / T4

Main update in this version
---------------------------
- T1/T2 evaluation is changed to follow the same logic as the provided scMRDR code:
  * T2: predicted target modality vs true target modality
  * T1: PCA on [true_target; predicted_target], then evaluate paired distance,
        FOSCTTM, Top-k ACC on cross-distance matrix
- T3/T4 keep TRACE's original dynamic evaluation pipeline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, roc_auc_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# Constants
# =============================================================================

MIN_COMMON_FEATURES = 50
N_PCS_EVAL = 30


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_write_h5ad(adata: ad.AnnData, path: os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    adata.write_h5ad(str(path))


def sanitize_adata_for_write(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    if isinstance(adata.X, np.matrix):
        adata.X = np.asarray(adata.X)
    for k in list(adata.layers.keys()):
        if isinstance(adata.layers[k], np.matrix):
            adata.layers[k] = np.asarray(adata.layers[k])
    return adata


def save_h5ad_safe(adata: ad.AnnData, path: os.PathLike) -> None:
    safe_write_h5ad(sanitize_adata_for_write(adata), path)


def to_dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def maybe_log1p(x: np.ndarray, assume_logged: bool) -> np.ndarray:
    if assume_logged:
        return x.astype(np.float32, copy=False)
    return np.log1p(np.clip(x, a_min=0, a_max=None)).astype(np.float32, copy=False)


def intersect_strings(*lists: Sequence[str]) -> List[str]:
    if not lists:
        return []
    out = set(lists[0])
    for arr in lists[1:]:
        out &= set(arr)
    return sorted(out)


def union_strings(*lists: Sequence[str]) -> List[str]:
    out = set()
    for arr in lists:
        out |= set(arr)
    return sorted(out)


def get_common_names(*arrays):
    if len(arrays) == 0:
        return []
    common = set(arrays[0])
    for arr in arrays[1:]:
        common &= set(arr)
    return sorted(common)


def save_npz_dict(arrays: Dict[str, np.ndarray], path: os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def kl_div(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(1)
    denom = w.sum().clamp_min(1.0) * pred.shape[1]
    return ((pred - target) ** 2 * w).sum() / denom


def entropy_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ent = -(alpha * torch.log(alpha + eps)).sum(dim=1)
    return ent.mean()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def knn_graph_indices(x: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    n_neighbors = int(min(n_neighbors, max(2, x.shape[0] - 1)))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    _, idx = nbrs.kneighbors(x)
    return idx


def localize_global_knn(batch_global_idx: np.ndarray, global_knn: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
    pos = {int(g): i for i, g in enumerate(batch_global_idx.tolist())}
    local_lists: List[List[int]] = []
    batch_set = set(pos.keys())
    for g in batch_global_idx.tolist():
        neigh = [pos[int(j)] for j in global_knn[int(g)].tolist() if int(j) in batch_set and int(j) != int(g)]
        if len(neigh) < min_neighbors:
            fallback = [i for i in range(len(batch_global_idx)) if i != pos[int(g)]]
            neigh = fallback[:max(min_neighbors, 1)]
        local_lists.append(neigh[:max(min_neighbors, 1)])
    k = min(len(x) for x in local_lists) if local_lists else 1
    if k <= 0:
        return np.zeros((len(batch_global_idx), 1), dtype=np.int64)
    return np.asarray([x[:k] for x in local_lists], dtype=np.int64)


def smoothness_loss(latent: torch.Tensor, knn_idx_local: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(knn_idx_local, dtype=torch.long, device=latent.device)
    center = latent.unsqueeze(1).expand(-1, idx.shape[1], -1)
    neigh = latent[idx]
    return ((center - neigh) ** 2).mean()


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device)
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    kxx = torch.exp(-xx / (2 * sigma * sigma))
    kyy = torch.exp(-yy / (2 * sigma * sigma))
    kxy = torch.exp(-xy / (2 * sigma * sigma))
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def sinkhorn_transport(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
    n, m = cost.shape
    a = torch.full((n,), 1.0 / max(n, 1), device=cost.device)
    b = torch.full((m,), 1.0 / max(m, 1), device=cost.device)
    K = torch.exp(-cost / epsilon).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v).clamp_min(1e-12)
        v = b / (K.t() @ u).clamp_min(1e-12)
    T = torch.diag(u) @ K @ torch.diag(v)
    return T / T.sum().clamp_min(1e-12)


def weighted_pair_loss(x: torch.Tensor, y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, y, p=2).pow(2)
    return (T * dist).sum()


def estimate_pseudotime_from_latent(h: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    adata = ad.AnnData(X=h)
    sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, max(3, h.shape[0] - 1)), use_rep="X")
    sc.tl.diffmap(adata)
    diffmap = np.asarray(adata.obsm["X_diffmap"])
    root_idx = int(np.argmin(diffmap[:, 0]))
    adata.uns["iroot"] = root_idx
    try:
        sc.tl.dpt(adata)
    except Exception:
        pass
    if "dpt_pseudotime" in adata.obs.columns:
        pt = np.asarray(adata.obs["dpt_pseudotime"]).astype(np.float32)
    else:
        pt = diffmap[:, 0].astype(np.float32)
        pt = pt - np.nanmin(pt)
        denom = np.nanmax(pt)
        if denom > 0:
            pt = pt / denom
    if np.isnan(pt).any():
        pt = np.nan_to_num(pt, nan=np.nanmedian(pt))
    return pt


def time_consistency_loss(h: torch.Tensor, t: torch.Tensor, k: int = 5) -> torch.Tensor:
    if h.shape[0] < 3:
        return torch.tensor(0.0, device=h.device)
    with torch.no_grad():
        dist = torch.cdist(h, h)
        knn_idx = dist.topk(k=min(k + 1, h.shape[0]), largest=False).indices[:, 1:]
    t_center = t.expand(-1, knn_idx.shape[1])
    t_neigh = t[knn_idx].squeeze(-1)
    return ((t_center - t_neigh) ** 2).mean()


def forward_margin_loss(h: torch.Tensor, t: torch.Tensor, knn_idx_local: np.ndarray, margin: float = 0.02) -> torch.Tensor:
    idx = torch.as_tensor(knn_idx_local, dtype=torch.long, device=h.device)
    t_i = t.squeeze(-1).unsqueeze(1).expand(-1, idx.shape[1])
    t_j = t.squeeze(-1)[idx]
    return F.relu(margin - (t_j - t_i)).mean()


def build_lag_bank(
    latent: np.ndarray,
    pseudotime: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    max_lag_bins: int = 4,
    k_per_bin: int = 4,
) -> np.ndarray:
    n, d = latent.shape
    bins = np.linspace(0, 1, max_lag_bins + 2)
    lag_bank = np.zeros((n, max_lag_bins, d), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(20, max(2, n - 1))).fit(latent)
    _, idx = nn.kneighbors(latent)
    if source_mask is None:
        source_mask = np.ones(n, dtype=bool)
    source_mask = np.asarray(source_mask).astype(bool)

    for i in range(n):
        pi = pseudotime[i]
        valid_past = source_mask & (pseudotime < pi)
        for b in range(max_lag_bins):
            lo = max(0.0, pi - bins[b + 1])
            hi = max(0.0, pi - bins[b])
            cand = np.where(valid_past & (pseudotime >= lo) & (pseudotime < hi))[0]
            if cand.size == 0:
                cand = np.array([j for j in idx[i].tolist() if source_mask[j] and pseudotime[j] < pi], dtype=int)
            if cand.size == 0:
                lag_bank[i, b] = latent[i]
                continue
            if cand.size > k_per_bin:
                c_h = latent[cand]
                dists = np.sum((c_h - latent[i]) ** 2, axis=1)
                cand = cand[np.argsort(dists)[:k_per_bin]]
            lag_bank[i, b] = latent[cand].mean(axis=0)
    return lag_bank


def build_transition_matrix(
    h: np.ndarray,
    t: np.ndarray,
    lag_score: Optional[np.ndarray],
    n_neighbors: int = 30,
    beta_time: float = 5.0,
    gamma_lag: float = 1.0,
) -> np.ndarray:
    idx = knn_graph_indices(h, n_neighbors=n_neighbors + 1)
    n = h.shape[0]
    P = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neigh = idx[i, 1:]
        d_h = np.sum((h[neigh] - h[i]) ** 2, axis=1)
        dt = t[neigh] - t[i]
        s = -d_h + beta_time * dt
        if lag_score is not None:
            s = s + gamma_lag * lag_score[i, neigh]
        s = s - np.max(s)
        p = np.exp(s)
        p = p / np.clip(p.sum(), 1e-8, None)
        P[i, neigh] = p
    return P


def hit_at_k(cross_dist: np.ndarray, k: int = 5) -> float:
    nq = cross_dist.shape[0]
    hits = []
    for i in range(nq):
        ord_idx = np.argsort(cross_dist[i, :])[: min(k, cross_dist.shape[1])]
        hits.append(i in ord_idx)
    return float(np.mean(hits)) if len(hits) > 0 else np.nan


# =============================================================================
# Evaluation helpers for T3 / T4
# =============================================================================

def _find_first_existing_key(container, candidates, kind="key", raise_error: bool = False):
    for k in candidates:
        if k in container:
            return k
    if raise_error:
        raise KeyError(f"Could not find any valid {kind}. Tried: {candidates}")
    return None


def _resolve_label_key(adata, preferred=None):
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["celltype", "cell_type", "annotation", "annot", "labels", "leiden"])
    return _find_first_existing_key(
        adata.obs.columns,
        candidates,
        kind="label key",
        raise_error=False,
    )


def interclass_pairwise_accuracy(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[valid], ref[valid]
    correct, total = 0.0, 0.0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            if ref[i] == ref[j]:
                continue
            w = abs(ref[i] - ref[j]) if weighted else 1.0
            is_correct = (pred[i] < pred[j]) if ref[i] < ref[j] else (pred[j] < pred[i])
            correct += w * float(is_correct)
            total += w
    return correct / total if total > 0 else np.nan


def interclass_pairwise_auc(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[valid], ref[valid]
    classes = np.sort(np.unique(ref))
    aucs, weights = [], []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            mask = (ref == a) | (ref == b)
            y = (ref[mask] == b).astype(int)
            s = pred[mask]
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, s)
            w = abs(b - a) if weighted else 1.0
            aucs.append(auc)
            weights.append(w)
    if len(aucs) == 0:
        return np.nan
    aucs = np.asarray(aucs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.average(aucs, weights=weights)


def compute_t3_metrics(adata, time_key: str, reference_time_key: Optional[str] = None, label_key: Optional[str] = None, stage_order_map: Optional[dict] = None):
    pred = np.asarray(adata.obs[time_key]).astype(float)
    if reference_time_key is not None and reference_time_key in adata.obs.columns:
        ref = np.asarray(adata.obs[reference_time_key]).astype(float)
    else:
        if label_key is None:
            label_key = _resolve_label_key(adata)
        if stage_order_map is None:
            return {"POA": np.nan, "POAUC": np.nan, "n_eval": 0}
        labels = adata.obs[label_key].astype(str)
        ref = labels.map(stage_order_map).to_numpy(dtype=float)
    valid = np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 3:
        return {"POA": np.nan, "POAUC": np.nan, "n_eval": int(valid.sum())}
    pred_v, ref_v = pred[valid], ref[valid]
    return {
        "POA": float(interclass_pairwise_accuracy(pred_v, ref_v, weighted=True)),
        "POAUC": float(interclass_pairwise_auc(pred_v, ref_v, weighted=True)),
        "n_eval": int(valid.sum()),
    }


def compute_t4_metrics(adata, transition_matrix: np.ndarray, lineage_key: str):
    if lineage_key is None:
        return {"F1": np.nan, "AUROC": np.nan, "n_eval": 0, "status": "skipped_no_lineage_key"}

    if lineage_key not in adata.obs.columns:
        return {
            "F1": np.nan,
            "AUROC": np.nan,
            "n_eval": 0,
            "status": f"skipped_lineage_key_not_found:{lineage_key}",
        }

    y_true_str = adata.obs[lineage_key].astype(str).to_numpy()
    class_order = sorted(pd.unique(y_true_str))
    if len(class_order) < 2:
        return {"F1": np.nan, "AUROC": np.nan, "n_eval": 0, "status": "skipped_less_than_2_classes"}

    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_true = np.array([class_to_idx[x] for x in y_true_str], dtype=int)

    if "trace_time" in adata.obs.columns:
        t = np.asarray(adata.obs["trace_time"]).astype(float)
    else:
        t = np.linspace(0, 1, adata.n_obs)

    terminal_mask = np.zeros(adata.n_obs, dtype=bool)
    for cls in class_order:
        cls_idx = np.where(y_true_str == cls)[0]
        if cls_idx.size == 0:
            continue
        top_k = max(1, int(0.1 * cls_idx.size))
        keep = cls_idx[np.argsort(t[cls_idx])[-top_k:]]
        terminal_mask[keep] = True

    proba = np.zeros((adata.n_obs, len(class_order)), dtype=float)
    for i, cls in enumerate(class_order):
        cls_terminal = terminal_mask & (y_true_str == cls)
        if cls_terminal.sum() == 0:
            continue
        proba[:, i] = transition_matrix[:, cls_terminal].sum(axis=1)

    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    proba = proba / row_sum

    y_pred = proba.argmax(axis=1)
    f1 = f1_score(y_true, y_pred, average="macro")

    if len(class_order) == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=np.arange(len(class_order)))
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")

    return {
        "F1": float(f1),
        "AUROC": float(auc),
        "n_eval": int(adata.n_obs),
        "status": "ok",
    }


# =============================================================================
# Data
# =============================================================================

@dataclasses.dataclass
class ModalitySpec:
    name: str
    path: str
    assume_logged: bool = True
    align_weight: float = 1.0


@dataclasses.dataclass
class LagEdge:
    source: str
    target: str
    weight: float = 1.0


class MultimodalRatioDataset(Dataset):
    def __init__(self, modalities: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], cell_ids: Sequence[str]):
        self.modalities = {k: v.astype(np.float32, copy=False) for k, v in modalities.items()}
        self.masks = {k: v.astype(np.float32, copy=False) for k, v in masks.items()}
        self.modality_names = sorted(self.modalities.keys())
        self.cell_ids = np.asarray(cell_ids)
        self.n_obs = len(self.cell_ids)

    def __len__(self) -> int:
        return self.n_obs

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "modalities": {m: torch.from_numpy(self.modalities[m][idx]) for m in self.modality_names},
            "masks": {m: torch.tensor(self.masks[m][idx], dtype=torch.float32) for m in self.modality_names},
        }


def collate_multimodal(batch: List[Dict[str, object]]) -> Dict[str, object]:
    modality_names = sorted(batch[0]["modalities"].keys())
    return {
        "index": torch.stack([b["index"] for b in batch], dim=0),
        "modalities": {m: torch.stack([b["modalities"][m] for b in batch], dim=0) for m in modality_names},
        "masks": {m: torch.stack([b["masks"][m] for b in batch], dim=0) for m in modality_names},
    }


def parse_manifest(path: os.PathLike):
    raw = load_json(path)
    modalities = {
        name: ModalitySpec(
            name=name,
            path=cfg["path"],
            assume_logged=cfg.get("assume_logged", True),
            align_weight=float(cfg.get("align_weight", 1.0)),
        )
        for name, cfg in raw["modalities"].items()
    }
    alignment_pairs = [tuple(x) for x in raw.get("alignment_pairs", [])]
    prediction_pairs = [tuple(x) for x in raw.get("prediction_pairs", [])]
    lag_edges = [LagEdge(source=e["source"], target=e["target"], weight=float(e.get("weight", 1.0))) for e in raw.get("lag_graph", [])]
    label_key = raw.get("label_key")
    future_label_key = raw.get("future_label_key")
    reference_time_key = raw.get("reference_time_key")
    stage_order_map = raw.get("stage_order_map")
    return modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map


def load_or_rebuild_modality(path: Path, global_path: str, cells: Sequence[str]) -> ad.AnnData:
    if path.exists():
        adata = sc.read_h5ad(str(path))
        return adata
    global_adata = sc.read_h5ad(str(global_path))
    common = [c for c in cells if c in global_adata.obs_names]
    if not common:
        raise ValueError(f"No requested cells found in global file: {global_path}")
    return global_adata[common].copy()


def build_multimodal_ratio_dataset(split_dir: os.PathLike, modality_specs: Dict[str, ModalitySpec], label_key: Optional[str] = None, standardize: bool = True):
    split_dir = Path(split_dir)
    split_info = load_json(split_dir / "split_info.json")
    train_cells = split_info["train_cells"]
    val_cells = split_info.get("val_cells", split_info.get("val_query_cells", split_info.get("val_query_atac_cells", [])))
    if len(train_cells) == 0:
        raise ValueError(f"{split_dir}: train_cells empty")
    if len(val_cells) == 0:
        raise ValueError(f"{split_dir}: val_cells empty")

    train_adatas: Dict[str, ad.AnnData] = {}
    val_adatas: Dict[str, ad.AnnData] = {}
    feature_sets_train: List[List[str]] = []
    feature_sets_val: List[List[str]] = []

    for m, spec in modality_specs.items():
        train_candidates = [split_dir / f"train_{m}.h5ad"]
        val_candidates = [split_dir / f"val_{m}.h5ad"]
        if m in ["rna", "s"]:
            train_candidates.extend([split_dir / "train_rna_ref.h5ad"])
            val_candidates.extend([split_dir / "val_true_rna.h5ad"])
        elif m == "atac":
            train_candidates.extend([split_dir / "train_atac_activity.h5ad", split_dir / "train_atac_full.h5ad"])
            val_candidates.extend([split_dir / "val_atac_activity.h5ad", split_dir / "val_query_atac.h5ad"])

        train_path = next((p for p in train_candidates if p.exists()), train_candidates[0])
        val_path = next((p for p in val_candidates if p.exists()), val_candidates[0])

        train_adatas[m] = load_or_rebuild_modality(train_path, spec.path, train_cells)
        val_adatas[m] = load_or_rebuild_modality(val_path, spec.path, val_cells)
        feature_sets_train.append(train_adatas[m].var_names.tolist())
        feature_sets_val.append(val_adatas[m].var_names.tolist())

    common_features = intersect_strings(*feature_sets_train, *feature_sets_val)
    if len(common_features) < 32:
        raise ValueError(f"Too few common features across selected modalities: {len(common_features)}")

    train_union = sorted(union_strings(*[a.obs_names.tolist() for a in train_adatas.values()]))
    val_union = sorted(union_strings(*[a.obs_names.tolist() for a in val_adatas.values()]))
    train_pos = {c: i for i, c in enumerate(train_union)}
    val_pos = {c: i for i, c in enumerate(val_union)}

    train_arrays: Dict[str, np.ndarray] = {}
    train_masks: Dict[str, np.ndarray] = {}
    val_arrays: Dict[str, np.ndarray] = {}
    val_masks: Dict[str, np.ndarray] = {}
    stats: Dict[str, dict] = {}

    for m, spec in modality_specs.items():
        tr = train_adatas[m][:, common_features].copy()
        va = val_adatas[m][:, common_features].copy()
        x_tr = maybe_log1p(to_dense(tr.X), spec.assume_logged)
        x_va = maybe_log1p(to_dense(va.X), spec.assume_logged)

        tr_full = np.zeros((len(train_union), len(common_features)), dtype=np.float32)
        va_full = np.zeros((len(val_union), len(common_features)), dtype=np.float32)
        tr_mask = np.zeros(len(train_union), dtype=np.float32)
        va_mask = np.zeros(len(val_union), dtype=np.float32)

        for j, c in enumerate(tr.obs_names.tolist()):
            tr_full[train_pos[c]] = x_tr[j]
            tr_mask[train_pos[c]] = 1.0
        for j, c in enumerate(va.obs_names.tolist()):
            va_full[val_pos[c]] = x_va[j]
            va_mask[val_pos[c]] = 1.0

        if standardize and tr_mask.sum() > 0:
            mu = tr_full[tr_mask > 0].mean(axis=0, keepdims=True)
            sd = tr_full[tr_mask > 0].std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            tr_full = ((tr_full - mu) / sd).astype(np.float32)
            va_full = ((va_full - mu) / sd).astype(np.float32)
            stats[m] = {"mean": mu.squeeze(0).astype(np.float32), "std": sd.squeeze(0).astype(np.float32)}
        else:
            stats[m] = {"mean": np.zeros(len(common_features), dtype=np.float32), "std": np.ones(len(common_features), dtype=np.float32)}

        train_arrays[m] = tr_full
        train_masks[m] = tr_mask
        val_arrays[m] = va_full
        val_masks[m] = va_mask

    labels = None
    label_ref = next(iter(val_adatas.values()))
    if label_key is not None and label_key in label_ref.obs.columns:
        try:
            labels = label_ref.obs.loc[val_union, label_key].astype(str).to_numpy()
        except Exception:
            labels = None

    ds_train = MultimodalRatioDataset(train_arrays, train_masks, train_union)
    meta = {
        "split_info": split_info,
        "train_cell_ids": np.asarray(train_union),
        "val_cell_ids": np.asarray(val_union),
        "common_features": np.asarray(common_features),
        "val_modalities": val_arrays,
        "val_masks": val_masks,
        "labels": labels,
        "label_key": label_key,
        "stats": stats,
    }
    return ds_train, meta, val_adatas


# =============================================================================
# Model
# =============================================================================

class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.1, last_activation: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or last_activation:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SplitVariationalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], shared_dim: int, private_dim: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = MLP([input_dim, *hidden_dims], dropout=dropout, last_activation=True)
        hdim = hidden_dims[-1]
        self.mu_shared = nn.Linear(hdim, shared_dim)
        self.lv_shared = nn.Linear(hdim, shared_dim)
        self.mu_private = nn.Linear(hdim, private_dim)
        self.lv_private = nn.Linear(hdim, private_dim)

    def _sample(self, mu: torch.Tensor, lv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lv = lv.clamp(-8, 8)
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, lv

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        hs, mus, lvs = self._sample(self.mu_shared(h), self.lv_shared(h))
        hp, mup, lvp = self._sample(self.mu_private(h), self.lv_private(h))
        return hs, hp, mus, lvs, mup, lvp


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = MLP([input_dim, *hidden_dims, output_dim], dropout=dropout, last_activation=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MaskAwareFusion(nn.Module):
    def __init__(self, shared_dim: int, n_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_modalities * shared_dim + n_modalities, max(shared_dim, 32)),
            nn.GELU(),
            nn.Linear(max(shared_dim, 32), n_modalities),
        )

    def forward(self, z_stack: torch.Tensor, mask_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, m, d = z_stack.shape
        gate_in = torch.cat([z_stack.reshape(b, m * d), mask_stack], dim=1)
        logits = self.gate(gate_in)
        logits = logits.masked_fill(mask_stack <= 0, -1e9)
        weights = F.softmax(logits, dim=1)
        h = torch.sum(weights.unsqueeze(-1) * z_stack, dim=1)
        return h, weights


class TimeHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h))


class DirectedLagModule(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, current_h: torch.Tensor, source_bank: torch.Tensor):
        b, k, d = source_bank.shape
        cur = current_h.unsqueeze(1).expand(b, k, d)
        diff = cur - source_bank
        score_in = torch.cat([cur, source_bank, diff], dim=-1)
        alpha = F.softmax(self.score(score_in).squeeze(-1), dim=1)
        ctx = torch.sum(alpha.unsqueeze(-1) * source_bank, dim=1)
        delta = self.update(torch.cat([current_h, ctx], dim=1))
        return current_h + delta, alpha


class TRACE(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        prediction_pairs: Sequence[Tuple[str, str]],
        lag_edges: Sequence[LagEdge],
        shared_dim: int = 32,
        private_dim: int = 16,
        enc_hidden: Sequence[int] = (256, 128),
        dec_hidden: Sequence[int] = (128, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_names = sorted(input_dims.keys())
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.encoders = nn.ModuleDict({
            m: SplitVariationalEncoder(d, enc_hidden, shared_dim, private_dim, dropout)
            for m, d in input_dims.items()
        })
        self.current_decoders = nn.ModuleDict({
            m: Decoder(shared_dim + private_dim, dec_hidden, d, dropout)
            for m, d in input_dims.items()
        })
        self.prediction_heads = nn.ModuleDict({
            f"{src}__to__{tgt}": Decoder(shared_dim + private_dim, dec_hidden, input_dims[tgt], dropout)
            for src, tgt in prediction_pairs
        })
        self.fusion = MaskAwareFusion(shared_dim=shared_dim, n_modalities=len(self.modality_names))
        self.shared_refine = MLP([shared_dim, shared_dim, shared_dim], dropout=dropout, last_activation=False)
        self.time_head = TimeHead(shared_dim)
        self.lag_modules = nn.ModuleDict({
            f"{e.source}__to__{e.target}": DirectedLagModule(shared_dim) for e in lag_edges
        })
        self.edge_weights = {f"{e.source}__to__{e.target}": float(e.weight) for e in lag_edges}
        self.target_to_edges: Dict[str, List[str]] = defaultdict(list)
        for e in lag_edges:
            self.target_to_edges[e.target].append(f"{e.source}__to__{e.target}")

    def encode(self, modalities: Dict[str, torch.Tensor]):
        shared, private = {}, {}
        mu_s, lv_s, mu_p, lv_p = {}, {}, {}, {}
        for m in self.modality_names:
            hs, hp, mus, lvs, mup, lvp = self.encoders[m](modalities[m])
            shared[m], private[m] = hs, hp
            mu_s[m], lv_s[m], mu_p[m], lv_p[m] = mus, lvs, mup, lvp
        return shared, private, mu_s, lv_s, mu_p, lv_p

    def fuse_shared(self, shared: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]):
        z_stack = torch.stack([shared[m] for m in self.modality_names], dim=1)
        mask_stack = torch.stack([masks[m] for m in self.modality_names], dim=1)
        h, fusion_weights = self.fusion(z_stack, mask_stack)
        h = self.shared_refine(h)
        return h, fusion_weights

    def decode_current(self, h_shared: torch.Tensor, private: Dict[str, torch.Tensor]):
        out = {}
        for m in self.modality_names:
            out[m] = self.current_decoders[m](torch.cat([h_shared, private[m]], dim=1))
        return out

    def decode_prediction(self, h_shared: torch.Tensor, private: Dict[str, torch.Tensor]):
        out = {}
        for key, head in self.prediction_heads.items():
            src, tgt = key.split("__to__")
            out[key] = head(torch.cat([h_shared, private[src]], dim=1))
        return out

    def apply_directed_lag(self, h: torch.Tensor, lag_banks: Dict[str, torch.Tensor]):
        lag_latent: Dict[str, torch.Tensor] = {}
        lag_alpha: Dict[str, torch.Tensor] = {}
        for target in self.modality_names:
            edge_names = self.target_to_edges.get(target, [])
            proposals = []
            weights = []
            for edge_name in edge_names:
                source = edge_name.split("__to__")[0]
                if source not in lag_banks:
                    continue
                prop, alpha = self.lag_modules[edge_name](h, lag_banks[source])
                proposals.append(prop)
                weights.append(self.edge_weights[edge_name])
                lag_alpha[edge_name] = alpha
            if proposals:
                w = torch.tensor(weights, dtype=h.dtype, device=h.device)
                w = w / w.sum().clamp_min(1e-8)
                stacked = torch.stack(proposals, dim=1)
                lag_latent[target] = torch.sum(stacked * w.view(1, -1, 1), dim=1)
            else:
                lag_latent[target] = h
        return lag_latent, lag_alpha

    def forward(self, modalities: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], lag_banks: Optional[Dict[str, torch.Tensor]] = None):
        shared, private, mu_s, lv_s, mu_p, lv_p = self.encode(modalities)
        h, fusion_weights = self.fuse_shared(shared, masks)
        t = self.time_head(h)
        current_rec = self.decode_current(h, private)
        pred_rec = self.decode_prediction(h, private)
        out = {
            "shared": shared,
            "private": private,
            "mu_s": mu_s,
            "lv_s": lv_s,
            "mu_p": mu_p,
            "lv_p": lv_p,
            "h": h,
            "t": t,
            "fusion_weights": fusion_weights,
            "current_rec": current_rec,
            "pred_rec": pred_rec,
        }
        if lag_banks is not None:
            lag_latent, lag_alpha = self.apply_directed_lag(h, lag_banks)
            out["lag_latent"] = lag_latent
            out["lag_alpha"] = lag_alpha
        return out


# =============================================================================
# Training
# =============================================================================

@dataclasses.dataclass
class TrainConfig:
    output_dir: str
    epochs_phase1: int = 80
    epochs_phase2: int = 60
    epochs_phase3: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    shared_dim: int = 32
    private_dim: int = 16
    dropout: float = 0.1
    mmd_weight: float = 0.5
    align_weight: float = 1.0
    recon_weight: float = 1.0
    pred_weight: float = 2.0
    kl_weight: float = 1e-3
    topo_weight: float = 0.05
    time_cons_weight: float = 0.1
    forward_weight: float = 0.1
    lag_weight: float = 0.3
    lag_entropy_weight: float = 0.05
    grad_clip: float = 5.0
    device: str = "cuda"
    num_workers: int = 0
    save_every: int = 25
    topo_k: int = 10
    topo_min_local: int = 2
    bank_warmup_epochs: int = 10
    lag_bins: int = 4


@dataclasses.dataclass
class FreezePlan:
    freeze_encoders: bool = False
    freeze_fusion: bool = False
    freeze_prediction: bool = False
    freeze_dynamics: bool = False


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def apply_freeze_plan(model: TRACE, plan: FreezePlan) -> None:
    set_requires_grad(model.encoders, not plan.freeze_encoders)
    set_requires_grad(model.fusion, not plan.freeze_fusion)
    set_requires_grad(model.shared_refine, not plan.freeze_fusion)
    set_requires_grad(model.current_decoders, not plan.freeze_encoders)
    set_requires_grad(model.prediction_heads, not plan.freeze_prediction)
    set_requires_grad(model.time_head, not plan.freeze_dynamics)
    set_requires_grad(model.lag_modules, not plan.freeze_dynamics)


def build_alignment_losses(out, modality_specs, alignment_pairs, masks):
    shared = out["shared"]
    device = next(iter(shared.values())).device
    mmd_total = torch.tensor(0.0, device=device)
    align_total = torch.tensor(0.0, device=device)
    pairs = list(alignment_pairs) if alignment_pairs else [
        (a, b) for i, a in enumerate(sorted(shared.keys()))
        for b in sorted(shared.keys())[i + 1:]
    ]
    for a, b in pairs:
        paired_mask = (masks[a] > 0) & (masks[b] > 0)
        if paired_mask.sum() < 2:
            continue
        za = shared[a][paired_mask]
        zb = shared[b][paired_mask]
        mmd_ab = compute_mmd_rbf(za, zb, sigma=1.0)
        cost = torch.cdist(za, zb, p=2).pow(2)
        T = sinkhorn_transport(cost, epsilon=0.05, n_iter=30)
        align_ab = weighted_pair_loss(za, zb, T)
        w = 0.5 * (modality_specs[a].align_weight + modality_specs[b].align_weight)
        mmd_total = mmd_total + w * mmd_ab
        align_total = align_total + w * align_ab
    return mmd_total, align_total


def build_kl_loss(out, modality_names):
    total = 0.0
    for m in modality_names:
        total = total + kl_div(out["mu_s"][m], out["lv_s"][m])
        total = total + kl_div(out["mu_p"][m], out["lv_p"][m])
    return total / max(len(modality_names), 1)


def build_reconstruction_loss(out, x, masks, modality_names):
    recon = torch.tensor(0.0, device=next(iter(x.values())).device)
    for m in modality_names:
        recon = recon + masked_mse(out["current_rec"][m], x[m], masks[m])
    return recon


def build_prediction_loss(out, x, masks, prediction_pairs):
    loss = torch.tensor(0.0, device=next(iter(x.values())).device)
    n_used = 0
    for src, tgt in prediction_pairs:
        key = f"{src}__to__{tgt}"
        target_mask = masks[tgt]
        source_mask = masks[src]
        paired_mask = ((target_mask > 0) & (source_mask > 0)).float()
        if paired_mask.sum() < 1:
            continue
        loss = loss + masked_mse(out["pred_rec"][key], x[tgt], paired_mask)
        n_used += 1
    if n_used == 0:
        return torch.tensor(0.0, device=next(iter(x.values())).device)
    return loss / n_used


def build_global_lag_artifacts(model: TRACE, ds: MultimodalRatioDataset, cfg: TrainConfig, device: torch.device):
    model.eval()
    with torch.no_grad():
        all_mod_t = {m: torch.from_numpy(ds.modalities[m]).to(device) for m in ds.modality_names}
        all_mask_t = {m: torch.from_numpy(ds.masks[m]).to(device) for m in ds.modality_names}
        out_all = model(all_mod_t, all_mask_t)
        h_np = out_all["h"].detach().cpu().numpy()
        t_np = out_all["t"].detach().cpu().numpy().reshape(-1)
        global_knn = knn_graph_indices(h_np, n_neighbors=cfg.topo_k + 1)
        lag_banks_np = {
            m: build_lag_bank(h_np, t_np, source_mask=ds.masks[m] > 0, max_lag_bins=cfg.lag_bins, k_per_bin=4)
            for m in ds.modality_names
        }
    return h_np, t_np, global_knn, lag_banks_np


def run_training_phase(
    model: TRACE,
    ds: MultimodalRatioDataset,
    modality_specs: Dict[str, ModalitySpec],
    alignment_pairs,
    prediction_pairs,
    cfg: TrainConfig,
    writer: SummaryWriter,
    phase_name: str,
    epochs: int,
    start_epoch: int,
    freeze_plan: FreezePlan,
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    apply_freeze_plan(model, freeze_plan)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    history = []
    best_loss = math.inf
    best_ckpt = None
    global_knn = None
    lag_banks_np = None

    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch - 1
        if phase_name == "phase3":
            if local_epoch > cfg.bank_warmup_epochs:
                _, _, global_knn, lag_banks_np = build_global_lag_artifacts(model, ds, cfg, device)
            else:
                with torch.no_grad():
                    all_mod_t = {m: torch.from_numpy(ds.modalities[m]).to(device) for m in ds.modality_names}
                    all_mask_t = {m: torch.from_numpy(ds.masks[m]).to(device) for m in ds.modality_names}
                    out_all = model(all_mod_t, all_mask_t)
                    global_knn = knn_graph_indices(out_all["h"].detach().cpu().numpy(), n_neighbors=cfg.topo_k + 1)
                    lag_banks_np = None

        model.train()
        epoch_stats = defaultdict(float)

        for batch in loader:
            batch_idx_np = batch["index"].detach().cpu().numpy()
            x = {m: batch["modalities"][m].to(device) for m in ds.modality_names}
            masks = {m: batch["masks"][m].to(device) for m in ds.modality_names}

            lag_banks_batch = None
            if phase_name == "phase3" and lag_banks_np is not None:
                lag_banks_batch = {
                    m: torch.from_numpy(lag_banks_np[m][batch_idx_np]).to(device)
                    for m in ds.modality_names
                }

            out = model(x, masks, lag_banks=lag_banks_batch)
            recon = build_reconstruction_loss(out, x, masks, ds.modality_names)
            pred = build_prediction_loss(out, x, masks, prediction_pairs)
            kl = build_kl_loss(out, ds.modality_names)
            mmd, align = build_alignment_losses(out, modality_specs, alignment_pairs, masks)

            topo = torch.tensor(0.0, device=device)
            time_cons = torch.tensor(0.0, device=device)
            forward = torch.tensor(0.0, device=device)
            lag_entropy = torch.tensor(0.0, device=device)
            lag_cons = torch.tensor(0.0, device=device)

            if phase_name == "phase3":
                if global_knn is not None:
                    knn_local = localize_global_knn(batch_idx_np, global_knn, min_neighbors=cfg.topo_min_local)
                    topo = smoothness_loss(out["h"], knn_local)
                    time_cons = time_consistency_loss(out["h"], out["t"], k=min(5, knn_local.shape[1]))
                    forward = forward_margin_loss(out["h"], out["t"], knn_local, margin=0.02)

                if "lag_alpha" in out:
                    lag_entropy_terms = []
                    lag_cons_terms = []
                    for edge_name, alpha in out["lag_alpha"].items():
                        lag_entropy_terms.append(entropy_loss(alpha))
                        target = edge_name.split("__to__")[1]
                        if target in out["lag_latent"]:
                            lag_cons_terms.append(F.mse_loss(out["lag_latent"][target], out["h"]))
                    if lag_entropy_terms:
                        lag_entropy = torch.stack(lag_entropy_terms).mean()
                    if lag_cons_terms:
                        lag_cons = torch.stack(lag_cons_terms).mean()

            if phase_name == "phase1":
                loss = (
                    cfg.recon_weight * recon
                    + cfg.mmd_weight * mmd
                    + cfg.align_weight * align
                    + cfg.kl_weight * kl
                )
            elif phase_name == "phase2":
                loss = (
                    cfg.recon_weight * recon
                    + cfg.pred_weight * pred
                    + 0.5 * cfg.mmd_weight * mmd
                    + 0.5 * cfg.align_weight * align
                    + cfg.kl_weight * kl
                )
            else:
                loss = (
                    cfg.recon_weight * recon
                    + cfg.pred_weight * pred
                    + 0.5 * cfg.mmd_weight * mmd
                    + 0.5 * cfg.align_weight * align
                    + cfg.kl_weight * kl
                    + cfg.topo_weight * topo
                    + cfg.time_cons_weight * time_cons
                    + cfg.forward_weight * forward
                    + cfg.lag_weight * lag_cons
                    + cfg.lag_entropy_weight * lag_entropy
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            epoch_stats["loss"] += float(loss.detach().cpu())
            epoch_stats["recon"] += float(recon.detach().cpu())
            epoch_stats["pred"] += float(pred.detach().cpu())
            epoch_stats["mmd"] += float(mmd.detach().cpu())
            epoch_stats["align"] += float(align.detach().cpu())
            epoch_stats["kl"] += float(kl.detach().cpu())
            epoch_stats["topo"] += float(topo.detach().cpu())
            epoch_stats["time_cons"] += float(time_cons.detach().cpu())
            epoch_stats["forward"] += float(forward.detach().cpu())
            epoch_stats["lag_cons"] += float(lag_cons.detach().cpu())
            epoch_stats["lag_entropy"] += float(lag_entropy.detach().cpu())

        denom = max(len(loader), 1)
        row = {k: v / denom for k, v in epoch_stats.items()}
        row["epoch"] = epoch
        row["phase"] = phase_name
        history.append(row)

        for k, v in row.items():
            if k not in ["epoch", "phase"]:
                writer.add_scalar(f"{phase_name}/{k}", v, epoch)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (local_epoch % cfg.save_every == 0) or (local_epoch == epochs):
            ckpt_path = Path(cfg.output_dir) / f"{phase_name}_epoch_{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "phase": phase_name}, ckpt_path)

        print(
            f"[{phase_name}] epoch={epoch} "
            f"loss={row['loss']:.4f} recon={row['recon']:.4f} pred={row['pred']:.4f} "
            f"align={row['align']:.4f} topo={row['topo']:.4f} lag={row['lag_cons']:.4f}"
        )

    if best_ckpt is not None:
        model.load_state_dict(best_ckpt)
    return model, history, start_epoch + epochs


# =============================================================================
# Inference and evaluation
# =============================================================================

def infer_on_arrays(model: TRACE, arrays: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], device: torch.device):
    model.eval()
    with torch.no_grad():
        x = {m: torch.from_numpy(arrays[m]).to(device) for m in sorted(arrays.keys())}
        ms = {m: torch.from_numpy(masks[m]).to(device) for m in sorted(masks.keys())}
        out = model(x, ms)
        pred = {k: v.detach().cpu().numpy() for k, v in out["pred_rec"].items()}
        current = {k: v.detach().cpu().numpy() for k, v in out["current_rec"].items()}
        h = out["h"].detach().cpu().numpy()
        t = out["t"].detach().cpu().numpy().reshape(-1)
        shared = {k: v.detach().cpu().numpy() for k, v in out["shared"].items()}
        private = {k: v.detach().cpu().numpy() for k, v in out["private"].items()}
    return {"pred": pred, "current": current, "h": h, "t": t, "shared": shared, "private": private}


def get_eval_prediction_pair(prediction_pairs, preferred_src="atac", preferred_tgt="rna"):
    if len(prediction_pairs) == 0:
        raise ValueError("prediction_pairs is empty, cannot evaluate T1/T2.")
    for src, tgt in prediction_pairs:
        if src == preferred_src and tgt == preferred_tgt:
            return src, tgt
    return prediction_pairs[0]


def build_pred_true_for_eval(
    meta,
    infer_res,
    val_adatas,
    src: str,
    tgt: str,
    output_dir: Path,
):
    key = f"{src}__to__{tgt}"

    if key not in infer_res["pred"]:
        raise KeyError(f"Prediction key not found in infer_res['pred']: {key}")

    if tgt not in val_adatas:
        raise KeyError(f"Target modality '{tgt}' not found in val_adatas")

    pred_all = infer_res["pred"][key]
    pred_cells_all = np.asarray(meta["val_cell_ids"])
    pred_features_all = np.asarray(meta["common_features"])

    pred_adata_all = ad.AnnData(
        X=pred_all,
        obs=pd.DataFrame(index=pred_cells_all.copy()),
        var=pd.DataFrame(index=pred_features_all.copy()),
    )

    true_adata = val_adatas[tgt].copy()

    common_cells = get_common_names(
        pred_adata_all.obs_names.tolist(),
        true_adata.obs_names.tolist(),
    )
    common_features = get_common_names(
        pred_adata_all.var_names.tolist(),
        true_adata.var_names.tolist(),
    )

    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(f"Too few common features for evaluation: {len(common_features)}")
    if len(common_cells) == 0:
        raise ValueError("No common cells between predicted target and true target.")

    pred_val = pred_adata_all[common_cells, common_features].copy()
    true_val = true_adata[common_cells, common_features].copy()

    save_h5ad_safe(pred_val, output_dir / f"pred_{src}_to_{tgt}_val.h5ad")
    save_h5ad_safe(true_val, output_dir / f"true_{tgt}_val.h5ad")

    return pred_val, true_val, common_cells, common_features


def evaluate_t2_like_scmrdr(pred_val, true_val, outdir: Path):
    common_features = get_common_names(
        pred_val.var_names.tolist(),
        true_val.var_names.tolist(),
    )
    common_cells = get_common_names(
        pred_val.obs_names.tolist(),
        true_val.obs_names.tolist(),
    )

    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(f"Too few common features for T2 evaluation: {len(common_features)}")
    if len(common_cells) == 0:
        raise ValueError("No common cells for T2 evaluation.")

    pred = pred_val[common_cells, common_features].copy()
    true = true_val[common_cells, common_features].copy()

    pred_mat = to_dense(pred.X).T
    true_mat = to_dense(true.X).T

    cell_cor = np.array(
        [safe_corr(pred_mat[:, i], true_mat[:, i]) for i in range(pred_mat.shape[1])],
        dtype=float
    )
    gene_cor = np.array(
        [safe_corr(pred_mat[i, :], true_mat[i, :]) for i in range(pred_mat.shape[0])],
        dtype=float
    )

    mse = float(np.nanmean((pred_mat - true_mat) ** 2))
    rmse_val = float(np.sqrt(mse))

    t2_metrics = pd.DataFrame({
        "metric": ["pearson_cell_mean", "pearson_gene_mean", "rmse"],
        "value": [
            float(np.nanmean(cell_cor)),
            float(np.nanmean(gene_cor)),
            rmse_val,
        ],
    })
    t2_metrics.to_csv(outdir / "T2_metrics.csv", index=False)

    pd.DataFrame({
        "cell": common_cells,
        "cellwise_pearson": cell_cor,
    }).to_csv(outdir / "T2_cellwise_pearson.csv", index=False)

    pd.DataFrame({
        "gene": common_features,
        "genewise_pearson": gene_cor,
    }).to_csv(outdir / "T2_genewise_pearson.csv", index=False)

    return t2_metrics, pred_mat, true_mat, common_cells, common_features


def evaluate_t1_from_true_pred_like_scmrdr(
    pred_mat: np.ndarray,
    true_mat: np.ndarray,
    common_cells,
    common_features,
    outdir: Path,
    seed: int = 2022,
):
    true_cells_by_features = true_mat.T
    pred_cells_by_features = pred_mat.T
    mix = np.vstack([true_cells_by_features, pred_cells_by_features])

    n_components = min(N_PCS_EVAL, mix.shape[0] - 1, mix.shape[1])
    if n_components < 2:
        raise ValueError("PCA components < 2, cannot evaluate T1.")

    pca = PCA(n_components=n_components, random_state=seed)
    emb = pca.fit_transform(mix)

    n_q = true_cells_by_features.shape[0]
    emb_true = emb[:n_q, :]
    emb_pred = emb[n_q:, :]

    dist_mat = pairwise_distances(
        np.vstack([emb_true, emb_pred]),
        metric="euclidean"
    )
    cross_dist = dist_mat[:n_q, n_q:(2 * n_q)]

    paired_dist = np.diag(cross_dist).astype(float)
    foscttm_each = np.array(
        [np.mean(cross_dist[i, :] < paired_dist[i]) for i in range(n_q)],
        dtype=float
    )

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
    t1_metrics.to_csv(outdir / "T1_metrics.csv", index=False)

    pd.DataFrame({
        "cell": list(common_cells),
        "FOSCTTM": foscttm_each,
        "paired_dist": paired_dist,
        "Top1_match": nn_match.astype(bool),
    }).to_csv(outdir / "T1_per_cell_metrics.csv", index=False)

    pca_df = pd.DataFrame(
        emb[:, :min(5, emb.shape[1])],
        columns=[f"PC{i+1}" for i in range(min(5, emb.shape[1]))]
    )
    pca_df["group"] = ["True_Target"] * n_q + ["Pred_Target"] * n_q
    pca_df["cell"] = [f"true_{c}" for c in common_cells] + [f"pred_{c}" for c in common_cells]
    pca_df.to_csv(outdir / "pca_true_pred_coords.csv", index=False)

    return t1_metrics


def build_eval_anndata(meta, infer_res, val_adatas, label_key: Optional[str], future_label_key: Optional[str]):
    adata_ref = next(iter(val_adatas.values())).copy()
    adata = ad.AnnData(X=infer_res["h"], obs=adata_ref.obs.loc[meta["val_cell_ids"]].copy())
    adata.obs_names = meta["val_cell_ids"].copy()
    adata.obsm["X_trace"] = infer_res["h"]
    adata.obs["trace_time"] = infer_res["t"]
    if label_key is not None and label_key in adata_ref.obs.columns:
        adata.obs[label_key] = adata_ref.obs.loc[adata.obs_names, label_key].astype(str).values
    if future_label_key is not None and future_label_key in adata_ref.obs.columns:
        adata.obs[future_label_key] = adata_ref.obs.loc[adata.obs_names, future_label_key].astype(str).values
    return adata


def evaluate_t3_t4(meta, infer_res, val_adatas, label_key, future_label_key, reference_time_key, stage_order_map, output_dir: Path):
    adata_eval = build_eval_anndata(meta, infer_res, val_adatas, label_key, future_label_key)
    lag_score = None
    P = build_transition_matrix(
        adata_eval.obsm["X_trace"],
        adata_eval.obs["trace_time"].to_numpy(dtype=float),
        lag_score,
        n_neighbors=30
    )
    adata_eval.obsp["trace_transition_matrix"] = sp.csr_matrix(P)

    try:
        t3 = compute_t3_metrics(
            adata_eval,
            time_key="trace_time",
            reference_time_key=reference_time_key,
            label_key=label_key,
            stage_order_map=stage_order_map,
        )
    except Exception as e:
        t3 = {
            "POA": np.nan,
            "POAUC": np.nan,
            "n_eval": 0,
            "status": f"failed:{type(e).__name__}:{e}",
        }

    try:
        if future_label_key is not None:
            t4 = compute_t4_metrics(adata_eval, P, lineage_key=future_label_key)
        else:
            t4 = {
                "F1": np.nan,
                "AUROC": np.nan,
                "n_eval": 0,
                "status": "skipped_no_future_label_key",
            }
    except Exception as e:
        t4 = {
            "F1": np.nan,
            "AUROC": np.nan,
            "n_eval": 0,
            "status": f"failed:{type(e).__name__}:{e}",
        }

    pd.DataFrame([t3]).to_csv(output_dir / "T3_metrics.csv", index=False)
    pd.DataFrame([t4]).to_csv(output_dir / "T4_metrics.csv", index=False)
    save_h5ad_safe(adata_eval, output_dir / "trace_eval_embeddings.h5ad")
    np.save(output_dir / "trace_transition_matrix.npy", P)
    return t3, t4


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TRACE for multimodal alignment and adaptive lag dynamics")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs-phase1", type=int, default=50)
    parser.add_argument("--epochs-phase2", type=int, default=50)
    parser.add_argument("--epochs-phase3", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument("--private-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mmd-weight", type=float, default=0.5)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--pred-weight", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--topo-weight", type=float, default=0.05)
    parser.add_argument("--time-cons-weight", type=float, default=0.1)
    parser.add_argument("--forward-weight", type=float, default=0.1)
    parser.add_argument("--lag-weight", type=float, default=0.3)
    parser.add_argument("--lag-entropy-weight", type=float, default=0.05)
    parser.add_argument("--topo-k", type=int, default=10)
    parser.add_argument("--lag-bins", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    cfg = TrainConfig(
        output_dir=args.output_dir,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        epochs_phase3=args.epochs_phase3,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        dropout=args.dropout,
        mmd_weight=args.mmd_weight,
        align_weight=args.align_weight,
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        kl_weight=args.kl_weight,
        topo_weight=args.topo_weight,
        time_cons_weight=args.time_cons_weight,
        forward_weight=args.forward_weight,
        lag_weight=args.lag_weight,
        lag_entropy_weight=args.lag_entropy_weight,
        device=args.device,
        topo_k=args.topo_k,
        lag_bins=args.lag_bins,
    )
    save_json(dataclasses.asdict(cfg), Path(cfg.output_dir) / "train_config.json")

    modalities, alignment_pairs, prediction_pairs, lag_edges, label_key, future_label_key, reference_time_key, stage_order_map = parse_manifest(args.manifest)
    ds_train, meta, val_adatas = build_multimodal_ratio_dataset(
        args.split_dir, modalities, label_key=label_key, standardize=True
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=str(Path(cfg.output_dir) / "tensorboard"))

    input_dims = {m: ds_train.modalities[m].shape[1] for m in ds_train.modality_names}
    model = TRACE(
        input_dims=input_dims,
        prediction_pairs=prediction_pairs,
        lag_edges=lag_edges,
        shared_dim=cfg.shared_dim,
        private_dim=cfg.private_dim,
        dropout=cfg.dropout,
    ).to(device)

    epoch_ptr = 1
    model, hist1, epoch_ptr = run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase1", epochs=cfg.epochs_phase1, start_epoch=epoch_ptr,
        freeze_plan=FreezePlan(
            freeze_encoders=False,
            freeze_fusion=False,
            freeze_prediction=False,
            freeze_dynamics=True
        ),
    )
    model, hist2, epoch_ptr = run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase2", epochs=cfg.epochs_phase2, start_epoch=epoch_ptr,
        freeze_plan=FreezePlan(
            freeze_encoders=False,
            freeze_fusion=False,
            freeze_prediction=False,
            freeze_dynamics=True
        ),
    )
    model, hist3, epoch_ptr = run_training_phase(
        model, ds_train, modalities, alignment_pairs, prediction_pairs, cfg, writer,
        phase_name="phase3", epochs=cfg.epochs_phase3, start_epoch=epoch_ptr,
        freeze_plan=FreezePlan(
            freeze_encoders=False,
            freeze_fusion=False,
            freeze_prediction=False,
            freeze_dynamics=False
        ),
    )

    hist = pd.concat([pd.DataFrame(hist1), pd.DataFrame(hist2), pd.DataFrame(hist3)], axis=0, ignore_index=True)
    hist.to_csv(Path(cfg.output_dir) / "training_history.csv", index=False)
    torch.save({"model": model.state_dict()}, Path(cfg.output_dir) / "trace_final.pt")

    infer_res = infer_on_arrays(model, meta["val_modalities"], meta["val_masks"], device)
    np.save(Path(cfg.output_dir) / "val_h.npy", infer_res["h"])
    np.save(Path(cfg.output_dir) / "val_time.npy", infer_res["t"])
    save_npz_dict(infer_res["shared"], Path(cfg.output_dir) / "val_shared_latents.npz")
    save_npz_dict(infer_res["private"], Path(cfg.output_dir) / "val_private_latents.npz")

    # -------------------------------------------------
    # T1 / T2: scMRDR-style evaluation
    # -------------------------------------------------
    eval_src, eval_tgt = get_eval_prediction_pair(
        prediction_pairs,
        preferred_src="atac",
        preferred_tgt="rna",
    )

    pred_val, true_val, common_cells, common_features = build_pred_true_for_eval(
        meta=meta,
        infer_res=infer_res,
        val_adatas=val_adatas,
        src=eval_src,
        tgt=eval_tgt,
        output_dir=Path(cfg.output_dir),
    )

    t2_metrics, pred_mat, true_mat, common_cells, common_features = evaluate_t2_like_scmrdr(
        pred_val=pred_val,
        true_val=true_val,
        outdir=Path(cfg.output_dir),
    )

    t1_metrics = evaluate_t1_from_true_pred_like_scmrdr(
        pred_mat=pred_mat,
        true_mat=true_mat,
        common_cells=common_cells,
        common_features=common_features,
        outdir=Path(cfg.output_dir),
        seed=args.seed,
    )

    # -------------------------------------------------
    # T3 / T4: original TRACE logic
    # -------------------------------------------------
    t3, t4 = evaluate_t3_t4(
        meta,
        infer_res,
        val_adatas,
        label_key,
        future_label_key,
        reference_time_key,
        stage_order_map,
        Path(cfg.output_dir),
    )

    t2_map = dict(zip(t2_metrics["metric"], t2_metrics["value"]))
    t1_map = dict(zip(t1_metrics["metric"], t1_metrics["value"]))

    summary = {
        "T1": {
            "eval_pair": f"{eval_src}->{eval_tgt}",
            "query_cells": int(len(common_cells)),
            "common_features": int(len(common_features)),
            "paired_embedding_distance_mean": float(t1_map["paired_embedding_distance_mean"]),
            "paired_embedding_distance_median": float(t1_map["paired_embedding_distance_median"]),
            "FOSCTTM": float(t1_map["FOSCTTM"]),
            "Top1_ACC": float(t1_map["Top1_ACC"]),
            "Top5_ACC": float(t1_map["Top5_ACC"]),
            "Top10_ACC": float(t1_map["Top10_ACC"]),
        },
        "T2": {
            "eval_pair": f"{eval_src}->{eval_tgt}",
            "query_cells": int(len(common_cells)),
            "common_features": int(len(common_features)),
            "pearson_cell_mean": float(t2_map["pearson_cell_mean"]),
            "pearson_gene_mean": float(t2_map["pearson_gene_mean"]),
            "rmse": float(t2_map["rmse"]),
        },
        "T3": t3,
        "T4": t4,
    }

    save_json(summary, Path(cfg.output_dir) / "summary_metrics.json")

    print("\n===== TRACE training finished =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
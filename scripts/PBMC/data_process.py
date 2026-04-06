# 6. Split generation, matching the R logic

def subset_and_copy(
    adata: ad.AnnData,
    cells: Sequence[str],
    features: Sequence[str] | None = None
) -> ad.AnnData:
    out = adata[list(cells)].copy()
    if features is not None:
        out = out[:, list(features)].copy()
    return out


def save_split_h5ads(
    outdir: os.PathLike,
    train_rna_ref: ad.AnnData,
    train_atac_full: ad.AnnData,
    train_atac_activity: ad.AnnData,
    val_query_atac: ad.AnnData,
    val_true_rna: ad.AnnData,
    val_atac_activity: ad.AnnData,
) -> None:
    outdir = Path(outdir)
    safe_write_h5ad(train_rna_ref, outdir / "train_rna_ref.h5ad")
    safe_write_h5ad(train_atac_full, outdir / "train_atac_full.h5ad")
    safe_write_h5ad(train_atac_activity, outdir / "train_atac_activity.h5ad")
    safe_write_h5ad(val_query_atac, outdir / "val_query_atac.h5ad")
    safe_write_h5ad(val_true_rna, outdir / "val_true_rna.h5ad")
    safe_write_h5ad(val_atac_activity, outdir / "val_atac_activity.h5ad")


def generate_splits(
    rna_qc_path: os.PathLike,
    atac_qc_path: os.PathLike,
    atac_gas_path: os.PathLike,
    out_root: os.PathLike,
    seed: int = 1234,
    val_frac: float = 0.2,
    single_fracs: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    rna_keep_prob: float = 0.5,
    min_train_rna_cells: int = 50,
    min_val_query_cells: int = 20,
    min_common_features: int = 50,
) -> pd.DataFrame:
    rna = sc.read_h5ad(str(rna_qc_path))
    atac_peak = sc.read_h5ad(str(atac_qc_path))
    atac_gas = sc.read_h5ad(str(atac_gas_path))

    # 细胞交集：三者都要有
    common_cells = intersect_features(
        rna.obs_names.tolist(),
        atac_peak.obs_names.tolist(),
        atac_gas.obs_names.tolist()
    )
    if len(common_cells) == 0:
        raise ValueError("No common cells shared by RNA_counts_qc, ATAC_counts_qc, and ATAC_gas.")

    rna = rna[common_cells].copy()
    atac_peak = atac_peak[common_cells].copy()
    atac_gas = atac_gas[common_cells].copy()

    # RNA 与 gene activity 的共同基因特征
    common_features = intersect_features(
        rna.var_names.tolist(),
        atac_gas.var_names.tolist()
    )
    if len(common_features) < min_common_features:
        raise ValueError(
            f"Too few common gene-level features between RNA and ATAC_gas: {len(common_features)}"
        )

    rna = rna[:, common_features].copy()
    atac_gas = atac_gas[:, common_features].copy()

    all_cells = np.array(common_cells, dtype=object)
    train_cells, val_cells = train_test_split(
        all_cells,
        test_size=val_frac,
        random_state=seed,
        shuffle=True
    )
    train_cells = np.array(sorted(train_cells.tolist()), dtype=object)
    val_cells = np.array(sorted(val_cells.tolist()), dtype=object)

    ensure_dir(out_root)
    summaries = []

    for sf in single_fracs:
        ratio_label = f"single_{int(round(sf * 100)):03d}"
        outdir = Path(out_root) / ratio_label
        ensure_dir(outdir)

        train_assign = assign_partial_modality(
            train_cells,
            single_frac=sf,
            rna_keep_prob=rna_keep_prob,
            seed=seed + int(round(sf * 1000)) + 11
        )
        val_assign = assign_partial_modality(
            val_cells,
            single_frac=sf,
            rna_keep_prob=rna_keep_prob,
            seed=seed + int(round(sf * 1000)) + 97
        )

        # 和 R 逻辑保持一致
        train_rna_ref_cells = sorted(train_assign.paired_cells + train_assign.rna_only_cells)
        train_atac_ref_cells = sorted(train_assign.paired_cells + train_assign.atac_only_cells)
        val_query_atac_cells = sorted(val_assign.paired_cells + val_assign.atac_only_cells)

        if (
            len(train_rna_ref_cells) < min_train_rna_cells
            or len(train_atac_ref_cells) < min_train_rna_cells
            or len(val_query_atac_cells) < min_val_query_cells
        ):
            meta = {
                "ratio_label": ratio_label,
                "single_frac": sf,
                "status": "skipped",
                "reason": "too_few_cells",
                "train_rna_ref": len(train_rna_ref_cells),
                "train_atac_ref": len(train_atac_ref_cells),
                "val_query_atac": len(val_query_atac_cells),
            }
            with open(outdir / "split_info.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            summaries.append(meta)
            continue

        # -------- 保存各类 split 文件 --------
        train_rna_ref = subset_and_copy(rna, train_rna_ref_cells)

        # train ATAC full: 保留整套训练 ATAC peak-level 数据（沿用你原来的命名）
        train_atac_full = subset_and_copy(atac_peak, train_cells)

        # 新增：train ATAC activity，只取 train ATAC reference cells
        train_atac_activity = subset_and_copy(atac_gas, train_atac_ref_cells)

        # val query peak-level ATAC
        val_query_atac = subset_and_copy(atac_peak, val_query_atac_cells)

        # val true RNA
        val_true_rna = subset_and_copy(rna, val_query_atac_cells)

        # val ATAC activity
        val_atac_activity = subset_and_copy(atac_gas, val_query_atac_cells)

        save_split_h5ads(
            outdir,
            train_rna_ref=train_rna_ref,
            train_atac_full=train_atac_full,
            train_atac_activity=train_atac_activity,
            val_query_atac=val_query_atac,
            val_true_rna=val_true_rna,
            val_atac_activity=val_atac_activity,
        )

        meta = {
            "ratio_label": ratio_label,
            "single_frac": sf,
            "status": "ok",
            "train_cells": train_cells.tolist(),
            "val_cells": val_cells.tolist(),
            "train_paired_cells": train_assign.paired_cells,
            "train_rna_only_cells": train_assign.rna_only_cells,
            "train_atac_only_cells": train_assign.atac_only_cells,
            "val_paired_cells": val_assign.paired_cells,
            "val_rna_only_cells": val_assign.rna_only_cells,
            "val_atac_only_cells": val_assign.atac_only_cells,
            "train_rna_ref_cells": train_rna_ref_cells,
            "train_atac_ref_cells": train_atac_ref_cells,
            "val_query_atac_cells": val_query_atac_cells,
            "common_gene_features": common_features,
            "n_train_rna_ref_cells": len(train_rna_ref_cells),
            "n_train_atac_ref_cells": len(train_atac_ref_cells),
            "n_val_query_atac_cells": len(val_query_atac_cells),
            "n_common_features": len(common_features),
        }
        with open(outdir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        summaries.append({
            "ratio_label": ratio_label,
            "single_frac": sf,
            "status": "ok",
            "train_paired": len(train_assign.paired_cells),
            "train_rna_only": len(train_assign.rna_only_cells),
            "train_atac_only": len(train_assign.atac_only_cells),
            "val_paired": len(val_assign.paired_cells),
            "val_rna_only": len(val_assign.rna_only_cells),
            "val_atac_only": len(val_assign.atac_only_cells),
            "train_rna_ref": len(train_rna_ref_cells),
            "train_atac_ref": len(train_atac_ref_cells),
            "val_query_atac": len(val_query_atac_cells),
            "n_common_features": len(common_features),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(Path(out_root) / "summary_all_ratios.csv", index=False)
    return summary_df
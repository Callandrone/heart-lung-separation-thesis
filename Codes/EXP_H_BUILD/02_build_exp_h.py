#!/usr/bin/env python3
"""
02_build_exp_h.py

Build the final EXP_H_FULL_BOTH benchmark starting from EXP_G_40x40.

Final benchmark
---------------
EXP_H_FULL_BOTH extends EXP_G_40x40 with abnormal acoustic cluster-0 HS sources and ICBHI "both" LS sources in both training and validation.

Important
---------
- Unlike the older EXP_G script, this script DOES add new sources also to validation.
- It keeps the processed split as a single source-disjoint split with fold_no=1.
- ICBHI "both" class is added in the LS-extended variants.
- No rhonchi are used, because ICBHI does not provide a rhonchi class in this selection.
- No HLS-CMDS target-domain source is used.

Default additions
-----------------
- both: train +10, val +5
- HS C0  : train +10, val +3

You can change these from CLI.

Example
-------
python 02_build_exp_h.py --both-train 10 --both-val 5 --c0-train 10 --c0-val 3 --overwrite
"""

from __future__ import annotations

import argparse
import os
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
DEFAULT_SNRS = [-6.0, -3.0, 0.0, 3.0, 6.0]
DEFAULT_HARD_RECORD_IDS = {"c0015", "a0092", "a0135", "e02065", "f0082", "a0167"}

BASE_SELECTED_DIR_NAME = "EXP_G_40x40_SELECTED"
BASE_HS_MANIFEST = "selected_hs_EXP_G_40x40.csv"
BASE_LS_MANIFEST = "selected_ls_EXP_G_40x40.csv"

VARIANTS: Dict[str, Dict[str, object]] = {
    "EXP_H_FULL_BOTH": {
        "selected_dir": "EXP_H_FULL_BOTH_SELECTED",
        "mix_dir": "MIX_EXP_H_FULL_BOTH_4K_SNR5",
        "processed_dir": "experiment_H_full_both",
        "progress_cluster_id": 51003,
        "add_hs_c0": True,
        "add_ls_both": True,
    },
}


import dataset_builder_utils as expb


def parse_snr_values(values: List[str]) -> List[float]:
    out: List[float] = []
    for v in values:
        x = float(str(v).strip())
        if x not in out:
            out.append(x)
    return out


def norm_text(v: object) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def safe_id(text: object) -> str:
    s = norm_text(text)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def class_key(v: object) -> str:
    return norm_text(v).lower().replace(" ", "_")


def norm_path_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace("\\\\", "/", regex=False).str.strip()


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {label}: {sorted(missing)}")


def path_set(df: pd.DataFrame) -> set[str]:
    if "source_path" not in df.columns:
        return set()
    return set(norm_path_series(df["source_path"]))


def record_set(df: pd.DataFrame) -> set[str]:
    if "record_id" not in df.columns:
        return set()
    return set(df["record_id"].dropna().astype(str))


def sort_by_quality(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    df = df.copy()
    if source_kind == "hs":
        cols = [c for c in ["quality_score", "seg_rms_p10", "heart_band_ratio_20_500", "fixed_rms"] if c in df.columns]
    else:
        cols = [c for c in ["quality_score", "seg_rms_p10", "crop_cycle_coverage_sec", "lung_band_ratio_50_1800", "fixed_rms"] if c in df.columns]
    if cols:
        return df.sort_values(cols, ascending=[False] * len(cols)).reset_index(drop=True)
    if "record_id" in df.columns:
        return df.sort_values("record_id").reset_index(drop=True)
    return df.sort_index().reset_index(drop=True)


def read_csv_checked(path: Path, required_cols: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, required_cols, label)
    df["source_path_norm"] = norm_path_series(df["source_path"])
    if "record_id" in df.columns:
        df["record_id"] = df["record_id"].astype(str)
    return df


def read_base_exp_g_manifests(project_root: Path, base_selected_root: Path | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = base_selected_root or (project_root / "dataset" / BASE_SELECTED_DIR_NAME)
    hs_path = root / BASE_HS_MANIFEST
    ls_path = root / BASE_LS_MANIFEST
    if not hs_path.exists():
        raise RuntimeError(f"Base EXP_G HS manifest not found: {hs_path}")
    if not ls_path.exists():
        raise RuntimeError(f"Base EXP_G LS manifest not found: {ls_path}")
    hs = pd.read_csv(hs_path)
    ls = pd.read_csv(ls_path)
    require_columns(hs, ["source_id", "source_path", "output_path", "split", "target_class"], str(hs_path))
    require_columns(ls, ["source_id", "source_path", "output_path", "split", "target_class"], str(ls_path))
    hs["source_path_norm"] = norm_path_series(hs["source_path"])
    ls["source_path_norm"] = norm_path_series(ls["source_path"])
    return hs, ls


def filter_unused(pool: pd.DataFrame, used_paths: set[str], used_records: set[str], exclude_record_ids: set[str] | None = None) -> pd.DataFrame:
    out = pool.copy()
    if "source_path_norm" not in out.columns:
        out["source_path_norm"] = norm_path_series(out["source_path"])
    out = out[~out["source_path_norm"].isin(used_paths)].copy()
    if "record_id" in out.columns and used_records:
        out = out[~out["record_id"].astype(str).isin(used_records)].copy()
    if exclude_record_ids and "record_id" in out.columns:
        out = out[~out["record_id"].astype(str).isin(exclude_record_ids)].copy()
    return out.reset_index(drop=True)


def split_selected_extra(pool: pd.DataFrame, n_train: int, n_val: int, seed: int, label: str, patient_safe: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total = n_train + n_val
    if total <= 0:
        return pd.DataFrame(), pd.DataFrame()
    if len(pool) < total:
        raise RuntimeError(f"Not enough {label} candidates: need={total}, available={len(pool)}")
    pool = sort_by_quality(pool, "ls" if label.upper().startswith("LS") else "hs").head(max(total * 3, total)).copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    pool["_rand"] = rng.random(len(pool))

    if patient_safe and "patient_id" in pool.columns:
        # Prefer validation patients not reused in train; if impossible, fall back to source-disjoint only.
        pool = pool.sort_values(["quality_score" if "quality_score" in pool.columns else "_rand", "_rand"], ascending=[False, True] if "quality_score" in pool.columns else [True, True]).reset_index(drop=True)
        val_rows = []
        used_patients = set()
        for _, row in pool.iterrows():
            p = norm_text(row.get("patient_id"))
            if p and p in used_patients:
                continue
            val_rows.append(row)
            if p:
                used_patients.add(p)
            if len(val_rows) >= n_val:
                break
        if len(val_rows) == n_val:
            val_idx_keys = {str(r.get("source_path_norm", r.get("source_path"))) for r in val_rows}
            train_pool = pool[~pool["source_path_norm"].astype(str).isin(val_idx_keys)].copy()
            if "patient_id" in train_pool.columns and used_patients:
                train_patient_disjoint = train_pool[~train_pool["patient_id"].astype(str).isin(used_patients)].copy()
                if len(train_patient_disjoint) >= n_train:
                    train_pool = train_patient_disjoint
            train = sort_by_quality(train_pool, "ls").head(n_train).copy()
            val = pd.DataFrame(val_rows).copy()
        else:
            selected = pool.sort_values("_rand").head(total).copy()
            val = selected.head(n_val).copy()
            train = selected.iloc[n_val:n_val + n_train].copy()
    else:
        selected = pool.sort_values("_rand").head(total).copy()
        val = selected.head(n_val).copy()
        train = selected.iloc[n_val:n_val + n_train].copy()

    train = train.drop(columns=["_rand"], errors="ignore").reset_index(drop=True)
    val = val.drop(columns=["_rand"], errors="ignore").reset_index(drop=True)
    train["split"] = "train"
    val["split"] = "val"
    return train, val


def select_hs_c0_extras(clustered: pd.DataFrame, n_train: int, n_val: int, used_paths: set[str], used_records: set[str], exclude_record_ids: set[str], allow_hard_records: bool, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pool = clustered[clustered["cluster_id"].astype(int).eq(0)].copy()
    excluded = set() if allow_hard_records else set(exclude_record_ids)
    pool = filter_unused(pool, used_paths, used_records, exclude_record_ids=excluded)
    train, val = split_selected_extra(pool, n_train, n_val, seed, "HS_C0")
    for df, split in [(train, "train"), (val, "val")]:
        if len(df):
            df["target_class"] = "abnormal_cluster_0"
            df["label_name"] = "abnormal"
            df["source_kind"] = "hs"
            df["cluster_id"] = 0
            df["cluster_acoustic_profile"] = df.get("cluster_acoustic_profile", pd.Series("clean_periodic_abnormal", index=df.index))
            df["selection_reason"] = f"extra_C0_{split}_EXP_H"
    return train, val


def is_both_value(x: object) -> bool:
    """Return True for ICBHI combined crackles+wheezes labels.

    The exact label can vary depending on the preprocessing script:
    common values are "both", "crackles_wheezes",
    "wheezes_crackles", or strings containing both terms.
    """
    k = class_key(x)
    if k == "both" or "both" in k:
        return True
    has_crackle = "crackle" in k or "crackles" in k
    has_wheeze = "wheeze" in k or "wheezes" in k
    return has_crackle and has_wheeze


def select_ls_both_extras(icbhi_quality: pd.DataFrame, n_train: int, n_val: int, used_paths: set[str], used_records: set[str], seed: int, no_fallback: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class_col = "crop_lung_class" if "crop_lung_class" in icbhi_quality.columns else "target_class"
    pool = icbhi_quality[icbhi_quality[class_col].map(is_both_value)].copy()
    pool = filter_unused(pool, used_paths, used_records)
    hard_pool = pool[pool.get("passes_hard_quality", pd.Series(True, index=pool.index)).astype(bool)].copy()
    if len(hard_pool) >= n_train + n_val:
        pool = hard_pool
        selection_mode = "icbhi_both_hard_quality_pool"
    else:
        if no_fallback:
            raise RuntimeError(f"Not enough hard-quality ICBHI both candidates: need={n_train+n_val}, available={len(hard_pool)}")
        selection_mode = "icbhi_both_fallback_ranked_pool"
    train, val = split_selected_extra(pool, n_train, n_val, seed, "LS_BOTH", patient_safe=True)
    for df, split in [(train, "train"), (val, "val")]:
        if len(df):
            df["target_class"] = "both"
            df["crop_lung_class"] = "both"
            df["source_kind"] = "ls"
            df["selection_reason"] = f"{selection_mode}_{split}_EXP_H"
    return train, val


def write_extended_manifest(expb, base_hs: pd.DataFrame, base_ls: pd.DataFrame, extra_hs: List[Tuple[str, pd.DataFrame]], extra_ls: List[Tuple[str, pd.DataFrame]], selected_root: Path, variant_name: str, overwrite: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected_root.exists() and overwrite:
        print(f"Deleting previous selected root: {selected_root}")
        shutil.rmtree(selected_root)
    selected_root.mkdir(parents=True, exist_ok=True)
    hs_dir = selected_root / f"HS_PHYSIONET_{variant_name}"
    ls_dir = selected_root / f"LS_ICBHI_{variant_name}"
    hs_dir.mkdir(parents=True, exist_ok=True)
    ls_dir.mkdir(parents=True, exist_ok=True)

    hs_rows = []
    for i, (_, row) in enumerate(tqdm(base_hs.iterrows(), total=len(base_hs), desc=f"{variant_name} base HS"), start=1):
        src = Path(str(row.get("output_path") or row.get("source_path")))
        source_id = str(row["source_id"])
        out_name = f"hs_baseG_{safe_id(source_id)}_{i:03d}.wav"
        out_path = hs_dir / out_name
        fixed_rms, was_repeated = expb.write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "source_id": source_id,
            "output_file": out_name,
            "output_path": str(out_path),
            "source_kind": "hs",
            "expH_origin": "base_EXP_G_40x40",
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(was_repeated),
        })
        hs_rows.append(d)

    for origin, df in extra_hs:
        offset = len(hs_rows)
        for j, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"{variant_name} {origin} HS"), start=1):
            src = Path(str(row["source_path"]))
            rec_id = safe_id(row.get("record_id", f"extra_{j}"))
            source_id = f"HSH_{origin}_{j:03d}_{rec_id}"
            out_name = f"hs_{origin}_{safe_id(source_id)}.wav"
            out_path = hs_dir / out_name
            fixed_rms, was_repeated = expb.write_fixed_copy(src, out_path)
            d = row.to_dict()
            d.update({
                "selected_id": offset + j,
                "source_id": source_id,
                "output_file": out_name,
                "output_path": str(out_path),
                "source_kind": "hs",
                "expH_origin": origin,
                "fixed_rms_after_write": fixed_rms,
                "was_repeated_to_15s_after_write": bool(was_repeated),
            })
            hs_rows.append(d)

    ls_rows = []
    for i, (_, row) in enumerate(tqdm(base_ls.iterrows(), total=len(base_ls), desc=f"{variant_name} base LS"), start=1):
        src = Path(str(row.get("output_path") or row.get("source_path")))
        source_id = str(row["source_id"])
        out_name = f"ls_baseG_{safe_id(source_id)}_{i:03d}.wav"
        out_path = ls_dir / out_name
        fixed_rms, was_repeated = expb.write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "source_id": source_id,
            "output_file": out_name,
            "output_path": str(out_path),
            "source_kind": "ls",
            "expH_origin": "base_EXP_G_40x40",
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(was_repeated),
        })
        ls_rows.append(d)

    for origin, df in extra_ls:
        offset = len(ls_rows)
        for j, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"{variant_name} {origin} LS"), start=1):
            src = Path(str(row["source_path"]))
            rec_id = safe_id(row.get("record_id", f"extra_{j}"))
            source_id = f"LSH_{origin}_{j:03d}_{rec_id}"
            out_name = f"ls_{origin}_{safe_id(source_id)}.wav"
            out_path = ls_dir / out_name
            fixed_rms, was_repeated = expb.write_fixed_copy(src, out_path)
            d = row.to_dict()
            d.update({
                "selected_id": offset + j,
                "source_id": source_id,
                "output_file": out_name,
                "output_path": str(out_path),
                "source_kind": "ls",
                "target_class": "both",
                "crop_lung_class": "both",
                "expH_origin": origin,
                "fixed_rms_after_write": fixed_rms,
                "was_repeated_to_15s_after_write": bool(was_repeated),
            })
            ls_rows.append(d)

    hs_manifest = pd.DataFrame(hs_rows)
    ls_manifest = pd.DataFrame(ls_rows)
    hs_manifest.to_csv(selected_root / f"selected_hs_{variant_name}.csv", index=False)
    ls_manifest.to_csv(selected_root / f"selected_ls_{variant_name}.csv", index=False)
    pd.concat([hs_manifest.assign(modality="HS"), ls_manifest.assign(modality="LS")], ignore_index=True, sort=False).to_csv(
        selected_root / f"selected_sources_{variant_name}_combined.csv", index=False
    )
    return hs_manifest, ls_manifest


def check_counts_and_write_summary(variant_name: str, selected_root: Path, mix_root: Path, processed_dir: Path, hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame, metadata: pd.DataFrame, manifest: pd.DataFrame, disjoint: Dict[str, object], args: argparse.Namespace, selection_report: Dict[str, object]) -> Dict[str, object]:
    expected = {
        "hs_train": int((hs_manifest["split"] == "train").sum()),
        "hs_val": int((hs_manifest["split"] == "val").sum()),
        "ls_train": int((ls_manifest["split"] == "train").sum()),
        "ls_val": int((ls_manifest["split"] == "val").sum()),
    }
    expected["train_bases"] = expected["hs_train"] * expected["ls_train"] * len(args.snrs)
    expected["val_bases"] = expected["hs_val"] * expected["ls_val"] * len(args.snrs)
    expected["train_segments"] = expected["train_bases"] * expb_expected_segments(args)
    expected["val_segments"] = expected["val_bases"] * expb_expected_segments(args)

    actual = {
        "train_bases": int((metadata["split"] == "train").sum()),
        "val_bases": int((metadata["split"] == "val").sum()),
        "train_segments": int((manifest["split"] == "train").sum()),
        "val_segments": int((manifest["split"] == "val").sum()),
    }
    for k in actual:
        if expected[k] != actual[k]:
            raise RuntimeError(f"{variant_name}: count mismatch for {k}: expected={expected[k]}, actual={actual[k]}")
    if float(metadata["residual_snr_m_vs_h_plus_l_db"].min()) < 50.0:
        raise RuntimeError(f"{variant_name}: full additivity below 50 dB")
    if "segment_additivity_snr_db" in manifest.columns and float(manifest["segment_additivity_snr_db"].min()) < 50.0:
        raise RuntimeError(f"{variant_name}: segment additivity below 50 dB")

    payload = {
        "variant_name": variant_name,
        "selected_root": str(selected_root),
        "mix_root": str(mix_root),
        "processed_dir": str(processed_dir),
        "split_csv": str(processed_dir / "source_disjoint_split_smoke.csv"),
        "expected": expected,
        "actual": actual,
        "selection_report": selection_report,
        "disjoint": disjoint,
    }
    lines = []
    lines.append(f"{variant_name} - EXP_H extended variant")
    lines.append("=" * 100)
    lines.append("Base = EXP_G_40x40. Additions are included in train and/or validation according to variant.")
    lines.append("ICBHI both is added in LS-extended variants. No rhonchi. No HLS-CMDS target-domain source is used.")
    lines.append("")
    lines.append(json.dumps(payload, indent=2, sort_keys=True))
    lines.append("")
    lines.append("HS selected distribution:")
    lines.append(str(pd.crosstab(hs_manifest["target_class"], hs_manifest["split"])))
    lines.append("")
    lines.append("HS origin distribution:")
    lines.append(str(pd.crosstab(hs_manifest.get("expH_origin", "unknown"), hs_manifest["split"])))
    lines.append("")
    lines.append("LS selected distribution:")
    lines.append(str(pd.crosstab(ls_manifest["target_class"], ls_manifest["split"])))
    lines.append("")
    lines.append("LS origin distribution:")
    lines.append(str(pd.crosstab(ls_manifest.get("expH_origin", "unknown"), ls_manifest["split"])))
    lines.append("")
    lines.append("model_config.py reminder:")
    lines.append(f'SUPERVISED_DIR = str(PROJECT_ROOT / "dataset" / "processed" / "{processed_dir.name}")')
    lines.append("USE_SOURCE_DISJOINT_SPLIT = True")
    lines.append('SOURCE_DISJOINT_SPLIT_CSV = SUPERVISED_DIR + "/source_disjoint_split_smoke.csv"')
    lines.append("N_FOLDS = 1")
    lines.append("ONLY_FOLD = 1")

    for p in [selected_root / "summary.txt", mix_root / "summary.txt", processed_dir / "summary.txt"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
    for p in [selected_root / "summary.json", mix_root / "summary.json", processed_dir / "summary.json"]:
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def expb_expected_segments(args: argparse.Namespace) -> int:
    # Kept as a function to make it obvious if you change the segmentation later.
    return 27


def build_variant(expb, variant_key: str, project_root: Path, base_selected_root: Path | None, clustered_csv: Path, icbhi_quality_csv: Path, snrs: List[float], overwrite: bool, allow_hard_records: bool, exclude_record_ids: set[str], no_fallback: bool, args: argparse.Namespace) -> Dict[str, object]:
    v = VARIANTS[variant_key]
    variant_name = variant_key
    selected_root = project_root / "dataset" / str(v["selected_dir"])
    mix_root = project_root / "dataset" / str(v["mix_dir"])
    processed_dir = project_root / "dataset" / "processed" / str(v["processed_dir"])
    progress_cluster_id = int(v["progress_cluster_id"])

    print("\n" + "=" * 100)
    print(f"BUILDING {variant_name}")
    print("=" * 100)
    print(f"Selected root : {selected_root}")
    print(f"Mix root      : {mix_root}")
    print(f"Processed dir : {processed_dir}")

    base_hs, base_ls = read_base_exp_g_manifests(project_root, base_selected_root)
    clustered = read_csv_checked(clustered_csv, ["cluster_id", "record_id", "source_path", "quality_score"], "PhysioNet clustered CSV")
    icbhi_quality = read_csv_checked(icbhi_quality_csv, ["source_path", "quality_score"], "ICBHI quality CSV")
    if "crop_lung_class" not in icbhi_quality.columns and "target_class" not in icbhi_quality.columns:
        raise RuntimeError("ICBHI quality CSV must contain crop_lung_class or target_class")

    used_hs_paths = path_set(base_hs)
    used_hs_records = record_set(base_hs)
    used_ls_paths = path_set(base_ls)
    used_ls_records = record_set(base_ls)

    extra_hs_pairs: List[Tuple[str, pd.DataFrame]] = []
    extra_ls_pairs: List[Tuple[str, pd.DataFrame]] = []
    selection_report: Dict[str, object] = {
        "base_hs_train": int((base_hs["split"] == "train").sum()),
        "base_hs_val": int((base_hs["split"] == "val").sum()),
        "base_ls_train": int((base_ls["split"] == "train").sum()),
        "base_ls_val": int((base_ls["split"] == "val").sum()),
        "add_hs_c0": bool(v["add_hs_c0"]),
        "add_ls_both": bool(v["add_ls_both"]),
    }

    if bool(v["add_hs_c0"]):
        c0_train, c0_val = select_hs_c0_extras(
            clustered=clustered,
            n_train=args.c0_train,
            n_val=args.c0_val,
            used_paths=used_hs_paths,
            used_records=used_hs_records,
            exclude_record_ids=exclude_record_ids,
            allow_hard_records=allow_hard_records,
            seed=args.seed + 101,
        )
        extra_hs_pairs.extend([("extra_C0_train", c0_train), ("extra_C0_val", c0_val)])
        used_hs_paths |= path_set(pd.concat([c0_train, c0_val], ignore_index=True, sort=False))
        used_hs_records |= record_set(pd.concat([c0_train, c0_val], ignore_index=True, sort=False))
        selection_report["selected_c0_train_record_ids"] = c0_train.get("record_id", pd.Series(dtype=str)).astype(str).tolist()
        selection_report["selected_c0_val_record_ids"] = c0_val.get("record_id", pd.Series(dtype=str)).astype(str).tolist()

    if bool(v["add_ls_both"]):
        both_train, both_val = select_ls_both_extras(
            icbhi_quality=icbhi_quality,
            n_train=args.both_train,
            n_val=args.both_val,
            used_paths=used_ls_paths,
            used_records=used_ls_records,
            seed=args.seed + 303,
            no_fallback=no_fallback,
        )
        extra_ls_pairs.extend([("extra_both_train", both_train), ("extra_both_val", both_val)])
        used_ls_paths |= path_set(pd.concat([both_train, both_val], ignore_index=True, sort=False))
        used_ls_records |= record_set(pd.concat([both_train, both_val], ignore_index=True, sort=False))
        selection_report["selected_both_train_record_ids"] = both_train.get("record_id", pd.Series(dtype=str)).astype(str).tolist()
        selection_report["selected_both_val_record_ids"] = both_val.get("record_id", pd.Series(dtype=str)).astype(str).tolist()
        selection_report["selected_both_train_patient_ids"] = both_train.get("patient_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
        selection_report["selected_both_val_patient_ids"] = both_val.get("patient_id", pd.Series(dtype=str)).dropna().astype(str).tolist()

    hs_manifest, ls_manifest = write_extended_manifest(
        expb=expb,
        base_hs=base_hs,
        base_ls=base_ls,
        extra_hs=extra_hs_pairs,
        extra_ls=extra_ls_pairs,
        selected_root=selected_root,
        variant_name=variant_name,
        overwrite=overwrite,
    )

    disjoint = expb.check_source_disjoint(hs_manifest, ls_manifest)
    print("\nSource-disjoint audit:")
    print(json.dumps(disjoint, indent=2, sort_keys=True))
    print("\nHS distribution:")
    print(pd.crosstab(hs_manifest["target_class"], hs_manifest["split"]).to_string())
    print("\nLS distribution:")
    print(pd.crosstab(ls_manifest["target_class"], ls_manifest["split"]).to_string())

    metadata, _ = expb.create_mixtures_for_cluster(
        cluster_id=progress_cluster_id,
        hs_manifest=hs_manifest,
        ls_manifest=ls_manifest,
        mix_root=mix_root,
        snrs=snrs,
        overwrite=overwrite,
    )
    manifest = expb.segment_processed_dataset(
        cluster_id=progress_cluster_id,
        mix_root=mix_root,
        processed_dir=processed_dir,
        overwrite=overwrite,
    )

    payload = check_counts_and_write_summary(
        variant_name=variant_name,
        selected_root=selected_root,
        mix_root=mix_root,
        processed_dir=processed_dir,
        hs_manifest=hs_manifest,
        ls_manifest=ls_manifest,
        metadata=metadata,
        manifest=manifest,
        disjoint=disjoint,
        args=args,
        selection_report=selection_report,
    )

    index_path = project_root / "outputs" / f"{variant_name}_dataset_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_csv(index_path, index=False)
    print("\n" + "=" * 100)
    print(f"{variant_name} CREATED")
    print("=" * 100)
    print(f"Processed dir: {processed_dir}")
    print(f"Summary      : {processed_dir / 'summary.txt'}")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--base-selected-root", type=Path, default=None)
    p.add_argument("--clustered-csv", type=Path, default=None)
    p.add_argument("--icbhi-quality-csv", type=Path, default=None)
    p.add_argument("--snrs", nargs="+", default=[str(int(x)) for x in DEFAULT_SNRS])
    p.add_argument("--c0-train", type=int, default=10)
    p.add_argument("--c0-val", type=int, default=3)
    p.add_argument("--both-train", type=int, default=10)
    p.add_argument("--both-val", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-hard-records", action="store_true")
    p.add_argument("--exclude-record-ids", nargs="*", default=sorted(DEFAULT_HARD_RECORD_IDS))
    p.add_argument("--no-fallback", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root: Path = args.project_root
    clustered_csv = args.clustered_csv or (project_root / "outputs" / "physionet_abnormal_acoustic_clustering" / "physionet_abnormal_clustered.csv")
    icbhi_quality_csv = args.icbhi_quality_csv or (project_root / "outputs" / "smoke_phy_ich_hsnormal_ls3_selection" / "icbhi_all_candidates_quality.csv")
    args.snrs = parse_snr_values(args.snrs)
    exclude_record_ids = set(str(x) for x in (args.exclude_record_ids or []))

    build_variant(
        expb=expb,
        variant_key="EXP_H_FULL_BOTH",
        project_root=project_root,
        base_selected_root=args.base_selected_root,
        clustered_csv=clustered_csv,
        icbhi_quality_csv=icbhi_quality_csv,
        snrs=args.snrs,
        overwrite=bool(args.overwrite),
        allow_hard_records=bool(args.allow_hard_records),
        exclude_record_ids=exclude_record_ids,
        no_fallback=bool(args.no_fallback),
        args=args,
    )


if __name__ == "__main__":
    main()

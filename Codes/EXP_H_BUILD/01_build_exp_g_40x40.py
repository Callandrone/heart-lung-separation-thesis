"""
01_build_exp_g_40x40.py

Build the EXP_G_40x40 precursor used by the final EXP_H benchmark from the frozen EXP_E source selection.

Final precursor
---------------
EXP_G_40x40 keeps the frozen EXP_E validation sources and expands the training pools to 40 HS and 40 LS sources.

Design principle
----------------
EXP_E is not modified. This script reads EXP_E selected-source manifests,
keeps the same validation sources, adds only train sources, regenerates
controlled synthetic mixtures M = H + L, then creates the processed segment
folder and source_disjoint_split_smoke.csv.

Default target distributions
----------------------------
EXP_E baseline:
  HS train = 12 normal + 8 abnormal_cluster_2 + 5 abnormal_cluster_1 = 25
  LS train = 10 normal + 10 wheezes + 10 crackles = 30
  HS val   = 5 normal + 3 abnormal_cluster_2 = 8  [unchanged]
  LS val   = 5 normal + 5 wheezes + 5 crackles = 15 [unchanged]

G heart expansion:
  HS train = 15 normal + 15 abnormal_cluster_2 + 10 abnormal_cluster_1 = 40

G lung expansion:
  LS train = 14 normal + 13 wheezes + 13 crackles = 40

Important
---------
- No ICBHI both.
- No HLS-CMDS target-domain sources.
- No rhonchi / pleural rub.
- No validation change.
- No mixed train/validation combinations.
- Source-disjointness is audited on source_path.
- Patient/session overlap for ICBHI is avoided when possible; if impossible,
  the script falls back to best-ranked source-only additions and records this
  in selection_reason.

Usage
-----
python 01_build_exp_g_40x40.py --overwrite
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


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
DEFAULT_SNRS = [-6.0, -3.0, 0.0, 3.0, 6.0]
DEFAULT_HARD_RECORD_IDS = {"c0015", "a0092", "a0135", "e02065", "f0082", "a0167"}

EXP_E_SELECTED_DIR_NAME = "EXP_E_25x30_SELECTED"
EXP_E_HS_MANIFEST = "selected_hs_experiment_E_25x30.csv"
EXP_E_LS_MANIFEST = "selected_ls_experiment_E_25x30.csv"

CLUSTER_PROFILE = {
    0: "clean_periodic_abnormal",
    1: "low_amplitude_temporally_unstable",
    2: "murmur_like_high_cardiac_band",
}

VARIANTS: Dict[str, Dict[str, object]] = {
    "G_40x40": {
        "experiment_name": "EXP_G_40x40",
        "selected_dir": "EXP_G_40x40_SELECTED",
        "mix_dir": "MIX_EXP_G_40x40_4K_SNR5",
        "processed_dir": "experiment_G_40x40",
        "progress_cluster_id": 40440,
        "hs_targets": {"normal": 15, "abnormal_cluster_2": 15, "abnormal_cluster_1": 10},
        "ls_targets": {"normal": 14, "wheezes": 13, "crackles": 13},
    },
}


# =============================================================================
# SHARED BENCHMARK UTILITIES
# =============================================================================

import dataset_builder_utils as expb


# =============================================================================
# BASIC UTILS
# =============================================================================

def parse_snr_values(values: List[str]) -> List[float]:
    out: List[float] = []
    for v in values:
        if str(v).strip().lower() == "no":
            raise ValueError("The 'No' SNR condition is intentionally disabled here. Use -6 -3 0 3 6.")
        x = float(v)
        if x not in out:
            out.append(x)
    return out


def safe_id(text: object) -> str:
    s = str(text)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {name}: {sorted(missing)}")


def norm_path_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace("\\\\", "/", regex=False).str.strip()


def norm_text(v: object) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def class_key(v: object) -> str:
    return norm_text(v).lower().replace(" ", "_")


def record_set(df: pd.DataFrame) -> set[str]:
    if "record_id" not in df.columns:
        return set()
    return set(df["record_id"].dropna().astype(str))


def path_set(df: pd.DataFrame) -> set[str]:
    if "source_path" not in df.columns:
        return set()
    return set(norm_path_series(df["source_path"]))


def sort_by_quality(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    df = df.copy()
    if source_kind == "hs":
        sort_cols = [c for c in ["quality_score", "seg_rms_p10", "heart_band_ratio_20_500", "fixed_rms"] if c in df.columns]
    else:
        sort_cols = [c for c in ["quality_score", "seg_rms_p10", "crop_cycle_coverage_sec", "lung_band_ratio_50_1800", "fixed_rms"] if c in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    if "record_id" in df.columns:
        return df.sort_values("record_id").reset_index(drop=True)
    return df.sort_index().reset_index(drop=True)


def read_quality_csv(path: Path, required_cols: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, required_cols, str(path))
    df["source_path_norm"] = norm_path_series(df["source_path"])
    if "record_id" in df.columns:
        df["record_id"] = df["record_id"].astype(str)
    return df


def read_exp_e_manifests(project_root: Path, exp_e_selected_root: Path | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_root = exp_e_selected_root or (project_root / "dataset" / EXP_E_SELECTED_DIR_NAME)
    hs_path = selected_root / EXP_E_HS_MANIFEST
    ls_path = selected_root / EXP_E_LS_MANIFEST
    if not hs_path.exists():
        raise RuntimeError(f"EXP_E HS manifest not found: {hs_path}")
    if not ls_path.exists():
        raise RuntimeError(f"EXP_E LS manifest not found: {ls_path}")
    hs = pd.read_csv(hs_path)
    ls = pd.read_csv(ls_path)
    require_columns(hs, ["source_id", "source_path", "output_path", "split", "target_class"], str(hs_path))
    require_columns(ls, ["source_id", "source_path", "output_path", "split", "target_class"], str(ls_path))
    return hs, ls


# =============================================================================
# SOURCE SELECTION
# =============================================================================

def _filter_unused(
    pool: pd.DataFrame,
    used_paths: set[str],
    used_records: set[str],
    exclude_record_ids: set[str] | None = None,
) -> pd.DataFrame:
    out = pool.copy()
    if "source_path_norm" not in out.columns:
        out["source_path_norm"] = norm_path_series(out["source_path"])
    out = out[~out["source_path_norm"].isin(used_paths)].copy()
    if "record_id" in out.columns and used_records:
        out = out[~out["record_id"].astype(str).isin(used_records)].copy()
    if exclude_record_ids and "record_id" in out.columns:
        out = out[~out["record_id"].astype(str).isin(exclude_record_ids)].copy()
    return out


def select_hs_normal_extras(
    physionet_quality: pd.DataFrame,
    n_extra: int,
    used_paths: set[str],
    used_records: set[str],
    no_fallback: bool,
) -> pd.DataFrame:
    if n_extra <= 0:
        return pd.DataFrame()
    pool = physionet_quality[physionet_quality.get("label_name", "").astype(str).str.lower().eq("normal")].copy()
    pool = _filter_unused(pool, used_paths=used_paths, used_records=used_records)
    hard_pool = pool[pool.get("passes_hard_quality", pd.Series(True, index=pool.index)).astype(bool)].copy()
    if len(hard_pool) >= n_extra:
        pool = hard_pool
        selection_mode = "normal_hard_quality_pool"
    else:
        if no_fallback:
            raise RuntimeError(f"Not enough hard-quality normal HS candidates: needed={n_extra}, available={len(hard_pool)}")
        selection_mode = "normal_fallback_ranked_pool"
    pool = sort_by_quality(pool, source_kind="hs")
    if len(pool) < n_extra:
        raise RuntimeError(f"Not enough extra normal HS candidates: needed={n_extra}, available={len(pool)}")
    selected = pool.head(n_extra).copy().reset_index(drop=True)
    selected["split"] = "train"
    selected["target_class"] = "normal"
    selected["label_name"] = "normal"
    selected["source_kind"] = "hs"
    selected["cluster_id"] = np.nan
    selected["cluster_acoustic_profile"] = "normal"
    selected["selection_reason"] = selection_mode
    selected["easy_rank_global"] = np.arange(1, len(selected) + 1)
    return selected


def select_hs_cluster_extras(
    clustered: pd.DataFrame,
    cluster_id: int,
    n_extra: int,
    used_paths: set[str],
    used_records: set[str],
    exclude_record_ids: set[str],
    allow_hard_records: bool,
    reason: str,
) -> pd.DataFrame:
    if n_extra <= 0:
        return pd.DataFrame()
    pool = clustered[clustered["cluster_id"].astype(int).eq(int(cluster_id))].copy()
    pool["source_path_norm"] = norm_path_series(pool["source_path"])
    if "record_id" in pool.columns:
        pool["record_id"] = pool["record_id"].astype(str)
    excluded = set() if allow_hard_records else set(exclude_record_ids)
    pool = _filter_unused(pool, used_paths=used_paths, used_records=used_records, exclude_record_ids=excluded)
    pool = sort_by_quality(pool, source_kind="hs")
    if len(pool) < n_extra:
        raise RuntimeError(
            f"Not enough extra HS cluster {cluster_id} candidates after filtering: "
            f"needed={n_extra}, available={len(pool)}"
        )
    selected = pool.head(n_extra).copy().reset_index(drop=True)
    selected["split"] = "train"
    selected["target_class"] = f"abnormal_cluster_{cluster_id}"
    selected["label_name"] = "abnormal"
    selected["source_kind"] = "hs"
    selected["cluster_id"] = int(cluster_id)
    if "cluster_acoustic_profile" not in selected.columns:
        selected["cluster_acoustic_profile"] = CLUSTER_PROFILE.get(int(cluster_id), f"cluster_{cluster_id}")
    selected["selection_reason"] = reason
    selected["easy_rank_global"] = np.arange(1, len(selected) + 1)
    return selected


def select_ls_class_extras(
    icbhi_quality: pd.DataFrame,
    class_name: str,
    n_extra: int,
    used_paths: set[str],
    used_records: set[str],
    current_ls_manifest: pd.DataFrame,
    no_fallback: bool,
) -> pd.DataFrame:
    if n_extra <= 0:
        return pd.DataFrame()
    cls = class_key(class_name)
    class_col = "crop_lung_class" if "crop_lung_class" in icbhi_quality.columns else "target_class"
    pool = icbhi_quality[icbhi_quality[class_col].map(class_key).eq(cls)].copy()
    pool = _filter_unused(pool, used_paths=used_paths, used_records=used_records)
    hard_pool = pool[pool.get("passes_hard_quality", pd.Series(True, index=pool.index)).astype(bool)].copy()
    if len(hard_pool) >= n_extra:
        pool = hard_pool
        selection_mode = f"{cls}_hard_quality_pool"
    else:
        if no_fallback:
            raise RuntimeError(f"Not enough hard-quality LS {cls} candidates: needed={n_extra}, available={len(hard_pool)}")
        selection_mode = f"{cls}_fallback_ranked_pool"

    val_df = current_ls_manifest[current_ls_manifest["split"].eq("val")].copy()
    cur_df = current_ls_manifest.copy()
    val_patients = set(val_df.get("patient_id", pd.Series(dtype=str)).dropna().astype(str))
    val_sessions = set(val_df.get("session_id", pd.Series(dtype=str)).dropna().astype(str))
    cur_patients = set(cur_df.get("patient_id", pd.Series(dtype=str)).dropna().astype(str))
    cur_sessions = set(cur_df.get("session_id", pd.Series(dtype=str)).dropna().astype(str))

    pool = sort_by_quality(pool, source_kind="ls")
    selected_rows = []
    selected_indices = set()

    def patient(row) -> str:
        return norm_text(row.get("patient_id", "unknown"))

    def session(row) -> str:
        if "session_id" in row.index and norm_text(row.get("session_id")):
            return norm_text(row.get("session_id"))
        return f"{patient(row)}_{norm_text(row.get('recording_code', 'unknown'))}"

    def try_pass(predicate) -> bool:
        nonlocal selected_rows, selected_indices
        for idx, row in pool.iterrows():
            if int(idx) in selected_indices:
                continue
            if predicate(row):
                selected_rows.append(row)
                selected_indices.add(int(idx))
                if len(selected_rows) >= n_extra:
                    return True
        return False

    if try_pass(lambda r: patient(r) not in val_patients and patient(r) not in cur_patients):
        mode_suffix = "patient_unique"
    elif try_pass(lambda r: session(r) not in val_sessions and session(r) not in cur_sessions):
        mode_suffix = "session_unique"
    else:
        if len(selected_rows) < n_extra:
            try_pass(lambda r: True)
        mode_suffix = "source_only_relaxed"

    if len(selected_rows) < n_extra:
        raise RuntimeError(f"Not enough extra LS {cls} candidates: needed={n_extra}, available={len(pool)}")

    selected = pd.DataFrame(selected_rows).copy().reset_index(drop=True)
    selected["split"] = "train"
    selected["target_class"] = cls
    selected["crop_lung_class"] = cls
    selected["source_kind"] = "ls"
    selected["selection_reason"] = f"{selection_mode}_{mode_suffix}"
    selected["easy_rank_global"] = np.arange(1, len(selected) + 1)
    return selected


# =============================================================================
# MANIFEST WRITING
# =============================================================================

def write_fixed_manifest(
    expb,
    base_hs: pd.DataFrame,
    base_ls: pd.DataFrame,
    extra_hs_by_origin: List[Tuple[str, str, pd.DataFrame]],
    extra_ls_by_origin: List[Tuple[str, str, pd.DataFrame]],
    selected_root: Path,
    variant_name: str,
    overwrite: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected_root.exists() and overwrite:
        print(f"Deleting previous selected root: {selected_root}")
        shutil.rmtree(selected_root)
    selected_root.mkdir(parents=True, exist_ok=True)

    hs_dir = selected_root / f"HS_PHYSIONET_{variant_name}"
    ls_dir = selected_root / f"LS_ICBHI_{variant_name}"
    hs_dir.mkdir(parents=True, exist_ok=True)
    ls_dir.mkdir(parents=True, exist_ok=True)

    hs_rows: List[Dict[str, object]] = []
    ls_rows: List[Dict[str, object]] = []

    for i, (_, row) in enumerate(tqdm(base_hs.iterrows(), total=len(base_hs), desc=f"{variant_name} base EXP_E HS"), start=1):
        src = Path(str(row.get("output_path") or row.get("source_path")))
        source_id = str(row["source_id"])
        out_name = f"hs_expEbase_{safe_id(source_id)}_{i:03d}.wav"
        out_path = hs_dir / out_name
        fixed_rms, was_repeated = expb.write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "source_id": source_id,
            "output_file": out_name,
            "output_path": str(out_path),
            "source_kind": "hs",
            "expG_origin": "base_EXP_E_25x30",
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(was_repeated),
        })
        hs_rows.append(d)

    def append_extra_hs(df: pd.DataFrame, prefix: str, origin: str) -> None:
        nonlocal hs_rows
        offset = len(hs_rows)
        for j, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"{variant_name} {origin} HS"), start=1):
            src = Path(str(row["source_path"]))
            rec_id = safe_id(row.get("record_id", f"extra_{j}"))
            source_id = f"{prefix}_{j:03d}_{rec_id}"
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
                "split": "train",
                "expG_origin": origin,
                "fixed_rms_after_write": fixed_rms,
                "was_repeated_to_15s_after_write": bool(was_repeated),
            })
            hs_rows.append(d)

    for prefix, origin, df in extra_hs_by_origin:
        append_extra_hs(df, prefix, origin)

    for i, (_, row) in enumerate(tqdm(base_ls.iterrows(), total=len(base_ls), desc=f"{variant_name} base EXP_E LS"), start=1):
        src = Path(str(row.get("output_path") or row.get("source_path")))
        source_id = str(row["source_id"])
        out_name = f"ls_expEbase_{safe_id(source_id)}_{i:03d}.wav"
        out_path = ls_dir / out_name
        fixed_rms, was_repeated = expb.write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "source_id": source_id,
            "output_file": out_name,
            "output_path": str(out_path),
            "source_kind": "ls",
            "expG_origin": "base_EXP_E_25x30",
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(was_repeated),
        })
        ls_rows.append(d)

    def append_extra_ls(df: pd.DataFrame, prefix: str, origin: str) -> None:
        nonlocal ls_rows
        offset = len(ls_rows)
        for j, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"{variant_name} {origin} LS"), start=1):
            src = Path(str(row["source_path"]))
            rec_id = safe_id(row.get("record_id", f"extra_{j}"))
            source_id = f"{prefix}_{j:03d}_{rec_id}"
            out_name = f"ls_{origin}_{safe_id(source_id)}.wav"
            out_path = ls_dir / out_name
            fixed_rms, was_repeated = expb.write_fixed_copy(src, out_path)
            cls = class_key(row.get("target_class") or row.get("crop_lung_class")) or "unknown"
            d = row.to_dict()
            d.update({
                "selected_id": offset + j,
                "source_id": source_id,
                "output_file": out_name,
                "output_path": str(out_path),
                "source_kind": "ls",
                "split": "train",
                "target_class": cls,
                "crop_lung_class": cls,
                "expG_origin": origin,
                "fixed_rms_after_write": fixed_rms,
                "was_repeated_to_15s_after_write": bool(was_repeated),
            })
            ls_rows.append(d)

    for prefix, origin, df in extra_ls_by_origin:
        append_extra_ls(df, prefix, origin)

    hs_manifest = pd.DataFrame(hs_rows)
    ls_manifest = pd.DataFrame(ls_rows)

    hs_manifest.to_csv(selected_root / f"selected_hs_{variant_name}.csv", index=False)
    ls_manifest.to_csv(selected_root / f"selected_ls_{variant_name}.csv", index=False)
    pd.concat([
        hs_manifest.assign(modality="HS"),
        ls_manifest.assign(modality="LS"),
    ], ignore_index=True, sort=False).to_csv(selected_root / f"selected_sources_{variant_name}_combined.csv", index=False)
    hs_manifest[hs_manifest.get("expG_origin", "").astype(str).str.startswith("extra_")].to_csv(selected_root / f"extra_hs_sources_{variant_name}.csv", index=False)
    ls_manifest[ls_manifest.get("expG_origin", "").astype(str).str.startswith("extra_")].to_csv(selected_root / f"extra_ls_sources_{variant_name}.csv", index=False)
    return hs_manifest, ls_manifest


# =============================================================================
# CHECKS / SUMMARIES
# =============================================================================

def count_by_split_class(df: pd.DataFrame, classes: List[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for cls in classes:
        out[cls] = {}
        for split in ["train", "val"]:
            out[cls][split] = int(((df["target_class"].map(class_key) == cls) & (df["split"] == split)).sum())
    return out


def assert_targets(hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame, hs_targets: Dict[str, int], ls_targets: Dict[str, int]) -> None:
    hs_train_counts = count_by_split_class(hs_manifest, list(hs_targets.keys()))
    ls_train_counts = count_by_split_class(ls_manifest, list(ls_targets.keys()))
    bad = []
    for cls, target in hs_targets.items():
        actual = hs_train_counts[cls]["train"]
        if actual != target:
            bad.append(f"HS {cls} train: expected {target}, got {actual}")
    for cls, target in ls_targets.items():
        actual = ls_train_counts[cls]["train"]
        if actual != target:
            bad.append(f"LS {cls} train: expected {target}, got {actual}")
    # EXP_E validation must remain unchanged.
    expected_hs_val = {"normal": 5, "abnormal_cluster_2": 3, "abnormal_cluster_1": 0}
    expected_ls_val = {"normal": 5, "wheezes": 5, "crackles": 5}
    for cls, target in expected_hs_val.items():
        actual = count_by_split_class(hs_manifest, [cls])[cls]["val"]
        if actual != target:
            bad.append(f"HS {cls} val: expected {target}, got {actual}")
    for cls, target in expected_ls_val.items():
        actual = count_by_split_class(ls_manifest, [cls])[cls]["val"]
        if actual != target:
            bad.append(f"LS {cls} val: expected {target}, got {actual}")
    if bad:
        raise RuntimeError("Target count check failed:\n" + "\n".join(bad))


def expected_counts(expb, hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame, snrs: List[float]) -> Dict[str, int]:
    hs_train = int((hs_manifest["split"] == "train").sum())
    hs_val = int((hs_manifest["split"] == "val").sum())
    ls_train = int((ls_manifest["split"] == "train").sum())
    ls_val = int((ls_manifest["split"] == "val").sum())
    n_snr = len(snrs)
    return {
        "hs_train": hs_train,
        "hs_val": hs_val,
        "ls_train": ls_train,
        "ls_val": ls_val,
        "train_bases": hs_train * ls_train * n_snr,
        "val_bases": hs_val * ls_val * n_snr,
        "train_segments": hs_train * ls_train * n_snr * expb.EXPECTED_SEGMENTS,
        "val_segments": hs_val * ls_val * n_snr * expb.EXPECTED_SEGMENTS,
    }


def write_summary(
    variant_name: str,
    selected_root: Path,
    mix_root: Path,
    processed_dir: Path,
    hs_manifest: pd.DataFrame,
    ls_manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    expected: Dict[str, int],
    selection_report: Dict[str, object],
    disjoint: Dict[str, object],
) -> None:
    lines = []
    lines.append(f"{variant_name} - controlled data scaling from frozen EXP_E / EXP6")
    lines.append("=" * 100)
    lines.append("EXP_E validation is unchanged. Only training sources are added.")
    lines.append("No ICBHI both, no HLS-CMDS target-domain sources, no rhonchi, no pleural rub.")
    lines.append("")
    lines.append(f"Selected root : {selected_root}")
    lines.append(f"Mix root      : {mix_root}")
    lines.append(f"Processed dir : {processed_dir}")
    lines.append(f"Split CSV     : {processed_dir / 'source_disjoint_split_smoke.csv'}")
    lines.append("")
    lines.append("Expected counts:")
    lines.append(json.dumps(expected, indent=2, sort_keys=True))
    lines.append("")
    lines.append("Selection report:")
    lines.append(json.dumps(selection_report, indent=2, sort_keys=True))
    lines.append("")
    lines.append("Source-disjoint audit:")
    lines.append(json.dumps(disjoint, indent=2, sort_keys=True))
    lines.append("")
    lines.append("HS selected distribution:")
    lines.append(str(pd.crosstab(hs_manifest["target_class"], hs_manifest["split"])))
    lines.append("")
    lines.append("HS origin distribution:")
    lines.append(str(pd.crosstab(hs_manifest.get("expG_origin", "unknown"), hs_manifest["split"])))
    lines.append("")
    lines.append("LS selected distribution:")
    lines.append(str(pd.crosstab(ls_manifest["target_class"], ls_manifest["split"])))
    lines.append("")
    lines.append("LS origin distribution:")
    lines.append(str(pd.crosstab(ls_manifest.get("expG_origin", "unknown"), ls_manifest["split"])))
    lines.append("")
    lines.append("Generated base triplets:")
    lines.append(str(metadata["split"].value_counts(dropna=False)))
    lines.append("")
    lines.append("Generated segments:")
    lines.append(str(manifest["split"].value_counts(dropna=False)))
    lines.append("")
    if "segment_additivity_snr_db" in manifest.columns:
        lines.append("Segment additivity:")
        lines.append(str(manifest[["segment_additivity_snr_db"]].describe().T))
        lines.append("")

    payload = {
        "variant_name": variant_name,
        "selected_root": str(selected_root),
        "mix_root": str(mix_root),
        "processed_dir": str(processed_dir),
        "split_csv": str(processed_dir / "source_disjoint_split_smoke.csv"),
        "expected": expected,
        "selection_report": selection_report,
        "disjoint": disjoint,
        "train_base_triplets": int((metadata["split"] == "train").sum()),
        "val_base_triplets": int((metadata["split"] == "val").sum()),
        "train_segments": int((manifest["split"] == "train").sum()),
        "val_segments": int((manifest["split"] == "val").sum()),
    }

    for p in [selected_root / "summary.txt", mix_root / "summary.txt", processed_dir / "summary.txt"]:
        p.write_text("\n".join(lines))
    for p in [selected_root / "summary.json", mix_root / "summary.json", processed_dir / "summary.json"]:
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))


# =============================================================================
# BUILD VARIANT
# =============================================================================

def build_variant(
    expb,
    variant_key: str,
    project_root: Path,
    exp_e_selected_root: Path | None,
    clustered_csv: Path,
    physionet_quality_csv: Path,
    icbhi_quality_csv: Path,
    snrs: List[float],
    overwrite: bool,
    allow_hard_records: bool,
    exclude_record_ids: set[str],
    no_fallback: bool,
) -> Dict[str, object]:
    v = VARIANTS[variant_key]
    variant_name = str(v["experiment_name"])
    hs_targets: Dict[str, int] = dict(v["hs_targets"])  # type: ignore[arg-type]
    ls_targets: Dict[str, int] = dict(v["ls_targets"])  # type: ignore[arg-type]

    selected_root = project_root / "dataset" / str(v["selected_dir"])
    mix_root = project_root / "dataset" / str(v["mix_dir"])
    processed_dir = project_root / "dataset" / "processed" / str(v["processed_dir"])
    progress_cluster_id = int(v["progress_cluster_id"])

    print("\n" + "=" * 100)
    print(f"BUILDING {variant_name}")
    print("=" * 100)
    print(f"HS train targets: {hs_targets}")
    print(f"LS train targets: {ls_targets}")
    print(f"Selected root    : {selected_root}")
    print(f"Mix root         : {mix_root}")
    print(f"Processed dir    : {processed_dir}")

    exp_e_hs, exp_e_ls = read_exp_e_manifests(project_root, exp_e_selected_root)
    clustered = read_quality_csv(clustered_csv, ["cluster_id", "record_id", "source_path", "quality_score"], "Clustered PhysioNet abnormal CSV")
    physionet_quality = read_quality_csv(physionet_quality_csv, ["source_path", "quality_score", "label_name"], "PhysioNet quality CSV")
    icbhi_quality = read_quality_csv(icbhi_quality_csv, ["source_path", "quality_score"], "ICBHI quality CSV")
    if "crop_lung_class" not in icbhi_quality.columns and "target_class" not in icbhi_quality.columns:
        raise RuntimeError("ICBHI quality CSV must contain crop_lung_class or target_class")

    used_hs_paths = path_set(exp_e_hs)
    used_hs_records = record_set(exp_e_hs)
    used_ls_paths = path_set(exp_e_ls)
    used_ls_records = record_set(exp_e_ls)

    def current_train_count(df: pd.DataFrame, cls: str) -> int:
        return int(((df["split"] == "train") & (df["target_class"].map(class_key) == class_key(cls))).sum())

    n_extra_hs_normal = hs_targets["normal"] - current_train_count(exp_e_hs, "normal")
    n_extra_hs_c2 = hs_targets["abnormal_cluster_2"] - current_train_count(exp_e_hs, "abnormal_cluster_2")
    n_extra_hs_c1 = hs_targets["abnormal_cluster_1"] - current_train_count(exp_e_hs, "abnormal_cluster_1")

    n_extra_ls_normal = ls_targets["normal"] - current_train_count(exp_e_ls, "normal")
    n_extra_ls_w = ls_targets["wheezes"] - current_train_count(exp_e_ls, "wheezes")
    n_extra_ls_c = ls_targets["crackles"] - current_train_count(exp_e_ls, "crackles")

    extras = [n_extra_hs_normal, n_extra_hs_c2, n_extra_hs_c1, n_extra_ls_normal, n_extra_ls_w, n_extra_ls_c]
    if min(extras) < 0:
        raise RuntimeError(f"Variant target is smaller than EXP_E current train counts. Extra counts: {extras}")

    extra_hs_normal = select_hs_normal_extras(
        physionet_quality=physionet_quality,
        n_extra=n_extra_hs_normal,
        used_paths=used_hs_paths,
        used_records=used_hs_records,
        no_fallback=no_fallback,
    )
    used_hs_paths |= path_set(extra_hs_normal)
    used_hs_records |= record_set(extra_hs_normal)

    extra_hs_c2 = select_hs_cluster_extras(
        clustered=clustered,
        cluster_id=2,
        n_extra=n_extra_hs_c2,
        used_paths=used_hs_paths,
        used_records=used_hs_records,
        exclude_record_ids=exclude_record_ids,
        allow_hard_records=allow_hard_records,
        reason="extra_C2_by_quality_score_no_C0",
    )
    used_hs_paths |= path_set(extra_hs_c2)
    used_hs_records |= record_set(extra_hs_c2)

    extra_hs_c1 = select_hs_cluster_extras(
        clustered=clustered,
        cluster_id=1,
        n_extra=n_extra_hs_c1,
        used_paths=used_hs_paths,
        used_records=used_hs_records,
        exclude_record_ids=exclude_record_ids,
        allow_hard_records=allow_hard_records,
        reason="extra_C1_easy_by_quality_score_no_hard_records",
    )
    used_hs_paths |= path_set(extra_hs_c1)
    used_hs_records |= record_set(extra_hs_c1)

    current_ls = exp_e_ls.copy()
    extra_ls_normal = select_ls_class_extras(
        icbhi_quality=icbhi_quality,
        class_name="normal",
        n_extra=n_extra_ls_normal,
        used_paths=used_ls_paths,
        used_records=used_ls_records,
        current_ls_manifest=current_ls,
        no_fallback=no_fallback,
    )
    used_ls_paths |= path_set(extra_ls_normal)
    used_ls_records |= record_set(extra_ls_normal)
    current_ls = pd.concat([current_ls, extra_ls_normal], ignore_index=True, sort=False)

    extra_ls_w = select_ls_class_extras(
        icbhi_quality=icbhi_quality,
        class_name="wheezes",
        n_extra=n_extra_ls_w,
        used_paths=used_ls_paths,
        used_records=used_ls_records,
        current_ls_manifest=current_ls,
        no_fallback=no_fallback,
    )
    used_ls_paths |= path_set(extra_ls_w)
    used_ls_records |= record_set(extra_ls_w)
    current_ls = pd.concat([current_ls, extra_ls_w], ignore_index=True, sort=False)

    extra_ls_c = select_ls_class_extras(
        icbhi_quality=icbhi_quality,
        class_name="crackles",
        n_extra=n_extra_ls_c,
        used_paths=used_ls_paths,
        used_records=used_ls_records,
        current_ls_manifest=current_ls,
        no_fallback=no_fallback,
    )

    selection_report = {
        "n_extra_hs_normal": int(n_extra_hs_normal),
        "n_extra_hs_c2": int(n_extra_hs_c2),
        "n_extra_hs_c1_easy": int(n_extra_hs_c1),
        "n_extra_ls_normal": int(n_extra_ls_normal),
        "n_extra_ls_wheezes": int(n_extra_ls_w),
        "n_extra_ls_crackles": int(n_extra_ls_c),
        "allow_hard_records": bool(allow_hard_records),
        "no_fallback": bool(no_fallback),
        "exclude_record_ids": sorted(list(exclude_record_ids)),
        "selected_extra_hs_normal_record_ids": extra_hs_normal.get("record_id", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_hs_c2_record_ids": extra_hs_c2.get("record_id", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_hs_c1_record_ids": extra_hs_c1.get("record_id", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_ls_normal_record_ids": extra_ls_normal.get("record_id", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_ls_wheezes_record_ids": extra_ls_w.get("record_id", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_ls_crackles_record_ids": extra_ls_c.get("record_id", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_ls_normal_reasons": extra_ls_normal.get("selection_reason", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_ls_wheezes_reasons": extra_ls_w.get("selection_reason", pd.Series(dtype=str)).astype(str).tolist(),
        "selected_extra_ls_crackles_reasons": extra_ls_c.get("selection_reason", pd.Series(dtype=str)).astype(str).tolist(),
    }
    print("\nSelection report:")
    print(json.dumps(selection_report, indent=2, sort_keys=True))

    hs_manifest, ls_manifest = write_fixed_manifest(
        expb=expb,
        base_hs=exp_e_hs,
        base_ls=exp_e_ls,
        extra_hs_by_origin=[
            ("HSG_NORMAL", "extra_normal_train", extra_hs_normal),
            ("HSG_C2", "extra_C2_train", extra_hs_c2),
            ("HSG_C1EASY", "extra_C1_easy_train", extra_hs_c1),
        ],
        extra_ls_by_origin=[
            ("LSG_NORMAL", "extra_normal_train", extra_ls_normal),
            ("LSG_W", "extra_wheezes_train", extra_ls_w),
            ("LSG_C", "extra_crackles_train", extra_ls_c),
        ],
        selected_root=selected_root,
        variant_name=variant_name,
        overwrite=overwrite,
    )

    disjoint = expb.check_source_disjoint(hs_manifest, ls_manifest)
    assert_targets(hs_manifest, ls_manifest, hs_targets, ls_targets)
    exp_counts = expected_counts(expb, hs_manifest, ls_manifest, snrs)

    print("\nSource-disjoint audit:")
    print(json.dumps(disjoint, indent=2, sort_keys=True))
    print("\nExpected counts:")
    print(json.dumps(exp_counts, indent=2, sort_keys=True))
    print("\nHS distribution:")
    print(pd.crosstab(hs_manifest["target_class"], hs_manifest["split"]).to_string())
    print("\nLS distribution:")
    print(pd.crosstab(ls_manifest["target_class"], ls_manifest["split"]).to_string())

    metadata, _split_df = expb.create_mixtures_for_cluster(
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

    actual_counts = {
        "train_bases": int((metadata["split"] == "train").sum()),
        "val_bases": int((metadata["split"] == "val").sum()),
        "train_segments": int((manifest["split"] == "train").sum()),
        "val_segments": int((manifest["split"] == "val").sum()),
    }
    expected_subset = {k: exp_counts[k] for k in actual_counts}
    if actual_counts != expected_subset:
        raise RuntimeError(f"Unexpected generated counts. Expected={expected_subset}, actual={actual_counts}")

    write_summary(
        variant_name=variant_name,
        selected_root=selected_root,
        mix_root=mix_root,
        processed_dir=processed_dir,
        hs_manifest=hs_manifest,
        ls_manifest=ls_manifest,
        metadata=metadata,
        manifest=manifest,
        expected=exp_counts,
        selection_report=selection_report,
        disjoint=disjoint,
    )

    result = {
        "variant_key": variant_key,
        "experiment_name": variant_name,
        "selected_root": str(selected_root),
        "mix_root": str(mix_root),
        "processed_dir": str(processed_dir),
        "split_csv": str(processed_dir / "source_disjoint_split_smoke.csv"),
        **exp_counts,
    }

    index_path = project_root / "outputs" / f"{variant_name}_dataset_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(index_path, index=False)

    print("\n" + "=" * 100)
    print(f"{variant_name} CREATED")
    print("=" * 100)
    print(f"Selected root : {selected_root}")
    print(f"Mix root      : {mix_root}")
    print(f"Processed dir : {processed_dir}")
    print(f"Split CSV     : {processed_dir / 'source_disjoint_split_smoke.csv'}")
    print("Base triplets:")
    print(metadata["split"].value_counts(dropna=False).to_string())
    print("Segments:")
    print(manifest["split"].value_counts(dropna=False).to_string())
    print("Additivity:")
    print(metadata[["corr_m_h_plus_l", "residual_snr_m_vs_h_plus_l_db"]].describe().to_string())
    print(f"Index CSV: {index_path}")
    print("=" * 100)
    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--exp-e-selected-root", type=Path, default=None)
    parser.add_argument("--clustered-csv", type=Path, default=None)
    parser.add_argument("--physionet-quality-csv", type=Path, default=None)
    parser.add_argument("--icbhi-quality-csv", type=Path, default=None)
    parser.add_argument("--snrs", nargs="+", default=[str(int(x)) for x in DEFAULT_SNRS])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-hard-records", action="store_true", help="Allow known hard PhysioNet record IDs. Not recommended.")
    parser.add_argument("--exclude-record-ids", nargs="*", default=sorted(DEFAULT_HARD_RECORD_IDS))
    parser.add_argument("--no-fallback", action="store_true", help="Fail if hard-quality pools are insufficient instead of using ranked fallback.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root: Path = args.project_root

    clustered_csv = args.clustered_csv or (project_root / "outputs" / "physionet_abnormal_acoustic_clustering" / "physionet_abnormal_clustered.csv")
    physionet_quality_csv = args.physionet_quality_csv or (project_root / "outputs" / "smoke_phy_ich_hsnormal_ls3_selection" / "physionet_all_candidates_quality.csv")
    icbhi_quality_csv = args.icbhi_quality_csv or (project_root / "outputs" / "smoke_phy_ich_hsnormal_ls3_selection" / "icbhi_all_candidates_quality.csv")
    snrs = parse_snr_values(args.snrs)
    exclude_record_ids = set(str(x) for x in (args.exclude_record_ids or []))

    build_variant(
        expb=expb,
        variant_key="G_40x40",
        project_root=project_root,
        exp_e_selected_root=args.exp_e_selected_root,
        clustered_csv=clustered_csv,
        physionet_quality_csv=physionet_quality_csv,
        icbhi_quality_csv=icbhi_quality_csv,
        snrs=snrs,
        overwrite=bool(args.overwrite),
        allow_hard_records=bool(args.allow_hard_records),
        exclude_record_ids=exclude_record_ids,
        no_fallback=bool(args.no_fallback),
    )


if __name__ == "__main__":
    main()

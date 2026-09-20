#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_hflung_physionet_hs_leakage.py

Audit whether the PhysioNet HS sources used in the HF_Lung-selected external
validation dataset overlap with the EXP_H/ICBHI HS sources used for training.

Typical use
-----------
1) Check if the current external dataset uses HS sources that are also in EXP_H train.
2) Optionally create a replacement PhysioNet HS candidate CSV excluding EXP_H train
   sources, to regenerate the HF_Lung external dataset without training leakage.

The script is deliberately conservative. It compares sources using:
- source identifiers;
- path/file stems;
- optional waveform fingerprints/correlation on fixed 15 s, 4 kHz, DC-removed audio.

It does NOT modify any dataset by itself.
"""

from __future__ import annotations

import argparse
import json
import re
from math import gcd
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

TARGET_SR = 4000
DURATION_SEC = 15.0
TARGET_LEN = int(TARGET_SR * DURATION_SEC)
EPS = 1e-10
AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff"}

REPO_ROOT = Path(__file__).resolve().parents[2]

PATH_COL_CANDIDATES = [
    "resolved_path", "resolved_audio_path", "output_path", "fixed_path", "fixed_audio_path",
    "processed_path", "source_path", "audio_path", "wav_path", "path", "file_path",
]
ID_COL_CANDIDATES = [
    "source_id", "original_source_id", "record_id", "file_id", "name", "output_file",
    "filename", "audio_file", "source_name",
]
CLASS_COL_CANDIDATES = [
    "target_class", "class_name", "label", "diagnosis", "record_class", "cluster_id",
]


def norm_text(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def safe_key(x: object) -> str:
    s = norm_text(x).replace("\\", "/")
    s = Path(s).stem if ("/" in s or "." in Path(s).name) else s
    s = s.lower().strip()
    s = re.sub(r"^(m_|h_|l_)", "", s)
    s = re.sub(r"_s\d{3}_orig$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_to_actual:
            return lower_to_actual[cand.lower()]
    if required:
        raise RuntimeError(f"Cannot find any of {list(candidates)}. Available columns: {df.columns.tolist()}")
    return None


def resolve_existing_path(value: object, root: Optional[Path] = None) -> Optional[Path]:
    raw = norm_text(value)
    if not raw or raw.lower() == "nan":
        return None
    p = Path(raw)
    if p.exists():
        return p
    if root is not None:
        p2 = root / raw
        if p2.exists():
            return p2
    return p if raw else None


def row_path_values(row: pd.Series, root: Optional[Path] = None) -> List[str]:
    vals: List[str] = []
    for c in PATH_COL_CANDIDATES:
        if c in row.index:
            p = resolve_existing_path(row[c], root=root)
            if p is not None:
                vals.append(str(p))
                vals.append(p.name)
                vals.append(p.stem)
    return [v for v in vals if norm_text(v)]


def row_id_values(row: pd.Series) -> List[str]:
    vals: List[str] = []
    for c in ID_COL_CANDIDATES:
        if c in row.index:
            v = norm_text(row[c])
            if v:
                vals.append(v)
                vals.append(Path(v).name)
                vals.append(Path(v).stem)
    return vals


def source_keys_from_row(row: pd.Series, root: Optional[Path] = None) -> set[str]:
    keys = set()
    for v in row_id_values(row) + row_path_values(row, root=root):
        k = safe_key(v)
        if k:
            keys.add(k)
    # remove overly generic keys
    keys = {k for k in keys if len(k) >= 3 and k not in {"unknown", "nan", "wav"}}
    return keys


def find_external_hs_csv(external_selected_root: Path, external_hs_csv: Optional[Path]) -> Path:
    if external_hs_csv is not None:
        p = Path(external_hs_csv)
        if not p.exists():
            raise FileNotFoundError(f"External HS CSV not found: {p}")
        return p

    root = Path(external_selected_root)
    if not root.exists():
        raise FileNotFoundError(f"External selected root not found: {root}")

    candidates = []
    candidates.extend(root.glob("selected_hs_physionet*.csv"))
    candidates.extend(root.glob("*selected*hs*.csv"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-discover selected HS CSV under {root}. "
            "Pass --external-hs-csv explicitly."
        )
    # Prefer the direct manifest over combined files.
    candidates = sorted(candidates, key=lambda p: ("combined" in p.name.lower(), p.name))
    return candidates[0]


def read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def infer_split_series(df: pd.DataFrame) -> pd.Series:
    if "split" in df.columns:
        return df["split"].astype(str).str.lower().str.strip()
    if "subset" in df.columns:
        return df["subset"].astype(str).str.lower().str.strip()
    if "set" in df.columns:
        return df["set"].astype(str).str.lower().str.strip()
    # If no split is provided, mark unknown instead of assuming train.
    return pd.Series(["unknown"] * len(df), index=df.index)


def load_audio_vec(path: Path, sr: int = TARGET_SR, target_len: int = TARGET_LEN) -> np.ndarray:
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if int(orig_sr) != int(sr):
        g = gcd(int(sr), int(orig_sr))
        y = resample_poly(y, int(sr) // g, int(orig_sr) // g).astype(np.float32)
    y = y - float(np.mean(y))
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(y) < target_len:
        reps = int(np.ceil(target_len / max(len(y), 1)))
        y = np.tile(y, reps)[:target_len]
    else:
        y = y[:target_len]
    y = y - float(np.mean(y))
    norm = float(np.linalg.norm(y))
    if norm < EPS:
        return np.zeros(target_len, dtype=np.float32)
    return (y / norm).astype(np.float32)


def first_existing_audio_path(row: pd.Series, root: Optional[Path] = None) -> Optional[Path]:
    for c in PATH_COL_CANDIDATES:
        if c in row.index:
            p = resolve_existing_path(row[c], root=root)
            if p is not None and p.exists() and p.suffix.lower() in AUDIO_EXTS:
                return p
    return None


def build_source_table(df: pd.DataFrame, name: str, root: Optional[Path] = None) -> pd.DataFrame:
    split_s = infer_split_series(df)
    path_col = pick_col(df, PATH_COL_CANDIDATES, required=False)
    id_col = pick_col(df, ID_COL_CANDIDATES, required=False)
    class_col = pick_col(df, CLASS_COL_CANDIDATES, required=False)

    rows = []
    for idx, row in df.iterrows():
        keys = source_keys_from_row(row, root=root)
        audio_path = first_existing_audio_path(row, root=root)
        source_id = norm_text(row.get(id_col)) if id_col else ""
        cls = norm_text(row.get(class_col)) if class_col else ""
        rows.append({
            "row_idx": int(idx),
            "dataset": name,
            "split": split_s.loc[idx],
            "source_id": source_id,
            "class": cls,
            "audio_path": str(audio_path) if audio_path is not None else "",
            "keys_json": json.dumps(sorted(keys)),
            "n_keys": len(keys),
        })
    return pd.DataFrame(rows)


def keys_from_table_row(row: pd.Series) -> set[str]:
    try:
        return set(json.loads(row["keys_json"]))
    except Exception:
        return set()


def compute_best_audio_match(
    ext_paths: List[str],
    ref_paths: List[str],
    threshold: float,
) -> Dict[str, Tuple[float, str]]:
    ref_vecs: Dict[str, np.ndarray] = {}
    for p in tqdm(ref_paths, desc="Loading EXP_H reference audio", dynamic_ncols=True):
        if p:
            try:
                ref_vecs[p] = load_audio_vec(Path(p))
            except Exception as e:
                print(f"[WARN] Could not load reference audio {p}: {e}")

    out: Dict[str, Tuple[float, str]] = {}
    for p in tqdm(ext_paths, desc="Audio matching external HS", dynamic_ncols=True):
        if not p:
            out[p] = (float("nan"), "")
            continue
        try:
            v = load_audio_vec(Path(p))
        except Exception as e:
            print(f"[WARN] Could not load external audio {p}: {e}")
            out[p] = (float("nan"), "")
            continue
        best_corr = -np.inf
        best_path = ""
        for rp, rv in ref_vecs.items():
            if len(v) != len(rv):
                continue
            corr = float(np.dot(v, rv))
            if corr > best_corr:
                best_corr = corr
                best_path = rp
        out[p] = (best_corr if np.isfinite(best_corr) else float("nan"), best_path)
    return out


def audit_overlap(
    ext_tab: pd.DataFrame,
    exph_tab: pd.DataFrame,
    audio_fingerprint: bool,
    corr_threshold: float,
) -> Tuple[pd.DataFrame, dict]:
    exph_rows = []
    for _, r in exph_tab.iterrows():
        exph_rows.append({
            "row_idx": int(r["row_idx"]),
            "split": str(r["split"]).lower().strip(),
            "source_id": r.get("source_id", ""),
            "audio_path": r.get("audio_path", ""),
            "keys": keys_from_table_row(r),
        })

    ext_audio_paths = ext_tab["audio_path"].fillna("").astype(str).tolist()
    ref_audio_paths = sorted({r["audio_path"] for r in exph_rows if r["audio_path"]})
    audio_matches = {}
    if audio_fingerprint and ext_audio_paths and ref_audio_paths:
        audio_matches = compute_best_audio_match(ext_audio_paths, ref_audio_paths, corr_threshold)

    audit_rows = []
    for _, erow in ext_tab.iterrows():
        ekeys = keys_from_table_row(erow)
        id_path_matches = []
        id_path_train = []
        id_path_val = []
        id_path_unknown = []
        for rr in exph_rows:
            inter = ekeys.intersection(rr["keys"])
            if inter:
                item = {
                    "split": rr["split"],
                    "source_id": rr["source_id"],
                    "audio_path": rr["audio_path"],
                    "matching_keys": sorted(inter),
                }
                id_path_matches.append(item)
                if rr["split"] == "train":
                    id_path_train.append(item)
                elif rr["split"] == "val":
                    id_path_val.append(item)
                else:
                    id_path_unknown.append(item)

        ap = str(erow.get("audio_path", ""))
        best_corr, best_path = audio_matches.get(ap, (float("nan"), ""))
        best_audio_split = ""
        if best_path:
            hit = exph_tab[exph_tab["audio_path"].astype(str).eq(best_path)]
            if len(hit):
                best_audio_split = str(hit.iloc[0]["split"]).lower().strip()

        audio_overlap = bool(np.isfinite(best_corr) and best_corr >= corr_threshold)
        train_overlap = bool(id_path_train) or (audio_overlap and best_audio_split == "train")
        val_overlap = bool(id_path_val) or (audio_overlap and best_audio_split == "val")
        any_overlap = bool(id_path_matches) or audio_overlap

        audit_rows.append({
            "external_row_idx": int(erow["row_idx"]),
            "external_source_id": erow.get("source_id", ""),
            "external_class": erow.get("class", ""),
            "external_audio_path": ap,
            "id_path_overlap_any": bool(id_path_matches),
            "id_path_overlap_train": bool(id_path_train),
            "id_path_overlap_val": bool(id_path_val),
            "id_path_overlap_unknown": bool(id_path_unknown),
            "id_path_matches_json": json.dumps(id_path_matches),
            "best_audio_corr": best_corr,
            "best_audio_match_path": best_path,
            "best_audio_match_split": best_audio_split,
            "audio_overlap_at_threshold": audio_overlap,
            "overlap_any": any_overlap,
            "overlap_train": train_overlap,
            "overlap_val": val_overlap,
            "verdict": "TRAIN_OVERLAP" if train_overlap else ("VAL_OVERLAP_ONLY" if val_overlap else ("NONTRAIN_OR_UNSEEN" if not any_overlap else "OVERLAP_UNKNOWN")),
        })

    audit_df = pd.DataFrame(audit_rows)
    summary = {
        "external_hs_count": int(len(audit_df)),
        "exph_hs_count": int(len(exph_tab)),
        "exph_train_count": int((exph_tab["split"].astype(str).str.lower() == "train").sum()),
        "exph_val_count": int((exph_tab["split"].astype(str).str.lower() == "val").sum()),
        "overlap_any_count": int(audit_df["overlap_any"].sum()),
        "overlap_train_count": int(audit_df["overlap_train"].sum()),
        "overlap_val_count": int(audit_df["overlap_val"].sum()),
        "audio_fingerprint_enabled": bool(audio_fingerprint),
        "audio_corr_threshold": float(corr_threshold),
    }
    summary["overall_verdict"] = (
        "TRAIN_LEAKAGE_RISK" if summary["overlap_train_count"] > 0
        else "NO_EXP_H_TRAIN_OVERLAP_DETECTED"
    )
    return audit_df, summary


def hard_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["passes_hard_quality", "quality_pass"]:
        if col in out.columns:
            mask = out[col].astype(str).str.lower().isin(["true", "1", "yes"]) | (out[col] == True)
            out = out[mask].copy()
    return out


def sort_quality(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["quality_score", "quality", "score", "seg_rms_p10", "heart_band_ratio_20_500", "fixed_rms"] if c in df.columns]
    if cols:
        return df.sort_values(cols, ascending=[False] * len(cols)).copy()
    return df.copy()


def create_candidate_csv(
    quality_csv: Path,
    exph_tab: pd.DataFrame,
    out_csv: Path,
    exclude_split: str,
    n_hs: int,
    audio_fingerprint: bool,
    corr_threshold: float,
) -> pd.DataFrame:
    qdf = read_csv(quality_csv)
    qdf = hard_quality_filter(qdf)
    qdf = sort_quality(qdf)

    # Build exclude table.
    split_norm = exph_tab["split"].astype(str).str.lower().str.strip()
    if exclude_split == "train":
        exclude_tab = exph_tab[split_norm.eq("train")].copy()
    elif exclude_split == "all":
        exclude_tab = exph_tab.copy()
    elif exclude_split == "none":
        exclude_tab = exph_tab.iloc[0:0].copy()
    else:
        raise RuntimeError(f"Invalid exclude split: {exclude_split}")

    exclude_keys = set()
    for _, r in exclude_tab.iterrows():
        exclude_keys.update(keys_from_table_row(r))

    exclude_audio_paths = sorted({p for p in exclude_tab["audio_path"].fillna("").astype(str).tolist() if p})
    exclude_vecs: Dict[str, np.ndarray] = {}
    if audio_fingerprint and exclude_audio_paths:
        for p in tqdm(exclude_audio_paths, desc=f"Loading exclude audio ({exclude_split})", dynamic_ncols=True):
            try:
                exclude_vecs[p] = load_audio_vec(Path(p))
            except Exception as e:
                print(f"[WARN] Could not load exclude audio {p}: {e}")

    selected_rows = []
    rejected = []
    for idx, row in tqdm(qdf.iterrows(), total=len(qdf), desc="Selecting non-training PhysioNet HS", dynamic_ncols=True):
        row_keys = source_keys_from_row(row)
        key_hit = bool(row_keys.intersection(exclude_keys))
        audio_path = first_existing_audio_path(row)
        best_corr = float("nan")
        best_match = ""
        audio_hit = False
        if audio_fingerprint and audio_path is not None and exclude_vecs:
            try:
                v = load_audio_vec(audio_path)
                best_corr = -np.inf
                for ep, ev in exclude_vecs.items():
                    corr = float(np.dot(v, ev))
                    if corr > best_corr:
                        best_corr = corr
                        best_match = ep
                audio_hit = bool(np.isfinite(best_corr) and best_corr >= corr_threshold)
            except Exception as e:
                print(f"[WARN] Could not audio-check candidate {audio_path}: {e}")

        if key_hit or audio_hit:
            rejected.append({
                "candidate_idx": int(idx),
                "reason": "key_hit" if key_hit else "audio_hit",
                "audio_path": str(audio_path) if audio_path is not None else "",
                "best_corr": best_corr,
                "best_match": best_match,
            })
            continue

        d = row.to_dict()
        d["audit_selected_for_hflung_regen"] = True
        d["audit_exclude_split"] = exclude_split
        d["audit_best_corr_to_excluded"] = best_corr
        d["audit_best_match_to_excluded"] = best_match
        selected_rows.append(d)
        if len(selected_rows) >= int(n_hs):
            break

    sel = pd.DataFrame(selected_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_csv, index=False)
    rej_csv = out_csv.with_name(out_csv.stem + "_rejected.csv")
    pd.DataFrame(rejected).to_csv(rej_csv, index=False)

    if len(sel) < int(n_hs):
        print(f"[WARN] Only selected {len(sel)} candidates, requested {n_hs}.")
    print(f"Replacement HS candidate CSV saved: {out_csv}")
    print(f"Rejected candidate log saved: {rej_csv}")
    return sel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=REPO_ROOT)
    p.add_argument("--external-selected-root", type=Path, default=None, help="Root like dataset/HFLUNG_SELECTED_20X20_EXTERNAL_VAL_SELECTED")
    p.add_argument("--external-hs-csv", type=Path, default=None, help="Explicit selected_hs_physionet*.csv")
    p.add_argument("--exph-hs-selected-csv", type=Path, default=None, help="EXP_H selected_hs_EXP_H_FULL_BOTH.csv")
    p.add_argument("--physionet-quality-csv", type=Path, default=None, help="PhysioNet quality CSV, needed only to create replacement HS CSV")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--audio-fingerprint", action="store_true", default=True)
    p.add_argument("--no-audio-fingerprint", dest="audio_fingerprint", action="store_false")
    p.add_argument("--corr-threshold", type=float, default=0.995)
    p.add_argument("--make-nontrain-hs-csv", type=Path, default=None, help="Output replacement candidate CSV excluding EXP_H train/all")
    p.add_argument("--exclude-exp-h-split", type=str, default="train", choices=["train", "all", "none"], help="For replacement CSV: exclude EXP_H train only, EXP_H all, or none")
    p.add_argument("--n-hs", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)

    external_selected_root = args.external_selected_root or (project_root / "dataset" / "HFLUNG_SELECTED_20X20_EXTERNAL_VAL_SELECTED")
    exph_hs_csv = args.exph_hs_selected_csv or (project_root / "dataset" / "EXP_H_FULL_BOTH_SELECTED" / "selected_hs_EXP_H_FULL_BOTH.csv")
    out_dir = args.out_dir or (project_root / "outputs" / "domain_gap" / "HFLUNG_PHYSIONET_20X20_HS_LEAKAGE_AUDIT")
    out_dir.mkdir(parents=True, exist_ok=True)

    external_hs_csv = find_external_hs_csv(external_selected_root, args.external_hs_csv)

    print("=" * 100)
    print("HF_LUNG/PHYSIONET EXTERNAL VALIDATION — HS LEAKAGE AUDIT")
    print("=" * 100)
    print(f"External selected root : {external_selected_root}")
    print(f"External HS CSV       : {external_hs_csv}")
    print(f"EXP_H HS selected CSV : {exph_hs_csv}")
    print(f"Output dir            : {out_dir}")
    print(f"Audio fingerprint     : {args.audio_fingerprint} | threshold={args.corr_threshold}")
    print()

    ext_df = read_csv(external_hs_csv)
    exph_df = read_csv(exph_hs_csv)

    ext_tab = build_source_table(ext_df, name="HF_Lung_external_HS")
    exph_tab = build_source_table(exph_df, name="EXP_H_HS")

    ext_tab.to_csv(out_dir / "external_hs_source_table.csv", index=False)
    exph_tab.to_csv(out_dir / "exph_hs_source_table.csv", index=False)

    audit_df, summary = audit_overlap(
        ext_tab=ext_tab,
        exph_tab=exph_tab,
        audio_fingerprint=bool(args.audio_fingerprint),
        corr_threshold=float(args.corr_threshold),
    )

    audit_csv = out_dir / "hflung_external_hs_vs_exph_overlap_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append("HF_Lung/PhysioNet external validation — HS overlap audit")
    lines.append("=" * 100)
    lines.append("")
    lines.append(json.dumps(summary, indent=2, sort_keys=True))
    lines.append("")
    lines.append("Verdict reading:")
    lines.append("- TRAIN_LEAKAGE_RISK: at least one external HS source matches EXP_H train.")
    lines.append("- NO_EXP_H_TRAIN_OVERLAP_DETECTED: no match against EXP_H train was found.")
    lines.append("- VAL_OVERLAP_ONLY rows mean the HS source was present in EXP_H validation, not training.")
    lines.append("")
    lines.append("Per-source verdict counts:")
    lines.append(str(audit_df["verdict"].value_counts(dropna=False)))
    lines.append("")
    lines.append(f"Audit CSV: {audit_csv}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("AUDIT SUMMARY")
    print("-" * 100)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print()
    print("Per-source verdict counts:")
    print(audit_df["verdict"].value_counts(dropna=False))
    print()
    print(f"Audit CSV    : {audit_csv}")
    print(f"Summary TXT  : {out_dir / 'summary.txt'}")
    print(f"Summary JSON : {summary_json}")

    if args.make_nontrain_hs_csv is not None:
        if args.physionet_quality_csv is None:
            raise RuntimeError("--make-nontrain-hs-csv requires --physionet-quality-csv")
        create_candidate_csv(
            quality_csv=Path(args.physionet_quality_csv),
            exph_tab=exph_tab,
            out_csv=Path(args.make_nontrain_hs_csv),
            exclude_split=str(args.exclude_exp_h_split),
            n_hs=int(args.n_hs),
            audio_fingerprint=bool(args.audio_fingerprint),
            corr_threshold=float(args.corr_threshold),
        )

    print("=" * 100)


if __name__ == "__main__":
    main()

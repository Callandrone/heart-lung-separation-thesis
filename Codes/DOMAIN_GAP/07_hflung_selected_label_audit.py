#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_hflung_selected_label_audit.py

Goal:
Attach HF_Lung event labels to the HF_Lung files selected by the similarity audit.

Inputs:
- hflung_files_ranked_by_similarity_to_selected_icbhi.csv
- HF_Lung raw/extracted folder with .wav and corresponding *_label.txt files

Outputs:
- per-file label summary
- group summaries for:
  top50
  very_close_p95_1
  close
  partial
  far

Important:
HF_Lung labels are event-level sound annotations, not patient-level diagnoses.
"normal-like" here means: no adventitious sound label found in the 15-second recording.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(REPO_ROOT))).expanduser().resolve()

DEFAULT_AUDIT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "domain_gap"
    / "HFLUNG_trend_vs_SELECTED_ICBHI"
)

DEFAULT_HFLUNG_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "HF_Lung_V1"
)

DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "domain_gap"
    / "HFLUNG_SELECTED_LABEL_SUMMARY"
)


ADVENTITIOUS_KEYWORDS = {
    "wheeze": ["wheeze", "wheezing"],
    "stridor": ["stridor"],
    "rhonchi": ["rhonchi", "rhonchus", "rhonchi_sound"],
    "crackle": ["crackle", "crackles", "crackling"],
}

BREATH_PHASE_KEYWORDS = {
    "inhalation": ["inhalation", "inspiration", "inspiratory", "inhale"],
    "exhalation": ["exhalation", "expiration", "expiratory", "exhale"],
}


def infer_file_type(name: str) -> str:
    name = str(name)
    if name.startswith("trunc_"):
        return "trunc"
    if name.startswith("steth_"):
        return "steth"
    return "other"


def infer_location(name: str) -> str:
    name = str(name)
    m = re.search(r"-L(\d+)_", name)
    if m:
        return "L" + m.group(1)
    return "unknown"


def infer_split(path: str) -> str:
    p = str(path).lower()
    if "/train/" in p:
        return "train"
    if "/test/" in p:
        return "test"
    return "unknown"


def resolve_audio_path(row, hflung_root: Path) -> Path:
    """
    Prefer absolute source_file if present.
    Otherwise use hflung_root / relative_path.
    """
    sf = Path(str(row["source_file"]))
    if sf.exists():
        return sf

    rel = Path(str(row["relative_path"]))
    p = hflung_root / rel
    if p.exists():
        return p

    # fallback: search by source_name
    matches = list(hflung_root.rglob(str(row["source_name"])))
    if matches:
        return matches[0]

    return sf


def find_label_path(audio_path: Path) -> Path | None:
    """
    README says label file has same beginning as audio file and suffix _label.
    Example:
        file.wav -> file_label.txt
    """
    candidates = [
        audio_path.with_name(audio_path.stem + "_label.txt"),
        audio_path.with_name(audio_path.stem + "_label.TXT"),
        audio_path.with_suffix("").with_name(audio_path.stem + "_label").with_suffix(".txt"),
    ]

    for c in candidates:
        if c.exists():
            return c

    # fallback: search in same folder by stem
    local = list(audio_path.parent.glob(audio_path.stem + "*label*.txt"))
    if local:
        return local[0]

    return None


def read_label_text(label_path: Path) -> str:
    try:
        return label_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def count_keywords(text: str, keyword_map: dict) -> dict:
    text_l = text.lower()
    out = {}

    for label, keys in keyword_map.items():
        count = 0
        for k in keys:
            # word-boundary-ish search
            count += len(re.findall(rf"(?<![a-zA-Z]){re.escape(k)}(?![a-zA-Z])", text_l))
        out[label] = count

    return out


def parse_label_rows(label_path: Path):
    """
    Best-effort parser.
    It keeps raw lines and tries to read time intervals if present.

    It does not assume one exact HF_Lung format.
    """
    text = read_label_text(label_path)
    rows = []

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        low = raw.lower()
        tokens = re.split(r"[,\t; ]+", low)

        numeric = []
        for t in tokens:
            try:
                numeric.append(float(t))
            except Exception:
                pass

        duration = None
        if len(numeric) >= 2:
            duration = max(0.0, numeric[1] - numeric[0])

        detected = []

        for label, keys in ADVENTITIOUS_KEYWORDS.items():
            if any(k in low for k in keys):
                detected.append(label)

        for label, keys in BREATH_PHASE_KEYWORDS.items():
            if any(k in low for k in keys):
                detected.append(label)

        rows.append({
            "raw_line": raw,
            "duration": duration,
            "detected_labels": "|".join(sorted(set(detected))),
        })

    return rows


def summarize_label_file(label_path: Path | None):
    if label_path is None or not label_path.exists():
        return {
            "label_file_found": False,
            "label_path": "",
            "n_label_lines": 0,
            "has_wheeze": False,
            "has_stridor": False,
            "has_rhonchi": False,
            "has_crackle": False,
            "has_continuous_adventitious": False,
            "has_discontinuous_adventitious": False,
            "has_any_adventitious": False,
            "has_inhalation": False,
            "has_exhalation": False,
            "wheeze_count": 0,
            "stridor_count": 0,
            "rhonchi_count": 0,
            "crackle_count": 0,
            "inhalation_count": 0,
            "exhalation_count": 0,
            "wheeze_duration": 0.0,
            "stridor_duration": 0.0,
            "rhonchi_duration": 0.0,
            "crackle_duration": 0.0,
            "condition_group": "label_missing",
        }

    text = read_label_text(label_path)
    adv_counts = count_keywords(text, ADVENTITIOUS_KEYWORDS)
    phase_counts = count_keywords(text, BREATH_PHASE_KEYWORDS)
    rows = parse_label_rows(label_path)

    durations = {
        "wheeze": 0.0,
        "stridor": 0.0,
        "rhonchi": 0.0,
        "crackle": 0.0,
    }

    for r in rows:
        dur = r["duration"]
        if dur is None:
            continue

        labels = str(r["detected_labels"]).split("|")
        for lab in durations:
            if lab in labels:
                durations[lab] += float(dur)

    has_wheeze = adv_counts["wheeze"] > 0
    has_stridor = adv_counts["stridor"] > 0
    has_rhonchi = adv_counts["rhonchi"] > 0
    has_crackle = adv_counts["crackle"] > 0

    has_cont = has_wheeze or has_stridor or has_rhonchi
    has_disc = has_crackle
    has_any = has_cont or has_disc

    if not has_any:
        condition = "no_adventitious_normal_like"
    elif has_cont and not has_disc:
        labs = []
        if has_wheeze:
            labs.append("wheeze")
        if has_stridor:
            labs.append("stridor")
        if has_rhonchi:
            labs.append("rhonchi")
        condition = "continuous_" + "_".join(labs)
    elif has_disc and not has_cont:
        condition = "crackles_only"
    else:
        labs = []
        if has_wheeze:
            labs.append("wheeze")
        if has_stridor:
            labs.append("stridor")
        if has_rhonchi:
            labs.append("rhonchi")
        if has_crackle:
            labs.append("crackle")
        condition = "mixed_" + "_".join(labs)

    return {
        "label_file_found": True,
        "label_path": str(label_path),
        "n_label_lines": len(rows),

        "has_wheeze": has_wheeze,
        "has_stridor": has_stridor,
        "has_rhonchi": has_rhonchi,
        "has_crackle": has_crackle,

        "has_continuous_adventitious": has_cont,
        "has_discontinuous_adventitious": has_disc,
        "has_any_adventitious": has_any,

        "has_inhalation": phase_counts["inhalation"] > 0,
        "has_exhalation": phase_counts["exhalation"] > 0,

        "wheeze_count": adv_counts["wheeze"],
        "stridor_count": adv_counts["stridor"],
        "rhonchi_count": adv_counts["rhonchi"],
        "crackle_count": adv_counts["crackle"],
        "inhalation_count": phase_counts["inhalation"],
        "exhalation_count": phase_counts["exhalation"],

        "wheeze_duration": durations["wheeze"],
        "stridor_duration": durations["stridor"],
        "rhonchi_duration": durations["rhonchi"],
        "crackle_duration": durations["crackle"],

        "condition_group": condition,
    }


def add_selection_groups(rank: pd.DataFrame) -> pd.DataFrame:
    rank = rank.copy()

    rank["is_top50"] = False
    rank.loc[rank.index[:50], "is_top50"] = True

    rank["is_very_close_p95_1"] = rank["frac_inside_selected_icbhi_p95"] >= 0.999999
    rank["is_close"] = rank["similarity_label"] == "close_to_selected_ICBHI"
    rank["is_partial"] = rank["similarity_label"] == "partial_overlap"
    rank["is_far"] = rank["similarity_label"] == "far_from_selected_ICBHI"

    return rank


def group_counts(df: pd.DataFrame, group_name: str, mask_col: str, field: str):
    sub = df[df[mask_col]].copy()

    vc = sub[field].value_counts(dropna=False)
    rows = []

    for value, count in vc.items():
        rows.append({
            "selection_group": group_name,
            "field": field,
            "value": value,
            "count": int(count),
            "fraction": float(count / len(sub)) if len(sub) else 0.0,
            "n_files_in_group": int(len(sub)),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_dir", type=str, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--hflung_root", type=str, default=DEFAULT_HFLUNG_ROOT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    hflung_root = Path(args.hflung_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_path = audit_dir / "hflung_files_ranked_by_similarity_to_selected_icbhi.csv"
    if not rank_path.exists():
        raise FileNotFoundError(rank_path)

    rank = pd.read_csv(rank_path)
    rank = add_selection_groups(rank)

    rank["file_type"] = rank["source_name"].apply(infer_file_type)
    rank["location"] = rank["source_name"].apply(infer_location)
    rank["split"] = rank["relative_path"].apply(infer_split)

    enriched_rows = []

    for i, row in rank.iterrows():
        audio_path = resolve_audio_path(row, hflung_root)
        label_path = find_label_path(audio_path)
        label_info = summarize_label_file(label_path)

        d = row.to_dict()
        d["resolved_audio_path"] = str(audio_path)
        d.update(label_info)
        enriched_rows.append(d)

    enriched = pd.DataFrame(enriched_rows)

    # Save per-file enriched table
    enriched_path = out_dir / "hflung_selected_files_with_labels.csv"
    enriched.to_csv(enriched_path, index=False)

    # Group summaries
    groups = [
        ("top50", "is_top50"),
        ("very_close_p95_1", "is_very_close_p95_1"),
        ("close_292", "is_close"),
        ("partial_193", "is_partial"),
        ("far_15", "is_far"),
    ]

    fields = [
        "condition_group",
        "has_any_adventitious",
        "has_wheeze",
        "has_stridor",
        "has_rhonchi",
        "has_crackle",
        "file_type",
        "location",
        "split",
    ]

    summary_rows = []
    for group_name, mask_col in groups:
        for field in fields:
            summary_rows.extend(group_counts(enriched, group_name, mask_col, field))

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "hflung_selected_label_group_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Compact condition pivot
    compact = (
        summary[summary["field"] == "condition_group"]
        .pivot_table(
            index="selection_group",
            columns="value",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
    )

    compact_path = out_dir / "hflung_selected_condition_pivot.csv"
    compact.to_csv(compact_path)

    # Also save adventitious boolean pivot
    bool_rows = []
    for group_name, mask_col in groups:
        sub = enriched[enriched[mask_col]].copy()
        n = len(sub)
        row = {
            "selection_group": group_name,
            "n_files": n,
            "label_files_found": int(sub["label_file_found"].sum()),
            "no_adventitious_normal_like": int((sub["condition_group"] == "no_adventitious_normal_like").sum()),
            "has_any_adventitious": int(sub["has_any_adventitious"].sum()),
            "has_wheeze": int(sub["has_wheeze"].sum()),
            "has_stridor": int(sub["has_stridor"].sum()),
            "has_rhonchi": int(sub["has_rhonchi"].sum()),
            "has_crackle": int(sub["has_crackle"].sum()),
        }

        for k in list(row.keys()):
            if k not in ["selection_group", "n_files"]:
                row[k + "_frac"] = float(row[k] / n) if n else 0.0

        bool_rows.append(row)

    bool_summary = pd.DataFrame(bool_rows)
    bool_path = out_dir / "hflung_selected_adventitious_boolean_summary.csv"
    bool_summary.to_csv(bool_path, index=False)

    # Print
    print("=" * 100)
    print("HF_Lung selected label summary")
    print("=" * 100)
    print(f"Rank file: {rank_path}")
    print(f"Output dir: {out_dir}")

    print("\nLabel files found:")
    print(f"{int(enriched['label_file_found'].sum())} / {len(enriched)}")

    print("\nAdventitious boolean summary:")
    print(bool_summary.to_string(index=False))

    print("\nCondition pivot:")
    print(compact.to_string())

    print("\nSaved:")
    print(enriched_path)
    print(summary_path)
    print(compact_path)
    print(bool_path)


if __name__ == "__main__":
    main()

"""
snr_filter.py
-------------
Exclude temporally uninformative windows from the dataset,
as identified by the full RMS/SNR analysis.

The analysis CSV contains only original segments:
    M_000028_s000_orig

Training and validation may also contain augmented versions:
    000028_s000_aug
    000028_s000_aug_...

When an original window is excluded, all augmented versions of that
window are excluded as well.
"""

import re
from pathlib import Path

import pandas as pd


def segment_window_key(name_or_sample_id: str) -> str:
    """
    Map original and augmented sample names to the same window key.

    Examples:
        M_000028_s000_orig -> 000028_s000
        000028_s000_orig   -> 000028_s000
        000028_s000_aug    -> 000028_s000
        000028_s000_aug_01 -> 000028_s000
    """
    name = Path(str(name_or_sample_id)).stem

    if name.startswith("M_"):
        name = name[2:]

    name = re.sub(r"_(orig|aug.*)$", "", name)

    return name


def load_excluded_window_keys(
    analysis_csv: str,
    threshold_db: float,
) -> set[str]:
    """
    Load the analysis CSV and return windows whose absolute local SNR
    exceeds the configured threshold.
    """
    path = Path(analysis_csv)

    if not path.exists():
        raise FileNotFoundError(
            f"Local-SNR analysis CSV not found: {path}"
        )

    df = pd.read_csv(
        path,
        usecols=["sample_id", "snr_local_db"],
    )

    df["snr_local_db"] = pd.to_numeric(
        df["snr_local_db"],
        errors="coerce",
    )

    bad = df[
        df["snr_local_db"].abs() > threshold_db
    ].dropna(subset=["sample_id", "snr_local_db"])

    excluded_keys = {
        segment_window_key(sample_id)
        for sample_id in bad["sample_id"].astype(str)
    }

    print()
    print("=" * 76)
    print("LOCAL-SNR DATASET FILTER")
    print("=" * 76)
    print(f"Analysis CSV              : {path}")
    print(f"Threshold                 : |SNR local| > {threshold_db:.1f} dB")
    print(f"Excluded original windows : {len(excluded_keys)}")
    print("=" * 76)

    return excluded_keys


def is_excluded_window(
    name_or_sample_id: str,
    excluded_keys: set[str],
) -> bool:
    """True when the segment belongs to an excluded window."""
    return segment_window_key(name_or_sample_id) in excluded_keys


def filter_dataset_indices(
    dataset_names: list[str],
    indices: list[int],
    excluded_keys: set[str],
) -> tuple[list[int], list[int]]:
    """
    Filter a list of TripletDataset indices.

    Returns:
        kept_indices, removed_indices
    """
    kept_indices = []
    removed_indices = []

    for idx in indices:
        name = dataset_names[idx]

        if is_excluded_window(name, excluded_keys):
            removed_indices.append(idx)
        else:
            kept_indices.append(idx)

    return kept_indices, removed_indices


def count_unique_windows(
    dataset_names: list[str],
    indices: list[int],
) -> int:
    """
    Count unique original windows represented by a list of indices.
    Training and validation may contain both original and augmented samples.
    """
    return len({
        segment_window_key(dataset_names[idx])
        for idx in indices
    })

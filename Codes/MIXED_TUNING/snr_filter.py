"""
snr_filter.py
-------------
Utility per escludere dal dataset le finestre temporalmente non informative,
identificate dall'analisi completa RMS/SNR.

Il CSV di analisi contiene soltanto i segmenti originali:
    M_000028_s000_orig

Durante training e validation possono però esistere anche versioni aumentate:
    000028_s000_aug
    000028_s000_aug_...

Se la finestra originale è problematica, vengono escluse anche tutte le
eventuali versioni augmentate della stessa finestra.
"""

import re
from pathlib import Path

import pandas as pd


def segment_window_key(name_or_sample_id: str) -> str:
    """
    Converte nomi originali o augmentati nella chiave della stessa finestra.

    Esempi:
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
    Carica il CSV prodotto dall'analisi completa e restituisce le finestre
    da escludere quando |SNR locale| supera la soglia impostata.
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
    """True se il segmento appartiene a una finestra esclusa."""
    return segment_window_key(name_or_sample_id) in excluded_keys


def filter_dataset_indices(
    dataset_names: list[str],
    indices: list[int],
    excluded_keys: set[str],
) -> tuple[list[int], list[int]]:
    """
    Filtra una lista di indici del TripletDataset.

    Restituisce:
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
    Conta le finestre originali uniche rappresentate in una lista di indici.
    Utile perché train/validation possono contenere orig + augmentations.
    """
    return len({
        segment_window_key(dataset_names[idx])
        for idx in indices
    })
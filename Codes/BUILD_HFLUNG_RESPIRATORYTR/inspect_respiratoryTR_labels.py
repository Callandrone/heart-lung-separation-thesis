#!/usr/bin/env python3
"""Inspect label sheets distributed with RespiratoryDatabase@TR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(REPO_ROOT))).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-xlsx",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "raw"
            / "respiratoryTR_p9z4h98s6j_v1"
            / "extracted"
            / "Labels.xlsx"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "domain_gap"
            / "RESPIRATORY_TR_REAL_MIXTURE_AUDIT"
        ),
    )
    return parser.parse_args()


def safe_sheet_name(sheet: str) -> str:
    return "".join(
        char if char.isalnum() or char in "_-" else "_"
        for char in sheet
    )


def main() -> None:
    args = parse_args()
    labels_xlsx = Path(args.labels_xlsx)
    out_dir = Path(args.out_dir)

    if not labels_xlsx.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_xlsx}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RESPIRATORY TR LABELS INSPECTION")
    print("=" * 100)
    print(f"Labels file: {labels_xlsx}")

    workbook = pd.ExcelFile(labels_xlsx)

    print("\nSheets:")
    for sheet in workbook.sheet_names:
        print(f" - {sheet}")

    for sheet in workbook.sheet_names:
        print("\n" + "=" * 100)
        print(f"SHEET: {sheet}")
        print("=" * 100)

        df = pd.read_excel(labels_xlsx, sheet_name=sheet)

        print(f"Shape: {df.shape}")
        print("Columns:")
        for column in df.columns:
            print(f" - {column!r}")

        out_csv = out_dir / f"labels_sheet_{safe_sheet_name(sheet)}.csv"
        df.to_csv(out_csv, index=False)

        print(f"Saved CSV: {out_csv}")
        print("\nHead:")
        print(df.head(20).to_string(index=False))
        print("\nNon-null counts:")
        print(df.notna().sum().to_string())

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()

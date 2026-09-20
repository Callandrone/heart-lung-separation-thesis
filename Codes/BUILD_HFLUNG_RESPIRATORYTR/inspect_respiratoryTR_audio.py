#!/usr/bin/env python3
"""Inspect the audio files contained in RespiratoryDatabase@TR."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract-root",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "raw"
            / "respiratoryTR_p9z4h98s6j_v1"
            / "extracted"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "domain_gap"
            / "RESPIRATORY_TR_REAL_MIXTURE_AUDIT"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_root = Path(args.extract_root)
    out_dir = Path(args.out_dir)

    if not extract_root.exists():
        raise FileNotFoundError(f"Extracted dataset root not found: {extract_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(extract_root.rglob("*.wav"))
    print(f"Audio files found: {len(audio_files)}")

    rows = []
    pattern = re.compile(
        r"^(?P<subject>[A-Za-z]\\d+)_(?P<side>[LR])(?P<loc>\\d+)$"
    )

    for path in tqdm(audio_files, desc="Inspecting audio"):
        match = pattern.match(path.stem)

        subject = match.group("subject") if match else ""
        side = match.group("side") if match else ""
        loc = match.group("loc") if match else ""
        location = f"{side}{loc}" if match else ""

        base = {
            "path": str(path),
            "relative_path": str(path.relative_to(extract_root)),
            "filename": path.name,
            "stem": path.stem,
            "subject": subject,
            "side": side,
            "loc": loc,
            "location": location,
        }

        try:
            info = sf.info(str(path))
            base.update(
                {
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "duration_sec": (
                        info.frames / info.samplerate if info.samplerate else None
                    ),
                    "format": info.format,
                    "subtype": info.subtype,
                    "parent": path.parent.name,
                }
            )
        except Exception as exc:
            base["error"] = str(exc)

        rows.append(base)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "respiratoryTR_audio_inventory.csv"
    df.to_csv(out_csv, index=False)

    print("=" * 100)
    print("RESPIRATORY TR AUDIO INVENTORY")
    print("=" * 100)
    print(f"Saved: {out_csv}")
    print(f"Audio files: {len(df)}")

    if df.empty:
        return

    print(f"Subjects: {df['subject'].nunique()}")

    if "samplerate" in df:
        print("\nSample rates:")
        print(df["samplerate"].value_counts(dropna=False).to_string())

    if "channels" in df:
        print("\nChannels:")
        print(df["channels"].value_counts(dropna=False).to_string())

    if "duration_sec" in df:
        print("\nDuration summary:")
        print(df["duration_sec"].describe().to_string())

    print("\nLocations:")
    print(df["location"].value_counts().sort_index().to_string())

    print("\nFiles per subject summary:")
    print(df.groupby("subject")["filename"].count().describe().to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build a processed real-mixture dataset from RespiratoryDatabase@TR.

Output contains only M_*.wav segments because clean H/L references are not available.
This is intended for an external real-mixture pseudo-label consistency audit,
not for supervised SI-SDR validation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def peak_normalise(x: np.ndarray, peak: float = 0.95, eps: float = 1e-8) -> np.ndarray:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    if p < eps:
        return x.astype(np.float32)
    return (x * (peak / p)).astype(np.float32)


def maybe_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr_in, sr_out)
        up = sr_out // g
        down = sr_in // g
        y = resample_poly(x, up, down)
        return y.astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Need to resample from {sr_in} to {sr_out}, but scipy resample failed: {e}"
        )


def parse_filename(path: Path) -> dict:
    # Expected: H002_L1.wav, H002_R3.wav
    m = re.match(r"^(?P<subject>[A-Za-z]\d+)_(?P<side>[LR])(?P<loc>\d+)$", path.stem)
    if not m:
        return {"subject": "", "side": "", "loc": "", "location": ""}
    subject = m.group("subject")
    side = m.group("side")
    loc = m.group("loc")
    return {
        "subject": subject,
        "side": side,
        "loc": loc,
        "location": f"{side}{loc}",
    }


def load_labels(labels_xlsx: Path | None) -> pd.DataFrame:
    if labels_xlsx is None or not labels_xlsx.exists():
        return pd.DataFrame(columns=["Patient ID", "Diagnosis"])

    # The useful sheet observed in this dataset is Sayfa1.
    xls = pd.ExcelFile(labels_xlsx)
    if "Sayfa1" in xls.sheet_names:
        df = pd.read_excel(labels_xlsx, sheet_name="Sayfa1")
    else:
        df = pd.read_excel(labels_xlsx, sheet_name=xls.sheet_names[0])

    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extract-root",
        default="/nas/home/pcallandrone/DeepLearning/dataset/raw/respiratoryTR_p9z4h98s6j_v1/extracted",
        help="Root folder containing extracted RespiratoryDatabase@TR files and Labels.xlsx.",
    )
    ap.add_argument(
        "--out-dir",
        default="/nas/home/pcallandrone/DeepLearning/dataset/processed/respiratoryTR_real_mixture_audit",
        help="Output processed dataset directory.",
    )
    ap.add_argument("--sr", type=int, default=4000)
    ap.add_argument("--seg-seconds", type=float, default=2.0)
    ap.add_argument("--hop-seconds", type=float, default=0.5)
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="Use only the first N seconds of each recording. Use <=0 to process full duration.",
    )
    ap.add_argument(
        "--normalise",
        choices=["none", "peak"],
        default="peak",
        help="Recording-level normalisation before segmentation.",
    )
    ap.add_argument("--target-peak", type=float, default=0.95)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    extract_root = Path(args.extract_root)
    out_dir = Path(args.out_dir)

    if not extract_root.exists():
        raise FileNotFoundError(f"extract-root not found: {extract_root}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output dir exists and is not empty: {out_dir}. Use --overwrite.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean old M_ files if overwrite.
    if args.overwrite:
        for p in out_dir.glob("M_*.wav"):
            p.unlink()

    labels_xlsx = extract_root / "Labels.xlsx"
    labels_df = load_labels(labels_xlsx)
    labels_map = {}
    if not labels_df.empty and "Patient ID" in labels_df.columns and "Diagnosis" in labels_df.columns:
        labels_map = dict(
            zip(
                labels_df["Patient ID"].astype(str).str.strip(),
                labels_df["Diagnosis"].astype(str).str.strip(),
            )
        )

    wav_files = sorted(extract_root.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No wav files found in {extract_root}")

    seg_samples = int(round(args.seg_seconds * args.sr))
    hop_samples = int(round(args.hop_seconds * args.sr))
    max_samples = int(round(args.max_seconds * args.sr)) if args.max_seconds and args.max_seconds > 0 else None

    manifest = []
    skipped = []

    for wav_path in tqdm(wav_files, desc="Building RespiratoryTR processed M segments"):
        meta = parse_filename(wav_path)
        subject = meta["subject"]
        diagnosis = labels_map.get(subject, "")

        try:
            audio, sr_in = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = maybe_resample(audio, sr_in=sr_in, sr_out=args.sr)

            if max_samples is not None:
                audio = audio[:max_samples]

            if args.normalise == "peak":
                audio = peak_normalise(audio, peak=args.target_peak)
            else:
                audio = audio.astype(np.float32)

            if len(audio) < seg_samples:
                skipped.append({"path": str(wav_path), "reason": "shorter_than_segment", "samples": len(audio)})
                continue

            n_segments = 1 + ((len(audio) - seg_samples) // hop_samples)

            for seg_idx in range(int(n_segments)):
                start = seg_idx * hop_samples
                end = start + seg_samples
                seg = audio[start:end].astype(np.float32)

                sample_id = f"M_{subject}_{meta['location']}_s{seg_idx:03d}_orig"
                out_path = out_dir / f"{sample_id}.wav"
                sf.write(str(out_path), seg, args.sr)

                manifest.append({
                    "sample_id": sample_id,
                    "filename": out_path.name,
                    "path": str(out_path),
                    "source_path": str(wav_path),
                    "source_filename": wav_path.name,
                    "subject": subject,
                    "diagnosis": diagnosis,
                    "side": meta["side"],
                    "loc": meta["loc"],
                    "location": meta["location"],
                    "source_sr": sr_in,
                    "sr": args.sr,
                    "source_duration_sec": len(audio) / args.sr,
                    "segment_index": seg_idx,
                    "start_sec": start / args.sr,
                    "end_sec": end / args.sr,
                    "normalise": args.normalise,
                })

        except Exception as e:
            skipped.append({"path": str(wav_path), "reason": repr(e)})

    manifest_df = pd.DataFrame(manifest)
    manifest_csv = out_dir / "manifest_real_mixture.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    skipped_df = pd.DataFrame(skipped)
    skipped_csv = out_dir / "skipped_files.csv"
    skipped_df.to_csv(skipped_csv, index=False)

    summary = {
        "wav_files": len(wav_files),
        "processed_segments": int(len(manifest_df)),
        "subjects": int(manifest_df["subject"].nunique()) if len(manifest_df) else 0,
        "locations": sorted(manifest_df["location"].dropna().unique().tolist()) if len(manifest_df) else [],
        "diagnoses": manifest_df["diagnosis"].value_counts(dropna=False).to_dict() if len(manifest_df) else {},
        "sr": args.sr,
        "seg_seconds": args.seg_seconds,
        "hop_seconds": args.hop_seconds,
        "max_seconds": args.max_seconds,
        "normalise": args.normalise,
        "out_dir": str(out_dir),
        "manifest_csv": str(manifest_csv),
        "skipped_files_csv": str(skipped_csv),
        "skipped_count": len(skipped_df),
    }

    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(summary_txt, "w") as f:
        f.write("RESPIRATORYDATABASE@TR REAL-MIXTURE PROCESSED DATASET\n")
        f.write("=" * 80 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("=" * 100)
    print("DATASET CREATED")
    print("=" * 100)
    print(f"Output dir : {out_dir}")
    print(f"Manifest   : {manifest_csv}")
    print(f"Summary    : {summary_txt}")
    print(f"Segments   : {len(manifest_df)}")
    print(f"Subjects   : {summary['subjects']}")
    print("Diagnoses:")
    print(manifest_df["diagnosis"].value_counts(dropna=False).to_string() if len(manifest_df) else "none")
    print("Locations:")
    print(manifest_df["location"].value_counts().sort_index().to_string() if len(manifest_df) else "none")
    print("Skipped:", len(skipped_df))
    print("=" * 100)


if __name__ == "__main__":
    main()

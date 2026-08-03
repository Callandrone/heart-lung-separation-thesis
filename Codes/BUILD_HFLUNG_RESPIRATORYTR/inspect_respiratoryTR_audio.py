from pathlib import Path
import re
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path("/nas/home/pcallandrone/DeepLearning")
EXTRACT = ROOT / "dataset/raw/respiratoryTR_p9z4h98s6j_v1/extracted"
OUT_DIR = ROOT / "outputs/domain_gap/RESPIRATORY_TR_REAL_MIXTURE_AUDIT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

audio_files = sorted(EXTRACT.rglob("*.wav"))

print("Audio files found:", len(audio_files))

rows = []

# Expected filename examples:
# H002_L1.wav, H002_R3.wav
pattern = re.compile(r"^(?P<subject>[A-Za-z]\d+)_(?P<side>[LR])(?P<loc>\d+)$")

for p in tqdm(audio_files, desc="Inspecting audio"):
    stem = p.stem
    m = pattern.match(stem)

    subject = ""
    side = ""
    loc = ""
    location = ""

    if m:
        subject = m.group("subject")
        side = m.group("side")
        loc = m.group("loc")
        location = f"{side}{loc}"

    try:
        info = sf.info(str(p))

        rows.append({
            "path": str(p),
            "relative_path": str(p.relative_to(EXTRACT)),
            "filename": p.name,
            "stem": stem,
            "subject": subject,
            "side": side,
            "loc": loc,
            "location": location,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_sec": info.frames / info.samplerate if info.samplerate else None,
            "format": info.format,
            "subtype": info.subtype,
            "parent": p.parent.name,
        })

    except Exception as e:
        rows.append({
            "path": str(p),
            "relative_path": str(p.relative_to(EXTRACT)),
            "filename": p.name,
            "stem": stem,
            "subject": subject,
            "side": side,
            "loc": loc,
            "location": location,
            "error": str(e),
        })

df = pd.DataFrame(rows)

out_csv = OUT_DIR / "respiratoryTR_audio_inventory.csv"
df.to_csv(out_csv, index=False)

print()
print("=" * 100)
print("RESPIRATORY TR AUDIO INVENTORY")
print("=" * 100)
print("Saved:", out_csv)
print()

print("Audio files:", len(df))
print("Subjects:", df["subject"].nunique())
print()

print("Sample rates:")
print(df["samplerate"].value_counts(dropna=False).to_string())
print()

print("Channels:")
print(df["channels"].value_counts(dropna=False).to_string())
print()

print("Duration summary:")
print(df["duration_sec"].describe().to_string())
print()

print("Locations:")
print(df["location"].value_counts().sort_index().to_string())
print()

print("Files per subject summary:")
sub_counts = df.groupby("subject")["filename"].count()
print(sub_counts.describe().to_string())
print()

print("First rows:")
print(df.head(30).to_string(index=False))
print("=" * 100)

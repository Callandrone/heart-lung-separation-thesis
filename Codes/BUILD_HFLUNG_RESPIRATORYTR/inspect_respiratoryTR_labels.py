from pathlib import Path
import pandas as pd

ROOT = Path("/nas/home/pcallandrone/DeepLearning")
EXTRACT = ROOT / "dataset/raw/respiratoryTR_p9z4h98s6j_v1/extracted"
LABELS_XLSX = EXTRACT / "Labels.xlsx"

OUT_DIR = ROOT / "outputs/domain_gap/RESPIRATORY_TR_REAL_MIXTURE_AUDIT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("RESPIRATORY TR LABELS INSPECTION")
print("=" * 100)
print("Labels file:", LABELS_XLSX)
print("Exists:", LABELS_XLSX.exists())
print()

xls = pd.ExcelFile(LABELS_XLSX)
print("Sheets:")
for s in xls.sheet_names:
    print(" -", s)
print()

for sheet in xls.sheet_names:
    print("=" * 100)
    print("SHEET:", sheet)
    print("=" * 100)

    df = pd.read_excel(LABELS_XLSX, sheet_name=sheet)

    print("Shape:", df.shape)
    print("Columns:")
    for c in df.columns:
        print(" -", repr(c))
    print()

    out_csv = OUT_DIR / f"labels_sheet_{sheet}.csv"
    safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in sheet)
    out_csv = OUT_DIR / f"labels_sheet_{safe_name}.csv"
    df.to_csv(out_csv, index=False)

    print("Saved CSV:", out_csv)
    print()
    print("Head:")
    print(df.head(20).to_string(index=False))
    print()

    print("Non-null counts:")
    print(df.notna().sum().to_string())
    print()

print("=" * 100)
print("DONE")
print("=" * 100)

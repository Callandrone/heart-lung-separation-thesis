"""Shared filesystem conventions; no model or scientific dependencies."""

import os
from pathlib import Path


def project_root() -> Path:
    return Path(os.environ.get("ESD_JASSNET_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()


def physical_fold() -> int:
    fold = int(os.environ.get("HLSCMDS_FOLD", "1"))
    if fold not in range(1, 6):
        raise ValueError("HLSCMDS_FOLD must be between 1 and 5.")
    return fold


def target_dir(root: Path, fold: int) -> Path:
    """Prefer the release layout; discover an existing historical layout."""
    explicit = os.environ.get("HLSCMDS_TARGET_DIR") or os.environ.get("HLSCMDS_DATASET_DIR")
    if explicit:
        return Path(explicit).expanduser()
    processed = root / "dataset" / "processed"
    canonical = processed / f"hlscmds_full_40x40_10x10_no_unused_fold{fold}"
    legacy = processed / f"torabi_full_40x40_10x10_no_unused_fold{fold}"
    return legacy if not canonical.exists() and legacy.exists() else canonical


def stage2_checkpoint(root: Path, fold: int) -> Path:
    return Path(os.environ.get(
        "ESD_JASSNET_STAGE2_CKPT",
        str(root / "outputs" / "checkpoints" / f"ESD_JASSNET_MIXED_FOLD{fold}" / "finetune_fold1_best.pt"),
    )).expanduser()


def pseudo_root(root: Path, fold: int) -> Path:
    """Fold-specific outputs prevent accidental reuse of another teacher's labels."""
    return root / "dataset" / "processed" / f"v1_real_ssl_pseudo_fold{fold}"

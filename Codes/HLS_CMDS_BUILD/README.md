# HLS-CMDS controlled-mixture fold construction

This directory contains the release builder used to create the controlled
HLS-CMDS target-domain datasets used for supervised adaptation and evaluation.

The builder uses only the isolated standalone heart-sound and lung-sound
recordings. The physical cardiopulmonary mixtures are not used to construct
these supervised folds.

## Input sources

The release repository contains the standalone sources under:

```text
HLS_CMDS_ALIGNED/
|-- HS/
`-- LS/
```

The builder automatically searches these repository-relative locations.
Alternative source directories can be supplied explicitly with `--hs-root`
and `--ls-root`.

## Source-disjoint protocol

With the final thesis configuration:

- 50 standalone HS sources are available;
- 50 standalone LS sources are available;
- five source-disjoint folds are generated;
- each fold uses 40 HS and 40 LS sources for training;
- each fold reserves 10 HS and 10 LS sources for validation;
- no HS or LS source occurs in both training and validation within a fold.

The controlled mixtures use heart-to-lung SNR values of `-6`, `-3`, `0`,
`+3` and `+6` dB.

Each source pair is converted to a 15-second, 4 kHz signal before mixture
construction. The resulting triplets are segmented into 2-second windows with
a 0.5-second hop.

With five SNR conditions, each fold contains:

- 8,000 training base triplets;
- 500 validation base triplets;
- 229,500 total 2-second segments.

## Usage

Generate all five folds:

```bash
python Codes/HLS_CMDS_BUILD/01_build_hlscmds_folds.py --overwrite
```

Generate only one fold:

```bash
python Codes/HLS_CMDS_BUILD/01_build_hlscmds_folds.py --only-fold 1 --overwrite
```

Use explicit standalone-source directories:

```bash
python Codes/HLS_CMDS_BUILD/01_build_hlscmds_folds.py --hs-root /path/to/HS --ls-root /path/to/LS --overwrite
```

The environment variable `ESD_JASSNET_ROOT` can be used to override the
default repository root.

## Outputs

For each generated fold the builder creates:

- selected-source manifests;
- complete additive mixture/reference triplets;
- source-disjoint train/validation assignments;
- segmented `M_*.wav`, `H_*.wav` and `L_*.wav` files;
- `manifest_synth_supervised.csv`;
- additivity diagnostics and build summaries.

Processed datasets are written below:

```text
dataset/processed/hlscmds_full_40x40_10x10_no_unused_fold<N>/
```

The historical filename `source_disjoint_split_smoke.csv` is intentionally
retained for compatibility with the training pipeline. It contains the actual
source-disjoint train/validation split and does not indicate that the published
experiment is a smoke test.

## Reproducibility

The builder checks source disjointness explicitly and aborts if a heart or lung
source appears in both training and validation. It also verifies waveform
additivity at both full-mixture and segmented-sample level.

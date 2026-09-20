# EXP_H benchmark construction

This directory contains the release version of the dataset-building code used to construct the controlled source-domain benchmark reported in the thesis.

The final benchmark is `EXP_H_FULL_BOTH`. Its construction is split into two steps:

1. `01_build_exp_g_40x40.py` builds the `EXP_G_40x40` precursor from the frozen EXP_E source selection;
2. `02_build_exp_h.py` extends that precursor with additional PhysioNet cluster-0 heart-sound sources and ICBHI recordings labelled as containing both crackles and wheezes.

`dataset_builder_utils.py` contains the shared waveform preparation, additive-mixture generation, source-disjointness checks and segmentation routines.

## Signal construction

The controlled mixtures follow the same processing chain used in the thesis:

- mono audio at 4 kHz;
- fixed 15 s source excerpts before mixture construction;
- heart-to-lung SNR conditions of `-6`, `-3`, `0`, `+3` and `+6` dB;
- one shared peak scaling factor applied to the mixture and both references, preserving exact additivity;
- 2 s segments with a 0.5 s hop;
- 27 segments for each 15 s base mixture.

The builders write both a source-disjoint split file and additivity diagnostics.

## Required inputs

The raw PhysioNet 2016 and ICBHI 2017 recordings are not duplicated in this directory. The scripts expect the intermediate source-selection tables produced during the thesis experiments.

For `01_build_exp_g_40x40.py`:

- frozen EXP_E HS and LS manifests;
- PhysioNet abnormal acoustic-cluster table;
- PhysioNet quality table;
- ICBHI quality table.

For `02_build_exp_h.py`:

- the `EXP_G_40x40` selected-source manifests produced by step 1;
- PhysioNet abnormal acoustic-cluster table;
- ICBHI quality table.

By default, the scripts look for the historical experiment directory names under `dataset/` and `outputs/`. Every required input path can also be provided explicitly through the command line.

## Step 1 — EXP_G_40x40

Example:

```bash
python Codes/EXP_H_BUILD/01_build_exp_g_40x40.py \
  --project-root /path/to/project \
  --exp-e-selected-root /path/to/EXP_E_25x30_SELECTED \
  --clustered-csv /path/to/physionet_abnormal_clustered.csv \
  --physionet-quality-csv /path/to/physionet_all_candidates_quality.csv \
  --icbhi-quality-csv /path/to/icbhi_all_candidates_quality.csv \
  --overwrite
```

The resulting source pools contain 40 training HS sources and 40 training LS sources while retaining the frozen EXP_E validation sources.

## Step 2 — final EXP_H

Example:

```bash
python Codes/EXP_H_BUILD/02_build_exp_h.py \
  --project-root /path/to/project \
  --base-selected-root /path/to/EXP_G_40x40_SELECTED \
  --clustered-csv /path/to/physionet_abnormal_clustered.csv \
  --icbhi-quality-csv /path/to/icbhi_all_candidates_quality.csv \
  --overwrite
```

With the final thesis defaults, the EXP_G precursor is extended by:

- 10 cluster-0 HS sources for training and 3 for validation;
- 10 ICBHI `both` LS sources for training and 5 for validation.

No HLS-CMDS target-domain recordings are used in EXP_H.

## Outputs

Each build stage writes:

- selected-source manifests;
- full controlled mixture/reference triplets;
- `metadata.csv`;
- `source_disjoint_split_smoke.csv`;
- segmented `M_*.wav`, `H_*.wav` and `L_*.wav` files;
- `manifest_synth_supervised.csv`;
- summary and audit files.

The final processed directory used by the training pipeline is:

```text
dataset/processed/experiment_H_full_both/
```

## Reproducibility notes

The builders preserve the source selection and waveform-construction logic of the original experiments while removing machine-specific paths and dependencies on historical exploratory script folders.

The environment variable `ESD_JASSNET_ROOT` can be used as the default project root. Command-line paths take precedence when explicitly supplied.

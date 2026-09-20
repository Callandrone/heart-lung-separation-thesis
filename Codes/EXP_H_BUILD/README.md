# EXP\_H benchmark construction

This directory contains the release version of the dataset-building code used to construct the controlled source-domain benchmark reported in the thesis.

The final benchmark is `EXP\_H\_FULL\_BOTH`. Its construction is split into two steps:

1. `01\_build\_exp\_g\_40x40.py` builds the `EXP\_G\_40x40` precursor from the frozen EXP\_E source selection;
2. `02\_build\_exp\_h.py` extends that precursor with additional PhysioNet cluster-0 heart-sound sources and ICBHI recordings labelled as containing both crackles and wheezes.

`dataset\_builder\_utils.py` contains the shared waveform preparation, additive-mixture generation, source-disjointness checks and segmentation routines.

## Signal construction

The controlled mixtures follow the same processing chain used in the thesis:

* mono audio at 4 kHz;
* fixed 15 s source excerpts before mixture construction;
* heart-to-lung SNR conditions of `-6`, `-3`, `0`, `+3` and `+6` dB;
* one shared peak scaling factor applied to the mixture and both references, with post-write additivity verified through residual-SNR diagnostics;
* 2 s segments with a 0.5 s hop;
* 27 segments for each 15 s base mixture.

The builders write both a source-disjoint split file and additivity diagnostics.

## Required inputs

The raw PhysioNet 2016 and ICBHI 2017 recordings are not duplicated in this directory. The scripts expect the intermediate source-selection tables produced during the thesis experiments.

For `01\_build\_exp\_g\_40x40.py`:

* frozen EXP\_E HS and LS manifests;
* PhysioNet abnormal acoustic-cluster table;
* PhysioNet quality table;
* ICBHI quality table.

For `02\_build\_exp\_h.py`:

* the `EXP\_G\_40x40` selected-source manifests produced by step 1;
* PhysioNet abnormal acoustic-cluster table;
* ICBHI quality table.

By default, the scripts look for the historical experiment directory names under `dataset/` and `outputs/`. Every required input path can also be provided explicitly through the command line.

## Step 1 — EXP\_G\_40x40

Example:

```bash
python Codes/EXP\_H\_BUILD/01\_build\_exp\_g\_40x40.py \\
  --project-root /path/to/project \\
  --exp-e-selected-root /path/to/EXP\_E\_25x30\_SELECTED \\
  --clustered-csv /path/to/physionet\_abnormal\_clustered.csv \\
  --physionet-quality-csv /path/to/physionet\_all\_candidates\_quality.csv \\
  --icbhi-quality-csv /path/to/icbhi\_all\_candidates\_quality.csv \\
  --overwrite
```

The resulting source pools contain 40 training HS sources and 40 training LS sources while retaining the frozen EXP\_E validation sources.

## Step 2 — final EXP\_H

Example:

```bash
python Codes/EXP\_H\_BUILD/02\_build\_exp\_h.py \\
  --project-root /path/to/project \\
  --base-selected-root /path/to/EXP\_G\_40x40\_SELECTED \\
  --clustered-csv /path/to/physionet\_abnormal\_clustered.csv \\
  --icbhi-quality-csv /path/to/icbhi\_all\_candidates\_quality.csv \\
  --overwrite
```

With the final thesis defaults, the EXP\_G precursor is extended by:

* 10 cluster-0 HS sources for training and 3 for validation;
* 10 ICBHI `both` LS sources for training and 5 for validation.

No HLS-CMDS target-domain recordings are used in EXP\_H.

## Outputs

Each build stage writes:

* selected-source manifests;
* full controlled mixture/reference triplets;
* `metadata.csv`;
* `source\_disjoint\_split\_smoke.csv`;
* segmented `M\_\*.wav`, `H\_\*.wav` and `L\_\*.wav` files;
* `manifest\_synth\_supervised.csv`;
* summary and audit files.

The final processed directory used by the training pipeline is:

```text
dataset/processed/experiment\_H\_full\_both/
```

## Reproducibility notes

The builders preserve the source selection and waveform-construction logic of the original experiments while removing machine-specific paths and dependencies on historical exploratory script folders.

The environment variable `ESD\_JASSNET\_ROOT` can be used as the default project root. Command-line paths take precedence when explicitly supplied.


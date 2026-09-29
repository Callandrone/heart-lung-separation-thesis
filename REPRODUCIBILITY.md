# Reproducing the workflow

This guide explains how to run the workflow implemented in this repository.

## Environment

Create and activate a virtual environment, then install the project dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optionally save the environment used for a new experiment, then run the existing
test suite:

```powershell
python tools/capture_environment.py --output outputs/environment.json
python -m unittest discover -s tests -v
```

Seeds are configured in the trainers. Record the hardware, GPU driver and
CUDA/PyTorch build alongside each run when GPU reproducibility matters. Retain
the effective configuration, source selections, split CSVs, teacher identity,
checkpoint checksum and evaluation outputs with the experiment.

## Paths and folds

`ESD_JASSNET_ROOT` selects the data/output root while code remains in this
repository. Explicit script arguments and dataset environment overrides take
precedence over defaults. Relative paths are relative to the current working
directory, except HS audio paths in HF_Lung manifests, which can also resolve
under the specified project root.

HLS-CMDS folds use
`dataset/processed/hlscmds_full_40x40_10x10_no_unused_fold<N>`.
`HLSCMDS_TARGET_DIR` overrides the target dataset directory;
`HLSCMDS_DATASET_DIR` is supported as a lower-priority alias.

`HLSCMDS_FOLD` selects the physical fold directory (1–5). Each directory contains
one split with internal `fold_no=1`, so leave `ONLY_FOLD=1` and `N_FOLDS=1`.
`ESD_JASSNET_SPLIT_FOLD` selects that internal label for explicit-checkpoint
evaluation and noise audits; its default is 1.

## Stage 1 — EXP_H supervised training

Build EXP_H using `Codes/EXP_H_BUILD/README.md`, then run the supervised trainer
with replay disabled:

```powershell
$env:ESD_JASSNET_TRAINING_MODE = "stage1"
python Codes/MIXED_TUNING/train_mixed_source_disjoint.py
```

This profile uses supervised-from-scratch training with learning rate `1e-4`,
30 epochs and patience 5. It writes
`outputs/checkpoints/EXP_H_FULL_BOTH/scratch_fold1_best.pt`.
`EXPH_DATA_DIR` and `EXPH_SPLIT_CSV` override its dataset and split inputs.

## Target-only scratch reference

Build the HLS-CMDS folds and select the physical fold:

```powershell
python Codes/HLS_CMDS_BUILD/01_build_hlscmds_folds.py
$env:HLSCMDS_FOLD = "1"
$env:ESD_JASSNET_TRAINING_MODE = "target_scratch"
python Codes/MIXED_TUNING/train_mixed_source_disjoint.py
```

This profile uses learning rate `1e-4`, 30 epochs and patience 5. For fold 1 it
writes `outputs/checkpoints/ESD_JASSNET_SCRATCH_FOLD1/scratch_fold1_best.pt`.

## Stage 2 — supervised target adaptation

For each physical fold, run supervised target adaptation with EXP_H replay:

```powershell
$env:HLSCMDS_FOLD = "1"
$env:ESD_JASSNET_TRAINING_MODE = "stage2"
python Codes/MIXED_TUNING/train_mixed_source_disjoint.py
```

Repeat with `HLSCMDS_FOLD=2` through `5`. Stage 2 writes
`outputs/checkpoints/ESD_JASSNET_MIXED_FOLD<N>/finetune_fold1_best.pt`.
`ESD_JASSNET_STAGE1_CKPT` overrides the Stage-1 initialization, and
`ESD_JASSNET_STAGE2_CKPT` overrides the Stage-2 checkpoint used by downstream
steps.

Both adaptation stages cap EXP_H replay at 30,000 segments. An epoch contains as
many optimization steps as the target loader. Stage 2 uses learning rate `1e-6`,
10 epochs, patience 3 and replay weight 0.05.

## Pseudo-label generation

Build the common mixture-only input once, then generate fold-specific
pseudo-labels with the corresponding Stage-2 checkpoint:

```powershell
python Codes/SSL_MIXED/dataset_generation/01_build_m1_segmented_only.py
$env:HLSCMDS_FOLD = "1"
python Codes/SSL_MIXED/dataset_generation/02_generate_v1_ssl_pseudo_labels.py
```

Repeat generation with `HLSCMDS_FOLD=2` through `5`. Existing builder outputs
require `--overwrite` to regenerate. The generator uses the fold's Stage-2
checkpoint by default. `ESD_JASSNET_STAGE2_CKPT` overrides it, while the
generator's `--teacher-ckpt` or `--teacher-experiment` option takes precedence.

Pseudo-labels are written under
`dataset/processed/v1_real_ssl_pseudo_fold<N>`. Set
`ESD_JASSNET_SSL_PSEUDO_DIR` to a `confident/` directory and
`ESD_JASSNET_SSL_MANIFEST` to its manifest to override these inputs. When only
the directory is overridden, the manifest defaults to
`manifest_pseudo_confident.csv` in its parent directory. Confidence-weighted
training requires complete manifest coverage. The pseudo-label audit accepts an
explicit `--pseudo-root`.

## Stage 3 — semi-supervised refinement

Train and evaluate each physical fold:

```powershell
$env:HLSCMDS_FOLD = "1"
python Codes/SSL_MIXED/train_mixed_ssl_source_disjoint.py
python Codes/SSL_MIXED/evaluate_polarity_control.py
```

Repeat with `HLSCMDS_FOLD=2` through `5`. Stage 3 uses learning rate `1e-6`,
10 epochs, patience 3, replay weight 0.05 and SSL weight 0.10. It allocates 30%
of non-target steps to SSL, producing total SSL probabilities 0.21, 0.15 and
0.09 at target probabilities 0.30, 0.50 and 0.70. Confidence weights range from
0.25 to 1.0.

## External validation

HF_Lung construction requires exactly one explicit `--hs-selected-csv` or
`--hs-quality-csv`. It checks source identifiers and paths against
`--exph-hs-selected-csv` before writing audio and rejects detected training
overlap. Run the waveform-fingerprint audit as well to check renamed copies.

```powershell
python Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_20x20_external_val.py --hs-selected-csv D:\inputs\audited_external_hs.csv --exph-hs-selected-csv D:\inputs\selected_hs_EXP_H_FULL_BOTH.csv
python Codes/BUILD_HFLUNG_RESPIRATORYTR/audit_hflung_physionet_hs_leakage.py --exph-hs-selected-csv D:\inputs\selected_hs_EXP_H_FULL_BOTH.csv
```

The final protocol uses 20 HS and 20 LS sources at {-6, -3, 0, +3, +6} dB; the
builder and audit default to that protocol. For another size, specify `--n-hs`,
`--n-ls`, a descriptive `--out-name`, and the matching audit
`--external-selected-root`. `--allow-exph-train-overlap` enables the explicitly
LS-only external protocol and records the overlap. The builder uses
similarity-ranked HF_Lung LS files.

## Evaluation and bootstrap

The same explicit-evaluation variables work in both evaluator directories:

```powershell
$env:ESD_JASSNET_EVAL_CKPT = "D:\experiments\finetune_fold1_best.pt"
$env:ESD_JASSNET_EVAL_DATA_DIR = "D:\datasets\external_validation"
$env:ESD_JASSNET_EVAL_SPLIT_CSV = "D:\datasets\external_validation\source_disjoint_split_smoke.csv"
$env:ESD_JASSNET_EVAL_RESULTS_DIR = "D:\results\external_validation"
$env:ESD_JASSNET_SPLIT_FOLD = "1"
python Codes/SSL_MIXED/evaluate_polarity_control.py
```

Explicit-checkpoint evaluation selects validation rows from the split CSV.
External HF_Lung builds contain validation-only rows. Clear these overrides
before returning to normal fold evaluation.

For error analysis, use `--fold 1` with each physical-fold directory and its
explicit checkpoint. The export infers `physical_fold` from `_fold<N>` in the
dataset directory; use `--physical-fold` for custom directory names. Pooled
exports must retain this column because sample/base IDs restart in each physical
fold.

Bootstrap defaults to the error-analysis output `final_mixed_ssl`. Compare
explicit runs with:

```powershell
python Codes/ERROR_ANALYSIS/02_bootstrap_ci_error_analysis.py --run SSL_baseline=final_mixed_ssl --run stage2=stage2_analysis
```

Pairing requires unique fold-qualified sample IDs. Do not pool datasets that
reuse identifiers into one run. Resampling is performed at base-mixture level
to account for overlapping windows.

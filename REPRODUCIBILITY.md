# Reproducing the release workflow

The release encodes the published hyperparameters. It does not contain enough
historical artifacts to regenerate every thesis result from a fresh clone.
The commands below describe the maintained release workflow; they do not certify
that a new run reproduces the original checkpoints or reported acceptance counts.

## Inputs that still need to be recovered

| Artifact | Why it is needed |
|---|---|
| Frozen EXP_E HS/LS manifests | Starting selection for EXP_G, then EXP_H |
| PhysioNet quality and abnormal-cluster tables | Reconstruct the screened HS pool |
| ICBHI quality/class-selection table | Reconstruct the screened LS pool |
| Original experiment configurations, seeds, dependency versions and checkpoints | Verify the published numerical results |
| Historical per-fold teacher/student mapping and pseudo-label manifests | Establish whether original folds used separate teachers or shared labels |
| Final HF_Lung selection and invocation | Establish the published subset size and HS overlap policy |
| Full evaluation exports and ablation configurations | Regenerate aggregate tables and paired comparisons |

Put recovered, small source-ID/split manifests in `reproducibility/manifests/`.
That directory is exempt from the general CSV ignore rule. Include provenance,
source checksums and a description of path columns. Do not invent selections from
the reported aggregate results. Historical absolute paths must be relocated or
resolved against the experiment root before reuse.

The upstream EXP_E selection and PhysioNet/ICBHI screening/clustering code is not
included. Also absent are the original physical-mixture coherence/V2 audit,
Montoro M7 NMF baseline, JASSNet comparison implementation, and a dedicated
RespiratoryDatabase@TR mixture-only inference runner. RespiratoryTR preparation
is included. Optional replay morphology augmentation references an unavailable
module and remains disabled in the final profiles.

## Environment

Install `requirements.txt` in a virtual environment. It describes dependencies,
not the original training environment. `requirements-verified.txt` records the
environment used for release regression checks (Windows, Python 3.14); it is not
a historical thesis lockfile or a CUDA installation recipe.

Record the environment for each new experiment:

```powershell
python tools/capture_environment.py --output outputs/environment.json
python -m unittest discover -s tests -v
```

Seeds are configured in the trainers. Bitwise GPU reproducibility is not promised;
record the hardware, GPU driver and CUDA/PyTorch build alongside each run. Retain
the effective configuration, source selections, split CSVs, teacher identity,
checkpoint checksum and evaluation outputs with the experiment.

## Paths and folds

`ESD_JASSNET_ROOT` selects the data/output root. Code stays in this repository.
Explicit script arguments and dataset environment overrides take precedence over
their defaults. Relative paths are relative to the current working directory,
except HS audio paths in HF_Lung manifests, which can also resolve under the
specified project root. Prefer absolute paths for relocated inputs.

New HLS-CMDS folds use `dataset/processed/hlscmds_full_40x40_10x10_no_unused_fold<N>`.
Training and noise-audit configuration discover an existing `torabi_*` directory
only when the canonical directory is absent. `HLSCMDS_TARGET_DIR` overrides both;
the older `HLSCMDS_DATASET_DIR` is supported as a lower-priority alias.

`HLSCMDS_FOLD` selects the **physical fold directory** (1–5). Each such directory
contains one split with **internal `fold_no=1`**. Leave `ONLY_FOLD=1` and
`N_FOLDS=1`. `ESD_JASSNET_SPLIT_FOLD` selects that internal label for explicit
checkpoint evaluation and noise audits; its default is 1.

Historical `TORABI_*` result labels and `source_disjoint_split_smoke.csv` remain
supported. The latter contains the actual split, not a reduced smoke test.

## Stage 1 and target-only scratch training

First build EXP_H using `Codes/EXP_H_BUILD/README.md` and the recovered inputs.
The following profile reuses the supervised trainer with replay disabled:

```powershell
$env:ESD_JASSNET_TRAINING_MODE = "stage1"
python Codes/MIXED_TUNING/train_mixed_source_disjoint.py
```

This selects EXP_H, supervised-from-scratch training, learning rate `1e-4`,
30 epochs and patience 5. It writes
`outputs/checkpoints/EXP_H_FULL_BOTH/scratch_fold1_best.pt`.
`EXPH_DATA_DIR` and `EXPH_SPLIT_CSV` override its inputs.
The old `pretrain` function is a pseudo-mixture training branch and is not this
supervised Stage-1 profile.

For the target-only scratch reference, build HLS-CMDS folds and select:

```powershell
python Codes/HLS_CMDS_BUILD/01_build_hlscmds_folds.py
$env:HLSCMDS_FOLD = "1"
$env:ESD_JASSNET_TRAINING_MODE = "target_scratch"
python Codes/MIXED_TUNING/train_mixed_source_disjoint.py
```

This uses the same 30-epoch/`1e-4`/patience-5 settings and writes
`outputs/checkpoints/ESD_JASSNET_SCRATCH_FOLD1/scratch_fold1_best.pt`.

## Stage 2, pseudo-labels and Stage 3

For each physical fold, use its supervised Stage-2 checkpoint as teacher and
student initialization. New pseudo-label output directories are fold-specific
to prevent accidental overwriting or reuse. This is the release workflow;
the historical teacher mapping must still be recovered.

```powershell
$env:HLSCMDS_FOLD = "1"
$env:ESD_JASSNET_TRAINING_MODE = "stage2"
python Codes/MIXED_TUNING/train_mixed_source_disjoint.py
python Codes/SSL_MIXED/dataset_generation/01_build_m1_segmented_only.py
python Codes/SSL_MIXED/dataset_generation/02_generate_v1_ssl_pseudo_labels.py
python Codes/SSL_MIXED/train_mixed_ssl_source_disjoint.py
python Codes/SSL_MIXED/evaluate_polarity_control.py
```

Build the common M-only input once. Repeat Stage 2, pseudo-label generation,
Stage 3 and evaluation with `HLSCMDS_FOLD=2` through `5`. Existing builder outputs
require an explicit `--overwrite` to regenerate.

Stage 2 writes `ESD_JASSNET_MIXED_FOLD<N>/finetune_fold1_best.pt`; both teacher
generation and Stage 3 use that checkpoint by default. `ESD_JASSNET_STAGE2_CKPT`
overrides it. The generator's explicit `--teacher-ckpt` or
`--teacher-experiment` takes precedence. A shared historical teacher must be
selected explicitly and recorded as such.

Pseudo-labels are written under `dataset/processed/v1_real_ssl_pseudo_fold<N>`.
For existing historical labels, set `ESD_JASSNET_SSL_PSEUDO_DIR` to the
`confident/` directory and `ESD_JASSNET_SSL_MANIFEST` to its manifest. When only
the directory is overridden, the manifest defaults to its parent's
`manifest_pseudo_confident.csv`. Verify the manifest's `teacher_ckpt` against the
intended teacher. Confidence-weighted training requires complete manifest coverage.
The pseudo-audit accepts an explicit `--pseudo-root` for historical locations.

Both stages cap EXP_H replay at 30,000 segments. An epoch contains as many
optimization steps as the target loader. Stage 3 allocates 30% of non-target
steps to SSL, producing total SSL probabilities 0.21, 0.15 and 0.09 at target
probabilities 0.30, 0.50 and 0.70. Confidence weights range from 0.25 to 1.0.
The reported 2,740/2,970 acceptance count belongs to the original teacher; a new
teacher can produce a different count.

## Explicit checkpoint and external evaluation

The same variables work in both evaluator directories:

```powershell
$env:ESD_JASSNET_EVAL_CKPT = "D:\experiments\finetune_fold1_best.pt"
$env:ESD_JASSNET_EVAL_DATA_DIR = "D:\datasets\external_validation"
$env:ESD_JASSNET_EVAL_SPLIT_CSV = "D:\datasets\external_validation\source_disjoint_split_smoke.csv"
$env:ESD_JASSNET_EVAL_RESULTS_DIR = "D:\results\external_validation"
$env:ESD_JASSNET_SPLIT_FOLD = "1"
python Codes/SSL_MIXED/evaluate_polarity_control.py
```

Explicit-checkpoint evaluation selects validation rows from the split CSV.
External HF_Lung builds contain validation-only rows. Clear these evaluation
overrides before returning to normal fold evaluation.

HF_Lung construction requires exactly one explicit `--hs-selected-csv` or
`--hs-quality-csv`. It checks source identifiers and paths against
`--exph-hs-selected-csv` before writing audio and rejects detected training
overlap. Run the separate waveform-fingerprint audit as well to check renamed
copies. Absence of detected recording overlap does not establish patient-level
independence. Reuse of EXP_H validation hearts must also be reported.

```powershell
python Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_25x25_external_val.py --hs-selected-csv D:\inputs\audited_external_hs.csv --exph-hs-selected-csv D:\inputs\selected_hs_EXP_H_FULL_BOTH.csv
python Codes/BUILD_HFLUNG_RESPIRATORYTR/audit_hflung_physionet_hs_leakage.py --exph-hs-selected-csv D:\inputs\selected_hs_EXP_H_FULL_BOTH.csv
```

The default external size is 25×25, and the audit defaults to that output.
For another size, specify `--n-hs`, `--n-ls`, a descriptive `--out-name`, and the
matching audit `--external-selected-root`. `--allow-exph-train-overlap` permits
an explicitly LS-only external protocol and records the overlap; it must not be
described as an unseen two-source benchmark. The builder uses similarity-ranked
HF_Lung LS files. The quality-balanced HF_Lung selection utility is a separate
coverage-analysis workflow, not its direct input.

## Error analysis and bootstrap

Use `--fold 1` with each physical fold directory and its explicit checkpoint.
The error-analysis export infers `physical_fold` from `_fold<N>` in the dataset
directory; use `--physical-fold` for custom directory names. Pooled exports must
retain this column because sample/base IDs restart in each physical fold.

Bootstrap defaults to the error analysis output `final_mixed_ssl`. Compare
explicit runs using:

```powershell
python Codes/ERROR_ANALYSIS/02_bootstrap_ci_error_analysis.py --run SSL_baseline=final_mixed_ssl --run stage2=stage2_analysis
```

Pairing requires unique fold-qualified sample IDs. Do not pool different datasets
that reuse identifiers into one run. Resampling remains at base-mixture level;
it accounts for overlapping windows, not every dependency induced by source reuse.

## Delivered examples and historical naming

`Deliverables/EXP_H` contains Stage-1 checkpoint examples. `Deliverables/TORABI`
contains final SSL examples for physical folds 1–5. Their JSON files retain the
original server paths and experiment names as provenance. The historical
`SSL - FOLD<N>_EXPERIMENT26GIUGNO_NUMBERONE` names describe original runs; the new
`ESD_JASSNET_SSL_FOLD<N>` directories describe release runs and are not asserted
to contain identical checkpoints. No bundled audio, metrics or reported result
table was changed by the release cleanup.

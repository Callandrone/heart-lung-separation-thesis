# Error analysis

This directory contains the post-hoc analyses used to inspect the final separator.

- `01_error_analysis_final_mixed_ssl.py`: computes per-sample separation metrics, failure diagnostics, rankings and optional qualitative artifacts for the final Mixed + SSL model.
- `02_bootstrap_ci_error_analysis.py`: computes clustered bootstrap confidence intervals using `base_id` as the resampling unit and supports paired comparisons against the SSL baseline.
- `03_white_noise_robustness_audit.py`: performs a zero-shot additive white-noise robustness audit without retraining the separator.

The analyses preserve the source-disjoint evaluation protocol and avoid treating overlapping windows derived from the same base recording as statistically independent.

For the white-noise audit, set `ESD_JASSNET_SSL_CHECKPOINT` to the checkpoint to evaluate. Dataset and output locations can optionally be overridden with `HLSCMDS_TARGET_DIR`, `HLSCMDS_SPLIT_CSV`, `HLSCMDS_FOLD` and `ERROR_ANALYSIS_OUT_DIR`.

`HLSCMDS_FOLD` selects the physical dataset directory. The noise audit's internal
split is selected separately by `ESD_JASSNET_SPLIT_FOLD` (default 1).
`HLSCMDS_DATASET_DIR` remains a lower-priority legacy alias for the dataset path.

The main error analysis accepts `--physical-fold` for custom dataset directory
names and otherwise infers it from `_fold<N>`. Preserve this column when pooling
folds: base/sample IDs restart in each physical directory.

Bootstrap defaults to `final_mixed_ssl`, matching the main analysis output.
Use repeated `--run LABEL=DIRECTORY` and `--baseline LABEL` arguments for comparisons.
Directories are relative to `--error-root`; historical ablations are not assumed.
See [reproducibility instructions](../../REPRODUCIBILITY.md) for an example.

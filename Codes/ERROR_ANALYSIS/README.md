# Error analysis

This directory contains the post-hoc analyses used to inspect the final separator.

- `01_error_analysis_final_mixed_ssl.py`: computes per-sample separation metrics, failure diagnostics, rankings and optional qualitative artifacts for the final Mixed + SSL model.
- `02_bootstrap_ci_error_analysis.py`: computes clustered bootstrap confidence intervals using `base_id` as the resampling unit and supports paired comparisons against the SSL baseline.
- `03_white_noise_robustness_audit.py`: performs a zero-shot additive white-noise robustness audit without retraining the separator.

The analyses preserve the source-disjoint evaluation protocol and avoid treating overlapping windows derived from the same base recording as statistically independent.

For the white-noise audit, set `ESD_JASSNET_SSL_CHECKPOINT` to the checkpoint to evaluate. Dataset and output locations can optionally be overridden with `HLSCMDS_DATASET_DIR`, `HLSCMDS_SPLIT_CSV`, `HLSCMDS_FOLD` and `ERROR_ANALYSIS_OUT_DIR`.

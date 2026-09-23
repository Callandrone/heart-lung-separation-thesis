# External-dataset preparation and audits

This directory contains the utilities used for the HF_Lung and
RespiratoryDatabase@TR external analyses reported in the thesis.

## Scripts

- `hflung_quality_score_and_selection_v2.py` scores HF_Lung recordings and
  creates the balanced lung-sound subset used by the acoustic coverage audit.
- `audit_hflung_physionet_hs_leakage.py` checks whether PhysioNet heart sources
  used in an external HF_Lung validation set overlap with EXP_H training sources.
- `build_hflung_selected_25x25_external_val.py` builds the controlled HF_Lung
  external validation mixtures used for waveform-level evaluation.
- `inspect_respiratoryTR_audio.py` inventories the RespiratoryDatabase@TR audio
  files and their recording properties.
- `inspect_respiratoryTR_labels.py` exports and inspects the label sheets
  distributed with RespiratoryDatabase@TR.
- `build_respiratoryTR_real_mixture_dataset.py` converts the real respiratory
  recordings into mixture-only 2-second segments for zero-shot consistency
  analysis. Clean heart/lung references are not available for this dataset.

## Paths

Defaults are resolved relative to the repository root. All relevant input and
output locations can also be supplied explicitly through command-line options.

For example:

```bash
python Codes/BUILD_HFLUNG_RESPIRATORYTR/hflung_quality_score_and_selection_v2.py --help
python Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_25x25_external_val.py --help
python Codes/BUILD_HFLUNG_RESPIRATORYTR/build_respiratoryTR_real_mixture_dataset.py --help
```

The historical filename `source_disjoint_split_smoke.csv` is intentionally
preserved where required for compatibility with the experimental pipeline; it
does not indicate that the published analysis is a smoke-test result.

## External validation protocol

The builder requires exactly one explicit `--hs-selected-csv` or `--hs-quality-csv`.
It checks source identifiers and paths against the EXP_H HS manifest supplied
through `--exph-hs-selected-csv` (default: the final EXP_H selected manifest).
Detected training overlap stops the build before output audio is written.
`--allow-exph-train-overlap` is reserved for an explicitly LS-only external
protocol and is recorded in the output audit. Run the standalone waveform audit
to check renamed copies as well; an identifier check does not prove independence.

Builder and leakage-audit defaults both refer to 25?25. The builder uses
similarity-ranked lung recordings from the domain-gap workflow; the balanced
quality-selection utility is a separate coverage-analysis workflow.
The exact historical selection remains unavailable.

Generated summaries list the supported evaluation environment variables.
See [the full external-validation instructions](../../REPRODUCIBILITY.md).

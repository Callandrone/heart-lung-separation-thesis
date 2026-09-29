# Domain-gap analysis

This directory contains the acoustic domain-gap analyses used in the thesis.

The scripts are ordered according to their role in the analysis:

- `00_domain_gap_full_audit_with_snr.py`: source/target acoustic-feature audit, PCA, standardised mean differences and local-SNR analysis.
- `01_domain_gap_feature_extraction_class_conditioned.py`: class-conditioned lung-sound feature analysis.
- `02_domain_gap_train_vs_all_stability_check.py`: stability check between training-only and training-plus-validation analyses.
- `03_domain_gap_classifier_diagnostic.py`: source-disjoint domain-classification diagnostic.
- `04_hflung_global_vs_exph_baseline.py`: comparison between EXP_H/ICBHI and external real-patient lung recordings.
- `05_hflung_similarity_ranking_vs_exph.py`: HF_Lung similarity ranking relative to the selected EXP_H/ICBHI reference.
- `06_torabi_vs_hflung_subdomains.py`: comparison between the HLS-CMDS target domain and HF_Lung acoustic subdomains.
- `07_hflung_selected_label_audit.py`: label audit of the selected HF_Lung subsets.
- `08_statistical_domain_gap_icbhi_torabi.py`: source-level statistical domain-gap analysis.
- `09_domain_gap_distance_vs_performance.py`: relationship between acoustic distance and separator performance.

Scripts `00`-`03` provide the primary source-target diagnostics. Scripts `04`-`07` extend the analysis to external real-patient lung recordings. Scripts `08`-`09` provide the source-level statistical analyses used for interpretation.

Some historical output identifiers retain the `TORABI_*` naming used during the experiments. These identifiers are preserved for compatibility with the original result files; in the thesis they refer to the HLS-CMDS target domain.

Input and output locations are supplied through command-line arguments or repository-relative defaults. The scripts do not modify the source datasets.

# ESD-JASSNet — Semi-Supervised Refinement

This directory contains the implementation of the final semi-supervised refinement stage of ESD-JASSNet.

The Stage-3 model is initialised from the supervised mixed-domain checkpoint produced by the previous adaptation stage and is refined using three sources of training data:

- controlled HLS-CMDS target-domain mixtures;
- low-weight EXP_H source-domain replay;
- confidence-filtered pseudo-labels generated from unlabelled HLS-CMDS V1 physical mixtures.

The final configuration corresponds to the ESD-JASSNet results reported in the thesis.

## Main files

- `encoder.py` — multi-scale convolutional waveform encoder;
- `separator.py` — JASSNet-inspired local/global separation network and source-mask estimation;
- `decoder.py` — shared waveform decoder;
- `model_config.py` — architecture, training and path configuration;
- `train_mixed_ssl_source_disjoint.py` — final Stage-3 training procedure;
- `evaluate_polarity_control.py` — source-level evaluation and mixture-informed inference calibration;
- `normalization.py` — global layer normalisation for intermediate feature tensors;
- `snr_filter.py` — dataset filtering utilities.

## Pseudo-label generation

The `dataset_generation/` directory contains the preprocessing chain used for the unlabelled HLS-CMDS V1 physical mixtures:

1. `01_build_m1_segmented_only.py` — prepares the segmented mixture-only dataset;
2. `02_generate_v1_ssl_pseudo_labels.py` — generates teacher pseudo-labels, applies confidence filtering and exports accepted/rejected manifests;
3. `03_audit_v1_ssl_pseudo_labels.py` — audits the accepted pseudo-labels through acoustic features, reference comparisons and qualitative examples.

In the final experiment, 2,740 of 2,970 candidate segments passed the confidence filter.

## Final training configuration

The final Stage-3 configuration uses:

- five source-disjoint HLS-CMDS folds;
- 2 s mono segments at 4 kHz;
- batch size 64;
- learning rate `1e-6`;
- maximum 10 epochs;
- early-stopping patience 3;
- EXP_H replay weight `0.05`;
- SSL pseudo-label weight `0.10`;
- SSL mixture-consistency weight `0.05`;
- progressive HLS-CMDS sampling probability `0.30 -> 0.50 -> 0.70`.

The model configuration contains 300,608 trainable parameters.

## Path configuration

Machine-specific paths are not hard-coded in the release configuration.

The following environment variables can be used to point the scripts to the required datasets and checkpoints:

- `ESD_JASSNET_ROOT` — experiment root;
- `HLSCMDS_FOLD` — target-domain fold (`1` to `5`);
- `HLSCMDS_TARGET_DIR` — controlled HLS-CMDS dataset;
- `HLSCMDS_SPLIT_CSV` — corresponding source-disjoint split;
- `EXPH_DATA_DIR` — EXP_H replay dataset;
- `EXPH_SPLIT_CSV` — EXP_H source-disjoint split;
- `ESD_JASSNET_STAGE2_CKPT` — Stage-2 mixed-domain checkpoint;
- `ESD_JASSNET_SSL_PSEUDO_DIR` — accepted pseudo-label directory;
- `ESD_JASSNET_SSL_MANIFEST` — confidence manifest;
- `ESD_JASSNET_SELECTED_CASES_CSV` — optional CSV selecting qualitative evaluation cases.

The original experiments were executed on a research server. The environment-variable interface allows the same scripts to be configured for a different filesystem without modifying the source code.

## Training

Before running Stage 3, provide the Stage-2 checkpoint and the required dataset locations.

Example on Windows PowerShell:

```powershell
$env:HLSCMDS_FOLD = "1"
$env:ESD_JASSNET_STAGE2_CKPT = "D:\path\to\finetune_fold1_best.pt"

python .\Codes\SSL_MIXED\train_mixed_ssl_source_disjoint.py
```

Repeat the experiment with `HLSCMDS_FOLD` set from `1` to `5` for the complete target-domain evaluation protocol.

## Evaluation

The final checkpoint can be evaluated with:

```powershell
python .\Codes\SSL_MIXED\evaluate_polarity_control.py
```

Inference-time evaluation can apply deterministic mixture-informed polarity and gain calibration, as described in the thesis.

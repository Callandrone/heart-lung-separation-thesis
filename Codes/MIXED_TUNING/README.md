# ESD-JASSNet — Mixed-Domain Adaptation

This directory contains the supervised target-domain adaptation stage of ESD-JASSNet.

The Stage-2 model is initialised from the source-domain EXP_H checkpoint and adapted to controlled HLS-CMDS mixtures while retaining low-weight EXP_H replay to reduce catastrophic forgetting.

The resulting checkpoints are used as teachers and initialisation points for the subsequent semi-supervised refinement stage in `../SSL_MIXED/`.

## Main files

- `encoder.py` — multi-scale convolutional waveform encoder;
- `separator.py` — JASSNet-inspired local/global separation network and source-mask estimation;
- `decoder.py` — shared waveform decoder;
- `model_config.py` — architecture, optimisation and path configuration;
- `train_mixed_source_disjoint.py` — Stage-2 mixed-domain adaptation procedure;
- `train_disjoint.py` — historical supervised trainer, also used for dataset helpers by the evaluator;
- `evaluate_polarity_control.py` — source-level evaluation and mixture-informed inference calibration;
- `normalization.py` — global layer normalisation for intermediate feature tensors;
- `snr_filter.py` — dataset filtering utilities.

## Experimental protocol

Adaptation is performed independently over five source-disjoint HLS-CMDS folds.

For each fold:

- 40 standalone heart-sound recordings and 40 standalone lung-sound recordings are used for target-domain training;
- 10 heart-sound recordings and 10 lung-sound recordings are reserved for validation;
- no standalone source occurs in both training and validation;
- EXP_H training data are replayed with a low weight to preserve source-domain separation performance.

The Stage-1 EXP_H checkpoint is used to initialise the model before target-domain adaptation.

## Final training configuration

The final Stage-2 configuration uses:

- 2 s mono segments at 4 kHz;
- batch size 64;
- Adam optimisation;
- learning rate `1e-6`;
- maximum 10 epochs;
- early-stopping patience 3;
- target-domain weight `1.0`;
- EXP_H replay weight `0.05`;
- progressive target sampling probability `0.30 -> 0.50 -> 0.70`;
- no training augmentation.

The ESD-JASSNet architecture contains 300,608 trainable parameters.

## Path configuration

Machine-specific paths are not hard-coded in the release configuration.

The main environment variables are:

- `ESD_JASSNET_ROOT` — experiment root;
- `HLSCMDS_FOLD` — target-domain fold (`1` to `5`);
- `HLSCMDS_TARGET_DIR` — controlled HLS-CMDS dataset for the selected fold;
- `HLSCMDS_SPLIT_CSV` — corresponding source-disjoint split;
- `EXPH_DATA_DIR` — EXP_H replay dataset;
- `EXPH_SPLIT_CSV` — EXP_H source-disjoint split;
- `ESD_JASSNET_STAGE1_CKPT` — Stage-1 EXP_H checkpoint;
- `ESD_JASSNET_SELECTED_CASES_CSV` — optional CSV selecting qualitative evaluation cases.

The original experiments were executed on a research server. Environment-variable configuration allows the scripts to be used on another filesystem without modifying the source code.

## Training

Example on Windows PowerShell:

```powershell
$env:HLSCMDS_FOLD = "1"
$env:ESD_JASSNET_TRAINING_MODE = "stage2"
$env:ESD_JASSNET_STAGE1_CKPT = "D:\path\to\scratch_fold1_best.pt"

python .\Codes\MIXED_TUNING\train_mixed_source_disjoint.py
```

Repeat with `HLSCMDS_FOLD` set from `1` to `5` for the complete target-domain protocol.

## Evaluation

A trained checkpoint can be evaluated with:

```powershell
python .\Codes\MIXED_TUNING\evaluate_polarity_control.py
```

Evaluation can apply the deterministic mixture-informed polarity and gain calibration described in the thesis.

## Source and scratch profiles

Use `ESD_JASSNET_TRAINING_MODE=stage1` for supervised EXP_H training or
`target_scratch` for the controlled HLS-CMDS scratch reference. Both use 30 epochs,
learning rate `1e-4`, patience 5 and no replay. Restore `stage2` before adaptation.
The Stage-1 profile produces the default checkpoint consumed by Stage 2.

Canonical dataset directories begin with `hlscmds_`; existing `torabi_` directories
are discovered as a fallback. `HLSCMDS_FOLD` selects a physical dataset directory;
its split CSV still uses internal `fold_no=1`.

For explicit checkpoint evaluation, set `ESD_JASSNET_EVAL_CKPT`. External data,
split and results locations use `ESD_JASSNET_EVAL_DATA_DIR`,
`ESD_JASSNET_EVAL_SPLIT_CSV` and `ESD_JASSNET_EVAL_RESULTS_DIR`.
Single-checkpoint evaluation retains only validation samples.
See [the complete workflow](../../REPRODUCIBILITY.md) for prerequisites and limitations.

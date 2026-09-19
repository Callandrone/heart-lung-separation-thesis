\# ESD-JASSNet — Semi-Supervised Refinement



This directory contains the implementation of the final semi-supervised refinement stage of ESD-JASSNet.



The Stage-3 model is initialised from the supervised mixed-domain checkpoint produced by the previous adaptation stage and is refined using three sources of training data:



\- controlled HLS-CMDS target-domain mixtures;

\- low-weight EXP\_H source-domain replay;

\- confidence-filtered pseudo-labels generated from unlabelled HLS-CMDS V1 physical mixtures.



The final configuration corresponds to the ESD-JASSNet results reported in the thesis.



\## Main files



\- `encoder.py` — multi-scale convolutional waveform encoder;

\- `separator.py` — JASSNet-inspired local/global separation network and source-mask estimation;

\- `decoder.py` — shared waveform decoder;

\- `model\_config.py` — architecture, training and path configuration;

\- `train\_mixed\_ssl\_source\_disjoint.py` — final Stage-3 training procedure;

\- `evaluate\_polarity\_control.py` — source-level evaluation and mixture-informed inference calibration;

\- `normalization.py` — waveform normalisation utilities;

\- `snr\_filter.py` — dataset filtering utilities.



\## Pseudo-label generation



The `dataset\_generation/` directory contains the preprocessing chain used for the unlabelled HLS-CMDS V1 physical mixtures:



1\. `01\_build\_m1\_segmented\_only.py` — prepares the segmented mixture-only dataset;

2\. `02\_generate\_v1\_ssl\_pseudo\_labels.py` — generates teacher pseudo-labels;

3\. `03\_audit\_v1\_ssl\_pseudo\_labels.py` — applies consistency and non-degeneracy checks and identifies the confident pseudo-label subset.



In the final experiment, 2,740 of 2,970 candidate segments passed the confidence filter.



\## Final training configuration



The final Stage-3 configuration uses:



\- five source-disjoint HLS-CMDS folds;

\- 2 s mono segments at 4 kHz;

\- batch size 64;

\- learning rate `1e-6`;

\- maximum 10 epochs;

\- early-stopping patience 3;

\- EXP\_H replay weight `0.05`;

\- SSL pseudo-label weight `0.10`;

\- SSL mixture-consistency weight `0.05`;

\- progressive HLS-CMDS sampling probability `0.30 -> 0.50 -> 0.70`.



The model configuration contains 300,608 trainable parameters.



\## Path configuration



Machine-specific paths are not hard-coded in the release configuration.



The following environment variables can be used to point the scripts to the required datasets and checkpoints:



\- `ESD\_JASSNET\_ROOT` — experiment root;

\- `HLSCMDS\_FOLD` — target-domain fold (`1` to `5`);

\- `HLSCMDS\_TARGET\_DIR` — controlled HLS-CMDS dataset;

\- `HLSCMDS\_SPLIT\_CSV` — corresponding source-disjoint split;

\- `EXPH\_DATA\_DIR` — EXP\_H replay dataset;

\- `EXPH\_SPLIT\_CSV` — EXP\_H source-disjoint split;

\- `ESD\_JASSNET\_STAGE2\_CKPT` — Stage-2 mixed-domain checkpoint;

\- `ESD\_JASSNET\_SSL\_PSEUDO\_DIR` — accepted pseudo-label directory;

\- `ESD\_JASSNET\_SSL\_MANIFEST` — confidence manifest.



The original experiments were executed on a research server. The environment-variable interface allows the same scripts to be configured for a different filesystem without modifying the source code.



\## Training



Before running Stage 3, provide the Stage-2 checkpoint and the required dataset locations.



Example on Windows PowerShell:



```powershell

$env:HLSCMDS\_FOLD = "1"

$env:ESD\_JASSNET\_STAGE2\_CKPT = "D:\\path\\to\\finetune\_fold1\_best.pt"



python .\\Codes\\SSL\_MIXED\\train\_mixed\_ssl\_source\_disjoint.py

```



Repeat the experiment with `HLSCMDS\_FOLD` set from `1` to `5` for the complete target-domain evaluation protocol.



\## Evaluation



The final checkpoint can be evaluated with:



```powershell

python .\\Codes\\SSL\_MIXED\\evaluate\_polarity\_control.py

```



Inference-time evaluation can apply deterministic mixture-informed polarity and gain calibration, as described in the thesis.


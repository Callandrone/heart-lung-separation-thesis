\# ESD-JASSNet — Mixed-Domain Adaptation



This directory contains the supervised target-domain adaptation stage of ESD-JASSNet.



The Stage-2 model is initialised from the source-domain EXP\_H checkpoint and adapted to controlled HLS-CMDS mixtures while retaining low-weight EXP\_H replay to reduce catastrophic forgetting.



The resulting checkpoints are used as teachers and initialisation points for the subsequent semi-supervised refinement stage in `../SSL\_MIXED/`.



\## Main files



\- `encoder.py` — multi-scale convolutional waveform encoder;

\- `separator.py` — JASSNet-inspired local/global separation network and source-mask estimation;

\- `decoder.py` — shared waveform decoder;

\- `model\_config.py` — architecture, optimisation and path configuration;

\- `train\_mixed\_source\_disjoint.py` — Stage-2 mixed-domain adaptation procedure;

\- `train\_disjoint.py` — shared supervised training utilities used by the mixed-domain pipeline;

\- `evaluate\_polarity\_control.py` — source-level evaluation and mixture-informed inference calibration;

\- `normalization.py` — waveform normalisation utilities;

\- `snr\_filter.py` — dataset filtering utilities.



\## Experimental protocol



Adaptation is performed independently over five source-disjoint HLS-CMDS folds.



For each fold:



\- 40 standalone heart-sound recordings and 40 standalone lung-sound recordings are used for target-domain training;

\- 10 heart-sound recordings and 10 lung-sound recordings are reserved for validation;

\- no standalone source occurs in both training and validation;

\- EXP\_H training data are replayed with a low weight to preserve source-domain separation performance.



The Stage-1 EXP\_H checkpoint is used to initialise the model before target-domain adaptation.



\## Final training configuration



The final Stage-2 configuration uses:



\- 2 s mono segments at 4 kHz;

\- batch size 64;

\- Adam optimisation;

\- learning rate `1e-6`;

\- maximum 10 epochs;

\- early-stopping patience 3;

\- target-domain weight `1.0`;

\- EXP\_H replay weight `0.05`;

\- progressive target sampling probability `0.30 -> 0.50 -> 0.70`;

\- no training augmentation.



The ESD-JASSNet architecture contains 300,608 trainable parameters.



\## Path configuration



Machine-specific paths are not hard-coded in the release configuration.



The main environment variables are:



\- `ESD\_JASSNET\_ROOT` — experiment root;

\- `HLSCMDS\_FOLD` — target-domain fold (`1` to `5`);

\- `HLSCMDS\_TARGET\_DIR` — controlled HLS-CMDS dataset for the selected fold;

\- `HLSCMDS\_SPLIT\_CSV` — corresponding source-disjoint split;

\- `EXPH\_DATA\_DIR` — EXP\_H replay dataset;

\- `EXPH\_SPLIT\_CSV` — EXP\_H source-disjoint split;

\- `ESD\_JASSNET\_STAGE1\_CKPT` — Stage-1 EXP\_H checkpoint;

\- `ESD\_JASSNET\_SELECTED\_CASES\_CSV` — optional CSV selecting qualitative evaluation cases.



The original experiments were executed on a research server. Environment-variable configuration allows the scripts to be used on another filesystem without modifying the source code.



\## Training



Example on Windows PowerShell:



```powershell

$env:HLSCMDS\_FOLD = "1"

$env:ESD\_JASSNET\_STAGE1\_CKPT = "D:\\path\\to\\scratch\_fold1\_best.pt"



python .\\Codes\\MIXED\_TUNING\\train\_mixed\_source\_disjoint.py

```



Repeat with `HLSCMDS\_FOLD` set from `1` to `5` for the complete target-domain protocol.



\## Evaluation



A trained checkpoint can be evaluated with:



```powershell

python .\\Codes\\MIXED\_TUNING\\evaluate\_polarity\_control.py

```



Evaluation can apply the deterministic mixture-informed polarity and gain calibration described in the thesis.


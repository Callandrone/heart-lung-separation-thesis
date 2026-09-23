"""Stage-2 mixed-domain adaptation configuration for ESD-JASSNet.

The architecture and optimisation settings match the final mixed fine-tuning
configuration reported in the thesis. Machine-specific paths can be overridden
through environment variables without editing this file.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_config import physical_fold, project_root, target_dir, stage2_checkpoint, pseudo_root

# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------
SR = 4000
SEG_SECONDS = 2.0
SEG_SAMPLES = int(SR * SEG_SECONDS)

KERNEL_SIZE = 16
STRIDE = 8
N_FILTERS = 128
N_LATENT = 64
DILATIONS = (1, 4, 8)
DROPOUT_P = 0.1
ENCODER_R = 3
DECODER_R = 2
DECODER_USE_INPUT_GLN = True
N_SOURCES = 2

# JASSNet-inspired separator
JASSNET_DIM = N_LATENT
JASSNET_NUM_MODULES = 4
JASSNET_EXPANSION = 2
JASSNET_ATTN_DIM = 64
JASSNET_LOCAL_CHUNK = 50
JASSNET_CONV_KERNEL = 3
JASSNET_RPE_KERNEL = 3
JASSNET_POSITIONAL_ENCODING = True

SEP_N_STACKS = 1
SEP_BLOCKS_PER_STACK = JASSNET_NUM_MODULES
SEP_TCN_BOTTLENECK = 64
ATTN_HEADS = 1
ATTN_WINDOW = JASSNET_LOCAL_CHUNK
N_GLOBAL_TOKENS = 0
MASK_SCALE = "ReLU_unbounded"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = project_root()
HLSCMDS_FOLD = physical_fold()

EXPERIMENT_NAME = f"ESD_JASSNET_MIXED_FOLD{HLSCMDS_FOLD}"
CKPT_DIR = str(PROJECT_ROOT / "outputs" / "checkpoints" / EXPERIMENT_NAME)
RESULTS_DIR = str(PROJECT_ROOT / "outputs" / "results" / EXPERIMENT_NAME)
EVAL_CKPT = os.environ.get("ESD_JASSNET_EVAL_CKPT")
EVAL_SPLIT_FOLD = int(os.environ.get("ESD_JASSNET_SPLIT_FOLD", "1"))

# Canonical release path, with discovery of existing historical torabi_* data.
SUPERVISED_DIR = str(target_dir(PROJECT_ROOT, HLSCMDS_FOLD))
SOURCE_DISJOINT_SPLIT_CSV = os.environ.get(
    "HLSCMDS_SPLIT_CSV",
    str(Path(SUPERVISED_DIR) / "source_disjoint_split_smoke.csv"),
)
USE_SOURCE_DISJOINT_SPLIT = True

# EXP_H source-domain replay dataset. Only the training split is used.
SYNTH_SUPERVISED_DIR = os.environ.get(
    "EXPH_DATA_DIR",
    str(PROJECT_ROOT / "dataset" / "processed" / "experiment_H_full_both"),
)
SYNTH_SOURCE_DISJOINT_SPLIT_CSV = os.environ.get(
    "EXPH_SPLIT_CSV",
    str(Path(SYNTH_SUPERVISED_DIR) / "source_disjoint_split_smoke.csv"),
)
SYNTH_REPLAY_FOLD_NO = 1
SYNTH_REPLAY_SPLIT = "train"
REQUIRE_SYNTH_TRAIN_SPLIT = True

# Stage-1 EXP_H checkpoint used to initialise mixed-domain adaptation.
PRETRAIN_CKPT = os.environ.get(
    "ESD_JASSNET_STAGE1_CKPT",
    str(
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "EXP_H_FULL_BOTH"
        / "scratch_fold1_best.pt"
    ),
)

# Compatibility placeholder used by the training code.
PSEUDO_DIR = ""

# -----------------------------------------------------------------------------
# Run mode and reproducibility
# -----------------------------------------------------------------------------
STAGE = "finetune"
N_FOLDS = 1
ONLY_FOLD = 1
SEED = 42
MODEL_SEED = 42
SMOKE_MAX_BASE_TRIPLETS = None

# -----------------------------------------------------------------------------
# Optimisation
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

# Stage 1 is performed separately; these fields are retained for compatibility.
PRETRAIN_EPOCHS = 0
PRETRAIN_LR = 1e-4
PRETRAIN_PATIENCE = 5

# Final Stage-2 mixed fine-tuning schedule reported in the thesis.
FINETUNE_MODE = "full"
FINETUNE_EPOCHS = 10
FINETUNE_PATIENCE = 3
FINETUNE_LR = 1e-6
FULL_FINETUNE_LR = 1e-6
SEPARATOR_FINETUNE_LR = 1e-6
WARMUP_EPOCHS = 0

# -----------------------------------------------------------------------------
# Mixed-domain adaptation
# -----------------------------------------------------------------------------
MIXED_TRAINING = True
MIXED_USE_CURRICULUM = True
MIXED_CURRICULUM = ((0, 0.30), (3, 0.50), (6, 0.70))
# Historical names: MIXED_V2_PROB and MIXED_WEIGHT_V2 refer to controlled
# additive target-domain mixtures from standalone HLS-CMDS HS/LS recordings,
# not to the physical HLS-CMDS V2 subset. Retained for training-code compatibility.
MIXED_V2_PROB = 0.50  # compatibility fallback when curriculum is disabled

MIXED_WEIGHT_V2 = 1.0
MIXED_WEIGHT_SYNTH = 0.05
MIXED_STEPS_PER_EPOCH = None
MIXED_SYNTH_MAX_SEGMENTS = 30000

# -----------------------------------------------------------------------------
# Loss configuration
# -----------------------------------------------------------------------------
LOSS_WEIGHT_H = 1.0
LOSS_WEIGHT_L = 1.0
LAMBDA_L1 = 0.0
LAMBDA_RMS = 1.0
LAMBDA_POLARITY = 0.1
LAMBDA_MIX = 0.0
LAMBDA_MIX_POLARITY = 0.0

# Compatibility options used by the training code.
TARGET_GAIN_MODE = "none"
TARGET_GAIN_H = 1.0
TARGET_GAIN_L = 1.0
TARGET_GAIN_REMOVE_DC = True
USE_BASE_TRIPLET_ALLOWLIST = False
BASE_TRIPLET_ALLOWLIST = ()
BASE_TRIPLET_DENYLIST = ()

# -----------------------------------------------------------------------------
# Filtering and augmentation
# -----------------------------------------------------------------------------
FILTER_LOCAL_SNR = False
LOCAL_SNR_THRESHOLD_DB = 15.0
LOCAL_SNR_ANALYSIS_CSV = ""

# The final mixed fine-tuning experiment uses no augmentation.
AUGMENT_TRAINING = False
USE_V1_RESYNTH_TRAIN_INJECTION = False
AUGMENT_REPLAY_TORABI_MORPH = False
AUGMENT_TARGET_RANDOMIZATION = False

# -----------------------------------------------------------------------------
# Evaluation / inference-time calibration
# -----------------------------------------------------------------------------
APPLY_MIXTURE_POLARITY_CALIBRATION = True
APPLY_MIXTURE_GAIN_CALIBRATION = True
EVALUATE_POST_TRAINING = False

# Explicit supervised profiles reuse the same trainer and architecture.
# These encode the published settings, not a recovered historical run manifest.
TRAINING_MODE = os.environ.get("ESD_JASSNET_TRAINING_MODE", "stage2")
if TRAINING_MODE not in {"stage1", "stage2", "target_scratch"}:
    raise ValueError("ESD_JASSNET_TRAINING_MODE must be stage1, stage2, or target_scratch")
if TRAINING_MODE in {"stage1", "target_scratch"}:
    STAGE = "supervised_from_scratch"
    PRETRAIN_CKPT = None
    MIXED_TRAINING = False
    FINETUNE_EPOCHS = 30
    FINETUNE_PATIENCE = 5
    FINETUNE_LR = FULL_FINETUNE_LR = SEPARATOR_FINETUNE_LR = 1e-4
    if TRAINING_MODE == "stage1":
        SUPERVISED_DIR = SYNTH_SUPERVISED_DIR
        SOURCE_DISJOINT_SPLIT_CSV = SYNTH_SOURCE_DISJOINT_SPLIT_CSV
        EXPERIMENT_NAME = "EXP_H_FULL_BOTH"
    else:
        EXPERIMENT_NAME = f"ESD_JASSNET_SCRATCH_FOLD{HLSCMDS_FOLD}"
    CKPT_DIR = str(PROJECT_ROOT / "outputs" / "checkpoints" / EXPERIMENT_NAME)
    RESULTS_DIR = str(PROJECT_ROOT / "outputs" / "results" / EXPERIMENT_NAME)

# Evaluation paths are configured separately from training inputs.
EVAL_DATA_DIR = os.environ.get("ESD_JASSNET_EVAL_DATA_DIR", SUPERVISED_DIR)
EVAL_SPLIT_CSV = os.environ.get(
    "ESD_JASSNET_EVAL_SPLIT_CSV",
    str(Path(EVAL_DATA_DIR) / "source_disjoint_split_smoke.csv")
    if "ESD_JASSNET_EVAL_DATA_DIR" in os.environ else SOURCE_DISJOINT_SPLIT_CSV,
)
EVAL_RESULTS_DIR = os.environ.get("ESD_JASSNET_EVAL_RESULTS_DIR", RESULTS_DIR)

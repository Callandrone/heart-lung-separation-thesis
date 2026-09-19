"""Final Stage-3 SSL refinement configuration for ESD-JASSNet.

The model and training hyperparameters match the configuration reported in the
thesis. Machine-specific paths can be overridden through environment variables
without editing this file.
"""

import os
from pathlib import Path

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
# By default, use the repository root. Set ESD_JASSNET_ROOT to point to the
# original experiment root (or another compatible data/output layout).
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(REPO_ROOT))).resolve()

HLSCMDS_FOLD = int(os.environ.get("HLSCMDS_FOLD", "1"))
if HLSCMDS_FOLD not in {1, 2, 3, 4, 5}:
    raise ValueError(f"HLSCMDS_FOLD must be one of 1, 2, 3, 4, 5; got {HLSCMDS_FOLD}.")

EXPERIMENT_NAME = f"ESD_JASSNET_SSL_FOLD{HLSCMDS_FOLD}"
CKPT_DIR = str(PROJECT_ROOT / "outputs" / "checkpoints" / EXPERIMENT_NAME)
RESULTS_DIR = str(PROJECT_ROOT / "outputs" / "results" / EXPERIMENT_NAME)
EVAL_CKPT = None

# Controlled HLS-CMDS target-domain dataset for the selected fold.
# The fallback directory name matches the original experiment layout.
SUPERVISED_DIR = os.environ.get(
    "HLSCMDS_TARGET_DIR",
    str(
        PROJECT_ROOT
        / "dataset"
        / "processed"
        / f"torabi_full_40x40_10x10_no_unused_fold{HLSCMDS_FOLD}"
    ),
)
SOURCE_DISJOINT_SPLIT_CSV = os.environ.get(
    "HLSCMDS_SPLIT_CSV",
    str(Path(SUPERVISED_DIR) / "source_disjoint_split_smoke.csv"),
)
USE_SOURCE_DISJOINT_SPLIT = True

# EXP_H replay dataset. Only the training split is used during adaptation.
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

# Stage-2 mixed-fine-tuning checkpoint used to initialise Stage 3.
# Set this explicitly before training, for example:
#   ESD_JASSNET_STAGE2_CKPT=/path/to/finetune_fold1_best.pt
PRETRAIN_CKPT = os.environ.get("ESD_JASSNET_STAGE2_CKPT")

# Confidence-filtered pseudo-label dataset generated from physical V1 mixtures.
SSL_PSEUDO_DIR = os.environ.get(
    "ESD_JASSNET_SSL_PSEUDO_DIR",
    str(
        PROJECT_ROOT
        / "dataset"
        / "processed"
        / "v1_real_ssl_pseudo_from_mixed_noaug"
        / "confident"
    ),
)
SSL_CONFIDENCE_MANIFEST = os.environ.get(
    "ESD_JASSNET_SSL_MANIFEST",
    str(
        PROJECT_ROOT
        / "dataset"
        / "processed"
        / "v1_real_ssl_pseudo_from_mixed_noaug"
        / "manifest_pseudo_confident.csv"
    ),
)

# Compatibility placeholder used by the training script.
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

# Stage 1 is performed separately. These values are retained for compatibility.
PRETRAIN_EPOCHS = 0
PRETRAIN_LR = 1e-4
PRETRAIN_PATIENCE = 5

# Final Stage-3 schedule reported in the thesis.
FINETUNE_MODE = "full"
FINETUNE_EPOCHS = 10
FINETUNE_PATIENCE = 3
FINETUNE_LR = 1e-6
FULL_FINETUNE_LR = 1e-6
SEPARATOR_FINETUNE_LR = 1e-6
WARMUP_EPOCHS = 0

# -----------------------------------------------------------------------------
# Mixed-domain SSL schedule
# -----------------------------------------------------------------------------
MIXED_TRAINING = True
MIXED_USE_CURRICULUM = True
MIXED_CURRICULUM = ((0, 0.30), (3, 0.50), (6, 0.70))
MIXED_V2_PROB = 0.50  # compatibility fallback when curriculum is disabled

MIXED_WEIGHT_V2 = 1.0
MIXED_WEIGHT_SYNTH = 0.05
MIXED_STEPS_PER_EPOCH = None
MIXED_SYNTH_MAX_SEGMENTS = 30000

USE_SSL_PSEUDO = True
SSL_PSEUDO_WEIGHT = 0.10
SSL_PSEUDO_PROB_IN_REMAINDER = 0.30
SSL_PSEUDO_MAX_SEGMENTS = None
SSL_LAMBDA_MIX = 0.05

# Confidence weighting for accepted pseudo-labels.
USE_SSL_CONFIDENCE_WEIGHTS = True
SSL_CONF_MIN_WEIGHT = 0.25
SSL_CONF_MAX_WEIGHT = 1.00
SSL_CONF_CORR_GOOD = 0.985
SSL_CONF_CORR_BAD = 0.950
SSL_CONF_NMSE_GOOD_DB = -18.0
SSL_CONF_NMSE_BAD_DB = -10.0
SSL_CONF_HL_CORR_BAD = 0.85
SSL_CONF_ABS_SNR_BAD_DB = 12.0

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

# Ablation-only weighting mechanisms are disabled in the final model.
USE_DYNAMIC_SOURCE_WEIGHTS = False
USE_SNR_SAMPLE_WEIGHTS = False

# -----------------------------------------------------------------------------
# Target-gain compatibility options
# -----------------------------------------------------------------------------
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

AUGMENT_TRAINING = False
USE_V1_RESYNTH_TRAIN_INJECTION = False
AUGMENT_REPLAY_TORABI_MORPH = False
AUGMENT_TARGET_RANDOMIZATION = False

# Compatibility values used only if target randomisation is enabled.
TARGET_RANDOM_PROB = 0.30
TARGET_RANDOM_PRESERVE_SOURCE_RMS = True
TARGET_RANDOM_PRESERVE_PEAK = True
TARGET_RANDOM_PEAK_VALUE = 0.95
TARGET_RANDOM_APPLY_LS_PROB = 0.90
TARGET_RANDOM_LS_EQ_PROB = 0.30
TARGET_RANDOM_LS_EQ_DB = (-2.0, 2.0)
TARGET_RANDOM_LS_TEXTURE_PROB = 0.25
TARGET_RANDOM_LS_TEXTURE_SNR_DB = (35.0, 45.0)
TARGET_RANDOM_LS_TEXTURE_LOW_HZ = 100.0
TARGET_RANDOM_LS_TEXTURE_HIGH_HZ = 1200.0
TARGET_RANDOM_SHARED_GAIN_PROB = 0.30
TARGET_RANDOM_SHARED_GAIN_DB = (-1.5, 1.5)

# -----------------------------------------------------------------------------
# Evaluation / inference-time calibration
# -----------------------------------------------------------------------------
APPLY_MIXTURE_POLARITY_CALIBRATION = True
APPLY_MIXTURE_GAIN_CALIBRATION = True
EVALUATE_POST_TRAINING = False

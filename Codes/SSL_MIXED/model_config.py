# ============================================================
# SSL PSEUDO-LABEL PILOT — MIXED NOAUG + EXP_H REPLAY + V1 SSL
# ============================================================

import os

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

# JASSNet-like separator compatibility fields
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

# ============================================================
# TORABI RE-EVALUATION — ORIGINAL HYBRID SSL
# ============================================================
PROJECT_ROOT = "/nas/home/pcallandrone/DeepLearning"

TORABI_DATA_FOLD = int(
    os.environ.get("TORABI_DATA_FOLD", "1")
)

if TORABI_DATA_FOLD not in {1, 2, 3, 4, 5}:
    raise ValueError(
        "TORABI_DATA_FOLD must be one of 1,2,3,4,5; "
        f"got {TORABI_DATA_FOLD}."
    )

# Exact checkpoint experiment directory for each physical fold.
EXPERIMENT_NAME = (
    f"SSL - FOLD{TORABI_DATA_FOLD}_"
    f"EXPERIMENT26GIUGNO_NUMBERONE"
)

# Evaluate Fold K on the corresponding physical Torabi Fold K.
SUPERVISED_DIR = (
    PROJECT_ROOT
    + "/dataset/processed/"
    + f"torabi_full_40x40_10x10_no_unused_fold"
    + str(TORABI_DATA_FOLD)
)

SOURCE_DISJOINT_SPLIT_CSV = (
    SUPERVISED_DIR
    + "/source_disjoint_split_smoke.csv"
)

USE_SOURCE_DISJOINT_SPLIT = True

# Folder containing finetune_fold1_best.pt.
CKPT_DIR = (
    PROJECT_ROOT
    + "/outputs/checkpoints/"
    + EXPERIMENT_NAME
)

# New output directory: does not overwrite the old retention results.
RESULTS_DIR = (
    PROJECT_ROOT
    + "/outputs/results/"
    + f"SSL_26GIUGNO_HYBRID_TORABI_REEVAL_FOLD"
    + str(TORABI_DATA_FOLD)
)

# Let evaluate_polarity_control resolve finetune_fold1_best.pt
# automatically from CKPT_DIR.
EVAL_CKPT = None

# Not used by evaluation. None also prevents accidental retraining.
PRETRAIN_CKPT = None
STAGE = "finetune"

N_FOLDS = 1
ONLY_FOLD = 1
SEED = 42
MODEL_SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

PRETRAIN_EPOCHS = 0
PRETRAIN_LR = 1e-4
PRETRAIN_PATIENCE = 5

FINETUNE_MODE = "full"
FINETUNE_EPOCHS = 10
FINETUNE_PATIENCE = 3
FINETUNE_LR = 1e-6
FULL_FINETUNE_LR = 1e-6
SEPARATOR_FINETUNE_LR = 1e-6
WARMUP_EPOCHS = 0

# Mixed-domain schedule: target remains dominant; replay/SSL act as weak regularizers.
MIXED_TRAINING = True
MIXED_USE_CURRICULUM = True
MIXED_CURRICULUM = ((0, 0.30), (3, 0.50), (6, 0.70))
MIXED_V2_PROB = 0.50
MIXED_WEIGHT_V2 = 1.0
MIXED_WEIGHT_SYNTH = 0.05
MIXED_STEPS_PER_EPOCH = None
MIXED_SYNTH_MAX_SEGMENTS = 30000


USE_SSL_PSEUDO = True
SSL_PSEUDO_DIR = PROJECT_ROOT + "/dataset/processed/v1_real_ssl_pseudo_from_mixed_noaug/confident"
SSL_PSEUDO_WEIGHT = 0.10
SSL_PSEUDO_PROB_IN_REMAINDER = 0.30
SSL_PSEUDO_MAX_SEGMENTS = None

# Nuovi flag SSLv2
USE_SSL_CONFIDENCE_WEIGHTS = True
SSL_CONFIDENCE_MANIFEST = PROJECT_ROOT + "/dataset/processed/v1_real_ssl_pseudo_from_mixed_noaug/manifest_pseudo_confident.csv"

SSL_CONF_MIN_WEIGHT = 0.25
SSL_CONF_MAX_WEIGHT = 1.00

SSL_CONF_CORR_GOOD = 0.985
SSL_CONF_CORR_BAD = 0.950

SSL_CONF_NMSE_GOOD_DB = -18.0
SSL_CONF_NMSE_BAD_DB = -10.0

SSL_CONF_HL_CORR_BAD = 0.85
SSL_CONF_ABS_SNR_BAD_DB = 12.0

# Mixture consistency solo per SSL batch
SSL_LAMBDA_MIX = 0.05

# Loss: keep same mixed baseline objective.
LOSS_WEIGHT_H = 1.0
LOSS_WEIGHT_L = 1.0
LAMBDA_L1 = 0.0
LAMBDA_RMS = 1.0
LAMBDA_POLARITY = 0.1
LAMBDA_MIX = 0.0
LAMBDA_MIX_POLARITY = 0.0

FILTER_LOCAL_SNR = False
LOCAL_SNR_THRESHOLD_DB = 15.0
LOCAL_SNR_ANALYSIS_CSV = ""

# No generic augmentation: this experiment only tests mild target randomization.
AUGMENT_TRAINING = False
USE_V1_RESYNTH_TRAIN_INJECTION = False
AUGMENT_REPLAY_TORABI_MORPH = False

# Final optional experiment: stochastic, training-only domain randomization on
# Torabi target train batches. Validation/test stay clean Torabi.
AUGMENT_TARGET_RANDOMIZATION = False
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

SMOKE_MAX_BASE_TRIPLETS = None

APPLY_MIXTURE_POLARITY_CALIBRATION = True
APPLY_MIXTURE_GAIN_CALIBRATION = True
EVALUATE_POST_TRAINING = False

# Compatibility placeholders used by training/evaluation code.
PSEUDO_DIR = ""

#ricalcolare i risultati perchè li ho persi come un picio
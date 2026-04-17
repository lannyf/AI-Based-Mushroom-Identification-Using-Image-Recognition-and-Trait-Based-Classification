# Segmentation runtime configuration

USE_MASK_FOR_TRAITS = True  # Enabled for test run

# Selection thresholds
MIN_MASK_CONFIDENCE = 0.50
MIN_FOREGROUND_AREA_RATIO = 0.02
MAX_NEAR_FULL_FRAME_RATIO = 0.85
MAX_FRAGMENTATION = 3  # allowed connected components after cleanup

# Quality gating thresholds
MAX_HOLE_RATIO = 0.10
MAX_BOUNDARY_IRREGULARITY = 0.35

# Cleanup parameters
CLEANUP_MIN_COMPONENT_AREA = 64  # pixels
CLEANUP_MORPH_ITER = 1

# Classifier-aware re-ranking
ENABLE_CLASSIFIER_RERANK = True
RERANK_TOP_N = 3

# Control whether segmentation runs at all (Stage A: keep False to avoid changing behaviour)
RUN_SEGMENTATION_METADATA = True

# Other
MASK_DTYPE = "uint8"  # expected mask dtype
MASK_ON_VALUE = 255
MASK_OFF_VALUE = 0

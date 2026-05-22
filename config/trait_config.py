# Trait extractor feature flags and thresholds.
#
# These flags control the staged rollout of the part-aware trait extractor.
# They live in a dedicated module because they are not segmentation concerns.

# Phase 1: Enable part-aware trait computation and output.
# When False, extract() runs the legacy single-mask / whole-image path.
ENABLE_PART_AWARE_TRAITS = False

# Phase 4: Allow high-confidence part-aware traits to drive key.xml auto-answers.
ENABLE_PART_AWARE_KEY_AUTOANSWERS = False

# Phase 5: Allow part-aware traits to affect database comparison.
ENABLE_PART_AWARE_DB_COMPARATOR = False

# Minimum confidence for a trait to be considered reliable enough for general use.
PART_AWARE_MIN_TRAIT_CONFIDENCE = 0.65

# Minimum confidence for a trait to be allowed to auto-answer a key.xml question.
PART_AWARE_MIN_AUTOANSWER_CONFIDENCE = 0.80

# Mask quality gate thresholds (used by YoloPartMasks).
MIN_AREA_RATIO = 0.01
MAX_FRAGMENTATION = 5
MAX_HOLE_RATIO = 0.20
MIN_CONFIDENCE = 0.30

# Part-specific quality gate overrides.
# Caps often have holes where the stem passes through → allow higher hole ratio.
CAP_MAX_HOLE_RATIO = 0.70
# Stems are thin and can fall below the global area ratio.
STEM_MIN_AREA_RATIO = 0.002
# Undersides (gills/pores) can also be relatively small in the frame.
UNDERSIDE_MIN_AREA_RATIO = 0.003

# Coral-specific quality and geometric filters (used by YoloPartMasks).
# Coral mushrooms are naturally fragmented and full of holes (gaps between branches),
# so they need relaxed quality gates but strict geometric validation.
CORAL_MAX_FRAGMENTATION = 20   # Allow many contours (branches).
CORAL_MAX_HOLE_RATIO = 0.85    # Coral is naturally full of holes.
CORAL_MAX_SOLIDITY = 0.85      # Reject if mask is too compact / blob-like.
CORAL_MIN_COMPLEXITY = 1.0     # Reject if outline is too smooth / circular.

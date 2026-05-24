# Trait extractor thresholds.
#
# These thresholds govern the part-aware trait extractor.
# They live in a dedicated module because they are not segmentation concerns.

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


# Trait Extractor Feedback for Kimi - 18 May 2026

## Context

This feedback is based on reviewing:

- `data/Trait_extractor/trait_extractor_issues_may18.md`
- `models/unified_pipeline.py`
- `models/mushroom_segmenter.py`
- `models/yolo_part_masks.py`
- `models/visual_trait_extractor.py`
- `models/key_tree_traversal.py`
- `config/trait_config.py`

The issue document is directionally strong. The main diagnosis is valid: case detection and trait extraction are currently using different evidence layers, which can produce inconsistent results such as `case=coral` while `detected_parts=[]`.

## Main Assessment

The highest-priority issue is the mismatch between raw YOLO instances and filtered part masks.

Current flow in `UnifiedPipeline.run()`:

1. Segment above and below images.
2. Call `detect_case()` on raw YOLO instances.
3. Extract traits through `_extract_traits_masked()`, which internally calls `build_part_masks()`.
4. `build_part_masks()` may reject masks that already influenced `detect_case()`.

This means the pipeline can route on raw detections that the trait extractor later considers invalid. That is a real architectural bug.

Recommended direction:

1. Build part masks immediately after segmentation.
2. Use those same filtered masks for both case detection and trait extraction.
3. Return case reasoning that reflects accepted evidence only.

This should be fixed before deeper work on color and shape accuracy, because otherwise benchmark results will remain noisy.

## Important Caveats

### 1. Do not just pass current `build_part_masks()` output into `detect_case()`

`build_part_masks()` currently keeps one merged cluster per part. When multiple same-class masks do not merge, the extra clusters are effectively ignored.

That is risky for coral detection because real coral masks may appear as multiple separated clusters. If case detection uses only the current filtered output, recall for real corals could get worse.

Suggested prerequisite:

- Preserve multiple accepted clusters per part, or
- Validate per instance/cluster first, then union accepted clusters into the final part mask.

Only after that should case detection depend fully on filtered mask evidence.

### 2. Feature flag mismatch

The issue document says results were generated with:

```python
ENABLE_PART_AWARE_TRAITS = True
```

But the checked-in config currently has:

```python
ENABLE_PART_AWARE_TRAITS = False
```

This is not necessarily wrong, because the reproduction script mutates the flag at runtime. But the doc should make clear that the benchmark depends on runtime flag override, not default app behavior.

### 3. "Empty traits" is partly a field-mapping problem

When no part masks survive, `_part_aware_extract()` falls back to whole-image analysis. It returns fields like:

- `dominant_color`
- `secondary_color`
- `cap_shape`
- `surface_texture`

But it does not populate compatibility aliases like:

- `cap_color`
- `whole_color`
- `stem_color`
- `underside_color`

So benchmark code that reads `cap_color` can report missing values even when fallback color data exists under `dominant_color`.

Suggested fix:

When no valid part masks survive, populate fallback aliases:

```python
cap_color = dominant_color
whole_color = dominant_color
hymenophore_type = "unknown"
stem_color = "unknown"
underside_color = "unknown"
```

This will make downstream comparison more consistent without pretending part-specific evidence exists.

### 4. Color analysis description needs correction

The issue document says the current color analysis uses K-means on RGB pixels. The code actually samples BGR pixels, converts to HSV, then runs K-means on HSV values.

The problem is still real, but the cause is probably not "RGB clustering" alone. More likely causes:

- poor masks or background leakage;
- largest-cluster-wins behavior;
- lighting and exposure differences;
- coarse extractor vocabulary versus finer database vocabulary;
- no semantic normalization between extracted names and database names.

Suggested direction:

- Add canonical color families for DB comparison.
- Keep raw color ratios as supporting evidence.
- Add ratio-aware overrides for obvious cases, for example high red ratio should prevent red Amanita from becoming white.
- Consider reporting both `dominant_color` and `color_family`.

### 5. Cap shape should be treated as low-confidence

Current cap shape comes from 2D contour metrics. That is too weak for species-level routing and database comparison. The benchmark result of 1/10 correct supports this.

Suggested direction:

- Keep cap shape as descriptive metadata.
- Do not let it drive key traversal or database conflict scoring unless confidence is high.
- Prefer CNN/LLM or user clarification for shape-dependent decisions.

### 6. Key routing impact needs clarification

The issue document says wrong `coarse_case=coral` routes the user to the coral key tree.

That may be true through the LLM tree navigation prompt, because `case_info` is passed into the prompt. But programmatic key auto-answering appears gated by `ENABLE_PART_AWARE_KEY_AUTOANSWERS` and by `trait_confidence["morphology_case"]`.

I did not see `trait_confidence["morphology_case"]` being set in `visual_trait_extractor.py`, so programmatic routing may not actually be auto-answering the root case from morphology.

Suggested doc clarification:

- Separate LLM prompt influence from deterministic key traversal behavior.
- Confirm whether the user-facing wrong path comes from LLM navigation, visible traits, pre-answers, or deterministic auto-answering.

## Recommended Implementation Order

1. Fix `build_part_masks()` so it does not discard non-merging same-class clusters.
2. In `UnifiedPipeline.run()`, build above/below part masks once immediately after segmentation.
3. Replace or extend `detect_case()` so it can use filtered part masks.
4. Pass the same part masks into `extract(image_bytes, part_masks=...)`.
5. Add fallback alias fields when no valid part masks survive.
6. Re-run the coral benchmark and record both raw YOLO detections and accepted mask evidence.
7. Only then tune color and shape logic.

## Suggested Case Detection Contract

Case detection should report both accepted evidence and rejected raw evidence.

Example output:

```python
{
    "case": "classical",
    "confidence": 0.75,
    "detected_parts": ["Cap", "Stem"],
    "accepted_parts": {
        "above": ["cap"],
        "below": ["cap", "stem"]
    },
    "rejected_raw_parts": {
        "above": ["Coral"],
        "below": []
    },
    "reasoning": "Filtered masks show cap/stem structure; raw coral detection was rejected by geometric filter."
}
```

This keeps debugging possible. If YOLO keeps producing coral false positives, the system can expose that as rejected evidence rather than silently losing it.

## Questions for Kimi

1. Should case detection be derived entirely from filtered part masks, or should high-confidence raw YOLO detections still contribute as weak evidence?
2. What is the best way to preserve multiple disjoint coral clusters without creating false positives from scattered noise?
3. Should `build_part_masks()` return multiple masks per part, or keep the current single-mask API and add internal union logic?
4. Should fallback whole-image traits populate `cap_color` and `whole_color`, or should they stay separate to avoid misleading downstream consumers?
5. For color matching, would canonical color families be enough, or should the extractor compute nearest DB color labels directly?
6. Should cap shape be removed from automatic DB/key scoring until a better shape model exists?

## Bottom Line

The issue document correctly identifies the biggest architectural problem: `detect_case()` and trait extraction are not using the same filtered evidence.

The next fix should focus on making segmentation evidence consistent across the pipeline. Fixing color and shape first would likely hide the deeper problem and make benchmark results harder to interpret.

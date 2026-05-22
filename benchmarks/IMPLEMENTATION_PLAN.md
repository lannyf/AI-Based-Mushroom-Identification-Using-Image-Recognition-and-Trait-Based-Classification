# Benchmark Refactor Implementation Plan (Revised)

## Overview

Refactor the comparative benchmark to evaluate exactly 4 system variants (A1, A2, B1, B2) plus a CNN reference baseline on 30 held-out specimens. Legacy standalone benchmark methods (tree, trait-DB) are removed from the orchestrator—they remain as internal pipeline components only.

## Goals

1. **Reduce benchmark to 30 paired specimens** (from 57)
2. **Guarantee no YOLO training leakage**
3. **Redesign LLM prompt** — subsystems are *tools* the LLM uses; only CNN is an independent AI peer
4. **Expand System A and B into oracle/non-oracle variants**:
   - **A1**: Standalone vision-LLM, no trait pre-knowledge
   - **A2**: Standalone vision-LLM + **flat dict of vision-only oracle traits** appended to user message
   - **B1**: Full unified pipeline with trait extractor
   - **B2**: Full unified pipeline with a **perfect trait-extractor oracle**: oracle traits replace the extractor output used by tree traversal, database comparison, and LLM synthesis
5. **Remove legacy standalone benchmarks** — no more standalone tree or trait-DB columns

---

## Step 1 — Reduce Benchmark to 30 Specimens

### Current State
- 57 specimens across 22 species in `benchmarks/evaluation_manifest.csv`
- All paired above/below photos

### Proposed 30-Specimen Subset

| Category | Specimens | Count |
|---|---|---|
| Confusing (BO.BA↔BO.ED) | BO.BA_010, BO.BA_011, BO.ED_013, BO.ED_014 | 4 |
| Confusing (CA.CI↔HY.PS) | CA.CI_016, CA.CI_017, HY.PS_019, HY.PS_020 | 4 |
| Confusing (other) | AM.MU_004, AM.MU_005, AM.VI_007, AM.VI_008, CR.CO_033, SP.CR_039 | 6 |
| Edge case | CO.CO_031 | 1 |
| OOD classical | AG.AU_001, GY.ES_041, GA.MA_043, PL.OS_045 | 4 |
| OOD diverse | CA.TU_047, HY.RE_049, RU.IN_054, RU.BA_056 | 4 |
| Puffball | LY.PE_037, CAL.GI_051 | 2 |
| Coral | RA.BO_025, RA.PA_028 | 2 |
| Easy | LA.HE_022, FO.BE_035 | 2 |
| Filler (confusing balance) | BO.ED_015 | 1 |
| **TOTAL** | | **30** |

**Note on CAL.GI_051**: This puffball intentionally reuses the same image for above and below (puffballs have no underside view). This is preserved from the original manifest.

### Implementation
- Create `benchmarks/evaluation_manifest_v2.csv` with the 30-row subset
- Update `ManifestDataset` to accept any manifest path (already does via CLI arg)
- Update Makefile / CLI default to use v2

---

## Step 2 — Anti-Leakage Verification

### Problem
The YOLO model (`best(1).pt`) was trained on a Roboflow-exported ZIP (~364 images). We must confirm no benchmark images from `data/raw/Benchmark/` were in that ZIP.

### Approach
1. **Source provenance** (primary): Benchmark = Svampeatlas/GBIF (Denmark); Training = iNaturalist, Danish Fungi Atlas, manual collection. Entirely different sources = no leakage.
2. **Hash-based check** (secondary): Compute SHA-256 of all 60 benchmark photos and check against training-set hashes if available.

### Implementation
- Create `benchmarks/validate_no_leakage.py` script
- Computes SHA-256 of all benchmark images
- Looks for extracted Roboflow training images first; if absent, uses `data/Yolov8/training_source_hashes.json`
- **Default**: `--allow-inconclusive` — prints warning, documents source-provenance rationale, exits 0
- **Optional**: `--strict-leakage-check` — exits non-zero if training data is unavailable (for CI/production)
- Fails loudly on any exact hash collision
- Integrate into `make benchmark-validate`

---

## Step 3 — Vision-Only Oracle Trait Loader

### Trait Filter

From `species_traits.xml`, **only traits observable in photos** are included:

| Include | Exclude |
|---|---|
| cap_shape, cap_colour, cap_surface, cap_size_cm, cap_margin | smell |
| gill_attachment, gill_density, gill_colour, gill_edge | habitat |
| stem_shape, stem_colour, stem_surface, stem_ring, stem_size_cm | season |
| underside_type (pores/gills/ridges/teeth) | substrate |
| flesh_colour | growth_pattern |

**Excluded entirely**: smell, habitat, season, substrate, growth_pattern, tree_association.

Surface texture is visible in photos — **include** cap_surface and stem_surface.
Flesh texture is not directly visible — **exclude**.

### OOD Coverage Verified

All 23 species in the 30-specimen subset exist in `species_traits.xml`:
AG.AU, AM.MU, AM.VI, BO.BA, BO.ED, CA.CI, HY.PS, CR.CO, SP.CR, CO.CO, GY.ES, GA.MA, PL.OS, CA.TU, HY.RE, RU.IN, RU.BA, CAL.GI, LY.PE, RA.BO, RA.PA, LA.HE, FO.BE.

### Critical: XML → Extractor Key Schema Mapping

`species_traits.xml` uses hierarchical categories and trait names. The trait extractor (`visual_trait_extractor.py`) outputs flat keys. The oracle loader **must** map between these schemas. This table is the authoritative mapping:

| XML Category | XML Trait Name | Extractor Output Key | Notes |
|---|---|---|---|
| CAP | shape | `cap_shape` | convex, flat, funnel, etc. |
| CAP | color | `cap_colour` | British spelling matches extractor |
| CAP | surface_texture | `cap_surface` | smooth, scaly, cracked, etc. |
| CAP | size_cm | `cap_size_cm` | range string, e.g. "4-12" |
| CAP | margin | `cap_margin` | smooth, wavy, striate, etc. |
| GILLS | attachment | `hymenophore_type` | **Unified key**: free→gills, pores→pores, etc. |
| GILLS | color | `underside_colour` | or `gill_colour` depending on case |
| GILLS | density | `gill_density` | crowded, distant, etc. |
| GILLS | edge | `gill_edge` | smooth, serrated, etc. |
| STEM | shape | `stem_shape` | cylindrical, club-shaped, etc. |
| STEM | color | `stem_colour` | British spelling |
| STEM | surface | `stem_surface` | smooth, reticulate, scaly, etc. |
| STEM | size_cm | `stem_size_cm` | range string |
| FLESH | color | `flesh_colour` | white, yellow, etc. |
| FLESH | texture | *(excluded)* | Not directly visible in photos |

**Case-Specific Keys** (the extractor uses different keys for non-classical morphologies):

| Morphology Case | Extractor Key | XML Source | Mapping Rule |
|---|---|---|---|
| puffball | `puffball_surface` | CAP/surface_texture | When case=puffball |
| puffball | `puffball_roundness` | CAP/shape | round→high, oval→medium |
| coral | `coral_branching` | *(no direct XML equivalent)* | Use CAP/shape as proxy: branched→dichotomous |
| coral | `coral_density` | *(no direct XML equivalent)* | Omit; not in XML |

**OR-Value Handling**: XML values like `convex|flat` or `white|cream` represent variability. For the extractor-shaped output (`get_extractor_output`), pick the **first value** (e.g., `convex` from `convex|flat`). For `get_species_trait_dict`, preserve the full OR-value string.

### Manual `stem_ring` Mapping

`species_traits.xml` does not have a dedicated `STEM/ring` trait. Ring presence is described in `STEM/surface` text (e.g. "smooth with thin ring"). To provide reliable oracle data, maintain a hardcoded mapping:

```python
_SPECIES_STEM_RING = {
    "AG.AU": "absent", "AM.MU": "present", "AM.VI": "present",
    "BO.BA": "absent", "BO.ED": "absent", "CA.CI": "absent",
    "HY.PS": "absent", "CR.CO": "absent", "SP.CR": "absent",
    "CO.CO": "absent", "GY.ES": "absent", "GA.MA": "present",
    "PL.OS": "absent", "CA.TU": "absent", "HY.RE": "absent",
    "RU.IN": "absent", "RU.BA": "absent", "CAL.GI": "absent",
    "LY.PE": "absent", "RA.BO": "absent", "RA.PA": "absent",
    "LA.HE": "absent", "FO.BE": "absent",
}
```

This mapping is used by both `get_flat_dict` (A2) and `get_extractor_output` (B2).

### New Module: `benchmarks/species_trait_oracle.py`

```python
class SpeciesTraitOracle:
    """Loads ground-truth trait dicts from species_traits.xml, filtered to vision-only."""

    # Authoritative mapping: XML (category, name) → extractor flat key
    XML_TO_EXTRACTOR_KEY = {
        ("CAP", "shape"): "cap_shape",
        ("CAP", "color"): "cap_colour",
        ("CAP", "surface_texture"): "cap_surface",
        ("CAP", "size_cm"): "cap_size_cm",
        ("CAP", "margin"): "cap_margin",
        ("GILLS", "attachment"): "hymenophore_type",
        ("GILLS", "color"): "underside_colour",
        ("GILLS", "density"): "gill_density",
        ("GILLS", "edge"): "gill_edge",
        ("STEM", "shape"): "stem_shape",
        ("STEM", "color"): "stem_colour",
        ("STEM", "surface"): "stem_surface",
        ("STEM", "size_cm"): "stem_size_cm",
        ("FLESH", "color"): "flesh_colour",
    }

    _SPECIES_STEM_RING = { ... }  # see mapping above

    # Vision-only traits for A2 flat dict (human-readable American spelling)
    VISION_TRAITS = {
        "cap": ["shape", "color", "surface_texture", "size_cm", "margin"],
        "gills": ["attachment", "density", "color", "edge"],
        "stem": ["shape", "color", "surface", "ring", "size_cm"],
        "flesh": ["color"],
        "underside": ["type"],  # inferred from gills/attachment
    }

    def __init__(self, traits_xml_path: str):
        self.traits_df = load_species_traits_xml(Path(traits_xml_path))

    def get_flat_dict(self, species_id: str) -> Dict[str, str]:
        """Return flat vision-only trait dict for A2.
        
        Uses human-readable keys like cap_shape, cap_color (American spelling
        for readability), gill_attachment, etc.
        
        Example for AM.MU:
        {
            "cap_shape": "convex",
            "cap_color": "red with white spots",
            "cap_surface": "smooth",
            "gill_attachment": "free",
            "gill_color": "white",
            "stem_shape": "cylindrical",
            "stem_color": "white",
            "stem_ring": "present",
            "flesh_color": "white",
        }
        """
        ...

    def get_extractor_output(
        self, species_id: str, case: str = "classical"
    ) -> Dict[str, Any]:
        """Return a dict shaped like the trait extractor's merged visible_traits output.

        B2 uses this as a perfect trait-extractor output. Keys match the extractor
        schema exactly (British spelling, flat structure). OR-values are resolved
        by taking the first alternative.

        Args:
            species_id: Target species ID.
            case: Pipeline morphology case from detect_case() — one of
                classical, puffball, coral, uncertain.
        """
        ...

    def get_species_trait_dict(self, species_id: str) -> Dict[str, str]:
        """Return the structured species-traits dict used when tree traversal
        needs a missing trait answer.

        This is the same as get_flat_dict but with OR-values preserved
        (e.g., "convex|flat") so the tree or LLM can see variability.
        """
        ...
```

---

## Step 4 — LLM Prompt Redesign (Tool-Based Architecture)

### New Design

```
LLM = central mycologist reasoner
Tools available:
  - vision_analysis: LLM's own visual examination of images
  - cnn_classifier: INDEPENDENT AI system. Simply produces an answer.
  - trait_extractor: Extracts morphological traits from segmented images
  - dichotomous_key: Navigates decision tree using extracted traits
  - trait_database: Compares extracted traits against known species profiles
```

The CNN is the only autonomous AI peer. All other subsystems are deterministic tools the LLM can reference to gather evidence.

### New Prompt: `TOOL_BASED_SYSTEM_PROMPT`

Replace `UNIFIED_SYSTEM_PROMPT` in `models/llm_classifier.py`:

```python
TOOL_BASED_SYSTEM_PROMPT = """You are an expert mycologist examining a mushroom specimen.
Your goal is to identify the species by combining your own visual analysis with evidence from available tools.

=== AVAILABLE TOOLS ===

1. vision_analysis — Your own expert visual examination of the provided images.
   This is your primary source of evidence. Trust your eyes first.

2. cnn_classifier — An independent AI vision system trained on mushroom images.
   This is a separate AI that simply produces an answer. It can be overconfident,
   especially on unusual lighting, odd angles, or out-of-distribution species.
   Treat its output as one signal among many, not as ground truth.

3. trait_extractor — A deterministic tool that measures morphological traits
   (cap colour, gill attachment, stem features, etc.) from segmented images.
   Its output depends on segmentation quality and lighting. Verify against images.

4. dichotomous_key — A deterministic tool that navigates a Swedish dichotomous
   identification key using extracted traits. Requires precise trait matching;
   wrong traits lead to wrong paths. May get stuck if traits are ambiguous.

5. trait_database — A deterministic tool that compares extracted traits against
   known species profiles. Uses coarse descriptions; may miss fine distinctions.

=== YOUR REASONING PROCESS ===

1. EXAMINE the images carefully. Form your own preliminary diagnosis.
2. REVIEW the cnn_classifier output. Evaluate: is it plausible? What are its weaknesses?
3. REVIEW the extracted traits from trait_extractor. Do they match what you see?
4. REVIEW the dichotomous_key result. Does the path make sense given the traits?
5. REVIEW the trait_database comparison. Does the best match align with your visual diagnosis?
6. SYNTHESIZE: Which hypothesis has the strongest evidence across ALL sources?
   - If tools agree with your visual analysis, this strengthens confidence.
   - If tools contradict your visual analysis, critically evaluate BOTH sides.
   - Consider: what if YOU are wrong? What if a tool is wrong? Which has stronger evidence?
   - Use the dichotomous key to test competing hypotheses.
7. CONCLUDE with a final species identification. Do not simply follow the majority.
   Do not stubbornly stick to your first impression. Choose the strongest hypothesis.

{key_tree_text}

Available Species ({species_count} total):
{species_list}

RESPONSE FORMAT (strict JSON):
{{
    "top_prediction": {{
        "species": "English name",
        "confidence": 0.82,
        "reasoning": "Why this species fits best"
    }},
    "predictions": [
        {{"species": "English name", "confidence": 0.82, "reasoning": "..."}},
        {{"species": "Alternative", "confidence": 0.10, "reasoning": "..."}}
    ],
    "all_signals": {{
        "cnn": "Species name or 'uncertain'",
        "trait_extractor": "Summary of key extracted traits",
        "dichotomous_key": "Species name or 'incomplete'",
        "trait_database": "Species name or 'no_match'",
        "llm_own_diagnosis": "What you observed from the images yourself"
    }},
    "tool_evaluation": {{
        "cnn_trust": "high|medium|low",
        "cnn_why": "Brief justification",
        "traits_trust": "high|medium|low",
        "traits_why": "Brief justification",
        "key_trust": "high|medium|low",
        "key_why": "Brief justification",
        "database_trust": "high|medium|low",
        "database_why": "Brief justification"
    }},
    "agreement_state": "agree|disagree|partial|inconclusive",
    "reasoning": "Detailed analysis: visual observations, tool outputs, critical evaluation, final conclusion",
    "safety_warnings": ["Any toxicity warnings"],
    "needs_clarification": false,
    "clarification_question": null,
    "confidence_in_id": 0.82,
    "ambiguous": false
}}

Be concise but thorough. Prioritize safety."""
```

Also update `_build_unified_user_input` to use "tool output" framing instead of "assistant report".

---

## Step 5 — System A1 / A2 Implementation

### A1 (`llm_a1`): Standalone vision-LLM, no oracle
- Exactly as current `LLMStandaloneRunner`
- Images → raw vision LLM → species prediction

### A2 (`llm_a2`): Standalone vision-LLM + flat oracle trait dict
- Same pipeline as A1, but the **user message** (not the system prompt) includes the oracle flat dict:

```python
# In llm_standalone_runner.py predict():
user_msg = "Identify the mushroom species from these two images."
if self.oracle_trait_provider:
    traits = self.oracle_trait_provider.get_flat_dict(specimen.species_id)
    traits_lines = "\n".join(f"  {k}: {v}" for k, v in traits.items())
    user_msg += f"\n\nKNOWN VISIBLE TRAITS:\n{traits_lines}\n\nUse these known traits as ground-truth reference. Compare them against your visual analysis."
```

**Why append to user message?** The standalone runner has a hardcoded `SYSTEM_PROMPT` that doesn't use `LLMClassifier`. Modifying the user message avoids changing system prompt behavior and is consistent with how real users would provide extra information.

### Implementation in `benchmarks/runners/llm_standalone_runner.py`

```python
class LLMStandaloneRunner:
    name = "llm_a1"  # or "llm_a2"

    def __init__(self, backend=None, oracle_trait_provider: Optional[SpeciesTraitOracle] = None):
        # oracle_trait_provider = None → A1
        # oracle_trait_provider set → A2; fetch flat traits per specimen in predict()
        ...
```

---

## Step 6 — System B1 / B2 Implementation

### B1 (`unified_b1`): Full pipeline with extracted traits
- Current `UnifiedRunner` behavior
- YOLO → trait extractor → CNN → tree → DB → LLM synthesis

### B2 (`unified_b2`): Full pipeline + perfect trait-extractor oracle
- Same pipeline shape as B1, but the trait signal consumed by downstream tools is replaced with oracle traits:
  1. **YOLO and case detection still run** on real images (needed to know morphology case for oracle key selection)
  2. **Real trait extractor runs** for diagnostics/metadata but its output is discarded for downstream tools
  3. Build `oracle_visible_traits` inside `UnifiedPipeline.run()` after case detection via `oracle_trait_provider.get_extractor_output(species_id, case=detected_case)`, normalized to the same key schema as `merged_traits`
  4. **Tree traversal** receives `oracle_visible_traits` as if it were the trait extractor's merged output
  5. **If tree is incomplete**: The LLM prompt includes the stuck node/question as context. The LLM uses this — alongside the oracle traits presented as perfect extractor output — to reason about the specimen. No separate LLM resolver method is needed; the existing synthesis prompt already receives tree status.
  6. **Database comparison** receives the same oracle trait signal used by the tree
  7. **Final LLM synthesis** sees the oracle traits in the trait-extractor output section, so it treats them as the trait extractor answers
  8. **No natural-language user description is injected** — the oracle is invisible to the LLM except as perfect trait_extractor output

### B2 Prompt Modification

The final LLM prompt presents oracle traits in the **same place and shape** as B1 presents extracted traits:

```
=== EXTRACTED TRAITS (for reference) ===
  cap_shape: convex
  cap_colour: brown
  hymenophore_type: pores
  underside_colour: yellow
  stem_colour: white
  stem_surface: reticulate
  flesh_colour: white
```

If the tree is incomplete (rare with perfect traits), the existing tree status section in the prompt already shows:

```
=== TOOL OUTPUT: dichotomous_key ===
  Status: incomplete
  Stuck at question: Does the mushroom have gills, pores, ridges, or teeth?
  Path so far: [...]
```

The LLM uses this stuck-node context directly in its synthesis reasoning. No additional oracle-specific injection is required beyond the perfect traits already shown in the extractor section.

### Implementation in `models/unified_pipeline.py`

```python
def run(
    self,
    above,
    below,
    species_id: Optional[str] = None,
    oracle_trait_provider: Optional[SpeciesTraitOracle] = None,
):
    # ... existing pipeline: YOLO segmentation, case detection ...
    detected_case = case.get("case", "unknown")

    # B1 uses merged extractor traits. B2 uses oracle traits in the same role.
    traits_for_tools = merged_traits
    oracle_used = False
    if oracle_trait_provider is not None and species_id:
        oracle_visible_traits = oracle_trait_provider.get_extractor_output(
            species_id,
            case=detected_case,
        )
        # Preserve morphology case so downstream logic sees the same case signal
        oracle_visible_traits["morphology_case"] = merged_traits.get("morphology_case", detected_case)
        traits_for_tools = oracle_visible_traits
        oracle_used = True

    # Navigate tree using traits_for_tools (extracted or oracle)
    tree_res = self.llm.navigate_tree(
        visible_traits=traits_for_tools,
        cnn_prediction=cnn_pred,
        case_info=case,
        images_b64=images_b64,
    )

    if tree_res.get("status") == "conclusion":
        comparison_target = tree_res.get("species")
    else:
        comparison_target = cnn_pred.get("species")

    db_res = self.comparator.compare(comparison_target, traits_for_tools)
    llm_res = self.llm.unified_classify(
        visible_traits=traits_for_tools,
        cnn_prediction=cnn_pred,
        tree_result=tree_res,
        db_result=db_res,
        case_info=case,
        images_b64=images_b64,
    )
```

### Implementation in `benchmarks/runners/unified_runner.py`

```python
class UnifiedRunner:
    def __init__(self, ..., oracle_trait_provider: Optional[SpeciesTraitOracle] = None):
        self.oracle_trait_provider = oracle_trait_provider

    def predict(self, specimen):
        result = self.pipeline.run(
            above,
            below,
            species_id=specimen.species_id,
            oracle_trait_provider=self.oracle_trait_provider,
        )
```

**Note**: The case-aware output requires that `UnifiedPipeline.run()` detects case from YOLO before selecting oracle traits. The runner must not pre-compute `oracle_visible_traits`; it passes the provider and species ID so the pipeline can query the oracle after case detection.

---

## Step 7 — Remove Legacy Standalone Benchmarks

### What Gets Removed from `run_comparative.py`
- Standalone `TreeRunner` benchmark column
- Standalone `TraitDBRunner` benchmark column

### What Stays
- `CNNRunner` — standalone reference baseline (CNN is an independent AI)
- `LLMStandaloneRunner` — becomes A1/A2
- `UnifiedRunner` — becomes B1/B2

### Updated `_build_per_specimen_record`

```python
def _build_per_specimen_record(specimen, cnn_res, a1_res, a2_res, b1_res, b2_res):
    ...
    return {
        "specimen_id": specimen.specimen_id,
        "species_id": gt,
        "scenario": specimen.scenario,
        "results": {
            "cnn": _result_dict(cnn_res),
            "a1": _result_dict(a1_res),
            "a2": _result_dict(a2_res),
            "b1": _result_dict(b1_res),
            "b2": _result_dict(b2_res),
        },
    }
```

---

## Step 8 — Benchmark Orchestrator Updates

### `benchmarks/run_comparative.py`

1. Remove `--methods` arg. Replace with:
   ```
   --variants {cnn,a1,a2,b1,b2,all}  (default: all)
   --manifest  (default: evaluation_manifest_v2.csv)
   ```

2. Run loop:
   ```python
   oracle = SpeciesTraitOracle(str(PROJECT_ROOT / "data" / "raw" / "species_traits.xml"))
   
   if "cnn" in selected:
       cnn_results = _run_method("cnn", CNNRunner(), specimens, single_photo=True)
   
   if "a1" in selected:
       a1_results = _run_method("a1", LLMStandaloneRunner(oracle_trait_provider=None), specimens, single_photo=False)
   
   if "a2" in selected:
       a2_results = _run_method("a2", LLMStandaloneRunner(
           oracle_trait_provider=oracle
       ), specimens, single_photo=False)
   
   if "b1" in selected:
       b1_results = _run_method("b1", UnifiedRunner(oracle_trait_provider=None), specimens, single_photo=False)
   
   if "b2" in selected:
       b2_results = _run_method("b2", UnifiedRunner(
           oracle_trait_provider=oracle
       ), specimens, single_photo=False)
   ```

3. Deprecate `--oracle-mode` flag (old `OracleKeyTree` path oracle). Remove or keep for backward compat but unused.

---

## Step 9 — Report Generation Updates

### `benchmarks/comparative_reports.py`

New Markdown report structure:

1. **Overall Accuracy Table**
   | Method | Accuracy | Coverage | Mean Time (ms) |
   |---|---|---|---|
   | CNN | ... | ... | ... |
   | A1 (LLM raw) | ... | ... | ... |
   | A2 (LLM + oracle dict) | ... | ... | ... |
   | B1 (Unified) | ... | ... | ... |
   | B2 (Unified + perfect trait oracle) | ... | ... | ... |

2. **Accuracy by Scenario**
   | Scenario | CNN | A1 | A2 | B1 | B2 |
   |---|---|---|---|---|---|
   | confusing | ... | ... | ... | ... | ... |
   | ood | ... | ... | ... | ... | ... |
   | puffball | ... | ... | ... | ... | ... |
   | coral | ... | ... | ... | ... | ... |
   | easy | ... | ... | ... | ... | ... |
   | edge_case | ... | ... | ... | ... | ... |

3. **Oracle Impact: A2 vs A1**
   - Delta = A2_accuracy - A1_accuracy
   - Shows how much raw LLM benefits from perfect vision-only trait knowledge

4. **Trait Extractor Impact: B1 vs B2**
   - `extractor_penalty = B1_accuracy - B2_accuracy`
   - **Positive penalty** = B1 performs worse than B2 → performance lost to trait extractor errors
   - **Negative penalty** = B1 outperforms B2 → extractor noise somehow helps (unlikely)
   - This isolates the cost of imperfect trait extraction on the unified pipeline

5. **Confusing Pair Breakdown**
   | Pair | CNN | A1 | A2 | B1 | B2 |
   |---|---|---|---|---|---|
   | BO.BA↔BO.ED | ... | ... | ... | ... | ... |
   | CA.CI↔HY.PS | ... | ... | ... | ... | ... |

### CSV Report Column Mapping

Old columns (remove): `tree_top`, `tree_correct`, `db_top`, `db_correct`, `llm_top`, `llm_correct`, `unified_top`, `unified_correct`

New columns (add): `a1_top`, `a1_correct`, `a2_top`, `a2_correct`, `b1_top`, `b1_correct`, `b2_top`, `b2_correct`

Kept columns: `cnn_top`, `cnn_correct`, `specimen_id`, `species_id`, `scenario`, `agreement`

---

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `benchmarks/evaluation_manifest_v2.csv` | **Create** | 30-specimen subset |
| `benchmarks/species_trait_oracle.py` | **Create** | Loads vision-only traits + builds A2 flat dicts, B2 extractor-shaped oracle outputs, and B2 species-trait dicts. Includes manual `_SPECIES_STEM_RING` mapping. |
| `benchmarks/validate_no_leakage.py` | **Create** | SHA-256 leakage checker. Defaults to `--allow-inconclusive` (warn + exit 0). Strict mode via `--strict-leakage-check`. |
| `models/llm_classifier.py` | **Modify** | Tool-based prompt + A2 dict injection; B2 sees oracle traits in trait-extractor output section |
| `benchmarks/runners/llm_standalone_runner.py` | **Modify** | A1/A2 variants with oracle_trait_provider param; A2 appends flat dict to user message |
| `benchmarks/runners/unified_runner.py` | **Modify** | B1/B2 variants with oracle_trait_provider param |
| `models/unified_pipeline.py` | **Modify** | Accept species_id + oracle_trait_provider; query case-aware oracle traits after case detection and use them as trait extractor output for tree, DB, and LLM synthesis |
| `benchmarks/run_comparative.py` | **Modify** | 4-variant + CNN orchestration; remove tree/db standalone; new CLI args |
| `benchmarks/comparative_reports.py` | **Modify** | New 5-table report structure; CSV column remap |
| `benchmarks/comparative_metrics.py` | **Modify** | Update comparison helpers for a1/a2/b1/b2; extractor_penalty = B1 - B2 |
| `Makefile` | **Modify** | Update benchmark targets |

### Makefile Target Changes

| Target | Action | New Behavior |
|---|---|---|
| `benchmark-all` | **Modify** | Runs `--variants all` instead of old `--methods all` |
| `benchmark-validate` | **Modify** | Adds `python -m benchmarks.validate_no_leakage` step |
| `benchmark-local` | **Remove** | Replaced by per-variant runs |
| `benchmark-unified` | **Remove** | Replaced by per-variant runs |

---

## Testing Strategy

1. `python -m benchmarks.validate_no_leakage` — confirm no training overlap (warns gracefully if training cache missing)
2. `python -m benchmarks.run_comparative --variants cnn` — CNN baseline only
3. `python -m benchmarks.run_comparative --variants a1` — A1 smoke test
4. `python -m benchmarks.run_comparative --variants a2` — A2 smoke test (check flat dict appended to user message)
5. `python -m benchmarks.run_comparative --variants b1` — B1 smoke test
6. `python -m benchmarks.run_comparative --variants b2` — B2 smoke test (check oracle traits feed tree traversal, DB comparison, final LLM synthesis)
7. `python -m benchmarks.run_comparative --variants all` — full 30-specimen run
8. Inspect `report.md` for all 5 tables and correct oracle-impact deltas

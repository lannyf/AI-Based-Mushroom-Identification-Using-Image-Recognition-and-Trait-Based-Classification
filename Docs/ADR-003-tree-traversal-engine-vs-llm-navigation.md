# ADR-003: Programmatic Tree Traversal Engine vs. LLM-as-Navigator

**Status:** Accepted  
**Date:** 2026-05-14  
**Context:** Unified pipeline architecture — decision-tree component  
**Related:** `Docs/09-hybrid-system.md`, `benchmarks/key_tree_oracle.py`

---

## Problem

The project uses a Swedish dichotomous key (`data/raw/key.xml`) as one of four signals feeding into a final LLM synthesis layer. A fundamental architectural question arose:

> **Should the LLM itself navigate the tree** (reading each question, comparing options against observed traits, choosing branches), **or should a programmatic engine traverse the tree** and pass the result to the LLM as a pre-digested signal?

The two approaches represent very different philosophies:
- **LLM-as-Navigator**: The LLM is the expert mycologist actively using the key, as a human would.
- **Engine-as-Navigator**: A deterministic subsystem traverses the key; the LLM synthesises its output along with CNN, trait-DB, and visual-trait signals.

## Decision

**Keep the programmatic `KeyTreeEngine` as the tree navigator.** The LLM receives the traversal *result* (conclusion or stuck-at-question) as one of several input signals. It does not actively navigate the tree node-by-node.

This decision was made after evaluating three candidate architectures (see Alternatives Considered below).

## Rationale

### 1. Speed constraint (primary driver)

The thesis runs on CPU-only hardware (AMD Ryzen 7 5825U). A single LLM call with images takes ~280 s (text-only ~115 s). The benchmark must process **57 specimens**.

| Architecture | LLM calls per specimen | Est. time per specimen | Total for 57 specimens |
|-------------|----------------------|----------------------|----------------------|
| **Engine navigates** (current) | 1 | ~280 s | ~4.5 h |
| **Iterative LLM** (LLM decides each node) | ~8–12 (tree depth) | ~30–45 min | ~28–42 h |
| **One-shot LLM path** | 1 (larger prompt) | ~300 s | ~4.8 h |

An overnight benchmark window is ~8 h. Only the engine-based and one-shot approaches fit.

### 2. Reproducibility and validation

A programmatic engine produces **deterministic** outputs for the same trait/CNN inputs. This is essential for:
- The oracle A/B experiment (perfect `pre_answers` must be verifiable)
- Debugging why a specimen failed
- McNemar statistical tests between methods

An LLM navigator introduces non-determinism (temperature, sampling) that would confound the benchmark.

### 3. The oracle experiment design

The thesis includes a **Key-Tree Oracle** experiment: 5 benchmark species receive perfect `pre_answers` (guaranteed tree conclusion), 5 receive none. This isolates the effect of tree completeness on final LLM synthesis quality.

This experiment is only meaningful if:
- The engine's traversal is the controlled variable.
- The LLM's synthesis is the measured outcome.

If the LLM were the navigator, the oracle would have to be redefined as "LLM receives the correct answer at each node" — a much harder intervention to implement cleanly.

### 4. Separation of concerns

The current pipeline has clean layers:

```
YOLO      → Segmentation masks
Traits    → Visual feature extraction (deterministic CV)
CNN       → Deep-learning classifier (probabilistic)
Tree      → Rule-based traversal (deterministic logic)
DB        → Structured trait lookup (deterministic SQL-like)
LLM       → Synthesis & reasoning (probabilistic, high-level)
```

The tree engine is a **deterministic rule-based subsystem**, analogous to a traditional expert system. The LLM is the **probabilistic synthesis layer**. Mixing these roles would blur the architectural boundary and make ablation studies harder.

### 5. Tree structure is already injected for LLM reasoning

The system prompt includes the full decision tree (`{key_tree_text}`). The LLM is instructed:

> *"Use the decision tree above to reason about which path best matches the observed traits."*

So the LLM **can** (and does) use the tree structure to critique or override the engine's conclusion when signals conflict. It is not blind to the tree — it simply does not execute the traversal itself.

---

## Trade-offs and Consequences

### Positive

- **Overnight benchmark feasible**: 57 specimens complete in ~4.5 h.
- **Deterministic oracle experiment**: Perfect `pre_answers` are guaranteed to produce a conclusion.
- **Clear ablation boundaries**: Each subsystem can be swapped or disabled independently.
- **Debuggable**: Tree paths are logged; a failed traversal is inspectable.

### Negative

- **LLM cannot recover from engine errors**: If `KeyTreeEngine` takes a wrong branch due to a heuristic misclassifying a trait, the LLM sees the wrong conclusion as "strong evidence" and may not override it.
- **Engine brittleness is inherited**: The tree's incompleteness (only ~12 of 23 benchmark species map to XML decisions) becomes the LLM's incompleteness.
- **Not a "true" expert system**: A purist view holds that the LLM should be the reasoning agent and the tree should be its tool, not a separate black box.

### Neutral / Accepted

- **Thesis reframing**: The experiment measures *"how much does a complete tree signal improve LLM synthesis?"* rather than *"can an LLM navigate a dichotomous key?"*. Both are valid research questions; the former is what the current infrastructure supports.

---

## Alternatives Considered

### Alternative A: Iterative LLM Navigation

At each tree node, the LLM receives the question + options + observed traits and returns the chosen branch. The system advances the node and repeats until a leaf is reached.

**Rejected**: 8–12 LLM calls per specimen × ~115 s = ~15–23 min per specimen. 57 specimens would take ~14–22 hours, exceeding the overnight window. Also introduces non-determinism at every node.

### Alternative B: One-Shot LLM Path Output

In a single LLM call, ask the model to output its complete chosen path through the tree (e.g., `["Slekt: Russula", "Färg: Röd", "Smak: Skarp", ...]`). A validator checks path validity against the XML structure.

**Rejected**: Requires significant new prompt engineering, response parsing, and path-validation logic. The gain (LLM as navigator) is marginal for the thesis because the LLM already sees the tree and can critique the engine's path. Would delay the benchmark by days.

### Alternative C: Keep Engine, Add LLM-Path Reasoning

Hybrid: engine traverses and produces a result; the LLM additionally outputs its *own* path through the tree for comparison.

**Rejected**: Doubles prompt size and parsing complexity for speculative value. The thesis does not need to compare engine paths vs. LLM paths — it needs to compare *complete* vs. *incomplete* tree signals.

---

## Implementation Notes

- `models/key_tree_traversal.py` — `KeyTreeEngine` with `start_session(..., pre_answers=...)`
- `benchmarks/key_tree_oracle.py` — `OracleKeyTree` that extracts perfect root-to-leaf paths from `key.xml`
- `models/llm_classifier.py` — `_build_unified_user_input()` formats the engine's `tree_result` for the LLM prompt
- `benchmarks/run_comparative.py` — `--oracle-mode` flag wires the oracle into `TreeRunner` and `UnifiedRunner`

---

## Future Work (Post-Thesis)

If the project moves to GPU-hosted inference or API-based LLMs (sub-second latency), **Alternative A** (iterative LLM navigation) becomes viable and would be a stronger demonstration of LLM-as-expert. At that point, the oracle experiment could be redefined as "LLM receives the correct answer at each ambiguous node."

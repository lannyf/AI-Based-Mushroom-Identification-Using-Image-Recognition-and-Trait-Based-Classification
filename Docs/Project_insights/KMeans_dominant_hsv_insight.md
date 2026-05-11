# Project Insight: K-Means Clustering in `_dominant_hsv()`

**Date:** 2026-05-07
**Source:** `models/visual_trait_extractor.py`, function `_dominant_hsv()`
**Context:** Method section — classical computer vision colour extraction

---

## 1. What `n_init=5` Means

When `n_init=5`, scikit-learn runs the full K-Means algorithm **5 separate times**, each with a different random initial placement of the cluster centres. After all 5 runs finish, it keeps the single result with the lowest **inertia**.

**Inertia** (within-cluster sum-of-squares, WCSS):

$$
\text{inertia} = \sum_{i=1}^{n} \min_{\mu_j \in C} \left( \|x_i - \mu_j\|^2 \right)
$$

In plain terms: for every pixel, compute its squared Euclidean distance to the nearest cluster centre, then sum all of those distances. The run with the **smallest total sum wins**.

**Why this matters:** K-Means is sensitive to initialisation. A bad random start can trap centroids in poor local minima (e.g., two centres competing for the same colour while another colour is ignored). Running 5 times and picking the best inertia makes the result far more stable.

---

## 2. What Inertia Actually Measures

Inertia measures the **total compactness across all clusters combined** — it is a single global number for the entire clustering, not a per-cluster metric.

- **Lower inertia** → on average, points sit closer to their assigned centre → tighter clusters overall.
- **Higher inertia** → points are spread farther from their centres → looser, more diffuse clusters.

**Important caveat:** inertia always decreases as you add more clusters. This bias is irrelevant here because `n_clusters` is fixed (4 for `_dominant_hsv`, 5 in `analyse_colours`), so lower inertia genuinely indicates a better arrangement.

---

## 3. What `random_state=42` Does

K-Means needs random numbers for the **initial centroid positions** (randomly sampled from the data points) and for tie-breaking.

Without `random_state`, the global NumPy RNG starts from a different state each program run, so initial centroids land in different spots. Even with `n_init=5`, results can vary across executions.

With `random_state=42`, scikit-learn creates a **dedicated, seeded RNG** isolated from the global one. The sequence of "random" choices is **exactly the same every time**. Therefore:

| Property | Without `random_state` | With `random_state=42` |
|----------|------------------------|------------------------|
| Initial centres | Different every run | Identical every run |
| 5 initialisations | Different every run | Identical every run |
| Winning run | May change | Always the same |
| Final cluster centres | Non-deterministic | Deterministic (byte-for-byte) |

**Note:** The value `42` is arbitrary — any integer would work. It is used here as a conventional fixed seed.

---

## 4. What Changes Between Runs (and What Doesn't)

**Fixed across all 5 runs:**
- Number of clusters (`n_clusters=4` or `5`)
- Input data (the same HSV pixel array)

**What changes:**
- The **initial random positions** of the cluster centres

Each run then iteratively refines its centres:
1. Assign each pixel to the nearest centre
2. Move each centre to the mean of its assigned pixels
3. Repeat until convergence

Because the starting positions differ, the algorithm can settle into **different local minima**. `n_init` simply picks the best one.

---

## 5. Practical Relevance to the Project

In `visual_trait_extractor.py`:
- `_dominant_hsv()` uses `n_clusters=4`, `n_init=5`, `random_state=42`
- `analyse_colours()` uses `n_clusters=5`, `n_init=5`, `random_state=42`

This configuration ensures:
1. **Reproducibility:** Every execution of the pipeline on the same image yields identical dominant/secondary colour predictions.
2. **Robustness:** The 5 initialisations protect against unlucky random starts that would misrepresent the cap colour.
3. **Determinism:** Benchmarks and thesis experiments are fully repeatable.

**Known limitation:** K-Means assumes roughly spherical clusters. A smooth colour gradient on a mushroom cap (e.g., yellow centre fading to orange edge) may be split into two clusters or merged with background pixels. For the project's use case, this is generally acceptable because the mushroom cap usually occupies the largest contiguous region and tends to dominate one or two clusters.

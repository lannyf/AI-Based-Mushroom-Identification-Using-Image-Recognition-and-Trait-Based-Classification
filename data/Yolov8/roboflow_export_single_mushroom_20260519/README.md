# Roboflow Export: Single-Mushroom Selection

Created from visual review on 2026-05-19.

Criteria:
- Include only images with one clear mushroom / one clear coral fruiting body.
- Include above/top-cap views, underside/hymenophore views, and coral mushrooms.
- Exclude obvious backgrounds, puffballs, bracket/polypore-heavy folders, and multi-mushroom/cluster images where the target is ambiguous.

Files are flat in `images/` and prefixed with category: `above__`, `underside__`, or `coral__`.

See `selection_manifest.csv` for original source paths and visual-review indices.

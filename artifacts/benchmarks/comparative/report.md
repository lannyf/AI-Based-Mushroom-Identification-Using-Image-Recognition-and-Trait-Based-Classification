# Comparative Benchmark Results

## 1. Overall Accuracy Comparison

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| cnn | 0.0% | 0.0% | 0.0 |
| tree | 36.4% | 38.6% | 494.2 |
| db | 5.3% | 100.0% | 2.7 |
| unified | 14.8% | 94.7% | 332165.0 |

## 2. McNemar Paired Comparison vs. Unified

| Comparison | Discordant | Unified Wins | Other Wins | p-value | Significant? |
|------------|------------|--------------|------------|---------|--------------|
| unified_vs_cnn | 0 | 0 | 0 | 1.0000 | No |
| unified_vs_tree | 3 | 1 | 2 | 1.0000 | No |
| unified_vs_db | 7 | 6 | 1 | 0.1250 | No |

## 3. Confusing-Pair Breakdown

| Pair | Specimens | CNN Acc | Tree Acc | DB Acc | Unified Acc | Agreement Rate |
|------|-----------|---------|----------|--------|-------------|----------------|
| AM.MU-NONE | 3 | 0% | 0% | 0% | 33% | 33% |
| AM.VI-NONE | 3 | 0% | 0% | 0% | 0% | 0% |
| BO.BA-BO.ED | 6 | 0% | 17% | 0% | 17% | 50% |
| CA.CI-HY.PS | 6 | 0% | 0% | 0% | 0% | 33% |
| CR.CO-NONE | 2 | 0% | 50% | 100% | 50% | 50% |
| NONE-SP.CR | 2 | 0% | 100% | 0% | 50% | 50% |

## 4. Cases Where Unified Outperformed All Standalone Methods

| Specimen | GT | CNN | Tree | DB | Unified | Reasoning |
|----------|----|-----|------|----|---------|-----------|
| AM.MU_005 | AM.MU | N/A | N/A | LA.HE | AM.MU | The morphological traits strongly point towards *Amanita mus… |
| HY.RE_049 | HY.RE | N/A | CA.AU | LA.HE | HY.RE | The database comparison provides the strongest evidence (67.… |

## 5. Cases Where Unified Was Wrong But a Standalone Method Was Right

| Specimen | GT | CNN | Tree | DB | Unified | Notes |
|----------|----|-----|------|----|---------|-------|
| CR.CO_034 | CR.CO | N/A | N/A | CR.CO | CA.TU | paired above_idx=1 below_idx=2; trumpet;dark;funnel; source=Svampeatlas/GBIF; occurrence=5052262151; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| SP.CR_040 | SP.CR | N/A | SP.CR | MO.ES | LE.VU | paired above_idx=2 below_idx=1; cauliflower;branched;edible; source=Svampeatlas/GBIF; occurrence=5898657877; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| CA.TU_048 | CA.TU | N/A | CA.TU | LE.LU | CA.CI | paired above_idx=1 below_idx=2; funnel_chanterelle;folds;hollow_stem; source=Svampeatlas/GBIF; occurrence=5898654001; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |

## 6. Agreement Statistics

| Agreement Level | Count | % | Avg Unified Accuracy |
|-----------------|-------|---|----------------------|
| agree | 7 | 12.3% | 28.6% |
| partial | 11 | 19.3% | 36.4% |
| disagree | 36 | 63.2% | 5.6% |
| inconclusive | 3 | 5.3% | 0.0% |

## 7. Accuracy by Scenario

| Scenario | N | CNN | Tree | DB | Unified |
|----------|---|-----|------|----|---------|
| coral | 6 | 0% | 0% | 0% | 0% |
| edge_case | 2 | 0% | 0% | 0% | 0% |
| puffball | 5 | 0% | 33% | 0% | 20% |
| easy | 5 | 0% | 0% | 0% | 0% |
| confusing | 22 | 0% | 50% | 9% | 20% |
| ood | 17 | 0% | 38% | 6% | 19% |

# Comparative Benchmark Results

## 1. Overall Accuracy Comparison

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| cnn | 52.6% | 100.0% | 111.5 |
| tree | 36.4% | 38.6% | 1038.2 |
| db | 5.3% | 100.0% | 2.5 |
| unified | 0.0% | 0.0% | 0.0 |

## 2. McNemar Paired Comparison vs. Unified

| Comparison | Discordant | Unified Wins | Other Wins | p-value | Significant? |
|------------|------------|--------------|------------|---------|--------------|
| unified_vs_cnn | 0 | 0 | 0 | 1.0000 | No |
| unified_vs_tree | 0 | 0 | 0 | 1.0000 | No |
| unified_vs_db | 0 | 0 | 0 | 1.0000 | No |

## 3. Confusing-Pair Breakdown

| Pair | Specimens | CNN Acc | Tree Acc | DB Acc | Unified Acc | Agreement Rate |
|------|-----------|---------|----------|--------|-------------|----------------|
| AM.MU-NONE | 3 | 100% | 33% | 0% | 0% | 33% |
| AM.VI-NONE | 3 | 100% | 100% | 0% | 0% | 100% |
| BO.BA-BO.ED | 6 | 83% | 17% | 0% | 0% | 17% |
| CA.CI-HY.PS | 6 | 33% | 17% | 0% | 0% | 33% |
| CR.CO-NONE | 2 | 100% | 100% | 100% | 0% | 100% |
| NONE-SP.CR | 2 | 100% | 0% | 0% | 0% | 0% |

## 4. Cases Where Unified Outperformed All Standalone Methods

*No such cases found.*

## 5. Cases Where Unified Was Wrong But a Standalone Method Was Right

*No such cases found.*

## 6. Agreement Statistics

| Agreement Level | Count | % | Avg Unified Accuracy |
|-----------------|-------|---|----------------------|
| agree | 3 | 5.3% | 0.0% |
| partial | 11 | 19.3% | 0.0% |
| disagree | 43 | 75.4% | 0.0% |
| inconclusive | 0 | 0.0% | 0.0% |

## 7. Accuracy by Scenario

| Scenario | N | CNN | Tree | DB | Unified |
|----------|---|-----|------|----|---------|
| edge_case | 2 | 50% | 0% | 0% | 0% |
| easy | 5 | 80% | 0% | 0% | 0% |
| coral | 6 | 67% | 0% | 0% | 0% |
| puffball | 5 | 40% | 0% | 0% | 0% |
| ood | 17 | 12% | 0% | 6% | 0% |
| confusing | 22 | 77% | 80% | 9% | 0% |

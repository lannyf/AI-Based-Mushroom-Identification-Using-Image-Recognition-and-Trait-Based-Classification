# Comparative Benchmark Results

## 1. Overall Accuracy Comparison

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| cnn | 47.0% | 100.0% | 123.4 |
| tree | 36.4% | 44.0% | 1118.2 |
| db | 2.0% | 100.0% | 2.4 |
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
| AM.MU-NONE | 5 | 40% | 40% | 0% | 0% | 60% |
| AM.VI-NONE | 5 | 60% | 20% | 0% | 0% | 20% |
| BO.BA-CA.CA | 5 | 60% | 0% | 0% | 0% | 40% |
| BO.ED-CA.CA | 5 | 60% | 40% | 0% | 0% | 40% |
| CA.CI-HY.PS | 12 | 58% | 50% | 0% | 0% | 58% |
| CA.CN-CR.CO | 5 | 100% | 100% | 20% | 0% | 100% |
| NONE-SP.CR | 6 | 100% | 0% | 0% | 0% | 0% |

## 4. Cases Where Unified Outperformed All Standalone Methods

*No such cases found.*

## 5. Cases Where Unified Was Wrong But a Standalone Method Was Right

*No such cases found.*

## 6. Agreement Statistics

| Agreement Level | Count | % | Avg Unified Accuracy |
|-----------------|-------|---|----------------------|
| agree | 2 | 2.0% | 0.0% |
| partial | 29 | 29.0% | 0.0% |
| disagree | 69 | 69.0% | 0.0% |
| inconclusive | 0 | 0.0% | 0.0% |

## 7. Accuracy by Scenario

| Scenario | N | CNN | Tree | DB | Unified |
|----------|---|-----|------|----|---------|
| coral | 10 | 30% | 0% | 0% | 0% |
| edge_case | 6 | 33% | 0% | 0% | 0% |
| easy | 12 | 42% | 0% | 0% | 0% |
| puffball | 5 | 60% | 0% | 0% | 0% |
| confusing | 43 | 67% | 64% | 2% | 0% |
| ood | 24 | 21% | 0% | 4% | 0% |

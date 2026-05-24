# Comparative Benchmark Results

## 1. Overall Accuracy

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| cnn | 53.3% | 100.0% | 151.8 |
| a1 | 20.0% | 100.0% | 17541.6 |
| a2 | 30.0% | 100.0% | 14661.7 |
| b1 | 0.0% | 0.0% | 65759.2 |
| b2 | 0.0% | 0.0% | 76263.7 |

## 2. Accuracy by Scenario

| Scenario | N | CNN | A1 | A2 | B1 | B2 |
|----------|---|-----|----|----|----|----|
| ood | 10 | 20% | 20% | 20% | 0% | 0% |
| confusing | 10 | 60% | 30% | 70% | 0% | 0% |
| easy | 10 | 80% | 10% | 0% | 0% | 0% |

## 3. Oracle Impact: A2 vs A1

*How much raw LLM benefits from perfect vision-only trait knowledge.*

- A1 accuracy: 20.0%
- A2 accuracy: 30.0%
- Delta (A2 - A1): +10.0%

## 4. Trait Extractor Impact: B1 vs B2

*Performance lost to imperfect trait extraction. Positive extractor_penalty = B1 performs worse than B2.*

- B1 accuracy (extracted traits): 0.0%
- B2 accuracy (oracle traits): 0.0%
- Extractor penalty (B1 - B2): +0.0%

## 5. Confusing Pair Breakdown

| Pair | N | CNN | A1 | A2 | B1 | B2 |
|------|---|-----|----|----|----|----|
| AM.MU-NONE | 1 | 100% | 100% | 100% | 0% | 0% |
| AM.VI-NONE | 1 | 100% | 0% | 100% | 0% | 0% |
| BO.BA-BO.ED | 4 | 75% | 0% | 75% | 0% | 0% |
| CA.CI-HY.PS | 4 | 25% | 50% | 50% | 0% | 0% |

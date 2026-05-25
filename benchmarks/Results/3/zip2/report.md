# Comparative Benchmark Results

## 1. Overall Accuracy

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| cnn | 53.3% | 100.0% | 157.9 |
| a1 | 20.0% | 100.0% | 17856.8 |
| a2 | 30.0% | 100.0% | 14811.7 |
| b1 | 40.0% | 100.0% | 74277.6 |
| b2 | 43.3% | 100.0% | 79308.6 |

## 2. Accuracy by Scenario

| Scenario | N | CNN | A1 | A2 | B1 | B2 |
|----------|---|-----|----|----|----|----|
| easy | 10 | 80% | 10% | 0% | 30% | 40% |
| ood | 10 | 20% | 20% | 20% | 10% | 10% |
| confusing | 10 | 60% | 30% | 70% | 80% | 80% |

## 3. Oracle Impact: A2 vs A1

*How much raw LLM benefits from perfect vision-only trait knowledge.*

- A1 accuracy: 20.0%
- A2 accuracy: 30.0%
- Delta (A2 - A1): +10.0%

## 4. Trait Extractor Impact: B1 vs B2

*Performance lost to imperfect trait extraction. Positive extractor_penalty = B1 performs worse than B2.*

- B1 accuracy (extracted traits): 40.0%
- B2 accuracy (oracle traits): 43.3%
- Extractor penalty (B1 - B2): -3.3%

## 5. Confusing Pair Breakdown

| Pair | N | CNN | A1 | A2 | B1 | B2 |
|------|---|-----|----|----|----|----|
| AM.MU-NONE | 1 | 100% | 100% | 100% | 100% | 100% |
| AM.VI-NONE | 1 | 100% | 0% | 100% | 100% | 100% |
| BO.BA-BO.ED | 4 | 75% | 0% | 75% | 75% | 75% |
| CA.CI-HY.PS | 4 | 25% | 50% | 50% | 75% | 75% |

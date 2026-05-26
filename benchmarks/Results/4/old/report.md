# Comparative Benchmark Results

## 1. Overall Accuracy

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| cnn | 56.7% | 100.0% | 560.7 |
| a1_vision | 23.3% | 100.0% | 18119.0 |
| a1_llm | 0.0% | 0.0% | 0.0 |
| a1_tree | 0.0% | 0.0% | 0.0 |
| a1_db | 0.0% | 0.0% | 0.0 |
| a2_llm | 36.7% | 100.0% | 15431.2 |
| a2_tree | 50.0% | 20.0% | 142.2 |
| a2_db | 73.3% | 100.0% | 5.7 |
| b1 | 0.0% | 0.0% | 1133.2 |
| b2 | 0.0% | 0.0% | 1101.5 |

## 2. System A1 — Extracted-Trait Components

| Method | Accuracy | Coverage |
|--------|----------|----------|
| a1_vision | 23.3% | 100.0% |
| a1_llm | 0.0% | 0.0% |
| a1_tree | 0.0% | 0.0% |
| a1_db | 0.0% | 0.0% |

## 3. System A2 — Oracle-Trait Components

| Method | Accuracy | Coverage |
|--------|----------|----------|
| a2_llm | 36.7% | 100.0% |
| a2_tree | 50.0% | 20.0% |
| a2_db | 73.3% | 100.0% |

## 4. System B — Unified Pipeline

| Method | Accuracy | Coverage |
|--------|----------|----------|
| b1 | 0.0% | 0.0% |
| b2 | 0.0% | 0.0% |

## 5. Oracle Benefit (A2 − A1) per Component

*How much each standalone component gains from perfect trait knowledge.*

| Component | A1 (extracted) | A2 (oracle) | Δ |
|-----------|----------------|-------------|---|
| llm | 0.0% | 36.7% | +36.7% |
| tree | 0.0% | 50.0% | +50.0% |
| db | 0.0% | 73.3% | +73.3% |

## 6. Synthesis Benefit (B − best standalone)

*Does the unified pipeline outperform the best standalone component in its system?*

| System | Best Standalone | Unified | Δ (synthesis) |
|--------|-----------------|---------|---------------|
| A1 → B1 | 23.3% | 0.0% | -23.3% |
| A2 → B2 | 73.3% | 0.0% | -73.3% |

## 7. Trait Extractor Impact (B1 vs B2)

*Performance lost to imperfect automatic trait extraction. Positive extractor_penalty = B1 performs worse than B2.*

- B1 accuracy (extracted traits): 0.0%
- B2 accuracy (oracle traits): 0.0%
- Extractor penalty (B1 − B2): +0.0%

## 8. Accuracy by Scenario

| Scenario | N | cnn | a1_vision | a1_llm | a1_tree | a1_db | a2_llm | a2_tree | a2_db | b1 | b2 |
|----------|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| ood | 10 | 20.0% | 10.0% | 0.0% | 0.0% | 0.0% | 20.0% | 0.0% | 80.0% | 0.0% | 0.0% |
| easy | 10 | 80.0% | 20.0% | 0.0% | 0.0% | 0.0% | 20.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| confusing | 10 | 70.0% | 40.0% | 0.0% | 0.0% | 0.0% | 70.0% | 100.0% | 40.0% | 0.0% | 0.0% |

## 9. Confusing Pair Breakdown

| Pair | N | cnn | a1_vision | a1_llm | a1_tree | a1_db | a2_llm | a2_tree | a2_db | b1 | b2 |
|------|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| AM.MU-NONE | 1 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| AM.VI-NONE | 1 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| BO.BA-BO.ED | 4 | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 75.0% | 25.0% | 0.0% | 0.0% | 0.0% |
| CA.CI-HY.PS | 4 | 75.0% | 50.0% | 0.0% | 0.0% | 0.0% | 50.0% | 25.0% | 50.0% | 0.0% | 0.0% |

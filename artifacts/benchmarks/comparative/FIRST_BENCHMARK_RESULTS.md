# First Comparative Benchmark Results

**Date:** 2026-05-14 20:05

## Summary by Method

| Method | Accuracy | Coverage | Mean Time |
|--------|----------|----------|-----------|
| CNN    | 52.6% (30/57) | 100% | 103 ms |
| Tree   | 36.4% (8/22) | 39% | 863 ms |
| Trait DB | 5.3% (3/57) | 100% | 2.4 ms |
| Unified | 0.0% (0/0) | 0% | 271 s |

> **Note:** Unified pipeline had **0% coverage** because Ollama/LLM calls failed or timed out for every specimen. The unified column below shows `N/A` for all rows.

---

## Per-Specimen Results

| # | Specimen | True | Scenario | CNN → | Tree → | DB → | Unified |
|---|----------|------|----------|-------|--------|------|---------|
| 1 | `AG.AU_001` | `AG.AU` | ood | ✅ `AG.AU` | *no coverage* | ❌ `TY.FE` | *no coverage* |
| 2 | `AG.AU_002` | `AG.AU` | ood | ✅ `AG.AU` | ❌ `CR.CO` | ❌ `CR.CO` | *no coverage* |
| 3 | `AG.AU_003` | `AG.AU` | ood | ❌ `AM.VI` | *no coverage* | ❌ `GY.ES` | *no coverage* |
| 4 | `AM.MU_004` | `AM.MU` | confusing | ✅ `AM.MU` | *no coverage* | ❌ `CR.CO` | *no coverage* |
| 5 | `AM.MU_005` | `AM.MU` | confusing | ✅ `AM.MU` | *no coverage* | ❌ `GY.ES` | *no coverage* |
| 6 | `AM.MU_006` | `AM.MU` | confusing | ✅ `AM.MU` | ✅ `AM.MU` | ❌ `CR.CO` | *no coverage* |
| 7 | `AM.VI_007` | `AM.VI` | confusing | ✅ `AM.VI` | ✅ `AM.VI` | ❌ `HY.RE` | *no coverage* |
| 8 | `AM.VI_008` | `AM.VI` | confusing | ✅ `AM.VI` | ✅ `AM.VI` | ❌ `HY.RE` | *no coverage* |
| 9 | `AM.VI_009` | `AM.VI` | confusing | ✅ `AM.VI` | ✅ `AM.VI` | ❌ `LE.LU` | *no coverage* |
| 10 | `BO.BA_010` | `BO.BA` | confusing | ✅ `BO.BA` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 11 | `BO.BA_011` | `BO.BA` | confusing | ✅ `BO.BA` | ❌ `CA.AU` | ❌ `HY.RE` | *no coverage* |
| 12 | `BO.BA_012` | `BO.BA` | confusing | ✅ `BO.BA` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 13 | `BO.ED_013` | `BO.ED` | confusing | ❌ `BO.BA` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 14 | `BO.ED_014` | `BO.ED` | confusing | ✅ `BO.ED` | ✅ `BO.ED` | ❌ `CA.CI` | *no coverage* |
| 15 | `BO.ED_015` | `BO.ED` | confusing | ✅ `BO.ED` | *no coverage* | ❌ `CR.CO` | *no coverage* |
| 16 | `CA.CI_016` | `CA.CI` | confusing | ❌ `SP.CR` | *no coverage* | ❌ `AR.ME` | *no coverage* |
| 17 | `CA.CI_017` | `CA.CI` | confusing | ❌ `BO.ED` | *no coverage* | ❌ `HY.RE` | *no coverage* |
| 18 | `CA.CI_018` | `CA.CI` | confusing | ❌ `BO.ED` | *no coverage* | ❌ `BO.ED` | *no coverage* |
| 19 | `HY.PS_019` | `HY.PS` | confusing | ✅ `HY.PS` | ✅ `HY.PS` | ❌ `CR.CO` | *no coverage* |
| 20 | `HY.PS_020` | `HY.PS` | confusing | ❌ `BO.BA` | ❌ `CA.AU` | ❌ `CA.CI` | *no coverage* |
| 21 | `HY.PS_021` | `HY.PS` | confusing | ✅ `HY.PS` | *no coverage* | ❌ `AR.ME` | *no coverage* |
| 22 | `LA.HE_022` | `LA.HE` | easy | ✅ `LA.HE` | ❌ `CA.TU` | ❌ `GY.ES` | *no coverage* |
| 23 | `LA.HE_023` | `LA.HE` | easy | ✅ `LA.HE` | ❌ `CA.TU` | ❌ `LE.LU` | *no coverage* |
| 24 | `LA.HE_024` | `LA.HE` | easy | ❌ `CO.CO` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 25 | `RA.BO_025` | `RA.BO` | coral | ❌ `SP.CR` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 26 | `RA.BO_026` | `RA.BO` | coral | ✅ `RA.BO` | ❌ `CA.TU` | ❌ `LE.LU` | *no coverage* |
| 27 | `RA.BO_027` | `RA.BO` | coral | ❌ `RA.PA` | *no coverage* | ❌ `HY.RE` | *no coverage* |
| 28 | `RA.PA_028` | `RA.PA` | coral | ✅ `RA.PA` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 29 | `RA.PA_029` | `RA.PA` | coral | ✅ `RA.PA` | *no coverage* | ❌ `GY.ES` | *no coverage* |
| 30 | `RA.PA_030` | `RA.PA` | coral | ✅ `RA.PA` | *no coverage* | ❌ `PL.OS` | *no coverage* |
| 31 | `CO.CO_031` | `CO.CO` | edge_case | ❌ `LA.VO` | *no coverage* | ❌ `BO.ED` | *no coverage* |
| 32 | `CO.CO_032` | `CO.CO` | edge_case | ✅ `CO.CO` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 33 | `CR.CO_033` | `CR.CO` | confusing | ✅ `CR.CO` | ✅ `CR.CO` | ✅ `CR.CO` | *no coverage* |
| 34 | `CR.CO_034` | `CR.CO` | confusing | ✅ `CR.CO` | ✅ `CR.CO` | ✅ `CR.CO` | *no coverage* |
| 35 | `FO.BE_035` | `FO.BE` | easy | ✅ `FO.BE` | *no coverage* | ❌ `CR.CO` | *no coverage* |
| 36 | `FO.BE_036` | `FO.BE` | easy | ✅ `FO.BE` | *no coverage* | ❌ `BO.ED` | *no coverage* |
| 37 | `LY.PE_037` | `LY.PE` | puffball | ✅ `LY.PE` | *no coverage* | ❌ `HY.RE` | *no coverage* |
| 38 | `LY.PE_038` | `LY.PE` | puffball | ✅ `LY.PE` | *no coverage* | ❌ `HY.RE` | *no coverage* |
| 39 | `SP.CR_039` | `SP.CR` | confusing | ✅ `SP.CR` | *no coverage* | ❌ `GY.ES` | *no coverage* |
| 40 | `SP.CR_040` | `SP.CR` | confusing | ✅ `SP.CR` | *no coverage* | ❌ `MO.ES` | *no coverage* |
| 41 | `GY.ES_041` | `GY.ES` | ood | ❌ `CO.CO` | *no coverage* | ❌ `CR.CO` | *no coverage* |
| 42 | `GY.ES_042` | `GY.ES` | ood | ❌ `SP.CR` | *no coverage* | ❌ `CA.CI` | *no coverage* |
| 43 | `GA.MA_043` | `GA.MA` | ood | ❌ `AM.VI` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 44 | `GA.MA_044` | `GA.MA` | ood | ❌ `CA.CI` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 45 | `PL.OS_045` | `PL.OS` | ood | ❌ `CR.CO` | ❌ `CR.CO` | ❌ `HY.RE` | *no coverage* |
| 46 | `PL.OS_046` | `PL.OS` | ood | ❌ `LA.HE` | ❌ `CR.CO` | ❌ `HY.CA` | *no coverage* |
| 47 | `CA.TU_047` | `CA.TU` | ood | ❌ `LA.HE` | *no coverage* | ❌ `CR.CO` | *no coverage* |
| 48 | `CA.TU_048` | `CA.TU` | ood | ❌ `LA.HE` | *no coverage* | ❌ `LE.LU` | *no coverage* |
| 49 | `HY.RE_049` | `HY.RE` | ood | ❌ `CA.CI` | ❌ `CA.AU` | ❌ `LA.DE` | *no coverage* |
| 50 | `HY.RE_050` | `HY.RE` | ood | ❌ `HY.PS` | *no coverage* | ✅ `HY.RE` | *no coverage* |
| 51 | `CAL.GI_051` | `CAL.GI` | puffball | ❌ `SP.CR` | ❌ `CA.TU` | ❌ `GY.ES` | *no coverage* |
| 52 | `CAL.GI_052` | `CAL.GI` | puffball | ❌ `LY.PE` | *no coverage* | ❌ `GY.ES` | *no coverage* |
| 53 | `CAL.GI_053` | `CAL.GI` | puffball | ❌ `LY.PE` | ❌ `CA.TU` | ❌ `BO.ED` | *no coverage* |
| 54 | `RU.IN_054` | `RU.IN` | ood | ❌ `BO.ED` | ❌ `BO.ED` | ❌ `CR.CO` | *no coverage* |
| 55 | `RU.IN_055` | `RU.IN` | ood | ❌ `AM.MU` | *no coverage* | ❌ `CR.CO` | *no coverage* |
| 56 | `RU.BA_056` | `RU.BA` | ood | ❌ `AM.MU` | ❌ `AM.MU` | ❌ `HY.RE` | *no coverage* |
| 57 | `RU.BA_057` | `RU.BA` | ood | ❌ `AM.MU` | ❌ `AM.MU` | ❌ `LE.LU` | *no coverage* |

---

## Legend

- **✅ `PRED`** — prediction returned and was correct
- **❌ `PRED`** — prediction returned but was wrong
- ***no coverage*** — method could not produce a prediction (e.g., tree got stuck, LLM timed out)

## Key Observations

1. **CNN (52.6%)** is the strongest single method. It correctly identifies easy species (HY.PS, AM.VI, CR.CO, SP.CR) but struggles with:
   - OOD species (GY.ES, GA.MA, PL.OS, CA.TU, HY.RE, CAL.GI, RU.IN, RU.BA) — as expected since they were never seen during training
   - Some confusing pairs (CA.CI misclassified as SP.CR or BO.ED)
   - Edge cases (CO.CO_031 predicted as LA.VO)

2. **Tree (36.4%, 38.6% coverage)** gets stuck on species without oracle answers. When it reaches a conclusion, it is often correct for species with oracle data (AM.VI, CR.CO, HY.PS).

3. **Trait DB (5.3%)** performs poorly on raw image inference because it relies on simplified visual traits that don't capture enough discriminative information.

4. **Unified (0%)** produced no usable predictions because every LLM call failed/timed out. This is the most critical issue to fix before the thesis benchmark is complete.

## Next Steps

- [ ] **Fix Unified pipeline** — debug why Ollama calls failed (timeout? model not loaded? memory?)
- [ ] **Add missing oracle answers** for AG.AU, AM.MU, FO.BE, HY.PS, LA.HE, RA.PA to improve tree coverage
- [ ] **Retrain CNN** after dataset cleaning (duplicates removed) — current accuracy may improve
- [ ] **Rerun full benchmark** once unified is working to get agreement statistics
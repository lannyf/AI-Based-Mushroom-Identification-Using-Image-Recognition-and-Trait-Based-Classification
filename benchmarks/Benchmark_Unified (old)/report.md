# Comparative Benchmark Results

## 1. System A — Standalone Methods

*Each method operates independently without access to the others.*

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| cnn | 52.6% | 100.0% | 119.6 |
| tree | 51.7% | 50.9% | 1745.3 |
| db | 5.3% | 100.0% | 5.4 |
| llm | 15.8% | 100.0% | 6749.1 |

## 2. System B — Unified LLM Synthesis

*The LLM aggregates signals from all System A subsystems into a single prediction.*

| Method | Accuracy | Coverage | Mean Time (ms) |
|--------|----------|----------|----------------|
| unified | 42.6% | 94.7% | 29261.0 |

## 3. Raw Accuracy Difference: System B vs System A

*Simple accuracy difference between Unified and each standalone method.*

| Comparison | Unified Acc | Standalone Acc | Difference |
|------------|-------------|----------------|------------|
| unified vs cnn | 42.6% | 52.6% | -10.0% |
| unified vs tree | 42.6% | 51.7% | -9.1% |
| unified vs db | 42.6% | 5.3% | +37.3% |
| unified vs llm | 42.6% | 15.8% | +26.8% |

## 4. Confusing-Pair Breakdown

| Pair | N | CNN | Tree | DB | LLM | Unified | Agr |
|------|---|-----|------|----|-----|---------|-----|
| AM.MU-NONE | 3 | 100% | 33% | 0% | 100% | 100% | 100% |
| AM.VI-NONE | 3 | 100% | 100% | 0% | 0% | 67% | 100% |
| BO.BA-BO.ED | 6 | 83% | 33% | 0% | 33% | 50% | 50% |
| CA.CI-HY.PS | 6 | 33% | 17% | 0% | 17% | 50% | 50% |
| CR.CO-NONE | 2 | 100% | 100% | 100% | 0% | 100% | 100% |
| NONE-SP.CR | 2 | 100% | 100% | 0% | 0% | 100% | 100% |

## 5. Cases Where System B Outperformed All System A Methods

| Specimen | GT | CNN | Tree | DB | LLM | Unified | Reasoning |
|----------|----|-----|------|----|-----|---------|-----------|
| BO.ED_013 | BO.ED | BO.BA | N/A | LE.LU | BO.PI | BO.ED | Strong agreement between the CNN prediction, tree traversal … |
| CA.CI_016 | CA.CI | SP.CR | N/A | AR.ME | MA.PR | CA.CI | The CNN prediction is the strongest signal, but the database… |
| RA.BO_025 | RA.BO | SP.CR | N/A | LE.LU | Coral Mushroom (Clavariaceae) | RA.BO | The database comparison identifies Druvfingersvamp as a pote… |
| HY.RE_049 | HY.RE | CA.CI | CA.AU | LA.HE | Panaeolus foenisporus | HY.RE | The CNN and database both point towards *Hedgehog Mushroom*,… |

## 6. Cases Where System B Was Wrong But a System A Method Was Right

| Specimen | GT | CNN | Tree | DB | LLM | Unified | Notes |
|----------|----|-----|------|----|-----|---------|-------|
| AG.AU_001 | AG.AU | AG.AU | N/A | TY.FE | AM.MU | LA.HE | paired above_idx=2 below_idx=3; ood;cap;gills;stem; source=Svampeatlas/GBIF; occurrence=5898623767; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| AG.AU_002 | AG.AU | AG.AU | CR.CO | CR.CO | Shiitake Mushroom (Lentinula edodes) | LA.VO | paired above_idx=1 below_idx=3; ood;cap;gills;stem; source=Svampeatlas/GBIF; occurrence=5898627776; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| BO.BA_011 | BO.BA | BO.BA | CA.AU | LA.HE | Graygill (Calvatia utriculosa) | AG.AU | paired above_idx=2 below_idx=1; bolete;pores;brown_cap; source=Svampeatlas/GBIF; occurrence=5898629025; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| BO.BA_012 | BO.BA | BO.BA | N/A | LE.LU | AG.CA | LE.VE | paired above_idx=1 below_idx=2; bolete;pores;brown_cap; source=Svampeatlas/GBIF; occurrence=5898633268; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| BO.ED_015 | BO.ED | BO.ED | N/A | CR.CO | BO.ED | CR.CO | paired above_idx=1 below_idx=2; bolete;pores;edible; source=Svampeatlas/GBIF; occurrence=5898640047; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| CA.CI_017 | CA.CI | BO.ED | N/A | HY.RE | CA.CI | HY.PS | paired above_idx=1 below_idx=2; chanterelle;folds;yellow; source=Svampeatlas/GBIF; occurrence=5898649134; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| RA.PA_030 | RA.PA | RA.PA | N/A | PL.OS | Coral Fungus (Ramaria flava) | RA.BO | paired above_idx=3 below_idx=2; coral;branched;toxic;confusing_pair; source=Svampeatlas/GBIF; occurrence=4465870669; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| CO.CO_032 | CO.CO | CO.CO | N/A | LE.LU | AM.PH | BO.BA | paired above_idx=1 below_idx=3; inkcap;gills;edge_case_age; source=Svampeatlas/GBIF; occurrence=5898651767; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view; orientation_corrected: swapped above/below after visual audit |
| FO.BE_035 | FO.BE | FO.BE | N/A | CR.CO | CA.CA | Birch Polypore | paired above_idx=2 below_idx=1; polypore;bracket;wood; source=Svampeatlas/GBIF; occurrence=5106480117; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| FO.BE_036 | FO.BE | FO.BE | N/A | LA.HE | Birch Bolete (Leccinum scabrum) | AL.OV | paired above_idx=1 below_idx=2; polypore;bracket;wood; source=Svampeatlas/GBIF; occurrence=6161868580; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| LY.PE_038 | LY.PE | LY.PE | N/A | HY.RE | Panaeolus fibrillosus | Jätteröksvamp | paired above_idx=2 below_idx=1; puffball;round;spines; source=Svampeatlas/GBIF; occurrence=5898626813; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| PL.OS_046 | PL.OS | LA.HE | CR.CO | HY.CA | PL.OS | LA.VO | paired above_idx=1 below_idx=2; oyster;shelf;gills;wood; source=Svampeatlas/GBIF; occurrence=6161874219; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| CA.TU_047 | CA.TU | LA.HE | CA.TU | CR.CO | Red-capped Gullinella | LA.HE | paired above_idx=1 below_idx=2; funnel_chanterelle;folds;hollow_stem; source=Svampeatlas/GBIF; occurrence=5898631275; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |
| CA.TU_048 | CA.TU | LA.HE | CA.TU | LE.LU | Jack O'Lantern Mushroom (Omphalotus olivascens) | CA.CI | paired above_idx=1 below_idx=2; funnel_chanterelle;folds;hollow_stem; source=Svampeatlas/GBIF; occurrence=5898654001; above=cap/top/whole view; below=underside/gills/pores/folds/stem/base view |

## 7. Agreement Statistics

| Agreement Level | Count | % | Avg System B Accuracy |
|-----------------|-------|---|-----------------------|
| agree | 2 | 3.5% | 100.0% |
| partial | 35 | 61.4% | 48.6% |
| disagree | 20 | 35.1% | 20.0% |
| inconclusive | 0 | 0.0% | 0.0% |

## 8. Accuracy by Scenario

| Scenario | N | CNN | Tree | DB | LLM | Unified |
|----------|---|-----|------|----|-----|---------|
| confusing | 22 | 77% | 85% | 9% | 27% | 71% |
| ood | 17 | 12% | 30% | 6% | 12% | 12% |
| easy | 5 | 80% | 0% | 0% | 0% | 40% |
| coral | 6 | 67% | 0% | 0% | 0% | 60% |
| puffball | 5 | 40% | 33% | 0% | 20% | 20% |
| edge_case | 2 | 50% | 0% | 0% | 0% | 0% |

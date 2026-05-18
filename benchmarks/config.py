"""Central configuration for the benchmark suite.

All file paths and name-mapping tables live here so they can be
imported by any benchmark module without hard-coding paths.
"""

from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIES_CSV = PROJECT_ROOT / "data" / "raw" / "species.csv"
KEY_XML = PROJECT_ROOT / "data" / "raw" / "key.xml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks"

# YOLO segmentation weights — fine-tuned 4-class model (Cap, Coral, Stem, Underside)
YOLO_WEIGHTS = PROJECT_ROOT / "data" / "Yolov8" / "best.pt"

# -----------------------------------------------------------------------------
# Folder name → species_id mapping
# -----------------------------------------------------------------------------
# Training images are organised in sub-folders using Swedish common names
# with Latin scientific names in parentheses.
FOLDER_TO_SPECIES_ID: Dict[str, str] = {
    "Kungschampinjon (Agaricus augustus)": "AG.AU",
    "Flugsvamp (Amanita muscaria)": "AM.MU",
    "Änglsvamp (Amanita virosa)": "AM.VI",
    "Brunsopp (Boletus badius)": "BO.BA",
    "Karljohan (Boletus edulis)": "BO.ED",
    "Kantarell (Cantharellus cibarius)": "CA.CI",
    "Spindelskivling (Coprinellus comatus)": "CO.CO",
    "Svart trumpetsvamp (Craterellus cornucopioides)": "CR.CO",
    "Björkticka (Fomitopsis betulina)": "FO.BE",
    "Falsk kantarell (Hygrophoropsis aurantiaca)": "HY.PS",
    "Lakritsriska (Lactarius helvus)": "LA.HE",
    "Mandelriska (Lactarius volemus)": "LA.VO",
    "Rökslöjpa (Lycoperdon perlatum)": "LY.PE",
    "Druvfingersvamp (Ramaria botrytis)": "RA.BO",
    "Blek fingersvamp (Ramaria pallida)": "RA.PA",
    "Blomkålssvamp (Sparassis crispa)": "SP.CR",
}

# Reverse lookup for convenience
SPECIES_ID_TO_FOLDER: Dict[str, str] = {
    v: k for k, v in FOLDER_TO_SPECIES_ID.items()
}

# -----------------------------------------------------------------------------
# CNN output name → species_id mapping
# -----------------------------------------------------------------------------
# The CNN emits English display names. This table bridges those names to the
# canonical species_id used everywhere else in the project.
CNN_NAME_TO_SPECIES_ID: Dict[str, str] = {
    "Fly Agaric": "AM.MU",
    "Chanterelle": "CA.CI",
    "False Chanterelle": "HY.PS",
    "Porcini": "BO.ED",
    "Bay Bolete": "BO.BA",
    "Amanita virosa": "AM.VI",
    "Black Trumpet": "CR.CO",
    "King Agaricus": "AG.AU",
    "Shaggy Inkcap": "CO.CO",
    "Birch Polypore": "FO.BE",
    "Lakrits Milkcap": "LA.HE",
    "Mandel Milkcap": "LA.VO",
    "Puffball": "LY.PE",
    "Clustered Coral": "RA.BO",
    "Pale Coral Fungus": "RA.PA",
    "Cauliflower Mushroom": "SP.CR",
}

# -----------------------------------------------------------------------------
# In-distribution species
# -----------------------------------------------------------------------------
# These are the 16 species used by the current CNN training set. Benchmark
# species outside this list are out-of-distribution (OOD) for CNN analysis.
IN_DISTRIBUTION_SPECIES: List[str] = [
    "AG.AU", "AM.MU", "AM.VI", "BO.BA", "BO.ED", "CA.CI", "CO.CO",
    "CR.CO", "FO.BE", "HY.PS", "LA.HE", "LA.VO", "LY.PE", "RA.BO",
    "RA.PA", "SP.CR",
]

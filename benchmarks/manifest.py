"""Dataset manifest loader for comparative paired-specimen benchmarks.

The manifest is a CSV file that lists every evaluation specimen, its photo
paths, ground-truth species, scenario category, and metadata.  This decouples
the evaluation dataset organisation from the benchmark code — you can rearrange
folders or add new specimens simply by editing the manifest.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class BenchmarkSpecimen:
    """A single mushroom evaluation case with one or two photos.

    Attributes:
        specimen_id: Unique identifier (e.g. "BO.ED_001").
        above_path: Path to the above-view photo (cap + stem), or None.
        below_path: Path to the below-view photo (underside + stem), or None.
        species_id: Canonical species identifier from species.csv.
        scenario: Evaluation category — ``easy``, ``confusing``, ``puffball``,
            ``coral``, ``ood``, or ``edge_case``.
        subset: ``id`` (in-distribution) or ``ood`` (out-of-distribution).
        confusing_pair_with: For confusing specimens, the species_id of the
            look-alike partner (e.g. ``HY.PS`` when this specimen is ``CA.CI``).
        notes: Free-text annotation for the thesis (lighting, quality, etc.).
    """

    specimen_id: str
    above_path: Optional[Path]
    below_path: Optional[Path]
    species_id: str
    scenario: str
    subset: str
    confusing_pair_with: Optional[str]
    notes: str

    def load_above_bytes(self) -> Optional[bytes]:
        if self.above_path and self.above_path.exists():
            return self.above_path.read_bytes()
        return None

    def load_below_bytes(self) -> Optional[bytes]:
        if self.below_path and self.below_path.exists():
            return self.below_path.read_bytes()
        return None

    def has_pair(self) -> bool:
        return self.above_path is not None and self.below_path is not None


class ManifestDataset:
    """Load and filter a benchmark manifest CSV."""

    SCENARIOS = {"easy", "confusing", "puffball", "coral", "ood", "edge_case"}
    SUBSETS = {"id", "ood"}

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.specimens: List[BenchmarkSpecimen] = []
        self._load()

    def _load(self) -> None:
        with open(self.manifest_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                above = self._resolve_path((row.get("above_image_path") or "").strip())
                below = self._resolve_path((row.get("below_image_path") or "").strip())
                cpw = (row.get("confusing_pair_with") or "").strip() or None

                sid = (row.get("specimen_id") or "").strip()
                if not sid:
                    continue
                self.specimens.append(
                    BenchmarkSpecimen(
                        specimen_id=sid,
                        above_path=above,
                        below_path=below,
                        species_id=(row.get("species_id") or "").strip(),
                        scenario=(row.get("scenario") or "easy").strip().lower(),
                        subset=(row.get("subset") or "id").strip().lower(),
                        confusing_pair_with=cpw,
                        notes=(row.get("notes") or "").strip(),
                    )
                )

    def _resolve_path(self, raw: str) -> Optional[Path]:
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = self.manifest_path.parent / p
        return p

    def __iter__(self) -> Iterator[BenchmarkSpecimen]:
        yield from self.specimens

    def __len__(self) -> int:
        return len(self.specimens)

    def by_scenario(self, scenario: str) -> List[BenchmarkSpecimen]:
        return [s for s in self.specimens if s.scenario == scenario]

    def by_subset(self, subset: str) -> List[BenchmarkSpecimen]:
        return [s for s in self.specimens if s.subset == subset]

    def paired_specimens(self) -> List[BenchmarkSpecimen]:
        return [s for s in self.specimens if s.has_pair()]

    def confusing_pairs(self) -> List[BenchmarkSpecimen]:
        return [s for s in self.specimens if s.scenario == "confusing"]

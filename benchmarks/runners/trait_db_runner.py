"""Benchmark runner for the trait-database comparator.

Scores every species in the database against the provided traits
(extracted or oracle) and returns the top-ranked species.
"""

import time

from benchmarks.config import DATA_RAW_DIR
from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from benchmarks.runners._trait_helper import get_merged_extracted_traits


class TraitDBRunner(BenchmarkRunner):
    """Wrapper around ``models.trait_database_comparator.TraitDatabaseComparator``.

    Supports two trait sources:
      - "extracted" (a1_db): merged computer-vision traits
      - "oracle"    (a2_db): ground-truth traits from SpeciesTraitOracle
    """

    name = "trait_db"

    def __init__(self, trait_source: str = "extracted", oracle_trait_provider=None):
        if trait_source not in ("extracted", "oracle"):
            raise ValueError(f"TraitDBRunner trait_source must be 'extracted' or 'oracle', got {trait_source}")

        from models.trait_database_comparator import TraitDatabaseComparator

        self.trait_source = trait_source
        self.oracle_trait_provider = oracle_trait_provider
        self.comparator = TraitDatabaseComparator(str(DATA_RAW_DIR))

    def predict(self, specimen) -> RunnerResult:
        """Rank all species by trait-match score and return the top hit.

        Args:
            specimen: BenchmarkSpecimen with ``above_path``, ``below_path``,
                and ``species_id``.

        Returns:
            ``RunnerResult`` where ``predictions`` contains the top-ranked
            species (highest database match score).
        """
        t0 = time.perf_counter()

        if self.trait_source == "extracted":
            visible_traits = get_merged_extracted_traits(specimen)
        elif self.trait_source == "oracle" and self.oracle_trait_provider is not None:
            visible_traits = self.oracle_trait_provider.get_extractor_output(
                specimen.species_id, case="classical"
            )
        else:
            visible_traits = {}

        ranked = self.comparator.rank_all_species(visible_traits)
        elapsed = (time.perf_counter() - t0) * 1000

        # Return top prediction only (the species with highest match score)
        if ranked:
            top = ranked[0]
            predictions = [(top["species_id"], top["score"])]
        else:
            predictions = []

        return RunnerResult(
            method_name=f"trait_db_{self.trait_source}",
            predictions=predictions,
            coverage=True,
            inference_time_ms=elapsed,
            metadata={
                "visible_traits": visible_traits,
                "trait_source": self.trait_source,
                "all_ranked": ranked,
            },
        )

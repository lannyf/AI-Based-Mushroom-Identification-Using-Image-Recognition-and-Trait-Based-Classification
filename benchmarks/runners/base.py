from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple






@dataclass
class RunnerResult:
    """
      Universal output envelope for every identification method in the benchmark.

      All comparative methods — cnn, tree, db, and unified — return this same
      type so the benchmark engine can treat them uniformly. The ranked
      `predictions` list is always sorted best-first; `coverage` tracks whether
      the method abstained; `metadata` holds method-specific diagnostics without
      breaking the common interface.
    """
    method_name: str
    predictions: List[Tuple[str, float]]
    coverage: bool = True
    inference_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_species(self) -> str:
        return self.predictions[0][0] if self.predictions else "unknown"

    @property
    def top_confidence(self) -> float:
        return self.predictions[0][1] if self.predictions else 0.0


class BenchmarkRunner(ABC):
    name: str = ""

    @abstractmethod
    def predict(self, sample) -> RunnerResult:
        pass

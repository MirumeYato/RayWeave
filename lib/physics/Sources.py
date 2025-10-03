from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from lib.data import ParticleState

class Source(ABC):
    """Initializes (or injects) particles into the Grid at t0."""

    @abstractmethod
    def emit(self, t0: float) -> Sequence[ParticleState]:
        """Return the particles to insert at time t0."""

# -----------------------------
# List of Sources:
# -----------------------------

class TwoPointSource(Source):
    """Emit exactly two particles with given (pos, direction, speed)."""

    def __init__(
        self,
        pos1: Sequence[float], dir1: Sequence[float], 
        pos2: Sequence[float], dir2: Sequence[float], 
        speed1: float = 1.0, speed2: float = 1.0
    ) -> None:
        self.p1 = ParticleState(np.array(pos1, float), np.array(dir1, float), speed1)
        self.p2 = ParticleState(np.array(pos2, float), np.array(dir2, float), speed2)

    def emit(self, t0: float) -> Sequence[ParticleState]:  # noqa: D401
        return [self.p1, self.p2]
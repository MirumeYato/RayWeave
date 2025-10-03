from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
# import numpy as np

from lib.data import ParticleState

class Grid(ABC):
    """Abstract spatial container for particles and fields.

    Concrete implementations may be uniform grids, AMR, octrees, etc.
    Only minimal API is required by Steps & Propagator.
    """

    @abstractmethod
    def add_particles(self, particles: Sequence[ParticleState]) -> List[int]:
        """Insert particles, return their assigned ids."""

    @abstractmethod
    def iter_particles(self) -> Iterable[ParticleState]:
        """Iterate over all particles currently in the grid."""

    @abstractmethod
    def get(self, pid: int) -> ParticleState:
        """Get particle by id."""

    @abstractmethod
    def update(self, pid: int, new_state: ParticleState) -> None:
        """Replace particle state (position, direction, etc.)."""

    @abstractmethod
    def n_particles(self) -> int:
        """Current number of particles in the grid."""

# -----------------------------
# List of Grids:
# -----------------------------

class UniformGrid2D(Grid):
    """Minimal uniform grid that simply stores particles in a dict.

    This is *not* spatially indexed; it's a simple container to satisfy the API.
    Replace with AMR/Octree later without changing Step/Propagator.
    """

    def __init__(self) -> None:
        self._particles: Dict[int, ParticleState] = {}
        self._next_id: int = 0

    def add_particles(self, particles: Sequence[ParticleState]) -> List[int]:
        ids: List[int] = []
        for p in particles:
            pid = self._next_id
            self._next_id += 1
            p.pid = pid
            # store a copy to decouple outside references
            self._particles[pid] = ParticleState(pos=p.pos.copy(), direction=p.direction.copy(), speed=p.speed, weight=p.weight, pid=pid)
            ids.append(pid)
        return ids

    def iter_particles(self) -> Iterable[ParticleState]:
        return list(self._particles.values())

    def get(self, pid: int) -> ParticleState:
        return self._particles[pid]

    def update(self, pid: int, new_state: ParticleState) -> None:
        new_state.pid = pid
        self._particles[pid] = ParticleState(
            pos=new_state.pos.copy(),
            direction=new_state.direction.copy(),
            speed=new_state.speed,
            weight=new_state.weight,
            pid=pid,
        )

    def n_particles(self) -> int:
        return len(self._particles)

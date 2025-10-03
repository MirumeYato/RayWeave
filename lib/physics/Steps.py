from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
# import numpy as np

from lib.physics import Grid
from lib.data import ParticleState

class Step(ABC):
    """A single update operation applied each timestep.

    Examples: straight-line streaming, scattering, absorption, boundary, etc.
    """

    @abstractmethod
    def apply(self, grid: Grid, t: float, dt: float) -> None:
        """Apply the step in-place to particles in the grid."""

# -----------------------------
# List of Steps:
# -----------------------------

class StraightLineStep(Step):
    """Move each particle along its direction by distance = speed * dt."""

    def apply(self, grid: Grid, t: float, dt: float) -> None:  # noqa: D401
        for p in list(grid.iter_particles()):
            new_pos = p.pos + p.direction * (p.speed * dt)
            grid.update(p.pid, ParticleState(pos=new_pos, direction=p.direction, speed=p.speed, weight=p.weight, pid=p.pid))  # type: ignore[arg-type]

from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
# import numpy as np

from lib.physics import Grid, Step, Source
from lib.results import BackupData

class Propagator(ABC):
    """Coordinates the full time-stepping pipeline."""

    @abstractmethod
    def run(
        self,
        grid: Grid,
        sources: Sequence[Source],
        steps: Sequence[Step],
        backup: BackupData,
        t_start: float,
        dt: float,
        n_steps: int,
    ) -> None:
        pass

# -----------------------------
# List of Propagators:
# -----------------------------


class SimplePropagator(Propagator):
    """Minimal propagator that runs each Step sequentially per timestep."""

    def run(
        self,
        grid: Grid,
        sources: Sequence[Source],
        steps: Sequence[Step],
        backup: BackupData,
        t_start: float,
        dt: float,
        n_steps: int,
    ) -> None:
        # 1) Emit
        for src in sources:
            grid.add_particles(src.emit(t_start))
        backup.on_emit(grid, t_start)

        # 2) Time integration
        t = t_start
        for k in range(1, n_steps + 1):
            for step in steps:
                step.apply(grid, t, dt)
            t = t_start + k * dt
            backup.on_step_end(grid, t)
        backup.finalize()
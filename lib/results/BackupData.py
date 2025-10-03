from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from lib.physics import Grid

class BackupData(ABC):
    """Collects and stores data during a run (tracks, detector stats, etc.)."""

    @abstractmethod
    def on_emit(self, grid: Grid, t: float) -> None:
        pass

    @abstractmethod
    def on_step_end(self, grid: Grid, t: float) -> None:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass

# -----------------------------
# List of BackupData classes:
# -----------------------------
    
class TrackBackup(BackupData):
    """Store particle positions at each step for track plotting.

    For simplicity, we pre-allocate after emission based on the number of particles.
    """

    def __init__(self, n_expected_steps: int) -> None:
        self.n_expected_steps = n_expected_steps
        self.t: List[float] = []
        self.tracks: Optional[np.ndarray] = None  # shape (P, T, 2)
        self._step_idx: int = 0

    def on_emit(self, grid: Grid, t: float) -> None:
        P = grid.n_particles()
        T = self.n_expected_steps + 1  # include t0
        self.tracks = np.full((P, T, 2), np.nan, dtype=float)
        # record initial positions
        for p in grid.iter_particles():
            assert p.pid is not None
            self.tracks[p.pid, 0, :] = p.pos  # type: ignore[index]
        self.t = [t]
        self._step_idx = 0

    def on_step_end(self, grid: Grid, t: float) -> None:
        if self.tracks is None:
            raise RuntimeError("Call on_emit before on_step_end")
        self._step_idx += 1
        for p in grid.iter_particles():
            assert p.pid is not None
            self.tracks[p.pid, self._step_idx, :] = p.pos  # type: ignore[index]
        self.t.append(t)

    def finalize(self) -> None:
        pass

    # Convenience accessors
    def get_tracks(self) -> np.ndarray:
        assert self.tracks is not None
        return self.tracks

    def get_times(self) -> List[float]:
        return self.t
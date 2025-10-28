from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
# import numpy as np

# from lib.physics import Grid
from lib.data import FieldState # ParticleState

class Step(ABC):
    """A single update operation applied each timestep.

    Examples: straight-line streaming, scattering, absorption, boundary, etc.
    """
    def setup(self, state: FieldState) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""
        pass

    def teardown(self) -> None:
        """Free big buffers; flush files, etc."""
        pass

    @abstractmethod
    def forward(self, state: FieldState) -> FieldState:
        """Pure transform: no I/O, no prints, no device syncs if possible."""
        ...

    def __call__(self, state: FieldState) -> FieldState:
        return self.forward(state)

# -----------------------------
# List of Steps:
# -----------------------------

class DummyField(Step):
    def __init__(self):
        pass

    def forward(self, state: FieldState) -> FieldState:
        return FieldState(state.field, state.t + state.dt, state.dt, state.meta)

class ShiftField(Step):
    def __init__(self, vx: float, vy: float = 0.0, vz: float = 0.0):
        self.v = (vx, vy, vz)

    def forward(self, state: FieldState) -> FieldState:
        # Example: naive periodic shift along x (replace with your grid_sample kernel)
        # state.field: [Q, Nx, Ny, Nz]
        vx, vy, vz = self.v
        dx = int(round(vx * state.dt))
        f = state.field
        if dx != 0:
            f = torch.roll(f, shifts=dx, dims=1)
        return FieldState(f, state.t + state.dt, state.dt, state.meta)

class Collide(Step):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def forward(self, state: FieldState) -> FieldState:
        # Example in-place-ish safe pattern (clone only if needed)
        f = state.field * (1.0 - self.alpha)
        return FieldState(f, state.t, state.dt, state.meta)

# class StraightLineStep(Step):
#     """Move each particle along its direction by distance = speed * dt."""

#     def apply(self, grid: Grid, t: float, dt: float) -> None:  # noqa: D401
#         for p in list(grid.iter_particles()):
#             new_pos = p.pos + p.direction * (p.speed * dt)
#             grid.update(p.pid, ParticleState(pos=new_pos, direction=p.direction, speed=p.speed, weight=p.weight, pid=p.pid))  # type: ignore[arg-type]

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Any

# from lib.grid.Grid import Grid
import torch

# -----------------------------
# Core Data Structures
# -----------------------------

@dataclass
class FieldState:
    """Container for the system state passed through Steps.

    Attributes
    ----------
    field : torch.Tensor
        The main field tensor (e.g., shape [B, C, Nx, Ny, Nz]). Must be on the
        same device as the engine.
    t : float
        Current simulation time (in user units).
    dt : float
        Current time-step size (may be updated each iteration by the engine
        when variable `dt` is used).
    meta : Dict[str, Any]
        Arbitrary metadata. Keep values small; observers can persist the heavy
        stuff out-of-band.
    """

    field: torch.Tensor
    dt: float
    meta: Dict[str, Any]

    def to(self, device: torch.device | str = "cpu", dtype: torch.dtype | None = None) -> "FieldState":
        dev = torch.device(device)
        if dtype:
            return replace(self, field=self.field.to(dev, dtype=dtype))
        else:
            return replace(self, field=self.field.to(dev))
    
    # @classmethod
    # def from_floats(cls, field: torch.Tensor, t: float, dt: float, meta: dict = None, device=None):
    #     """
    #     >>> state = FieldState.from_floats(field, t=0.0, dt=0.01, meta={"id": 1})
    #     """
    #     device = device or field.device
    #     return cls(
    #         field=field,
    #         t=torch.tensor(t, device=device, dtype=torch.float32),
    #         dt=torch.tensor(dt, device=device, dtype=torch.float32),
    #         meta=meta or {}
    #     )

# @dataclass
# class FieldState:
#     # Always use 4D shape [Q, Nx, Ny, Nz]; set Ny=Nz=1 for 1D, Nz=1 for 2D.
#     field: Grid      # float32 [Q, Nx, Ny, Nz] (contiguous, device-ready)
#     t: float                 # current time
#     dt: float                # step size
#     meta: Dict[str, Any]     # nside, L_max, grid spacing, etc.


# =====================================
# needs refactoring

# @dataclass
# class ParticleState:
#     """State of a single photon/particle in 2D.

#     Attributes
#     ----------
#     pos : np.ndarray
#         (2,) position vector [x, y].
#     direction : np.ndarray
#         (2,) unit vector direction of motion.
#     speed : float
#         Propagation speed (use c=1.0 in demo units or 3e8 for SI).
#     weight : float
#         Optional weight (for future use: attenuation, importance, etc.).
#     pid : Optional[int]
#         Unique id assigned by the Grid when inserted.
#     """
#     pos: np.ndarray
#     direction: np.ndarray
#     speed: float = 1.0
#     weight: float = 1.0
#     pid: Optional[int] = field(default=None)

#     def __post_init__(self) -> None:
#         self.pos = np.asarray(self.pos, dtype=float).reshape(2)
#         d = np.asarray(self.direction, dtype=float).reshape(2)
#         n = np.linalg.norm(d)
#         if n == 0:
#             raise ValueError("direction must be non-zero")
#         self.direction = d / n  # ensure unit vector
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Any


import torch

# -----------------------------
# Core Data Structures
# -----------------------------
Field = torch.Tensor

@dataclass
class FieldState:
    """Container for the system state passed through Steps.

    Attributes
    ----------
    field : torch.Tensor
        The main field tensor (e.g., shape [B, C, Nx, Ny, Nz]). Must be on the
        same device as the engine.
    dt : float
        Current time-step size (may be updated each iteration by the engine
        when variable `dt` is used).
    meta : Dict[str, Any]
        Arbitrary metadata. Keep values small; observers can persist the heavy
        stuff out-of-band.
    """
    field: Field
    dt: float
    meta: Dict[str, Any]

    def to(self, device: torch.device | str = "cpu", dtype: torch.dtype | None = None) -> "FieldState":
        dev = torch.device(device)
        if dtype:
            return replace(self, field=self.field.to(dev, dtype=dtype))
        else:
            return replace(self, field=self.field.to(dev))
    
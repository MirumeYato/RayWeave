from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

# -----------------------------
# Core Data Structures
# -----------------------------

@dataclass
class ParticleState:
    """State of a single photon/particle in 2D.

    Attributes
    ----------
    pos : np.ndarray
        (2,) position vector [x, y].
    direction : np.ndarray
        (2,) unit vector direction of motion.
    speed : float
        Propagation speed (use c=1.0 in demo units or 3e8 for SI).
    weight : float
        Optional weight (for future use: attenuation, importance, etc.).
    pid : Optional[int]
        Unique id assigned by the Grid when inserted.
    """
    pos: np.ndarray
    direction: np.ndarray
    speed: float = 1.0
    weight: float = 1.0
    pid: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float).reshape(2)
        d = np.asarray(self.direction, dtype=float).reshape(2)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("direction must be non-zero")
        self.direction = d / n  # ensure unit vector
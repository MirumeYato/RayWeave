from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# from lib.physics import Grid
from lib.State import FieldState, Field # ParticleState

class Step(ABC):
    """A single split-operator (e.g., drift, collision, source, etc.).

    Subclasses implement `__call__(state)` to return a new `FieldState`.
    `setup/teardown` are lifecycle hooks that receive the **mutable** `state` in
    place and may prepare buffers.

    IMPORTANT: For the compiled fast path to work well, keep `__call__` free of
    Python-side side effects and shape changes.
    """

    name: str = "Step"

    def __init__(self, device = "cpu", verbose = 0):
        # Grid for interpolation
        self.dt = None
        self.verbose = verbose
        self.device = device

    def setup(self, state: FieldState) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""
        self.setup_dt(state)
        pass

    @abstractmethod
    def forward(self, field: Field) -> Field:
        """Pure transform: no I/O, no prints, no device syncs if possible."""
        ...

    def __call__(self, field: Field) -> Field:
        return self.forward(field)

    def teardown(self) -> None:
        """Free big buffers; flush files, etc."""
        pass    
    
    def setup_dt(self, state: FieldState) -> None: 
        """Store delta t (amount of time step in some units) from FieldState"""
        self.dt = float(state.dt)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
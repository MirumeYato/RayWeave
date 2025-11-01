from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# from lib.physics import Grid
from lib.State import FieldState # ParticleState

class Step(ABC):
    """A single split-operator (e.g., drift, collision, source, etc.).

    Subclasses implement `__call__(state)` to return a new `FieldState`.
    `setup/teardown` are lifecycle hooks that receive the **mutable** `state` in
    place and may prepare buffers.

    IMPORTANT: For the compiled fast path to work well, keep `__call__` free of
    Python-side side effects and shape changes.
    """

    name: str = "Step"

    def setup(self, state: FieldState) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""
        pass

    @abstractmethod
    def forward(self, state: FieldState) -> FieldState:
        """Pure transform: no I/O, no prints, no device syncs if possible."""
        ...

    def __call__(self, state: FieldState) -> FieldState:
        return self.forward(state)

    def teardown(self) -> None:
        """Free big buffers; flush files, etc."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
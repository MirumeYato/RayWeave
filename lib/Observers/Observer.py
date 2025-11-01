from abc import ABC

from ..State import FieldState

class Observer(ABC):
    """
    Collects and stores data during a run (tracks, detector stats, etc.).
    Receives callbacks during the run to record sparse diagnostics.
    """
    def on_setup(self, state: FieldState) -> None: pass
    def on_step_end(self, step_idx: int, state: FieldState) -> None: pass
    def on_teardown(self) -> None: pass
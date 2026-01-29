from abc import ABC, abstractmethod

from ..State import FieldState, Field

class Observer(ABC):
    """
    Collects and stores data during a run (tracks, detector stats, etc.).
    Receives callbacks during the run to record sparse diagnostics.
    """
    def __init__(self, every: int = 50):
        self.every = every

    def sync_every(self, chunk_size: int = None) -> None:
        """
        Method for syncronize counter "every" with chunk_size. 
        In chunk mode we can not update state in non multiple to chink_size iteration.
        
        :param chunk_size: Number of iterations, that are compiled without observers.
        :type chunk_size: int
        """
        if not chunk_size: 
            if self.every % chunk_size and self.every > chunk_size: 
                self.every = self.every - self.every % chunk_size
        pass

    def on_setup(self, state: FieldState) -> None: pass
    @abstractmethod
    def on_step_end(self, step_idx: int, field: Field) -> None: pass
    def on_teardown(self) -> None: pass
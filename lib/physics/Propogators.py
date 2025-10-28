from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
# import numpy as np
from tqdm import trange

from lib.physics import Grid, Step, Source
from lib.results import BackupData

class Propagator(ABC):
    """Coordinates the full time-stepping pipeline."""

    def set_Source(self, Source: Source):
        self.field = Source.get_field() # stores some massive like np.array or torch.tensor
        self.params = Source.params # [nside:int , shape: Tuple[int], L_max: int]
        # self.media_params = Media.params # maybe in future [mu_t, mu_s]

    def run(self):
        self.Setps = self.pipeline()
        # do all needed preliminary actions
        for Step in self.Setps:
            print(f"[DEBUG] Starting procedure of all needed preliminary actions. Current step is {Step.__name__}")
            Step.initialize(self)

        # Start main loop
        print(f"[DEBUG] Run main loop")
        for time_step in trange(self.n_time_steps):
            for Step in self.Setps:
                self.field = Step.update(self.field, time_step)
        
        # do all needed final actions
        for Step in self.Setps:
            print(f"[DEBUG] Starting procedure of all needed final actions. Current step is {Step.__name__}")
            Step.close()

    @abstractmethod
    def pipeline(self) -> None:
        # for example
        # >>> self.field = Shift_field(self.field) # some changing of field
        # >>> self.field = Rotate_field(self.field) # some changing of field
        # >>> Save2Dplot(self.field, properties_of_plot) # saves picture of field or its projection 
        # >>> Store_field_in_certain_volume(self.field, volume) # saves sum(field(volume)) per time as np.array in .npz file
        # >>> self.print_step(metrix = 'total_energy')

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
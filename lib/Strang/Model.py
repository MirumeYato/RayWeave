from __future__ import annotations

"""
Model/Engine split for Strang-type time integration with pluggable Steps and
Observers. This is **not** a neural-network API; `Model` here is a convenience
wrapper around a numerical engine.

Key ideas
---------
- `Engine` defines `run(init_state) -> FieldState`.
- `StrangEngine` implements a second-order Strang splitting loop over `steps`.
- `Model` subclasses `StrangEngine` but exposes a familiar, high-level API:
    * Build via `SequentialModel([...])` (like nn.Sequential)
    * OR subclass `Model` and implement `__init__` + `forward(state)`
- `Observer`s capture only important data during the run (hits, projections,
  accumulators), to avoid saving every time-step to disk/memory.
- `dt` is constant for now, but the engine already supports a future
  `dt_provider(i, state)` for variable time-step logic.

This file is intentionally self-contained (FieldState, Step, Observer, Engine,
StrangEngine, Model, SequentialModel, and a few example Steps/Observers).
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import torch

# import numpy as np
from tqdm import trange

from .Engine import StrangEngine as SEngine
from lib import Step, Source, Observer # Grid
from lib.State import FieldState

# -----------------------------------------------------------------------------
# High-level Model API
# -----------------------------------------------------------------------------

class Model(SEngine):
    """User-facing numerical model (not a NN!).

    Two construction modes
    ----------------------
    1) **Sequential**: Pass a list of Step instances as `layers/steps`.
    2) **Subclassing**: Define your own `__init__` (configure layers) and
       implement `forward(state)` to describe a single time-step.

    Notes
    -----
    - `forward(state)` must be *pure* and avoid shape changes to benefit from
      compilation and CUDA graph capture.
    - Access the current time step as `state.dt`.
    - Use Observers for outputs; avoid storing full trajectories.
    """

    def __init__(
        self,
        steps: Optional[List[Step]] = None,
        *,
        num_time_steps: int,
        dt: float,
        observers: Optional[List[Observer]] = None,
        device: str = "cuda",
        compile_fused: bool = False,
        use_cuda_graph: bool = False,
        dt_provider: Optional[Callable[[int, FieldState], float]] = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            steps=steps or [],
            num_time_steps=num_time_steps,
            dt=dt,
            observers=observers,
            device=device,
            compile_fused=compile_fused,
            use_cuda_graph=use_cuda_graph,
            dt_provider=dt_provider,
        )
        self.layers = self.steps  # alias familiar to NN users
        self.name = name or self.__class__.__name__

    # --- user-extensible per-time-step logic ----------------------------------
    def forward(self, state: FieldState) -> FieldState:
        """Default: sequentially apply `self.layers`.

        Subclasses may override to implement custom compositions for a single
        time step (e.g., A(dt/2) -> B(dt) -> A(dt/2)).
        """
        for layer in self.layers:
            state = layer(state)
        return state

    # Ensure the engine compiles the *Model* forward
    def _fused_step(self, state: FieldState) -> FieldState:  # type: ignore[override]
        return self.forward(state)

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(s) for s in self.layers)
        return f"{self.name}(\n  {inner}\n)"


class Sequential(Model):
    """Convenience wrapper: `Sequential(layers, **engine_kwargs)`."""

    def __init__(
        self,
        layers: Sequence[Step],
        *,
        num_time_steps: int,
        dt: float,
        observers: Optional[List[Observer]] = None,
        device: str = "cuda",
        compile_fused: bool = False,
        use_cuda_graph: bool = False,
        dt_provider: Optional[Callable[[int, FieldState], float]] = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            steps=list(layers),
            num_time_steps=num_time_steps,
            dt=dt,
            observers=observers,
            device=device,
            compile_fused=compile_fused,
            use_cuda_graph=use_cuda_graph,
            dt_provider=dt_provider,
            name=name or "Sequential",
        )

# -----------------------------------------------------------------------------
# Usage examples (as documentation)
# -----------------------------------------------------------------------------

EXAMPLE_DOC = r"""
USAGE
=====

1) Build via Sequential
-----------------------

>>> layers = [DriftStep(vx=1), DecayStep(lambd=0.05)]
>>> obs = [EnergySumObserver()]
>>> model = Sequential(layers, num_time_steps=100, dt=0.01, observers=obs, device="cuda")
>>> final_state = model.run(FieldState(field=torch.ones(1,1,16,16,16), t=0.0, dt=0.01, meta={}))
>>> print(obs[0].values[:3])  # accumulated energy per step

2) Subclass Model and implement forward
---------------------------------------

>>> class ABA_Strang(Model):
...     def __init__(self, A: Step, B: Step, *, num_time_steps:int, dt:float, **kw):
...         super().__init__(steps=[], num_time_steps=num_time_steps, dt=dt, **kw)
...         self.A, self.B = A, B
...
...     def forward(self, state: FieldState) -> FieldState:
...         # symmetric composition per step (A half, B full, A half)
...         # Here we assume Steps read `state.dt` and internally use a factor.
...         state = self.A(state)  # interpreted as A(dt/2) if A uses internal factor
...         state = self.B(state)  # B(dt)
...         state = self.A(state)  # A(dt/2)
...         return state

>>> model = ABA_Strang(DriftStep(1), DecayStep(0.05), num_time_steps=100, dt=0.01)
>>> final_state = model.run(FieldState(field=torch.ones(1,1,16,16,16), t=0.0, dt=0.01, meta={}))

Notes
-----
- For variable dt in the future, pass `dt_provider=lambda i, s: ...` when
  constructing the model/engine. During the run, the engine updates `state.dt` at
  the beginning of each iteration.
- Keep `forward` and `Step.__call__` side-effect free and shape-stable if you
  want `torch.compile`+CUDA Graphs to shine.
"""
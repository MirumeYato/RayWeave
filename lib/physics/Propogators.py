from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

import torch
import torch._dynamo
torch._dynamo.config.verbose = True
# torch._dynamo.config.log_level = logging.DEBUG

# import numpy as np
from tqdm import trange

from lib.physics import Step, Source # Grid
from lib.results import Observer
from lib.data import FieldState

class Propagator:
    def __init__(self,
                 steps: List[Step],
                 n_steps: int,
                 dt: float,
                 observers: Optional[List[Observer]] = None,
                 device: str = "cuda",
                 compile_fused: bool = True,
                 use_cuda_graph: bool = False):
        self.steps = steps
        self.n_steps = n_steps
        self.dt = dt
        self.device = torch.device(device)
        self.observers = observers or []
        self.compile_fused = compile_fused
        self.use_cuda_graph = use_cuda_graph
        self._compiled_step: Optional[Callable] = None
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_state_ref: Optional[FieldState] = None  # for graph replay

    def _fused_step(self, state: FieldState) -> FieldState:
        # Apply all Steps; keep pure & tensor-only to help compilation.
        for s in self.steps:
            state = s(state)
        return state

    def _maybe_compile(self):
        fn = self._fused_step
        if self.compile_fused:
            # fullgraph=True helps avoid graph breaks when shapes are static
            fn = torch.compile(fn, fullgraph=True)  # PyTorch 2.x compiler :contentReference[oaicite:8]{index=8}
        self._compiled_step = fn

    def run(self, init_state: FieldState) -> FieldState:
        # Move tensors & set dt
        f = init_state.field.to(self.device, non_blocking=True)
        state = FieldState(f.contiguous(), init_state.t, self.dt, dict(init_state.meta))

        # Setup hooks & steps
        for s in self.steps: s.setup(state)
        for ob in self.observers: ob.on_setup(state)

        self._maybe_compile()

        if self.use_cuda_graph and self.device.type == "cuda":
            print("[DEBUG]: Starting cuda pipeline")
            # Warm-up allocations and capture one iteration
            static_state = FieldState(state.field, state.t, state.dt, state.meta)
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            torch.cuda.set_stream(stream)
            # graph capture requires static memory addresses; avoid new tensors inside step
            with torch.cuda.graph(g):
                out_state = self._compiled_step(static_state)
                # store result into preallocated buffer to keep addresses fixed
                static_state.field.copy_(out_state.field)
                static_state.t = out_state.t
            torch.cuda.set_stream(torch.cuda.default_stream())
            self._cuda_graph = g
            self._static_state_ref = static_state

            for i in range(self.n_steps):
                g.replay()  # super-low overhead replay :contentReference[oaicite:9]{index=9}
                for ob in self.observers:
                    ob.on_step_end(i, self._static_state_ref)
            state = self._static_state_ref
        else:
            print("[DEBUG]: Starting CPU pipeline")
            step_fn = self._compiled_step
            for i in range(self.n_steps):
                state = step_fn(state)
                for ob in self.observers:
                    ob.on_step_end(i, state)

        # Teardown
        for ob in self.observers: ob.on_teardown()
        for s in self.steps: s.teardown()
        return state
    
# -----------------------------
# List of Propagators:
# -----------------------------

from lib.physics.Steps import ShiftField, Collide, DummyField
from lib.results.Observer import EnergyLogger

def make_propagator(dt: float, n_steps: int, device) -> Propagator:
    steps: List[Step] = [
        DummyField(),
        # ShiftField(vx=+1.0),
        # Collide(alpha=0.01),
        # ... add RotateField, Boundary, SourceInject, etc.
    ]
    observers = [EnergyLogger(every=1)]
    return Propagator(steps, n_steps, dt, observers,
                      device=device, compile_fused=True, use_cuda_graph=True)

# class SimplePropagator(Propagator):
#     """Minimal propagator that runs each Step sequentially per timestep."""

#     def run(
#         self,
#         grid: Grid,
#         sources: Sequence[Source],
#         steps: Sequence[Step],
#         backup: BackupData,
#         t_start: float,
#         dt: float,
#         n_steps: int,
#     ) -> None:
#         # 1) Emit
#         for src in sources:
#             grid.add_particles(src.emit(t_start))
#         backup.on_emit(grid, t_start)

#         # 2) Time integration
#         t = t_start
#         for k in range(1, n_steps + 1):
#             for step in steps:
#                 step.apply(grid, t, dt)
#             t = t_start + k * dt
#             backup.on_step_end(grid, t)
#         backup.finalize()
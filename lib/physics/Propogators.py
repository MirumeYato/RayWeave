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
        device = torch.device(self.device)
        assert device.type == ("cuda" if self.use_cuda_graph else device.type)
        with torch.inference_mode():
            # -- move & normalize --
            f = init_state.field.to(device, non_blocking=True).contiguous()
            state = FieldState(f, init_state.t, self.dt, dict(init_state.meta))

            # Setup hooks & steps
            for s in self.steps: s.setup(state)
            for ob in self.observers: ob.on_setup(state)

            # -- compile fused step once --
            self._maybe_compile()                     # sets self._compiled_step
            step_fn = self._compiled_step

            # ---- WARMUP OUTSIDE CAPTURE ----
            # warmup runs allow dynamo/inductor to finish compilation and allocate kernels
            for _ in range(2):
                tmp = step_fn(state)
                # write back into the original buffers to keep addresses stable
                state.field.copy_(tmp.field)
                state.t = tmp.t

            if self.use_cuda_graph and device.type == "cuda":
                print("[DEBUG]: Run cuda propogator")
                g = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()

                # capture one iteration, writing result back into same storage
                with torch.cuda.graph(g):
                    out_state = step_fn(state)
                    state.field.copy_(out_state.field)
                    state.t = out_state.t

                # main loop (fast replay)
                for i in range(self.n_steps):
                    g.replay()
                    for ob in self.observers: ob.on_step_end(i, state)
            else:
                print("[DEBUG]: Run simple propogator")
                for i in range(self.n_steps):
                    out_state = step_fn(state)
                    state.field.copy_(out_state.field)
                    state.t = out_state.t
                    for ob in self.observers: ob.on_step_end(i, state)

            # -- teardown --
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
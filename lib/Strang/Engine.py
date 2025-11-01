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

from lib import Step, Observer
from lib.State import FieldState

class Engine(ABC):
    """Base running pipeline class for solving your equation.

    Must implement `run(init_state) -> FieldState`.
    `dt` may be not constant in future; see `dt_provider` in `StrangEngine`.
    """
    @abstractmethod
    def run(self, init_state: FieldState) -> FieldState:
        ...

class StrangEngine(Engine):
    """
    Second-order Strang splitting time integrator over `steps`.

    If `compile_fused` is True, the per-step function is compiled with
    `torch.compile(fullgraph=True)` (PyTorch 2.x). If `use_cuda_graph` is True
    and device is CUDA, a CUDA Graph replay loop is used for the inner time
    stepping (fast when shapes & allocations are stable).

    Parameters
    ----------
    steps : List[Step]
    Ordered list of split operators applied each time step.
    num_time_steps : int
    Number of time steps to perform.
    dt : float
    Constant time step for now. See `dt_provider` for a future extension.
    observers : Optional[List[Observer]]
    Callbacks to record sparse diagnostics.
    device : str
    Device string for the main run (e.g., "cuda" or "cpu").
    compile_fused : bool
    Use `torch.compile` on the fused step.
    use_cuda_graph : bool
    Use CUDA graphs (only if device is CUDA and kernels are capture-safe).
    dt_provider : Optional[Callable[[int, FieldState], float]]
    (Future) function that returns the `dt` to use at step `i` given the
    current state. If provided, it overrides the constant `dt` during run.

    Example of usage:
    --------
    >>> SEngine = StrangEngine(steps, num_time_steps, dt, observers)
    >>> final_state = SEngine.run(init_state) # also do jobs of observers like saveing some data or debug prints
    """
    def __init__(self,
                steps: List[Step],
                num_time_steps: int,
                dt: float,
                observers: Optional[List[Observer]] = None,
                device: str = "cuda",
                compile_fused: bool = True,
                use_cuda_graph: bool = False,
                dt_provider: Optional[Callable[[int, FieldState], float]] = None,
                )-> None:
        self.steps = list(steps)
        self.num_time_steps = int(num_time_steps)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.observers = list(observers or [])
        self.compile_fused = bool(compile_fused)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.dt_provider = dt_provider

        self._compiled_step: Optional[Callable[[FieldState], FieldState]] = None
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None  # for graph replay

    # ---- overridable fused step -------------------------------------------------
    def _fused_step(self, state: FieldState) -> FieldState:
        """Default fused step: apply all steps sequentially.

        `Model` overrides this to call `forward`.
        """
        # Apply all Steps; keep pure & tensor-only to help compilation.
        for s in self.steps:
            state = s(state)
        return state
    
    def _maybe_compile(self) -> Callable[[FieldState], FieldState]:
        fn = self._fused_step
        if self.compile_fused and hasattr(torch, "compile"):
            # fullgraph=True helps avoid graph breaks when shapes are static
            try:
                fn = torch.compile(fn, fullgraph=True) # type: ignore[attr-defined] # PyTorch 2.x compiler :contentReference[oaicite:8]{index=8}
            except Exception as e: # pragma: no cover
                print(f"[WARN] torch.compile failed; falling back. Reason: {e}")
                fn = self._fused_step
        return fn

    # ---- main run --------------------------------------------------------------
    def run(self, init_state: FieldState) -> FieldState:
        device = self.device
        use_graph = self.use_cuda_graph and device.type == "cuda"
        
        with torch.inference_mode():
            # Move state to device & clone a working instance
            f = init_state.field.to(device, non_blocking=True).contiguous()
            state = FieldState(f, float(init_state.t), float(self.dt), dict(init_state.meta))

            # Setup hooks & steps
            for s in self.steps: s.setup(state)
            for ob in self.observers: ob.on_setup(state)

            # -- compile fused step once --
            step_fn = self._maybe_compile()     # sets self._compiled_step

            # ---- WARMUP OUTSIDE CAPTURE ----
            # warmup runs allow dynamo/inductor to finish compilation and allocate kernels
            for _ in range(2):
                tmp: FieldState = step_fn(state)
                # write back into the original buffers to keep addresses stable
                state.field.copy_(tmp.field)
                state.t = tmp.t

            # -- Time-stepping loop --
            if use_graph:
                print("[DEBUG]: Run with CUDA Graph")
                g = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                # capture one iteration, writing result back into same storage
                with torch.cuda.graph(g):
                    out_state: FieldState = step_fn(state)
                    state.field.copy_(out_state.field)
                    state.t = out_state.t
                # main loop (fast replay)
                for i in range(self.num_time_steps):
                    # Optional variable dt
                    if self.dt_provider is not None:
                        state.dt = float(self.dt_provider(i, state))
                    g.replay()
                    for ob in self.observers: ob.on_step_end(i, state)
            else:
                print("[DEBUG]: Run (eager/compiled, no CUDA Graph)")
                for i in range(self.num_time_steps):
                    if self.dt_provider is not None:
                        state.dt = float(self.dt_provider(i, state))
                    out_state = step_fn(state)
                    state.field.copy_(out_state.field)
                    state.t = out_state.t
                    for ob in self.observers: ob.on_step_end(i, state)

            # -- Teardown --
            for ob in self.observers: ob.on_teardown()
            for s in self.steps: s.teardown()
            
            return state


# =====================================
# needs refactoring

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
#         num_time_steps: int,
#     ) -> None:
#         # 1) Emit
#         for src in sources:
#             grid.add_particles(src.emit(t_start))
#         backup.on_emit(grid, t_start)

#         # 2) Time integration
#         t = t_start
#         for k in range(1, num_time_steps + 1):
#             for step in steps:
#                 step.apply(grid, t, dt)
#             t = t_start + k * dt
#             backup.on_step_end(grid, t)
#         backup.finalize()
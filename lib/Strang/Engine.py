from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

import torch
import torch._dynamo
# torch._dynamo.config.verbose = True
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.log_level = logging.DEBUG

# import numpy as np
from tqdm import tqdm, trange

from lib import Step, Observer
from lib.State import FieldState, Field

# from lib.tools import performance
# from lib.tools.mem_plot_profiler import profile_memory_usage, log_event

class Engine(ABC):
    """Base running pipeline class for solving your equation.

    Must implement `run(init_state) -> FieldState`.
    """
    @abstractmethod
    def run(self, init_state: FieldState) -> FieldState:
        ...

class LoopEngine(Engine):
    """
    Second-order Strang splitting time integrator over `steps`. 
    Designed via simple "for" cycles (loop over steps in each time step). 

    Parameters
    ----------
    steps : List[Step]
    Ordered list of split operators applied each time step.
    num_time_steps : int
    Number of time steps to perform.
    dt : float
    Constant time step
    observers : Optional[List[Observer]]
    Callbacks to record sparse diagnostics.
    device : str
    Device string for the main run (e.g., "cuda" or "cpu").

    Example of usage:
    --------
    >>> LEngine = LoopEngine(steps, num_time_steps, dt, observers)
    >>> final_state = LEngine.run(init_state) # also do jobs of observers like saveing some data or debug prints
    """
    def __init__(self,
                steps: List[Step],
                num_time_steps: int,
                dt: float,
                observers: Optional[List[Observer]] = None,
                device: str = "cuda",
                verbose: int = 0
                )-> None:
        self.num_time_steps = int(num_time_steps)
        self.dt = float(dt)

        self.steps = list(steps)
        self.observers = list(observers or [])

        self.device = torch.device(device)
        self.verbose = verbose

    
    # @performance
    # @profile_memory_usage(interval=0.00001, verbose=1)
    def __simulation_loop(self, field: Field, **kwargs) -> Field:
        if self.verbose: 
            print("[DEBUG]: Run (dummy cycles)")
            # log_event("Start", **kwargs)
        for i in trange(self.num_time_steps, desc = "LoopEngine simulation:"):
            for s in self.steps: 
                field = s.forward(field)
            for ob in self.observers: ob.on_step_end(i, field)
        # if self.verbose: log_event("Fin", **kwargs)
        return field

    def run(self, init_state: FieldState) -> FieldState:
        device = self.device
        
        with torch.inference_mode():
            # Move state to device & clone a working instance
            field = init_state.field.to(device, non_blocking=True).contiguous()
            state = FieldState(field, float(self.dt), dict(init_state.meta))

            # -- Setup hooks & steps --
            for s in self.steps: s.setup(state)
            for ob in self.observers: ob.on_setup(state)

            # -- Main sim,ulation loop --
            field = self.__simulation_loop(field)
            state = FieldState(field, float(self.dt), dict(init_state.meta))

            # -- Teardown --
            for ob in self.observers: ob.on_teardown()
            for s in self.steps: s.teardown()
            
            return state

class TorchEngine(LoopEngine):
    """
    Second-order Strang splitting time integrator over `steps`.

    Optimized TorchEngine with Time-Step Batching (Super-stepping).
    
    Instead of calling the GPU 10,000 times for 10,000 steps, we call it
    100 times, where each call performs 100 steps internally.

    Parameters
    ----------
    steps : List[Step]
    Ordered list of split operators applied each time step.
    num_time_steps : int
    Number of time steps to perform.
    dt : float
    Constant time step for now.
    observers : Optional[List[Observer]]
    Callbacks to record sparse diagnostics.
    device : str
    Device string for the main run (e.g., "cuda" or "cpu").
    chunk_size : int
    Size of time iterations for one compiled simulation process (during this chunked time range there will be not any possibility to run observers).

    Example of usage:
    --------
    >>> TEngine = TorchEngine(steps, num_time_steps, dt, observers, chunk_size)
    >>> final_state = TEngine.run(init_state) # also do jobs of observers like saveing some data or debug prints
    """
    def __init__(self,
                steps: List[Step],
                num_time_steps: int,
                dt: float,
                observers: Optional[List[Observer]] = None,                
                chunk_size: int = 100,
                device: str = "cuda",
                verbose: int = 0
                )-> None:
        super().__init__(steps, num_time_steps, dt, observers, device, verbose)
        
        self.chunk_size = chunk_size
        self._compiled_chunk = None

        # Check for meaningful batching
        if self.num_time_steps < self.chunk_size:
            self.chunk_size = self.num_time_steps
            if self.verbose:
                print(f"[INFO] Reduced chunk_size to {self.chunk_size} to match total steps.")

    # ---- overridable fused step -------------------------------------------------
    def _make_compiled_chunk(self, field: Field):
        """
        Creates a function that runs `self.chunk_size` steps in a loop,
        then compiles that entire loop.
        """
        if self._compiled_chunk is not None:
            return self._compiled_chunk

        # 1. Define the 'Super Step' (Pure PyTorch logic, no side effects)
        def block_step(f: torch.Tensor) -> torch.Tensor:
            # We hardcode the loop range here so the compiler unrolls/optimizes it
            for _ in range(self.chunk_size):
                for s in self.steps:    # ← hope all steps are torch ops / scripted
                    f = s.forward(f)
            return f

        # 2. Compile it
        if self.verbose:
            print(f"[INFO] Compiling graph for block of {self.chunk_size} steps...")
        
        try:
            self._compiled_chunk = torch.compile(
                block_step,
                mode="reduce-overhead",  # Enforces CUDA Graphs where possible # "default",
                fullgraph=True,          # We promise there are no python fallbacks in block_step
                dynamic=False            # We promise shapes won't change
                # Test options.
                # options={
                #     "triton.cudagraphs": True, # try to enable cuda graph inside dynamo
                #     # "traceable_collection": True,   # sometimes helps with list of modules
                # }
            )
        except Exception as e:
            print(f"[WARN] torch.compile failed; falling back to eager execution. Reason: {e}")
            self._compiled_chunk = block_step # Fallback to uncompiled python loop
            
        return self._compiled_chunk

    # ---- main run --------------------------------------------------------------
    
    # @torch.inference_mode()
    # @profile_memory_usage(interval=0.00001, verbose=1)
    def run(self, init_state: FieldState, **kwargs) -> FieldState:
        with torch.inference_mode():
            device = self.device
            
            # 1. Prepare State
            # Ensure we are contiguous and on correct device
            f = init_state.field.to(device, non_blocking=True).contiguous()
            state = FieldState(f, float(self.dt), dict(init_state.meta))

            # 2. Setup
            for s in self.steps: s.setup(state)
            for ob in self.observers: 
                ob.sync_every(self.chunk_size)
                ob.on_setup(state)

            # 3. Compile the "Super Step"
            # We trigger compilation on the first actual tensor
            compiled_step_block = self._make_compiled_chunk(f)

            # 4. Simulation Loop
            total_steps = self.num_time_steps
            
            # Calculate how many "blocks" we need
            # e.g., if total=1000, chunk=100 -> 10 iterations
            num_blocks = total_steps // self.chunk_size
            remainder = total_steps % self.chunk_size

            if self.verbose:
                print(f"[DEBUG] Running {num_blocks} blocks of {self.chunk_size} steps each.")

            # -- WARMUP (Optional but recommended for benchmarking) --
            # Runs the compiled kernel once to burn-in CUDA graphs before timing starts
            if self.verbose > 0:
                _ = compiled_step_block(f.clone())

            # -- MAIN LOOP --
            current_time_step = 0
            
            # We iterate over blocks, not individual steps
            pbar = tqdm(total=total_steps, disable=not self.verbose, desc="Simulating")
            
            # log_event("Start", **kwargs)
            for _ in range(num_blocks):
                # This acts as a synchronization barrier for the CUDA Graph memory pool
                # torch.compiler.cudagraph_mark_step_begin()
                # A. Run the heavy computation (Pure GPU, no Python overhead)
                f = compiled_step_block(f.clone())
                
                # B. Update counters
                current_time_step += self.chunk_size
                pbar.update(self.chunk_size)

                # C. Run Observers (Sparse Diagnostics)
                # Note: Observers only see the state at the END of the chunk.
                if self.observers:
                    for ob in self.observers:
                        ob.on_step_end(current_time_step, f)

            # log_event("Fin", **kwargs)

            # -- REMAINDER --
            # If total_steps wasn't divisible by chunk_size, finish the rest eagerly
            if remainder > 0:
                if self.verbose:
                    print(f"[DEBUG] Running remaining {remainder} steps eagerly.")
                for _ in range(remainder):
                    for s in self.steps:
                        f = s.forward(f)
                    current_time_step += 1
                    if self.observers:
                        for ob in self.observers:
                            ob.on_step_end(current_time_step, f)
            
            pbar.close()

            # 5. Teardown
            state = FieldState(f, float(self.dt), dict(init_state.meta))
            for ob in self.observers: ob.on_teardown()
            for s in self.steps: s.teardown()

            return state
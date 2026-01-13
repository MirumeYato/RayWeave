from __future__ import annotations

from typing import List

import torch
import numpy as np

# Models pakages
from lib import Step
from lib.Strang.Engine import StrangEngine

# Custom Observers and Steps
from lib.Observers.Loggers import EnergyLogger
from lib.Observers.Plot import Plot3D, EnergyPlotter
from lib.Steps.Streaming import Streaming
from lib.Steps.Collision import Collision
# from lib.Steps.dummy import DummyPropagate

from lib.grid.Angle import Angle
from lib.tools.func_HenyeyGreenshtein import alm_HenyeyGreenstein, expand_repeating_al_to_alm as expand_lm

# Function-like run
def make_combined_model(
        mu_a: float, mu_s: float, g: float, 
        speed: float, dt: float, bin_size: float,
        n_steps: int, Quadrature: Angle, 
        device, obs_type = None, vebrose=0) -> StrangEngine:
    """
    Simpliest propogater pipeline example. Do nothing, just pushes same FieldState further
    """
    steps: List[Step] = [
        Collision(g, Quadrature, mu_a, mu_s, device=device, vebrose=vebrose),
        Streaming(speed, bin_size, Quadrature, device=device, vebrose=vebrose),
        Collision(g, Quadrature, mu_a, mu_s, device=device, vebrose=vebrose)
    ]
    # Each observer impacts on calculation time. Make every parameter big as possible
    if obs_type is None: observers = [] 
    elif obs_type == 'plotting': observers = [Plot3D(every=1, vebrose=vebrose)]
    else: observers = [
        EnergyLogger(every=n_steps//10), 
        Plot3D(every=n_steps//10, vebrose=vebrose), 
        EnergyPlotter(n_steps, every=n_steps//100)]
    return StrangEngine(steps, n_steps, dt, observers,
                    device=device, compile_fused=False, use_cuda_graph=False)

def get_analytical_solution(
        initial_field: torch.Tensor, 
        speed: float, dt: float, point_s0: int,
        Quadrature: Angle, 
        device) -> torch.Tensor:
    J = initial_field.sum().abs().detach().cpu().numpy() # intensity in point like delta t source.

    # TODO: needs some imprelemtation

    final_field = NotImplemented
    return final_field

from __future__ import annotations

from typing import List

import torch
import numpy as np

# Models pakages
from lib import Step
from lib.Strang.Engine import StrangEngine

# Custom Observers and Steps
from lib.Observers.Loggers import EnergyLogger
from lib.Observers.Plot import PlotMollviewInPoint, EnergyPlotter
from lib.Steps.Collision import Collision
# from lib.Steps.dummy import DummyPropagate

from lib.grid.Angle import Angle
from lib.tools.func_HenyeyGreenshtein import alm_HenyeyGreenstein, expand_repeating_al_to_alm as expand_lm

# Function-like run
def make_collision_model(
        mu_a: float, mu_s: float, g: float, dt: float, 
        n_steps: int, Quadrature: Angle, c:int, 
        device, obs_type = None, vebrose=0) -> StrangEngine:
    """
    Simpliest propogater pipeline example. Do nothing, just pushes same FieldState further
    """
    steps: List[Step] = [
        Collision(g, Quadrature, mu_a, mu_s, device=device, vebrose=vebrose)
    ]
    # Each observer impacts on calculation time. Make every parameter big as possible
    if obs_type is None: observers = [] 
    elif obs_type == 'plotting': observers = [PlotMollviewInPoint([c,c,c], Quadrature, every=1)]
    else: observers = [
        EnergyLogger(every=n_steps//10), 
        PlotMollviewInPoint([c,c,c], Quadrature, every=n_steps//10), 
        EnergyPlotter(n_steps, every=n_steps//100)]
    return StrangEngine(steps, n_steps, dt, observers,
                    device=device, compile_fused=False, use_cuda_graph=False)

def get_analytical_solution(
        initial_field: torch.Tensor,
        mu_absorb: float, mu_scatter: float, g: float, 
        speed: float, dt: float, Lmax: int, point_s0: int,
        Quadrature: Angle, 
        device) -> torch.Tensor:
    J = initial_field.sum().abs().detach().cpu().numpy() # intensity in point like delta t source.

    g_l = alm_HenyeyGreenstein(g=g, L_max=Lmax, device=device).to(dtype=initial_field.dtype) # {L+1}
    lambda_l = speed * (-(mu_absorb + mu_scatter) + mu_scatter * g_l) # {L+1}
    exp_lm = torch.exp(lambda_l * np.complex128(dt / 2.0)) # {L+1}
    exp_lm = expand_lm(exp_lm, Lmax) # {(L+1)^2}

    shperical_harmonics, shperical_harmonics_H = Quadrature.get_spherical_harmonics(Lmax=Lmax, dtype=initial_field.dtype)

    final_field = speed * J * torch.einsum('p,qp->q', shperical_harmonics_H[point_s0, :] * exp_lm, shperical_harmonics) * Quadrature.get_weights() # [Q]
    return final_field

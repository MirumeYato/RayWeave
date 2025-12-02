from .Step import Step
from lib.State import FieldState
from lib.grid.Angle import Angle
from lib.tools.func_HenyeyGreenshtein import alm_HenyeyGreenstein

from abc import abstractmethod

import torch

import numpy as np

class Collision(Step):
    """
    Henyey-Greenstein scattering without moving. 
    """
    def __init__(self, anisotropy_coef: float, mu_absorb = 0, mu_scatter = 0, speed = 1,
                 device = None, vebrose = 0):
        # Grid for interpolation
        self.shifted_grid = None
        self.vebrose = vebrose
        self.device = device

        self.mu_absorb = mu_absorb      # Absorbtion length
        self.mu_scatter = mu_scatter    # Scattering length
        self.speed = speed    # Speed of light
        self.g = anisotropy_coef    # Anisotropy coefitient of Henyey-Greenstein function

    def setup(self, state: FieldState, fQuadrature: function, Lmax: int, dtype_float = torch.float64, dtype_complex = torch.complex128) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""

        # Check dimentions
        n_size = state.field.shape[0]             # angular
        spatial_dim = len(state.field.shape[1:])  # number of space dimensions (1D, 2D, 3D ...)
        spatial_size = state.field.shape[-1]                 # bin number for space coordinates

        # Define quadrature algo
        Quadrature: Angle = fQuadrature(n_size, 
                device = self.device, verbose = self.vebrose, dtype = dtype_float)

        # Pre-compute spherical harmonisc
        self.shperical_harmonics, self.shperical_harmonics_H = Quadrature.get_spherical_harmonics(Lmax=Lmax, dtype=dtype_complex)
        
        # Define weights for numerical integration via quadrature
        weights = Quadrature.get_weights()
        # Type fix. For speed and correct type usage.
        if isinstance(weights, (int, float)):
            self.weights = torch.full_like(state.field, float(weights)) # If solution is Chebishev-like 
        else:
            self.weights = weights.to(device=state.field.device, dtype=state.field.dtype)

        # Pre-calculate solution for Henyey-Greenstein's coefficients evolution.
        g_l = alm_HenyeyGreenstein(g=self.g, L_max=Lmax, device=self.device).to(dtype = dtype_float)
        lambda_l = self.speed * (-(self.mu_absorb + self.mu_scatter) + self.mu_scatter * g_l)
        self.exp_lm = torch.exp(lambda_l * (state.dt / 2.0)) 
        # TODO: extend for using source (blm = map2alm(source_map)) also
        # Originally alm_star = exp_lm * alm + c*(exp_lm-1)/lambda_l * blm
        #       P.S. For delta(t)-like sources easier to init state.field = source_map, so blm=0 in any (r,t,s)
        #            So alm_star = exp_lm * alm (calculation is simplier)

        if self.vebrose: print(f"""
    [DEBUG]: Setup stage
        Sucsesfully spherical harmonisc was pre-computed with shape: {self.shperical_harmonics.shape}.
        Angular dimention is {Quadrature.num_bins}, Spatial dimention is {spatial_size}\n""")

    def forward(self, state: FieldState) -> FieldState:
        
        # Core transform: a_lm = sum_p map(p) * Y_lm(p) * w(p)
        alm = torch.matmul(state.field * self.weights, self.shperical_harmonics_H)

        # Spectral filtering: multiply by f_lm per (l,m)
        alm_scattered = torch.einsum('pijk,p->pijk', alm, self.exp_lm)  # A_pv * exp_lm[:, None, None, None]

        # just summation for lm. No need quadrature
        scattered_field = torch.einsum('p,qp->q', alm_scattered, self.shperical_harmonics) # [Q,1,N,N,N]
        
        return FieldState(scattered_field, state.t + state.dt, state.dt, state.meta)
    
        # Lines below does not work with torch.compile . Use them only when compile_fused=False, use_cuda_graph=False
        #     if self.vebrose: print(f""" 
        # [DEBUG]: Setup stage
        #     Field sucsesfully scattered in time dt={state.dt}.
        #     Initial shape: {state.field.shape}.
        #     Result shape:  {scattered_field.shape}.\n""")
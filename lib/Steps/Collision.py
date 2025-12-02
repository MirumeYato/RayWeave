from .Step import Step
from lib.State import FieldState
from lib.grid.Angle import Angle
from lib.tools.func_HenyeyGreenshtein import alm_HenyeyGreenstein, expand_f_l_to_lm

from abc import abstractmethod

import torch

import numpy as np

class Collision(Step):
    """
    Henyey-Greenstein scattering without moving. 
    """
    def __init__(self, anisotropy_coef: float, Quadrature: Angle,
                 mu_absorb = 0., mu_scatter = 0., speed = 1.,
                 dtype_float = torch.float64, dtype_complex = torch.complex128,
                 device = "cpu", vebrose = 0):
        # Grid for interpolation
        self.shifted_grid = None
        self.vebrose = vebrose
        self.device = device
        self.dtype_float = dtype_float 
        self.dtype_complex = dtype_complex

        self.mu_absorb = mu_absorb      # Absorbtion length
        self.mu_scatter = mu_scatter    # Scattering length
        self.speed = speed    # Speed of light
        self.g = anisotropy_coef    # Anisotropy coefitient of Henyey-Greenstein function

        self.Quadrature:Angle = Quadrature

    def setup(self, state: FieldState) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""

        Lmax = state.meta["L_max"]

        # Check dimentions
        n_size = state.field.shape[0]             # angular
        spatial_dim = len(state.field.shape[1:])  # number of space dimensions (1D, 2D, 3D ...)
        spatial_size = state.field.shape[-1]                 # bin number for space coordinates

        # Pre-compute spherical harmonisc
        self.shperical_harmonics, self.shperical_harmonics_H = self.Quadrature.get_spherical_harmonics(Lmax=Lmax, dtype=self.dtype_complex)
        
        # Define weights for numerical integration via quadrature
        weights = self.Quadrature.get_weights()
        # Type fix. For speed and correct type usage.
        # TODO: really not well designed now. To match memory used here
        if isinstance(weights, (int, float)):
            self.weights = torch.full_like(state.field, float(weights), 
                    device=state.field.device, dtype=state.field.dtype) # If solution is Chebishev-like 
        else:
            self.weights = weights.to(device=state.field.device, dtype=state.field.dtype) 

        # Pre-calculate solution for Henyey-Greenstein's coefficients evolution.
        g_l = alm_HenyeyGreenstein(g=self.g, L_max=Lmax, device=self.device).to(dtype=self.dtype_float)
        lambda_l = self.speed * (-(self.mu_absorb + self.mu_scatter) + self.mu_scatter * g_l)
        exp_lm = torch.exp(lambda_l * (state.dt / 2.0)) 
        self.exp_lm = expand_f_l_to_lm(exp_lm, Lmax)

        #################################################################
        # TODO: extend for using source (blm = map2alm(source_map)) also
        # Originally alm_star = exp_lm * alm + c*(exp_lm-1)/lambda_l * blm
        #       P.S. For delta(t)-like sources easier to init state.field = source_map, so blm=0 in any (r,t,s)
        #            So alm_star = exp_lm * alm (calculation is simplier)
        # TODO: what if dt is not constant? Need adding possibility to 
        #       precalculate self.exp_lm for all other predefined dt.
        #
        #       I think that here is no real need to change dt in real time. 
        #       If dt was choosen bad, I think better to solve it adding some 
        #       logger of error or initially try to make prediction od error and print it out
        #       (look more same todo by "dt!=const")

        if self.vebrose: print(f"""
    [DEBUG]: Setup stage
        Sucsesfully spherical harmonisc was pre-computed with shape: {self.shperical_harmonics.shape}.
        Angular dimention is {self.Quadrature.num_bins}, Spatial dimention is {spatial_size}\n""")

    def forward(self, state: FieldState) -> FieldState:
        
        # A_pv = sum_q  w * I_qv * conj(Y_qp)
        # einsum shapes: (q) , (qv) , (qp) -> (pv)
        alm = torch.einsum('qijk,qp->pijk', state.field * self.weights, self.shperical_harmonics_H) # (P, V), complex

        # Spectral filtering: multiply by f_lm per (l,m)
        alm_scattered = torch.einsum('pijk,p->pijk', alm, self.exp_lm)  # A_pv * exp_lm[:, None, None, None] # TODO: what if dt is not constant? Need adding possibility to reinit self.exp_lm

        # just summation for lm. No need quadrature
        scattered_field = torch.einsum('pijk,qp->qijk', alm_scattered, self.shperical_harmonics) # [Q,N,N,N]
        
        return FieldState(scattered_field, state.dt, state.meta)
    
        # Lines below does not work with torch.compile . Use them only when compile_fused=False, use_cuda_graph=False
        #     if self.vebrose: print(f""" 
        # [DEBUG]: Setup stage
        #     Field sucsesfully scattered in time dt={state.dt}.
        #     Initial shape: {state.field.shape}.
        #     Result shape:  {scattered_field.shape}.\n""")
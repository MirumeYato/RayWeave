from .Step import Step
from lib.State import FieldState
from lib.grid.Angle import Angle
from lib.tools.func_HenyeyGreenshtein import alm_HenyeyGreenstein, expand_f_l_to_lm

from lib.tools.mem_plot_profiler import profile_memory_usage, log_event

import torch
import numpy as np
import gc

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

        self.mu_absorb = np.float64(mu_absorb)      # Absorbtion length
        self.mu_scatter = np.float64(mu_scatter)    # Scattering length
        self.speed = np.float64(speed)    # Speed of light
        self.g = np.float64(anisotropy_coef)    # Anisotropy coefitient of Henyey-Greenstein function

        self.Quadrature:Angle = Quadrature

    # @profile_memory_usage(interval=0.00001)
    def setup(self, state: FieldState, **kwargs) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""
        log_event("start", **kwargs)
        Lmax = state.meta["L_max"]

        # Check dimentions
        n_size = state.field.shape[0]             # angular
        spatial_dim = len(state.field.shape[1:])  # number of space dimensions (1D, 2D, 3D ...)
        spatial_size = state.field.shape[-1]                 # bin number for space coordinates

        # if self.vebrose: log_event("before SH", **kwargs)

        # Pre-compute spherical harmonisc. shape {Quadrature.num_bins, (L+1)**2}
        self.shperical_harmonics, self.shperical_harmonics_H = self.Quadrature.get_spherical_harmonics(Lmax=Lmax, dtype=self.dtype_complex)
        if self.vebrose: 
            print(f"L_max is {Lmax}, num of angle directions {self.Quadrature.num_bins}\nShperical harmonics shae is: {self.shperical_harmonics.shape}")
            # log_event("SH precalculated", **kwargs)
        
        # Define weights for numerical integration via quadrature
        weights = self.Quadrature.get_weights()
        # Type fix. For speed and correct type usage.
        # TODO: really not well designed now. To match memory used here
        if isinstance(weights, (int, float, np.float64)):
            self.weights = torch.full_like(state.field, np.float64(weights), 
                    device=state.field.device, dtype=state.field.dtype) # If solution is Chebishev-like 
        else:
            self.weights = weights.to(device=state.field.device, dtype=state.field.dtype) 

        # if self.vebrose: log_event("checked weights", **kwargs)

        # Pre-calculate solution for Henyey-Greenstein's coefficients evolution.
        g_l = alm_HenyeyGreenstein(g=self.g, L_max=Lmax, device=self.device).to(dtype=self.dtype_complex)
        lambda_l = self.speed * (-(self.mu_absorb + self.mu_scatter) + self.mu_scatter * g_l)
        exp_lm = torch.exp(lambda_l * np.complex128(state.dt / 2.0)) 
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

        if self.vebrose: 
            print(f"""
    [DEBUG]: Setup stage
        Sucsesfully spherical harmonisc was pre-computed with shape: {self.shperical_harmonics.shape}.
        Angular dimention is {self.Quadrature.num_bins}, Spatial dimention is {spatial_size}\n""")
            # log_event("precalculated exp(lambda dt)", **kwargs)

    def forward(self, state: FieldState) -> FieldState:
        
        # A_pv = sum_q  w * I_qv * conj(Y_qp)
        # einsum shapes: (qv) , (qp) -> (pv)
        alm = torch.einsum('qijk,qp->pijk', state.field * self.weights, self.shperical_harmonics_H) # (P, V), complex

        # Spectral filtering: multiply by f_lm per (l,m)
        alm_scattered = torch.einsum('pijk,p->pijk', alm, self.exp_lm)  # A_pv * exp_lm[:, None, None, None] # TODO: what if dt is not constant? Need adding possibility to reinit self.exp_lm

        # just summation for lm. No need quadrature
        scattered_field = torch.einsum('pijk,qp->qijk', alm_scattered, self.shperical_harmonics) # [Q,N,N,N]

        # scattered_field.real.relu_() # make zero of negative real items
        # scattered_field.imag.zero_()
        
        return FieldState(scattered_field, state.dt, state.meta)
    
        # Lines below does not work with torch.compile . Use them only when compile_fused=False, use_cuda_graph=False
        #     if self.vebrose: print(f""" 
        # [DEBUG]: Setup stage
        #     Field sucsesfully scattered in time dt={state.dt}.
        #     Initial shape: {state.field.shape}.
        #     Result shape:  {scattered_field.shape}.\n""")

    def teardown(self):
        del self.exp_lm
        del self.shperical_harmonics
        del self.shperical_harmonics_H        
        gc.collect()
        
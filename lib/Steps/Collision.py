from .Step import Step
from lib.State import FieldState, Field
from lib.grid.Angle import Angle
from lib.tools.func_HenyeyGreenstein import eigenvalues_HenyeyGreenstein, expand_repeating_al_to_alm as expand_lm

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
                 device = "cpu", verbose = 0):        
        super().__init__(device, verbose)

        # Grid for interpolation
        self.dtype_float = dtype_float 
        self.dtype_complex = dtype_complex

        self.mu_absorb = np.float64(mu_absorb)      # Absorbtion length
        self.mu_scatter = np.float64(mu_scatter)    # Scattering length
        # self.speed = np.float64(speed)              # Speed of light
        self.g = np.float64(anisotropy_coef)    # Anisotropy coefitient of Henyey-Greenstein function

        self.Quadrature:Angle = Quadrature # Class for producing angular operation like integration.

    def setup(self, state: FieldState, **kwargs) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""
        self.setup_dt(state)
        Lmax = state.meta["L_max"]

        # Check dimentions
        n_size = state.field.shape[0]             # angular
        spatial_dim = len(state.field.shape[1:])  # number of space dimensions (1D, 2D, 3D ...)
        spatial_size = state.field.shape[-1]                 # bin number for space coordinates

        # Pre-compute spherical harmonisc. shape {Quadrature.num_bins, (L+1)**2}
        self.shperical_harmonics, self.shperical_harmonics_H = self.Quadrature.get_spherical_harmonics(Lmax=Lmax, dtype=self.dtype_complex)
        if self.verbose: 
            print(f"[INFO]: L_max is {Lmax}, num of angle directions {self.Quadrature.num_bins}\\nShperical harmonics shape is: {self.shperical_harmonics.shape}")
        
        # Define weights for numerical integration via quadrature
        weights = self.Quadrature.get_weights()
        # Type fix. For speed and correct type usage.
        # TODO: really not well designed now. To match memory used here
        if isinstance(weights, (int, float, np.float64)):
            self.weights = torch.full_like(state.field, np.float64(weights), 
                    device=state.field.device, dtype=state.field.dtype) # If solution is Chebishev-like 
        else:
            self.weights = weights.to(device=state.field.device, dtype=state.field.dtype)

        # g^l are the exact Legendre moments / eigenvalues of the HG collision operator
        g_l = eigenvalues_HenyeyGreenstein(g=self.g, L_max=Lmax, device=self.device).to(dtype=self.dtype_complex)
        
        # lambda_l is the exact eigenvalue for the collision operator for degree l.
        lambda_l = -(self.mu_absorb + self.mu_scatter) + self.mu_scatter * g_l
        
        # Using full dt, not dt/2 — Strang splitting compensation is handled at engine level
        exp_lm = torch.exp(lambda_l * self.dt / 2.0)  # {L+1} 
        self.exp_lm = expand_lm(exp_lm, Lmax)   # {(L+1)^2}


        #################################################################
        # TODO: extend for using source (blm = map2alm(source_map)) also
        # Originally alm_star = exp_lm * alm + c*(exp_lm-1)/lambda_l * blm
        #       P.S. For delta(t)-like sources easier to init state.field = source_map, so blm=0 in any (r,t,s)
        #            So alm_star = exp_lm * alm (calculation is simplier)

        if self.verbose: 
            print(f"""
    [DEBUG]: Setup stage
        Sucsesfully spherical harmonisc was pre-computed with shape: {self.shperical_harmonics.shape}.
        Sucsesfully exp(lambda_l dt) was pre-computed with shape: {self.exp_lm.shape}.
        Angular dimention is {self.Quadrature.num_bins}, Spatial dimention is {spatial_size}\n""")

    def forward(self, field: Field, **kwargs) -> Field:
        # A_pv = sum_q  w * I_qv * conj(Y_qp)
        # einsum shapes: (qv) , (qp) -> (pv)
        alm = torch.einsum('qijk,qp->pijk', field * self.weights, self.shperical_harmonics_H) # (P, V), complex
        
        # Spectral filtering: multiply by f_lm per (l,m)
        alm_scattered = torch.einsum('pijk,p->pijk', alm, self.exp_lm)  # A_pv * exp_lm[:, None, None, None] # TODO: what if dt is not constant? Need adding possibility to reinit self.exp_lm

        # just summation for lm. No need quadrature
        scattered_field = torch.einsum('pijk,qp->qijk', alm_scattered, self.shperical_harmonics) # [Q,N,N,N]
        
        # scattered_field.real.relu_() # make zero of negative real items
        # scattered_field.imag.zero_()
        
        return scattered_field

    def teardown(self):
        del self.exp_lm
        del self.shperical_harmonics
        del self.shperical_harmonics_H        
        gc.collect()


import healpy as hp
class CollisionHP(Step):
    """
    Henyey-Greenstein scattering without moving. implemented on HealPix.
    """
    def __init__(self, anisotropy_coef: float, Quadrature: Angle,
                 mu_absorb = 0., mu_scatter = 0., speed = 1.,
                 dtype_float = torch.float64, dtype_complex = torch.complex128,
                 device = "cpu", verbose = 0):        
        super().__init__(device, verbose)

        # Grid for interpolation
        self.dtype_float = dtype_float 
        self.dtype_complex = dtype_complex

        self.mu_absorb = np.float64(mu_absorb)      # Absorbtion length
        self.mu_scatter = np.float64(mu_scatter)    # Scattering length
        # self.speed = np.float64(speed)              # Speed of light
        self.g = np.float64(anisotropy_coef)    # Anisotropy coefitient of Henyey-Greenstein function
        self.n_side = Quadrature.n_side

    def setup(self, state: FieldState, **kwargs) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""
        self.setup_dt(state)
        self.Lmax = state.meta["L_max"]
        l_arr, m_arr = hp.Alm.getlm(self.Lmax)

        # Check dimentions
        n_size = state.field.shape[0]             # angular
        spatial_dim = len(state.field.shape[1:])  # number of space dimensions (1D, 2D, 3D ...)
        spatial_size = state.field.shape[-1]                 # bin number for space coordinates
        

        # Pre-calculate solution for Henyey-Greenstein's coefficients evolution.
        lambda_l = -(self.mu_absorb + self.mu_scatter) + self.mu_scatter * (self.g ** l_arr)
        self.exp_lm = np.exp(lambda_l * self.dt / 2.0)

    def forward(self, field: Field, **kwargs) -> Field:
        # A_pv = sum_q  w * I_qv * conj(Y_qp)
        alm = hp.map2alm(field.detach().cpu().numpy()[:, 0, 0, 0], lmax=self.Lmax, iter=1, pol=False)
        
        # Spectral filtering: multiply by f_lm per (l,m)
        alm_scattered = self.exp_lm * alm # A_pv * exp_lm[:, None, None, None] # TODO: what if dt is not constant? Need adding possibility to reinit self.exp_lm

        # just summation for lm. No need quadrature
        scattered_field = field
        scattered_field[:, 0, 0, 0] = torch.tensor(hp.alm2map(alm_scattered, nside=self.n_side, lmax=self.Lmax, verbose=False), device=self.device, dtype=self.dtype_complex)
        
        return scattered_field

    def teardown(self):
        del self.exp_lm     
        gc.collect()
        
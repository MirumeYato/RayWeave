from .Step import Step
from lib.State import FieldState
from lib.grid.Angle import Angle
from lib.grid.tools import prepare_grid

import torch
import torch.nn.functional as F

import healpy as hp
import numpy as np
import gc

class Streaming(Step):
    """
    Simple propagation via interpolation
    """
    def __init__(self, speed: float, bin_size: float, Quadrature: Angle, device = "cpu", vebrose = 0):
        # Grid for interpolation
        self.shifted_grid = None
        self.bin_size = bin_size
        self.speed = speed

        self.vebrose = vebrose
        self.device = device
        self.Quadrature = Quadrature

    def setup(self, state: FieldState) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""
        self.dtype = state.field.dtype

        # Check dimentions
        n_size = state.field.shape[0]            # angular
        spatial_dim = len(state.field.shape[1:]) # number of space dimensions (1D, 2D, 3D ...)
        spatial_size = state.field.shape[-1]     # bin number for space coordinates

        # Pre-calc grid with shifts
        scaled_module_of_shift = self.speed * state.dt / self.bin_size
        # Get massive with all pixels norm vectors.
        dirs = self.Quadrature.get_nodes_coord()             # [Q,3], Q - angles dim.
        # Scale vectors by velocity
        shifts = dirs.to(self.device, dtype=state.field.real.dtype) * scaled_module_of_shift  # [Q,3]
        # Get shifted grid (where we will calculate new field values)
        self.shifted_grid = prepare_grid(shifts, spatial_size=spatial_size, device=self.device)      # (Q, N, N, N, 3)
        # TODO: (look more same todo by "dt!=const")

        if self.vebrose: print(f"""
    [DEBUG]: Setup stage
        Sucsesfully created shifted_grid with shape: {self.shifted_grid.shape}.
        Angular dimention is {self.Quadrature.num_bins}, Spatial dimention is {spatial_size}\n""")

    def forward(self, state: FieldState) -> FieldState:
        propagated_field = F.grid_sample( 
            state.field.real.unsqueeze(1), self.shifted_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)  # [Q,1,N,N,N].squeeze(1) -> [Q,N,N,N]

        ##############################################################
        # TODO: Here no any dt multiplication (because it is very slow).
        #       But than what if dt is not constant?
        #       Better will be precalculate all possible self.shifted_grid * dt
        #       and just coose here correct one by addresing by index of dt array.
        #       (look more same todo by "dt!=const")
        
        return FieldState(propagated_field.to(dtype=self.dtype), state.dt, state.meta)
    
        # Lines below does not work with torch.compile . Use them only when compile_fused=False, use_cuda_graph=False
        #     if self.vebrose: print(f""" 
        # [DEBUG]: Setup stage
        #     Field sucsesfully propogated in time dt={state.dt}.
        #     Initial shape: {state.field.shape}.
        #     Result shape:  {propagated_field.shape}.\n""")
    
    def teardown(self):
        del self.shifted_grid     
        gc.collect()
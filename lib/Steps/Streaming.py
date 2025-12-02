from .Step import Step
from lib.State import FieldState
from lib.grid.Angle import Angle
from lib.grid.tools import prepare_grid

import torch
import torch.nn.functional as F

import healpy as hp
import numpy as np

class Streaming(Step):
    """
    Simple propagation via interpolation
    """
    def __init__(self, speed = 1., device = None, vebrose = 0):
        # Grid for interpolation
        self.shifted_grid = None
        self.vebrose = vebrose
        self.speed = speed
        self.device = device

    def setup(self, state: FieldState, fQuadrature: Angle, dtype_float) -> None:
        """Allocate reusable buffers or precompute constants (on correct device)."""

        # Check dimentions
        n_size = state.field.shape[0]            # angular
        spatial_dim = len(state.field.shape[1:]) # number of space dimensions (1D, 2D, 3D ...)
        spatial_size = state.field.shape[-1]     # bin number for space coordinates

        # Define quadrature algo
        Quadrature: Angle = fQuadrature(n_size, 
                device = self.device, verbose = self.vebrose, dtype = dtype_float)

        # Pre-calc grid with shifts
        scale = 10.0 / (spatial_size - 1) * self.speed  # hardcode scaling. Needs refactoring
        # Get massive with all pixels norm vectors.
        dirs = Quadrature.get_nodes_coord()             # [Q,3], Q - angles dim.
        # Scale vectors by velocity
        shifts = torch.from_numpy(dirs.astype(np.float32)).to(self.device, dtype=dtype_float) * scale # [Q,3]
        # Get shifted grid (where we will calculate new field values)
        self.shifted_grid = prepare_grid(shifts, spatial_size=spatial_size, device=self.device)       # (Q, N, N, N, 3)

        if self.vebrose: print(f"""
    [DEBUG]: Setup stage
        Sucsesfully created shifted_grid with shape: {self.shifted_grid.shape}.
        Angular dimention is {Quadrature.num_bins}, Spatial dimention is {spatial_size}\n""")

    def forward(self, state: FieldState) -> FieldState:
        propagated_field = F.grid_sample( 
            state.field.unsqueeze(1), self.shifted_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)  # [Q,1,N,N,N]
        
        return FieldState(propagated_field, state.t + state.dt, state.dt, state.meta)
    
        # Lines below does not work with torch.compile . Use them only when compile_fused=False, use_cuda_graph=False
        #     if self.vebrose: print(f""" 
        # [DEBUG]: Setup stage
        #     Field sucsesfully propogated in time dt={state.dt}.
        #     Initial shape: {state.field.shape}.
        #     Result shape:  {propagated_field.shape}.\n""")
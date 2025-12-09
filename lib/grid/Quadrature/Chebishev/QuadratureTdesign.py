import os
import torch
import numpy as np

from ...Angle import Angle
from lib import PATH

class QuadratureTdesign(Angle):
    """
    Docstring for QuadratureTdesign
    
    :param n_size: number of quadratures nodes (or possible directions). For now implemented only for N = nonsym: 240, 1014, 10083 sym: 1038, 10014, 52978 .
    :type n_size: int
    """
    def __init__(self, n_size = 1, device=None, verbose=0, dtype = torch.float64):
        if verbose: print(f"[DEBUG]: Choosen number of nodes is {n_size}") # TODO: make this as func with variability of nodes

        if n_size == 240: self.fname = 'hs021.00240' # no sym
        elif n_size == 1014: self.fname = 'sf044.01014' 
        elif n_size == 10083: self.fname = 'sf141.10083' 
        elif n_size == 1038: self.fname = 'ss045.01038' # sym
        elif n_size == 10014: self.fname = 'ss141.10014'
        elif n_size == 52978: self.fname = 'ss325.52978'
        else: raise NotImplemented

        super().__init__(n_size, device, verbose, dtype)

    def __get_nodes_coord(self):
        coord = np.loadtxt(os.path.join(PATH,'cache', self.fname))

        # Convert your flat coord list into Nx3 matrix efficiently
        directions = np.asarray(coord, dtype=np.float64).reshape(-1, 3)  # shape (N, 3)

        # Normalize if needed (just directions)
        # norms = np.linalg.norm(directions, axis=1, keepdims=True)
        # directions = directions / norms
        return directions
    
    def get_nodes_coord(self) -> torch.Tensor:
        """
        Get directions of all pixels using T-design grid.

        :return: Unit vectors (x, y, z) representing pixel center directions.
        :rtype: torch.Tensor, shape (n_dirs, 3)
        """
        return torch.from_numpy(self.__get_nodes_coord()).to(device = self.device, dtype = self.dtype)
    
    def get_weights(self) -> np.float64:
        """Weights for Chebishev method of integration"""
        weights = np.float64((4.0 * np.pi)) / np.float64(self.n_directions)
        return weights
    
    def get_nodes_angles(self) -> tuple[torch.Tensor]:
        """
        :return: grid like arrays of theta and phi (theta ∈ [0,π], phi ∈ [0,2π))
        :rtype: tuple
        """
        directions = self.__get_nodes_coord()
        x, y, z = directions.T

        # More stable than np.arccos(z)
        r_xy = np.hypot(x, y)          # sqrt(x^2 + y^2) with better stability
        theta = np.arctan2(r_xy, z)    # θ ∈ [0, π]

        phi = np.arctan2(y, x)         # φ ∈ (-π, π]
        phi = np.mod(phi, np.float64(2. * np.pi))   # φ ∈ [0, 2π)
        return torch.from_numpy(theta).to(device = self.device, dtype = self.dtype), \
            torch.from_numpy(phi).to(device = self.device, dtype = self.dtype)
from ...Angle import Angle

import torch
import numpy as np
import healpy as hp

class QuadratureHEALPix(Angle):
    """
    Docstring for QuadratureHEALPix
    
    :param n_size: healpix nside parameter, that needs for calculation of pixel number
    :type n_size: int
    """
    def __init__(self, n_size = 1, device=None, verbose=0, dtype = torch.float64):
        self.n_side = n_size
        self.n_directions = hp.nside2npix(n_size)
        print(f"[DEBUG]: number of pixel is {self.n_directions}")
        super().__init__(self.n_directions, device, verbose, dtype)

    def get_nodes_coord(self) -> torch.Tensor:
        """
        Get directions of all pixels using HEALPix pixel centers.

        :return: Unit vectors (x, y, z) representing pixel center directions.
        :rtype: torch.Tensor, shape (n_dirs, 3)
        """
        ipix = np.arange(self.n_directions)
        # Get unit vectors for pixel centers
        directions = np.array(hp.pix2vec(self.n_side, ipix)).T  # shape (n_dirs, 3)
        return torch.from_numpy(directions).to(device = self.device, dtype = self.dtype)
    
    def get_weights(self) -> float:
        """Weights for Chebishev method of integration"""
        weights = (4.0 * np.pi) / self.n_directions
        return weights
    
    def get_nodes_angles(self) -> tuple[torch.Tensor]:
        """
        :return: grid like arrays of theta and phi (theta ∈ [0,π], phi ∈ [0,2π))
        :rtype: tuple
        """        
        # than we should choose correct greed (kinda uniform). But how to do this correctly is discussible...
        theta, phi = hp.pix2ang(self.n_side, np.arange(self.n_directions)) 
        return torch.from_numpy(theta).to(device = self.device, dtype = self.dtype), \
            torch.from_numpy(phi).to(device = self.device, dtype = self.dtype)

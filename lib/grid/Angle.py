from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import torch

from lib.tools.sht import compute_SH

class  Angle(ABC):
    def __init__(self, n_size: int = 1, device = "cpu", verbose = 0, dtype = torch.float64):
        self.n_directions = n_size
        self.device = device
        self.dtype = dtype

    @property
    def num_bins(self) -> int:
        """How many bins (channels or directions) needed to store angular histogram."""
        return self.n_directions
    
    @abstractmethod
    def get_nodes_coord(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_nodes_angles(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    def get_spherical_harmonics(self, Lmax, dtype=torch.complex128):
        theta, phi = self.get_nodes_angles()

        Y = compute_SH(theta=theta, phi=phi, L_max=Lmax, device=self.device, dtype=dtype)
        Y_H = torch.conj(Y)
        return Y, Y_H

    # @abstractmethod
    # def show_hist(ang_arr: np.ndarray):
    #     pass

# --- Angle: HEALPix helper (keeps anything angular in one place) -----------

class Angle3D(Angle):
    """HEALPix-based angular discretization & utilities."""
    def __init__(self, healpix_nside: int = 3):
        if healpix_nside <= 0 or not isinstance(healpix_nside, int):
            raise ValueError("healpix_nside must be an integer")
        self.nside = healpix_nside
        self.n_dirs = hp.nside2npix(self.nside)

    def vec2pix(self, dirs: np.ndarray) -> np.ndarray: # is it really needed?
        """
        Vectorized: map direction vectors (N,3) -> HEALPix pixel indices (N,).
        Keeps raw counts semantics (no normalization).
        """
        v = dirs.astype(np.float64, copy=False)
        # Normalize to unit vectors (safe guard)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        v = v / n
        # Spherical
        theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))         # [0, pi]
        phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2.0 * np.pi) # [0, 2pi)
        return hp.ang2pix(self.nside, theta, phi)
    
    def _angle2pix(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Vectorized: map angels (N,) -> HEALPix pixel indices (N,).
        Keeps raw counts semantics (no normalization).
        """
        return hp.ang2pix(self.nside, theta, phi)
    
    def get_hpix_arr(self):
        """ Returns hpix array from theta, phi, weight"""
        return NotImplemented
    
    def get_all_vecs(self):
        """
        Get directions of all pixels using HEALPix pixel centers.

        Returns
        -------
        dirs : np.ndarray, shape (n_dirs, 3)
            Unit vectors (x, y, z) representing pixel center directions.
        """
        ipix = np.arange(self.n_dirs)
        # Get unit vectors for pixel centers
        return np.array(hp.pix2vec(self.nside, ipix)).T  # shape (n_dirs, 3)
    
    def show_hist(hpix_arr: np.ndarray):
        """ Shows Mollweide projection of input array """
        hp.mollview(hpix_arr, title="Mollview image RING")
        hp.graticule()
from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

class  Angle(ABC):
    def __init__(self, n: int = 1):
        self.n_dirs = n
    @property
    def channels(self) -> int:
        """How many channels needed to store angular histogram."""
        return self.n_dirs
    
    def _angle2pix(self, phi: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Vectorized: map angele (N,) -> HEALPix pixel indices (N,).
        Keeps raw counts semantics (no normalization).
        """
        return phi
    
    def zero_hist(self, *shape_prefix: int, dtype=np.float32) -> np.ndarray:
        """Allocate an angular histogram tensor with a given leading shape."""
        return np.zeros((*shape_prefix, self.n_dirs), dtype=dtype)
    
    @abstractmethod
    def get_all_vecs(self):
        pass

    @abstractmethod
    def show_hist(ang_arr: np.ndarray):
        pass

class Angle2D(Angle):
    """HEALPix-based angular discretization & utilities."""
    def __init__(self, angle_arr_lenth: int = 3):
        if angle_arr_lenth <= 0 or not isinstance(angle_arr_lenth, int):
            raise ValueError("angle_arr_lenth must be an integer")
        self.n_dirs = angle_arr_lenth

    def get_all_vecs(self):
        return NotImplemented
    
    def show_hist(ang_arr: np.ndarray):
        """ Shows Mollweide projection of input array """
        plt.hist(ang_arr)

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
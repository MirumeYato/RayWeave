from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch

from lib.grid.Angle import Angle
from lib.tools.func_HenyeyGreenstein import map_HenyeyGreenstein

# from lib.data import ParticleState

class Source(ABC):
    """Initializes (or injects) particles into the Grid at t0."""
    pass
    # @abstractmethod
    # def emit(self, t0: float) -> Sequence[ParticleState]:
    #     """Return the particles to insert at time t0."""

def make_dummy_source(Q, N, device='cpu'):
    return torch.zeros((Q, N, N, N), device=device)

def make_hg_source(Angle: Angle, device, N = 1, c = 0, c2 = 0, g = 0.1):
    Q = Angle.num_bins
    field_tensor = torch.zeros((Q, N, N, N), dtype=torch.complex128, device=device)
    
    # 1. Get the full grids of angles
    # Assuming get_nodes_angles() returns full arrays/tensors for all bins
    thetas, phis = Angle.get_nodes_angles() 
    
    # 2. Extract the target direction where the max should be
    theta_target = thetas[c2]
    phi_target = phis[c2]
    
    # 3. Calculate the relative angle (alpha) between all directions and the target direction
    # Using the spherical law of cosines: cos(α) = cos(θ1)cos(θ2) + sin(θ1)sin(θ2)cos(φ1 - φ2)
    cos_alpha = (torch.cos(thetas) * torch.cos(theta_target) + 
                 torch.sin(thetas) * torch.sin(theta_target) * torch.cos(phis - phi_target))
    
    # Clamp cos_alpha to avoid any floating-point edge cases outside [-1, 1] before acos
    cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
    relative_thetas = torch.acos(cos_alpha)
    
    # 4. Pass the relative angles to the HG mapping function
    # This guarantees that at index c2, relative_theta is 0, yielding the maximum HG value.
    hg_values = map_HenyeyGreenstein(g, relative_thetas)
    
    # 5. Normalize and assign to the source spatial position [c, c, c]
    field_tensor[:, c, c, c] = hg_values / hg_values.sum()

    return field_tensor

def make_point_source(Angle: Angle, device, N = 1, c = 0, c2 = 0):
    Q = Angle.num_bins
    field_tensor = torch.zeros((Q, N, N, N), dtype=torch.complex128, device=device)
    
    # Adding sources
    field_tensor[c2, c, c, c] = 1.0 # point-like source in the middle

    return field_tensor


# =====================================
# needs refactoring

# -----------------------------
# List of Sources:
# -----------------------------

# class TwoPointSource(Source):
#     """Emit exactly two particles with given (pos, direction, speed)."""

#     def __init__(
#         self,
#         pos1: Sequence[float], dir1: Sequence[float], 
#         pos2: Sequence[float], dir2: Sequence[float], 
#         speed1: float = 1.0, speed2: float = 1.0
#     ) -> None:
#         self.p1 = ParticleState(np.array(pos1, float), np.array(dir1, float), speed1)
#         self.p2 = ParticleState(np.array(pos2, float), np.array(dir2, float), speed2)

#     def emit(self, t0: float) -> Sequence[ParticleState]:  # noqa: D401
#         return [self.p1, self.p2]
    







    # TBD. some materials for tests
# def angular_quadrant_weights(dirs: np.ndarray, axis=np.array([1.0, 0.0, 0.0]), tol: float = 1e-2):
#     """
#     Assigns weight = 1 if direction forms angles pi/4, 3pi/4, 5pi/4, or 7pi/4 with the given axis.
#     Otherwise, weight = 0.

#     Parameters
#     ----------
#     dirs : np.ndarray
#         Array of direction vectors (N, 3) or (N, 2).
#     axis : np.ndarray
#         Reference axis vector.
#     tol : float
#         Angular tolerance in radians.

#     Returns
#     -------
#     w : np.ndarray
#         Binary weight array of shape (N,).
#     angles : np.ndarray
#         Angle of each direction relative to the axis (radians).
#     """
#     axis = axis / np.linalg.norm(axis)
#     dots = np.clip(dirs @ axis, -1.0, 1.0)
#     angles = np.arccos(dots)

#     # Target angles
#     target_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])

#     # Compute weights
#     w = np.zeros_like(angles)
#     print(angles)
#     print(target_angles)
#     for target in target_angles:
#         w[np.isclose(angles, target, atol=tol)] = 1

#     return w, angles

# def angular_uniform_weights(dirs: np.ndarray, axis=np.array([1.0, 0.0, 0.0])):
#     """
#     Assigns weight = 1 if direction forms angles pi/4, 3pi/4, 5pi/4, or 7pi/4 with the given axis.
#     Otherwise, weight = 0.

#     Parameters
#     ----------
#     dirs : np.ndarray
#         Array of direction vectors (N, 3) or (N, 2).
#     axis : np.ndarray
#         Reference axis vector.
#     tol : float
#         Angular tolerance in radians.

#     Returns
#     -------
#     w : np.ndarray
#         Binary weight array of shape (N,).
#     angles : np.ndarray
#         Angle of each direction relative to the axis (radians).
#     """
#     axis = axis / np.linalg.norm(axis)
#     dots = np.clip(dirs @ axis, -1.0, 1.0)
#     angles = np.arccos(dots)

#     # Compute weights
#     w = np.ones_like(dirs[:,0])
#     return w, angles

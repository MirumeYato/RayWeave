from __future__ import annotations

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

# from lib.data import ParticleState

class Source(ABC):
    """Initializes (or injects) particles into the Grid at t0."""

    # @abstractmethod
    # def emit(self, t0: float) -> Sequence[ParticleState]:
    #     """Return the particles to insert at time t0."""

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

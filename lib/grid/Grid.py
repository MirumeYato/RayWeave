# from __future__ import annotations

# from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

# import numpy as np
# import healpy as hp
import torch

# from lib.data import ParticleState
# from lib.physics.Angles import Angle

# ====================================

Grid = torch.Tensor




# ====================================

# class Grid(ABC):
#     """Abstract spatial container for particles and fields.

#     Concrete implementations may be uniform grids, AMR, octrees, etc.
#     Only minimal API is required by Steps & Propagator.
#     """

#     @abstractmethod
#     def add_particles(self, particles: Sequence[ParticleState]) -> None:
#         """Insert particles on grid (castom IC statement)."""       

#     @abstractmethod
#     def get(self) -> ParticleState:
#         """Get grid object."""

#     @abstractmethod
#     def update(self) -> None:
#         """Updates grid (state RTE solution at new time)"""

# -----------------------------
# List of Grids:
# -----------------------------

# class ParticleDict2D(Grid):
#     """Minimal uniform grid (not really) that simply stores particles in a dict.

#     This is *not* spatially indexed; it's a simple container to satisfy the API.
#     Replace with AMR/Octree later without changing Step/Propagator.
#     """

#     def __init__(self) -> None:
#         self._particles: Dict[int, ParticleState] = {}
#         self._next_id: int = 0

#     def add_particles(self, particles: Sequence[ParticleState]) -> List[int]:
#         """Insert particles, return their assigned ids.""" 
#         ids: List[int] = []
#         for p in particles:
#             pid = self._next_id
#             self._next_id += 1
#             p.pid = pid
#             # store a copy to decouple outside references
#             self._particles[pid] = ParticleState(pos=p.pos.copy(), direction=p.direction.copy(), speed=p.speed, weight=p.weight, pid=pid)
#             ids.append(pid)
#         return ids

#     def iter_particles(self) -> Iterable[ParticleState]:
#         """Iterate over all particles currently in the grid."""
#         return list(self._particles.values())

#     def get(self, pid: int) -> ParticleState:
#         """Get particle by id."""
#         return self._particles[pid]

#     def update(self, pid: int, new_state: ParticleState) -> None:
#         """Replace particle state (position, direction, etc.)."""
#         new_state.pid = pid
#         self._particles[pid] = ParticleState(
#             pos=new_state.pos.copy(),
#             direction=new_state.direction.copy(),
#             speed=new_state.speed,
#             weight=new_state.weight,
#             pid=pid,
#         )

#     def n_particles(self) -> int:
#         """Current number of particles in the grid."""
#         return len(self._particles)


# class MeshGrid(Grid):
#     """
#     Uniform 3D grid with a single state tensor:
#         state[..., 0:2]          -> two scalar params (accumulated sums)
#         state[..., 2:2+n_dirs]   -> HEALPix angular counts (raw, unnormalized)

#     """

#     def __init__(
#         self,
#         box_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
#         n_cells: Tuple[int, int, int] = (16, 16, 16),
#         angle: Angle = None,
#     ):
#         self.box_size = np.asarray(box_size, dtype=np.float64)
#         self.n_cells = np.asarray(n_cells, dtype=np.int32)
#         if (self.n_cells <= 0).any():
#             raise ValueError("n_cells must be positive in each dimension")
#         self.cell_size = self.box_size / self.n_cells

#         self.angle = angle or Angle(healpix_nside=4)
#         self.C = 2 + self.angle.channels  # 2 params + angular bins

#         # Single contiguous state tensor
#         self.state = np.zeros((*self.n_cells, self.C), dtype=np.float32)

#         # Update config (kept simple; can be changed at runtime)
#         self._update_cfg = dict(
#             normalize_params=False,   # average params by intensity if True
#             scatter_fraction=0.0,     # isotropic scattering fraction in [0,1]
#             attenuation=0.0,          # exponential attenuation coeff per step
#         )

#     # ------------------- geometry helpers (no storage duplication) ----------

#     @property
#     def centers(self) -> np.ndarray:
#         """Return (Nx,Ny,Nz,3) array of cell centers (computed on demand)."""
#         axes = [
#             (np.arange(n) + 0.5) * cs
#             for n, cs in zip(self.n_cells, self.cell_size)
#         ]
#         X, Y, Z = np.meshgrid(*axes, indexing="ij")
#         return np.stack([X, Y, Z], axis=-1)

#     # Accessors that keep correlation by shared indexing:
#     def get_params(self) -> np.ndarray:
#         """(Nx,Ny,Nz,2) – accumulated parameter sums (not normalized)."""
#         return self.state[..., :2]

#     def get_dirs(self) -> np.ndarray:
#         """(Nx,Ny,Nz,n_dirs) – HEALPix raw counts (intensity)."""
#         return self.state[..., 2:2 + self.angle.channels]

#     def get_counts(self) -> np.ndarray:
#         """(Nx,Ny,Nz,1) – total intensity per cell (sum over HEALPix bins)."""
#         return self.get_dirs().sum(axis=-1, keepdims=True)

#     # ------------------- configuration for update() -------------------------

#     def configure_update(self, **kwargs):
#         """Set update parameters (e.g., normalize_params, scatter_fraction)."""
#         self._update_cfg.update(kwargs)

#     # ------------------- core API ------------------------------------------

#     def add_particles(self, particles: Sequence[ParticleState]) -> None:
#         """Vectorized deposit: positions -> cell indices; dirs -> HEALPix bins."""
#         if len(particles) == 0:
#             return

#         positions = np.stack([p.pos for p in particles], axis=0)  # (N,3)
#         directions = np.stack([p.dir for p in particles], axis=0) # (N,3)
#         params = np.stack([p.params for p in particles], axis=0)  # (N,2)

#         # Cell indices (clip to box)
#         rel = positions / self.box_size
#         idx = np.floor(rel * self.n_cells).astype(np.int64)
#         idx = np.clip(idx, 0, self.n_cells - 1)  # (N,3)

#         # HEALPix bin indices (vectorized)
#         pix = self.angle.vec2pix(directions).astype(np.int64)     # (N,)

#         # Flattened cell indices
#         flat = np.ravel_multi_index(idx.T, self.n_cells)          # (N,)

#         # Scatter-add into the single state tensor (flattened view)
#         S = self.state.reshape(-1, self.C)

#         # params channels (two scalars)
#         np.add.at(S[:, 0], flat, params[:, 0])
#         np.add.at(S[:, 1], flat, params[:, 1])

#         # angular counts: add 1 at column (2 + pix)
#         cols = 2 + pix
#         np.add.at(S, (flat, cols), 1.0)

#     def update(self) -> None:
#         """
#         Minimal, vectorized RTE-like step:
#           - optional attenuation
#           - optional isotropic scattering fraction
#           - optional parameter normalization by intensity
#         Streaming/advection between cells is intentionally omitted in this dummy.
#         """
#         dirs = self.get_dirs()              # view into self.state
#         params = self.get_params()          # view into self.state
#         counts = self.get_counts()          # (Nx,Ny,Nz,1)

#         # 1) attenuation (Beer–Lambert): I <- I * exp(-atten)
#         atten = float(self._update_cfg.get("attenuation", 0.0))
#         if atten > 0.0:
#             np.multiply(dirs, np.exp(-atten, dtype=np.float32), out=dirs)

#         # 2) isotropic scattering fraction: mix with uniform over bins
#         sf = float(self._update_cfg.get("scatter_fraction", 0.0))
#         if sf > 0.0:
#             # target uniform distribution per cell
#             uniform = counts / float(self.angle.channels)  # (Nx,Ny,Nz,1)
#             # broadcast to angular channels
#             dirs *= (1.0 - sf)
#             dirs += sf * np.broadcast_to(uniform, dirs.shape)

#         # 3) (optional) params normalization by intensity (turn sums -> means)
#         if bool(self._update_cfg.get("normalize_params", False)):
#             denom = np.maximum(counts, 1e-8)   # avoid divide by zero
#             params /= denom  # in-place view; if you prefer, write to a copy

#     def get(self) -> Dict[str, Any]:
#         """Lightweight snapshot access (no copies beyond what's necessary)."""
#         return dict(
#             centers=self.centers,          # computed on the fly
#             params=self.get_params(),      # view
#             dirs=self.get_dirs(),          # view
#             counts=self.get_counts(),      # derived
#             cell_size=self.cell_size.copy(),
#             n_cells=self.n_cells.copy(),
#             healpix_nside=self.angle.nside,
#         )
    



# import torch
# import ocnn

# class OCNNGrid(Grid):
#     def __init__(self, max_depth: int, healpix_nside: int, device: torch.device = "cpu"):
#         self.max_depth = max_depth
#         self.healpix_nside = healpix_nside
#         self.n_dirs = hp.nside2npix(healpix_nside)
#         self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Create a blank octree in ocnn
#         # The ocnn library uses its own Octree descriptor class
#         self.octree = ocnn.octree.create(max_depth=self.max_depth, device=self.device)

#         # Features per node: we allocate node feature tensor
#         # Suppose ocnn supports a feature tensor `node_feat` of shape (N_nodes, C)
#         # We'll use first n_dirs dims for angular bins, then 2 dims for scalar params
#         # Start with zeros
#         self.node_feat = torch.zeros((0, self.n_dirs + 2), device=self.device)

#     def add_particles(self, particles: Sequence["ParticleState"]) -> None:
#         """
#         Deposit particles into octree. We'll convert their positions to leaf nodes,
#         then accumulate direction + scalar params into features.
#         """
#         # Example: get positions tensor, directions, params
#         pos = torch.stack([torch.tensor(p.pos, device=self.device) for p in particles])
#         dirs = torch.stack([torch.tensor(p.dir, device=self.device) for p in particles])
#         params = torch.stack([torch.tensor(p.params, device=self.device) for p in particles])

#         # Use ocnn method to map particles to node indices:
#         # ocnn may offer something like `octree.point2node(octree, pos)` -> (node_idx, ...)
#         node_idxs = ocnn.octree.point2node(self.octree, pos)  # placeholder API

#         # Convert directions to HEALPix pixel indices (vectorized)
#         # Healpy functions might be Python; for performance you may implement your own
#         # Here's a naive approach (for sketch)
#         theta, phi = hp.vec2ang(dirs.cpu().numpy())
#         pix = hp.ang2pix(self.healpix_nside, theta, phi)
#         pix = torch.tensor(pix, dtype=torch.long, device=self.device)

#         # Now we accumulate into node_feat:
#         # Expand node_feat if needed
#         Nn = self.octree.node_num  # number of nodes in this octree
#         if self.node_feat.shape[0] < Nn:
#             new = torch.zeros((Nn, self.n_dirs + 2), device=self.device)
#             new[: self.node_feat.shape[0], :] = self.node_feat
#             self.node_feat = new

#         # Accumulate direction counts
#         self.node_feat[node_idxs, pix] += 1.0
#         # Accumulate scalar params into last two dims
#         self.node_feat[node_idxs, self.n_dirs : self.n_dirs + 2] += params

#     def update(self) -> None:
#         """
#         After accumulation, normalize angular histograms and parameters, and then
#         apply RTE advancement (scattering, propagation) using tensor ops.
#         """
#         # Normalize angular bins per node
#         angular = self.node_feat[:, : self.n_dirs]
#         sums = angular.sum(dim=1, keepdim=True)
#         # Prevent divide-by-zero
#         mask = sums > 0
#         angular[mask] = angular[mask] / sums[mask]
#         # Scalar params normalization if needed, e.g. average
#         # Maybe you want to divide by particle counts stored elsewhere
#         # For simplicity, do nothing now.

#         # Then you can do propagation: e.g. angular redistribution between neighbor nodes
#         # You can use ocnn neighbor queries, or define your own adjacency for the octree,
#         # and then do something like tensor scatter/gather to move flux from one node to neighbors.

#     def get(self):
#         """Return a snapshot: node positions, features, etc."""
#         # You can extract from ocnn: e.g. `octree.node_xyz` or similar
#         xyz = ocnn.octree.get_xyz(self.octree)  # placeholder
#         return {
#             "xyz": xyz.detach().cpu().numpy(),
#             "feature": self.node_feat.detach().cpu().numpy(),
#             "octree": self.octree  # you might return the descriptor for further custom ops
#         }
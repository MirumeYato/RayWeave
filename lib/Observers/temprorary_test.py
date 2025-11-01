from __future__ import annotations

from typing import Dict, Any, List

from lib import Observer
from lib.State import FieldState

# class EnergySumObserver(Observer):
#     """Accumulates total energy per step (sum over the entire field)."""

#     def __init__(self) -> None:
#         self.values: List[float] = []

#     def on_setup(self, state: FieldState) -> None:
#         self.values.clear()

#     def on_step_end(self, step_idx: int, state: FieldState) -> None:
#         val = float(state.field.sum().detach().cpu())
#         self.values.append(val)


# class DetectorHitObserver(Observer):
#     """Records when energy at a given voxel crosses a threshold (toy detector)."""

#     def __init__(self, voxel_xyz: tuple[int, int, int], threshold: float) -> None:
#         self.xyz = tuple(int(x) for x in voxel_xyz)
#         self.th = float(threshold)
#         self.hits: List[Dict[str, Any]] = []

#     def on_setup(self, state: FieldState) -> None:
#         self.hits.clear()

#     def on_step_end(self, step_idx: int, state: FieldState) -> None:
#         x, y, z = self.xyz
#         # assumes field shape [B, C, Nx, Ny, Nz]; adapt indexing to your layout
#         val = float(state.field[..., x, y, z].sum().detach().cpu())
#         if val >= self.th:
#             self.hits.append({"step": step_idx, "t": state.t, "value": val})



# -----------------------------
# List of BackupData classes:
# -----------------------------

class EnergyLogger(Observer):
    def __init__(self, every: int = 50):
        self.every = every
    def on_step_end(self, step_idx: int, state: FieldState) -> None:
        if step_idx % self.every == 0:
            # cheap summary; no sync if possible
            total = state.field.sum().detach().item()
            print(f"[{step_idx}] total_energy={total:.6e}")

# class TrackBackup:
#     """Store particle positions at each step for track plotting.

#     For simplicity, we pre-allocate after emission based on the number of particles.
#     """

#     def __init__(self, n_expected_steps: int) -> None:
#         self.n_expected_steps = n_expected_steps
#         self.t: List[float] = []
#         self.tracks: Optional[np.ndarray] = None  # shape (P, T, 2)
#         self._step_idx: int = 0

#     def on_emit(self, grid: Grid, t: float) -> None:
#         P = grid.n_particles()
#         T = self.n_expected_steps + 1  # include t0
#         self.tracks = np.full((P, T, 2), np.nan, dtype=float)
#         # record initial positions
#         for p in grid.iter_particles():
#             assert p.pid is not None
#             self.tracks[p.pid, 0, :] = p.pos  # type: ignore[index]
#         self.t = [t]
#         self._step_idx = 0

#     def on_step_end(self, grid: Grid, t: float) -> None:
#         if self.tracks is None:
#             raise RuntimeError("Call on_emit before on_step_end")
#         self._step_idx += 1
#         for p in grid.iter_particles():
#             assert p.pid is not None
#             self.tracks[p.pid, self._step_idx, :] = p.pos  # type: ignore[index]
#         self.t.append(t)

#     def finalize(self) -> None:
#         pass

#     # Convenience accessors
#     def get_tracks(self) -> np.ndarray:
#         assert self.tracks is not None
#         return self.tracks

#     def get_times(self) -> List[float]:
#         return self.t
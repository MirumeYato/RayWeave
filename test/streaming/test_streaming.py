import pytest
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from lib.State import FieldState
from lib.Steps.Streaming import Streaming
from lib.Strang.Engine import LoopEngine
from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign
from lib.Sources.Source import make_point_source
from lib import Observer
from lib.Observers.Plot import Plot3D

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'test', 'streaming'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


class StreamingPhysicsObserver(Observer):
    """
    Tracks three physics metrics at each timestep:
      - Energy conservation error  (1 - I_current / I_initial)
      - Trajectory CoM MAE (pixels)
      - 99% intensity containment radius (smearing)
    Saves a 3-panel plot on teardown.
    """
    def __init__(self, dt: float, grid_size: int, center: int,
                 dirs: torch.Tensor, device: torch.device, grid_name: str = ""):
        super().__init__(every=1)
        self.dt = dt
        self.grid_size = grid_size
        self.center_voxel = center
        self.dirs = dirs.cpu().numpy()
        self.device = device
        self.grid_name = grid_name

        self.metrics = {"time": [], "intensities": [], "maes": [], "radii_95": []}

        # Pre-compute pixel coordinate grid for Center of Mass
        self.Z, self.Y, self.X = torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=torch.float32),
            torch.arange(grid_size, device=device, dtype=torch.float32),
            torch.arange(grid_size, device=device, dtype=torch.float32),
            indexing='ij'
        )

    def on_setup(self, state: FieldState) -> None:
        self.initial_sum = state.field.real.sum().item()

    def on_step_end(self, step_idx: int, field: torch.Tensor, initial_flag=False) -> None:
        # We assume evaluating only a single angular directory (field.real[c2])
        # We need the single active angular bin index. 
        # For simplicity, we just take the sum over angular bins if it was a point source,
        # but the physics observer focuses on center trajectory. Here we use the whole field 
        # and sum over angles.
        current_time = (step_idx + 1) * self.dt
        field_data = field.real.sum(dim=0)   # sum over angular bins -> [N,N,N]
        i_current = field_data.sum().item()

        self.metrics["time"].append(current_time)
        self.metrics["intensities"].append(
            abs(1.0 - (i_current / self.initial_sum)) if self.initial_sum > 0 else 0
        )

        if i_current > 1e-8:
            z_com = (field_data * self.Z).sum().item() / i_current
            y_com = (field_data * self.Y).sum().item() / i_current
            x_com = (field_data * self.X).sum().item() / i_current
            com = np.array([x_com, y_com, z_com])

            # Analytical position: center + direction * (speed=1) * time
            true_pos = np.array([self.center_voxel] * 3) + self.dirs * current_time
            print(f"Step {step_idx+1}: CoM = {com}, True Pos = {true_pos}, Intensity = {i_current:.4f}")
            mae = np.linalg.norm(com - true_pos)
            self.metrics["maes"].append(mae)

            # 99% containment radius
            dists = torch.sqrt(
                (self.Z - z_com)**2 + (self.Y - y_com)**2 + (self.X - x_com)**2
            ).flatten()
            flat_field = field_data.flatten()
            sorted_indices = torch.argsort(dists)
            cumsum = torch.cumsum(flat_field[sorted_indices], dim=0)
            idx_99 = torch.searchsorted(cumsum, 0.99 * i_current).item()
            idx_99 = min(idx_99, len(dists) - 1)
            self.metrics["radii_95"].append(dists[sorted_indices[idx_99]].item())
        else:
            self.metrics["maes"].append(np.nan)
            self.metrics["radii_95"].append(np.nan)

    def on_teardown(self) -> None:
        times = self.metrics["time"]
        intensities = self.metrics["intensities"]
        maes = self.metrics["maes"]
        radii_95 = self.metrics["radii_95"]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Streaming Physics Validation — {self.grid_name}", fontsize=13)

        axs[0].plot(times, intensities, marker='o', color='red', linewidth=2)
        axs[0].set_title("Energy Conservation")
        axs[0].set_ylabel(r"$1 - I_{current}/I_{initial}$")
        axs[0].set_xlabel("Time")
        axs[0].grid(True)

        axs[1].plot(times, maes, marker='s', color='blue', linewidth=2)
        axs[1].set_title("Trajectory Accuracy (CoM)")
        axs[1].set_ylabel("MAE (pixels)")
        axs[1].set_xlabel("Time")
        axs[1].grid(True)

        axs[2].plot(times, radii_95, marker='^', color='green', linewidth=2)
        axs[2].set_title("Smearing Effect")
        axs[2].set_ylabel("Radius containing 99% Intensity (px)")
        axs[2].set_xlabel("Time")
        axs[2].grid(True)

        plt.tight_layout()
        fname = f"streaming_physics_{self.grid_name}.png" if self.grid_name else "streaming_physics.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        plt.close()

    def get_max_energy_loss(self):
        return max(self.metrics["intensities"])

    def get_max_trajectory_mae(self):
        return max(np.nan_to_num(self.metrics["maes"]))


# ─────────────────────────── parametrize ────────────────────────────────────

def get_angular_binnings():
    return [
        (QuadratureHEALPix, 2),
    ]

@pytest.fixture(scope="module")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════ test ════════════════════════════════════════════

@pytest.mark.parametrize("GridClass, n_size", get_angular_binnings())
@pytest.mark.parametrize("grid_size", [80])
@pytest.mark.parametrize("n_steps", [50])
def test_streaming_physics(device, GridClass, n_size, grid_size, n_steps):
    """
    Validates streaming physics over many timesteps:
      - Energy is conserved (< 10% loss).
      - CoM trajectory follows the analytical path (MAE < 0.5 px).
    Saves a 3-panel metrics plot and a 3-D heat-map GIF to output/test/streaming/.
    """
    Angle = GridClass(n_size=n_size, device=device)
    center_voxel = grid_size // 2
    active_angle_idx = Angle.num_bins//2 +1

    field_tensor = torch.zeros(
        (Angle.num_bins, grid_size, grid_size, grid_size),
        device=device, dtype=torch.float32
    )
    field_tensor[active_angle_idx, center_voxel, center_voxel, center_voxel] = 1.0

    dirs = Angle.get_nodes_coord()[active_angle_idx]
    print(f"Active direction: {dirs}")

    dt = (grid_size // 2 * 0.6) / n_steps
    state = FieldState(field=field_tensor, dt=dt, meta={"L_max": 10})

    grid_name = f"{GridClass.__name__}_{n_size}_g{grid_size}"

    # Physics observer → metrics plot
    phys_obs = StreamingPhysicsObserver(dt, grid_size, center_voxel, dirs, device, grid_name)

    # 3D heat-map GIF (every ~5 steps)
    plot3d_obs = Plot3D(every=max(1, n_steps // 10), output_directory=OUTPUT_DIR, verbose=0)

    steps = [Streaming(speed=1.0, bin_size=1.0, Quadrature=Angle, device=device)]
    propagator = LoopEngine(steps, n_steps, dt, [phys_obs, plot3d_obs], device=device)

    with torch.no_grad():
        final_state = propagator.run(state)

    # ── assertions ──────────────────────────────────────────────────────────
    assert not torch.isnan(final_state.field).any(), "Simulation diverged (NaN values)"
    assert phys_obs.get_max_energy_loss() < 0.1, \
        f"Severe energy loss: {phys_obs.get_max_energy_loss()*100:.1f}%"
    assert phys_obs.get_max_trajectory_mae() < 0.5, \
        f"Trajectory deviation too high, max MAE = {phys_obs.get_max_trajectory_mae():.3f} px"

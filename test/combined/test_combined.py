import pytest
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from lib.State import FieldState
from lib.Steps.Collision import Collision
from lib.Steps.Streaming import Streaming
from lib.Strang.Engine import LoopEngine as StrangEngine
from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign
from lib.Sources.Source import make_point_source, make_hg_source
from lib import Observer
from lib.Observers.Plot import Plot3D

BASE_OUTPUT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'test', 'combined'
))


class CombinedPhysicsObserver(Observer):
    """
    Tracks energy conservation and field positivity during Strang-split simulation.
    Saves a 2-panel plot on teardown.
    """
    def __init__(self, dt: float, output_directory: str, grid_name: str):
        super().__init__(every=1)
        self.dt = dt
        self.output_dir = output_directory
        self.grid_name = grid_name
        self.metrics = {"time": [], "energy_error": [], "min_value": []}

    def on_setup(self, state: FieldState) -> None:
        self.initial_sum = state.field.real.sum().item()

    def on_step_end(self, step_idx: int, field: torch.Tensor, initial_flag=False) -> None:
        current_sum = field.real.sum().item()
        self.metrics["time"].append((step_idx + 1) * self.dt)
        self.metrics["energy_error"].append(
            abs(1.0 - current_sum / self.initial_sum) if self.initial_sum > 0 else 0.0
        )
        self.metrics["min_value"].append(field.real.min().item())

    def on_teardown(self) -> None:
        times = self.metrics["time"]

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Combined (Strang) Simulation — {self.grid_name}", fontsize=13)

        axs[0].plot(times, self.metrics["energy_error"], color='red', linewidth=2)
        axs[0].set_title("Energy Conservation Error")
        axs[0].set_ylabel(r"$|1 - I_{current}/I_{initial}|$")
        axs[0].set_xlabel("Time")
        axs[0].grid(True)

        axs[1].plot(times, self.metrics["min_value"], color='navy', linewidth=2)
        axs[1].axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        axs[1].set_title("Field Minimum (negativity indicator)")
        axs[1].set_ylabel("min(field)")
        axs[1].set_xlabel("Time")
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"combined_metrics_{self.grid_name}.png"), dpi=150)
        plt.close()

    def get_max_energy_error(self):
        return max(self.metrics["energy_error"])


# ─────────────────────────── parametrize ────────────────────────────────────

def get_angular_binnings():
    return [
        (QuadratureTdesign, 240, 10),
        (QuadratureTdesign, 1014, 22)
    ]

@pytest.fixture(scope="module")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.parametrize("GridClass, n_size, Lmax", get_angular_binnings())
def test_combined_robustness(device, GridClass, n_size, Lmax):
    """
    Ensures Collision + Streaming + Collision (Strang splitting) runs without
    NaNs or crashes and conserves energy to within 10%.
    Saves per-grid metrics plot and 3D GIF into output/test/combined/<grid>/
    """
    # Architecture params
    grid_size = 20
    g = 0.9
    mu_a = 0.0
    mu_s = 1.0
    speed = 1.0 # 3.e8 limits tests, keep it normalized avoiding instabilities if possible
    
    n_time_steps = 10
    dt = (grid_size // 2 * 0.6) / n_time_steps

    # ── per-test output subfolder ─────────────────────────────────────────
    grid_name = f"{GridClass.__name__}_{n_size}"
    output_dir = os.path.join(BASE_OUTPUT, grid_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── setup field ────────────────────────────────────────────────────────
    Angle = GridClass(n_size=n_size, device=device)
    c = grid_size // 2
    field_tensor = make_hg_source(Angle, device, N=grid_size, c=c, c2=Angle.num_bins//2, g=0.1)

    state = FieldState(field=field_tensor, dt=dt, meta={"L_max": Lmax})

    # ── observers ─────────────────────────────────────────────────────────
    phys_obs = CombinedPhysicsObserver(dt, output_dir, grid_name)
    plot3d_obs = Plot3D(every=max(1, n_time_steps // 5), output_directory=output_dir, verbose=0, azimut=150, elev=45)

    # ── Strang splitting: C/2 → S → C/2 ─────────────────────────────────
    steps = [
        Collision(g, Angle, mu_a, mu_s, device=device, verbose=0),
        Streaming(speed, 0., Angle, device=device, verbose=0),
        Collision(g, Angle, mu_a, mu_s, device=device, verbose=0)
    ]

    propagator = StrangEngine(steps, n_time_steps, dt, [phys_obs, plot3d_obs], device=device)

    with torch.no_grad():
        final_state = propagator.run(state)

    # ── assertions ────────────────────────────────────────────────────────
    assert not torch.isnan(final_state.field).any(), \
        f"[{grid_name}] Combined simulation yielded NaNs"
    assert phys_obs.get_max_energy_error() < 0.1, \
        f"[{grid_name}] Energy not conserved: max error = {phys_obs.get_max_energy_error():.2e}"

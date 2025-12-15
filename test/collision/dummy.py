from __future__ import annotations

from typing import List

# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..', '..'))
sys.path.insert(0, PATH)
#===============================#
# Retry with fixed-size frames (avoid bbox_inches='tight' which changed image sizes).

import torch

from lib.State import FieldState
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign as Angle3D

from lib.tools.func_HenyeyGreenshtein import map_HenyeyGreenstein
from lib.tools.mem_plot_profiler import profile_memory_usage, log_event

# Models pakages
from lib import Step
from lib.Strang.Engine import StrangEngine

# Custom Observers and Steps
from lib.Observers.Loggers import EnergyLogger
from lib.Observers.Plot import PlotMollviewInPoint, EnergyPlotter
from lib.Steps.Collision import Collision
# from lib.Steps.dummy import DummyPropagate

def make_hg_source(Angle, device):
    Q = Angle.num_bins
    N = 10
    field_tensor = torch.zeros((Q, N, N, N), dtype=torch.complex128, device=device)
    # Adding sources
    c = 0#N // 2
    thetas, __ = Angle.get_nodes_angles()
    field_tensor[:, c, c, c] = map_HenyeyGreenstein(0.9, thetas)

    return field_tensor

def make_point_source(Angle, device):
    Q = Angle.num_bins
    N = 1
    field_tensor = torch.zeros((Q, N, N, N), dtype=torch.complex128, device=device)
    # Adding sources
    c = 0#N // 2
    c2 = Q // 2 + 5
    field_tensor[c2, c, c, c] = 1.0 # point-like source in the middle

    return field_tensor

@profile_memory_usage(interval=0.0001)
def main(**kwargs):
    log_event("Simulation Start", **kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Source Init (needs refactoring)
    Angle = Angle3D(n_size = 240, device=device) # 10083 1014 240
    
    # field_tensor = make_point_source(Angle, device)
    field_tensor = make_hg_source(Angle, device)
    c = 0

    # Init FieldState
    state = FieldState(
        field=field_tensor,
        dt=1e-7,
        meta={
            "L_max": 100,
        }
    )

    n_time_steps = 10
    log_event("Field precalculated", **kwargs)

    # Function-like run
    def make_dev_dummy_model(dt: float, n_steps: int, Quadrature, device) -> StrangEngine:
        """
        Simpliest propogater pipeline example. Do nothing, just pushes same FieldState further
        """
        steps: List[Step] = [
            # DummyPropagate(device),
            # here can be any steps you want
            Collision(-0.9, Quadrature, 0., 1., device=device, vebrose=1)
        ]
        # Each observer impacts on calculation time. Make every parameter big as possible
        # observers = [EnergyLogger(every=n_time_steps//10), PlotMollviewInPoint([c,c,c], Quadrature, every=n_time_steps//10), EnergyPlotter(n_time_steps, every=n_time_steps//100)] # 
        observers = []
        return StrangEngine(steps, n_steps, dt, observers,
                        device=device, compile_fused=False, use_cuda_graph=False)

    propogator = make_dev_dummy_model(state.dt, n_time_steps, Angle, device=device)
    final_state = propogator.run(state)
    log_event("end", **kwargs)

    # from lib.tools.plot_tools import maps_rmae

    # # Accuracy plot
    # # mae = torch.abs((final_state.field[:, c, c, c] - state.field[:, c, c, c]))
    # mae = torch.abs((final_state.field[:, c, c, c].real - state.field[:, c, c, c]) / state.field[:, c, c, c])
    # # print(f"Mae in pixel with intencity = 1 is {mae[c2]}")
    # print(f"Max Mae is {mae.max()}")
    # maps_rmae(mae.real.detach().cpu().numpy(), 1, rmae_flag=True)

if __name__ == "__main__":
    with torch.no_grad():
        main()
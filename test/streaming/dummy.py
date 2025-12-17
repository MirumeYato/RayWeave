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
# from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix as Angle3D
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign as Angle3D

from lib.tools.func_HenyeyGreenshtein import map_HenyeyGreenstein
# from lib.tools.mem_plot_profiler import profile_memory_usage, log_event

# Models pakages
from lib import Step
from lib.Strang.Engine import StrangEngine

# Custom Observers and Steps
from lib.Observers.Loggers import EnergyLogger
from lib.Observers.Plot import PlotMollviewInPoint, EnergyPlotter
from lib.Steps.Collision import Collision
# from lib.Steps.dummy import DummyPropagate
from lib.Sources.Source import make_point_source
from streaming_model import make_streaming_model

# @profile_memory_usage(interval=0.0001)
def main(**kwargs):
    # log_event("Simulation Start", **kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Source Init
    Angle = Angle3D(n_size = 240, device=device)
    
    c = 80 // 2
    c2 = 0#Angle.num_bins // 2
    field_tensor = make_point_source(Angle, device, N=80, c=c, c2=c2)

    # Init FieldState
    state = FieldState(
        field=field_tensor,
        dt=1e-7,
        meta={
            "L_max": 10,
        }
    )

    # Initial energy
    w0 = state.field.real.sum().detach().item()

    n_time_steps = 100
    # log_event("Field precalculated", **kwargs)

    speed = 1.
    bin_size = 5e-6

    propogator = make_streaming_model(
        speed, state.dt, bin_size, 
        n_time_steps, Angle, 
        device=device, obs_type='other')
    final_state = propogator.run(state)
    # log_event("end", **kwargs)

    # from lib.tools.plot_tools import maps_rmae

    # # Accuracy plot
    # # mae = torch.abs((final_state.field[:, c, c, c] - state.field[:, c, c, c]))
    # mae = torch.abs((final_state.field[:, c, c, c].real - state.field[:, c, c, c]) / state.field[:, c, c, c])
    # # print(f"Mae in pixel with intencity = 1 is {mae[c2]}")
    # print(f"Max Mae is {mae.max()}")
    # maps_rmae(mae.real.detach().cpu().numpy(), 1, rmae_flag=True)

if __name__ == "__main__":
    # with torch.no_grad():
    main()
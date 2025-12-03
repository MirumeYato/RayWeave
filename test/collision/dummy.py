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

# Models pakages
from lib import Step
from lib.Strang.Engine import StrangEngine

# Custom Observers and Steps
from lib.Observers.Loggers import EnergyLogger
from lib.Observers.Plot import PlotMollviewInPoint
from lib.Steps.Collision import Collision
# from lib.Steps.dummy import DummyPropagate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Source Init (needs refactoring)
Angle = Angle3D(n_size = 240, device=device)
Q = Angle.num_bins
N = 10

# Empty field
field_tensor = torch.zeros((Q, N, N, N), dtype=torch.complex128, device=device)
# Adding sources
c = N // 2
# c2 = Q // 2 + 5
# field_tensor[c2, c, c, c] = 1.0 # point-like source in the middle
thetas, __ = Angle.get_nodes_angles()
field_tensor[:, c, c, c] = map_HenyeyGreenstein(0.9, thetas)

# Init FieldState
state = FieldState(
    field=field_tensor,
    dt=1e-5,
    meta={
        "L_max": 10,
    }
)

# Function-like run
def make_dev_dummy_model(dt: float, n_steps: int, Quadrature, device) -> StrangEngine:
    """
    Simpliest propogater pipeline example. Do nothing, just pushes same FieldState further
    """
    steps: List[Step] = [
        # DummyPropagate(device),
        # here can be any steps you want
        Collision(-0.9, Quadrature, 0., 1., device=device)
    ]
    # Each observer impacts on calculation time. Make every parameter big as possible
    observers = [EnergyLogger(every=500), PlotMollviewInPoint([c,c,c], Quadrature, every=1000)]
    return StrangEngine(steps, n_steps, dt, observers,
                      device=device, compile_fused=False, use_cuda_graph=False)

propogator = make_dev_dummy_model(state.dt, 50000, Angle, device=device)
final_state = propogator.run(state)

# from lib.tools.plot_tools import maps_rmae

# Accuracy plot
mae = torch.abs((final_state.field[:, c, c, c] - state.field[:, c, c, c]))
# mae = torch.abs((final_state.field[:, c, c, c] - state.field[:, c, c, c])/state.field[:, c, c, c])
# print(f"Mae in pixel with intencity = 1 is {mae[c2]}")
print(f"Max Mae is {mae.max()}")
# maps_rmae(mae.real.detach().cpu().numpy(), 10, rmae_flag=False)
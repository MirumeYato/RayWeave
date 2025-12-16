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

from lib.Sources.Source import make_point_source
from collision_model import make_collision_model, get_analytical_solution as solution

# Parameters to test:
g_fin = -0.1, 0.1, 0.5, 0.9
mu_a =  0.1, 1, 10
mu_s =  0.1, 1, 10
dt = 1e-7, 1e-5, 1e-3


def main(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Source Init
    Angle = Angle3D(n_size = 240, device=device)
    
    point_r0 = 0
    point_s0 = Angle.num_bins // 2 + 5
    field_tensor = make_point_source(Angle, device, c=point_r0, c2=point_s0)

    # Init FieldState
    Lmax = 10
    state = FieldState(
        field=field_tensor,
        dt=dt,
        meta={
            "L_max": Lmax,
        }
    )

    # Initial energy
    w0 = state.field.real.sum().detach().item()

    # Numerical solution
    
    n_time_steps = 1
    propogator = make_collision_model(mu_a, mu_s, g_fin, 
                state.dt, n_time_steps, point_r0, Angle, device=device)
    final_state = propogator.run(state)

    # Theoretical solution
    speed = 1.
    theory_field = solution(
        state.field[:, point_r0, point_r0, point_r0], mu_a, mu_s, g_fin, speed, dt, Lmax, point_s0,
        Angle, device=device) 

    # Compare solutions
    from lib.tools.plot_tools import maps_rmae

    # Accuracy plot
    rmae = torch.abs((final_state.field[:, point_r0, point_r0, point_r0].real - theory_field) / theory_field)
    maps_rmae(rmae.real.detach().cpu().numpy(), 1, rmae_flag=True)

if __name__ == "__main__":
    with torch.no_grad():
        main()
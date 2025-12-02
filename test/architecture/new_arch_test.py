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

from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix as Angle3D

from models.dummy import make_dev_dummy_model, model, model_seq
from lib.State import FieldState

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Source Init
Angle = Angle3D(n_size = 2, device=device)
Q = Angle.num_bins
N = 10

# Empty field
field_tensor = torch.zeros((Q, N, N, N), dtype=torch.float32, device=device)
# Adding sources
c = N // 2
c2 = Q // 2 + 5
field_tensor[c2, c, c, c] = 1.0 # point-like source in the middle

# Init FieldState
state = FieldState(
    field=field_tensor,
    dt=1.,
    meta={
        "L_max": 2*3-3,
    }
)

propogator = make_dev_dummy_model(state.dt, 10, device=device)
propogator.run(state)

print("test model approuach")

final_state = model.run(FieldState(field=field_tensor, dt=0.01, meta={}))

print("test sequential approuach")

final_state = model_seq.run(FieldState(field=field_tensor, dt=0.01, meta={}))
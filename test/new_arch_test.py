# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#
# Retry with fixed-size frames (avoid bbox_inches='tight' which changed image sizes).

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import numpy as np
import torch
from math import pi
import healpy as hp

from lib.physics.Angles import Angle3D
from tqdm import trange, tqdm

from lib.physics.Propogators import make_propagator
from lib.data import FieldState

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dt = 1.
nside = 2


N = 10
Angle = Angle3D(healpix_nside = nside)
Q = Angle.num_channels
field_tensor = torch.zeros((Q, N, N, N), dtype=torch.float32, device=device)
c = N // 2
c2 = Q // 2 + 5
field_tensor[c2, c, c, c] = 1.0 # point-like source in the middl

state = FieldState(
    field=field_tensor,
    t=0.0,
    dt=dt,
    meta={
        "nside": nside,
        "L_max": 2*3-3,
    }
)


propogator = make_propagator(dt, 10, device=device)
propogator.run(state)

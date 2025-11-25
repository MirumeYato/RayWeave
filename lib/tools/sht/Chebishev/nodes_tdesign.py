from lib import PATH
from ...sht import compute_SH

import os
import torch
import numpy as np

def get_nodes_weights():
    print(f"[DEBUG]: number of nodes is {240}") # TODO: make this as func with variability of nodes

    coord = np.loadtxt(os.path.join(PATH,'cache','hs021.00240'))

    # Convert your flat coord list into Nx3 matrix efficiently
    dirs = np.asarray(coord, dtype=np.float64).reshape(-1, 3)  # shape (N, 3)

    # Normalize if needed (just directions)
    # norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    # dirs = dirs / norms

    x, y, z = dirs.T

    # More stable than np.arccos(z)
    r_xy = np.hypot(x, y)          # sqrt(x^2 + y^2) with better stability
    theta = np.arctan2(r_xy, z)    # θ ∈ [0, π]

    phi = np.arctan2(y, x)         # φ ∈ (-π, π]
    phi = np.mod(phi, 2 * np.pi)   # φ ∈ [0, 2π)

    weights = (4.0 * np.pi) / 240

    return theta, phi, weights

def get_spherical_harmonics(Lmax, device, dtype=torch.complex128):
    theta, phi, weights = get_nodes_weights()
    Y = compute_SH(theta=theta, phi=phi, L_max=Lmax, device=device, dtype=dtype)
    Y_H = torch.conj(Y)
    return Y, Y_H, weights, theta
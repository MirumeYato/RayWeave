import numpy as np
import healpy as hp
import torch

def get_lanczos_filter_hp(lmax):
    """Lanczos filter weights for the healpy alm ordering."""
    l, _ = hp.Alm.getlm(lmax)
    sigma = np.ones_like(l, dtype=float)
    non_zero = l > 0
    x = np.pi * l[non_zero] / lmax
    sigma[non_zero] = np.sin(x) / x
    return sigma

def get_lanczos_filter_custom(lmax):
    """Lanczos filter weights for the custom (L+1)^2 flat alm ordering."""
    l_vals = np.array([l for l in range(lmax + 1) for _ in range(-l, l + 1)])
    sigma = np.ones(len(l_vals), dtype=float)
    non_zero = l_vals > 0
    x = np.pi * l_vals[non_zero] / lmax
    sigma[non_zero] = np.sin(x) / x
    return torch.from_numpy(sigma) # The output shape is ((lmax + 1)**2,).
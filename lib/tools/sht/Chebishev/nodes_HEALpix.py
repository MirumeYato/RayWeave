from ...sht import compute_SH

import torch
import numpy as np
import healpy as hp

def get_nodes_weights(nside = 5)-> tuple:
    """
    Docstring for get_nodes_weights
    
    :param nside: healpix nside parameter, that needs for calculation of pixel number
    :type nside: int
    :return: grid like arrays of theta and phi (theta ∈ [0,π], phi ∈ [0,2π)). Also returns weights for Chebishev method of integration
    :rtype: tuple
    """
    pix_num = hp.nside2npix(nside)
    print(f"[DEBUG]: number of pixel is {pix_num}")
    weights = (4.0 * np.pi) / pix_num
    # than we should choose correct greed (kinda uniform). But how to do this correctly is discussible...
    theta, phi = hp.pix2ang(nside, np.arange(pix_num)) 

    return theta, phi, weights

def get_spherical_harmonics(nside, Lmax, device, dtype=torch.complex128):
    theta, phi, weights = get_nodes_weights(nside = nside)

    Y = compute_SH(theta=theta, phi=phi, L_max=Lmax, device=device, dtype=dtype)
    Y_H = torch.conj(Y)
    return Y, Y_H, weights, theta

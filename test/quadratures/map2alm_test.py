# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..', '..'))
print(PATH)
sys.path.insert(0, PATH)
#===============================#
# Retry with fixed-size frames (avoid bbox_inches='tight' which changed image sizes).
from lib.tools import map_HenyeyGreenstein, alm_HenyeyGreenstein
from lib.tools.func_HenyeyGreenshtein import alm_diagonalize as alm_diag
from sht import map2alm
from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign

import matplotlib.pyplot as plt

import numpy as np
import torch
from math import pi
import healpy as hp

OUTPUT = os.path.abspath(os.path.join(PATH, 'output', 'test', 'sht_debug'))
os.makedirs(OUTPUT, exist_ok=True)


if __name__ == "__main__":

    nside, Lmax = 5, 20
    g = 0.9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Quadrature Methods
    qTdesign = QuadratureTdesign(240, device=device)
    qHEALPix = QuadratureHEALPix(nside, device=device)
    # Spherical Harmonics (grids are different!)
    sh_hp, shH_hp = qHEALPix.get_spherical_harmonics(Lmax) 
    sh_td, shH_td = qTdesign.get_spherical_harmonics(Lmax)
    # Weights (grids are different!) 
    w_hp = qHEALPix.get_weights()
    w_td = qTdesign.get_weights()
    # Thetas (grids are different!)
    theta_hp, __ = qHEALPix.get_nodes_angles()
    theta_td, __ = qTdesign.get_nodes_angles()
    # Meaning of HenyeyGreenshtein function in nodes points
    map_true_hp = map_HenyeyGreenstein(g, theta_hp)
    map_true_td = map_HenyeyGreenstein(g, theta_td)
    # Meaning of HenyeyGreenshtein shperical coefitients
    alm_true  = alm_HenyeyGreenstein(g, Lmax, device=device).detach().cpu().numpy()

    # Heal pix standart method
    alm_orig_hp = hp.map2alm(map_true_hp.detach().cpu().numpy(), lmax=Lmax, mmax=0, iter = 0, pol = False)
    # Our custom
    alm_hp = map2alm(map_true_hp, Y_H=shH_hp, weights=w_hp)
    alm_td = map2alm(map_true_td, Y_H=shH_td, weights=w_td)

    # Transform to alm_true shape
    alm_hp = alm_diag(alm_hp.detach().cpu().numpy(), L_max=Lmax)
    alm_td = alm_diag(alm_td.detach().cpu().numpy(), L_max=Lmax)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.abs((alm_orig_hp - alm_true)/alm_true),
                linewidth=5, label=r"$\frac{|a_{\ell m}^{OrigHP} - a_{\ell m}^{true}|}{a_{\ell m}^{true}}$", color='C0')

    # If we want to check custom method (But we should normalize on)
    plt.plot(np.abs((alm_hp - alm_true)/alm_true),
                linewidth=2, linestyle="--", label=r"$\frac{|a_{\ell m}^{CustHP} - a_{\ell m}^{true}|}{a_{\ell m}^{true}}$", color='C1')
    plt.plot(np.abs((alm_td - alm_true)/alm_true),
                linewidth=2, linestyle="--", label=r"$\frac{|a_{\ell m}^{tdesign} - a_{\ell m}^{true}|}{a_{\ell m}^{true}}$", color='C2')
    plt.title("MAE of alm reconstruction from map true", fontsize=14)
    plt.xlabel(r"Harmonic index $i$", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    # plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_map_true2alm.png"), dpi=150)
    plt.close()
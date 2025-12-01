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
from lib.tools import map_HenyeyGreenstein
from sht import map2map_xn
from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign

import matplotlib.pyplot as plt

import numpy as np
import torch
from math import pi
import healpy as hp

OUTPUT = os.path.abspath(os.path.join(PATH, 'output', 'test', 'sht_debug'))
os.makedirs(OUTPUT, exist_ok=True)


def hp_loop(map_hp, n, Lmax, nside):
    for _ in range(n):
        alm_hp = hp.map2alm(map_hp, lmax=Lmax, mmax=0, iter = 0, pol = False)
        map_hp = hp.alm2map(alm_hp, nside=nside, lmax=Lmax, mmax=0, pol = False)
    return map_hp


if __name__ == "__main__":

    N = 10
    nside, Lmax = 5, 11
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
    
    # Heal pix standart methods
    map_orig_hp = hp_loop(map_true_hp.detach().cpu().numpy(), N, Lmax, nside)
    # Our custom
    map_hp = map2map_xn(map_true_hp.to(torch.complex128), n=N, Y_H=shH_hp, Y=sh_hp, weights=w_hp).detach().cpu().numpy()
    map_td = map2map_xn(map_true_td.to(torch.complex128), n=N, Y_H=shH_td, Y=sh_td, weights=w_td).detach().cpu().numpy()

    map_true_hp = map_true_hp.detach().cpu().numpy()
    map_true_td = map_true_td.detach().cpu().numpy()

    # Accuracy Plot
    ### Get map (fHenyeyGreenshtein values) from alm (g**l)
    # in our collision step we a planning to  get alm form map, 
    # do some collision operation on alm for getting a^{star}_lm (alm on moment dt/2 later)
    # than we will get it back into map. 
    # And we will repeat it many many times. So here inportant to check
    # commutative error of each such tasport from map to alm and vice versa

    plt.figure(figsize=(8, 5))
    plt.plot(np.abs((map_orig_hp - map_true_hp)/map_true_hp), linewidth=5, label=r"$\frac{|f^{OrigHP} - f^{true}|}{f^{true}}$", color='C0')

    # If we want to check custom method (But we should normalize on)
    plt.plot(np.abs((map_hp - map_true_hp)/map_true_hp),
                linewidth=2, linestyle="--", label=r"$\frac{|f^{CustHP} - f^{true}|}{f^{true}}$", color='C1')
    plt.plot(np.abs((map_td - map_true_td)/map_true_td),
                linewidth=2, linestyle="--", label=r"$\frac{|f^{tdesign} - f^{true}|}{f^{true}}$", color='C2')
    plt.title(f"Round-trip consistency: map → alm → map ({N} times)", fontsize=14)
    plt.xlabel("Pixel index", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"Mae_comparison_map2map_x{N}.png"), dpi=150)
    plt.close()
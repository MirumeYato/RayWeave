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
from lib.tools.sht import compute_SH, map_HenyeyGreenshtein, alm_HenyeyGreenshtein, map2alm
from lib.tools.sht.Chebishev.nodes_HEALpix import get_spherical_harmonics as sht_hp
from lib.tools.sht.Chebishev.nodes_tdesign import get_spherical_harmonics as sht_td

import matplotlib.pyplot as plt

import numpy as np
import torch
from math import pi
import healpy as hp

OUTPUT = os.path.abspath(os.path.join(PATH, 'output', 'test', 'sht_debug'))
os.makedirs(OUTPUT, exist_ok=True)


if __name__ == "__main__":

    nside, Lmax = 5, 21
    g = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sh_hp, shH_hp, w_hp, theta_hp = sht_hp(nside, Lmax, device)
    sh_td, shH_td, w_td, theta_td = sht_td(Lmax, device)

    map_true_hp = map_HenyeyGreenshtein(g, theta_hp)
    map_true_td = map_HenyeyGreenshtein(g, theta_td)
    alm_true  = alm_HenyeyGreenshtein(g, Lmax)

    alm_orig_hp = hp.map2alm(map_true_hp, lmax=Lmax, mmax=0, iter = 0, pol = False)
    alm_hp = map2alm(map_true_hp, L_max=Lmax, device=device, Y_H=shH_hp, weights=w_hp)
    alm_td = map2alm(map_true_td, L_max=Lmax, device=device, Y_H=shH_td, weights=w_td)

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
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_map_true2alm.png"), dpi=150)
    plt.close()
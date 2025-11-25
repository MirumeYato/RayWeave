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
from lib.tools.sht import compute_SH, map_HenyeyGreenshtein, alm_HenyeyGreenshtein, alm2map
from lib.tools.sht.Chebishev.nodes_HEALpix import get_spherical_harmonics as sht_hp
from lib.tools.sht.Chebishev.nodes_tdesign import get_spherical_harmonics as sht_td

import matplotlib.pyplot as plt

import numpy as np
import torch
from math import pi
import healpy as hp

OUTPUT = os.path.abspath(os.path.join(PATH, 'output', 'test', 'sht_debug'))
os.makedirs(OUTPUT, exist_ok=True)

def alm_transform(alm_no_m, L_max, lenth)-> np.ndarray:
    alm = np.zeros(lenth)
    i=0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # print(l,m)
            if m == 0: alm[i] = np.real(alm_no_m[l])
            else : alm[i] = 0
            i+=1
    return alm

if __name__ == "__main__":

    nside, Lmax = 5, 21
    g = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sh_hp, shH_hp, w_hp, theta_hp = sht_hp(nside, Lmax, device)
    sh_td, shH_td, w_td, theta_td = sht_td(Lmax, device)

    map_true_hp = map_HenyeyGreenshtein(g, theta_hp)
    map_true_td = map_HenyeyGreenshtein(g, theta_td)
    alm_true  = alm_HenyeyGreenshtein(g, Lmax)

    alm_true_hp = alm_transform(alm_true, L_max=Lmax, lenth=sh_hp.shape[1])
    alm_true_td = alm_transform(alm_true, L_max=Lmax, lenth=sh_td.shape[1])
    

    map_orig_hp = hp.alm2map(alm_true.astype(np.complex128), nside=nside, lmax=Lmax, mmax=0, pol = False)
    map_hp = alm2map(alm_true_hp, device = device, Y=sh_hp)
    map_td = alm2map(alm_true_td, device = device, Y=sh_td)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.abs((map_orig_hp - map_true_hp)/map_true_hp), linewidth=5, label=r"$\frac{|f^{OrigHP} - f^{true}|}{f^{true}}$", color='C0')

    # If we want to check custom method (But we should normalize on)
    plt.plot(np.abs((map_hp - map_true_hp)/map_true_hp),
                linewidth=2, linestyle="--", label=r"$\frac{|f^{CustHP} - f^{true}|}{f^{true}}$", color='C1')
    plt.plot(np.abs((map_td - map_true_td)/map_true_td),
                linewidth=2, linestyle="--", label=r"$\frac{|f^{tdesign} - f^{true}|}{f^{true}}$", color='C2')
    plt.title("MAE of map reconstruction from alm true", fontsize=14)
    plt.xlabel("Pixel index", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_alm_true2map.png"), dpi=150)
    plt.close()
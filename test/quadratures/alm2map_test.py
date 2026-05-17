import os
import pytest
import matplotlib.pyplot as plt
import numpy as np
import torch
import healpy as hp

from lib.tools import map_HenyeyGreenstein, alm_HenyeyGreenstein
from lib.tools.func_HenyeyGreenstein import alm_transform_torch as alm_transform
from sht import alm2map
from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign

OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'test', 'quadratures'))
os.makedirs(OUTPUT, exist_ok=True)

@pytest.fixture(scope="module")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_alm2map(device):
    nside, Lmax = 5, 100
    g = 0.9

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
    
    map_true_hp = map_HenyeyGreenstein(g, theta_hp).detach().cpu().numpy()
    map_true_td = map_HenyeyGreenstein(g, theta_td).detach().cpu().numpy()
    # Meaning of HenyeyGreenshtein shperical coefitients
    
    alm_true  = alm_HenyeyGreenstein(g, Lmax, device=device)

    alm_true_hp = alm_transform(alm_true, L_max=Lmax, length=sh_hp.shape[1])
    alm_true_td = alm_transform(alm_true, L_max=Lmax, length=sh_td.shape[1])    

    map_orig_hp = hp.alm2map(alm_true.detach().cpu().numpy().astype(np.complex128), nside=nside, lmax=Lmax, mmax=0, pol=False)
    
    map_hp = alm2map(alm_true_hp, Y=sh_hp, dtype=torch.complex128).detach().cpu().numpy()
    map_td = alm2map(alm_true_td, Y=sh_td, dtype=torch.complex128).detach().cpu().numpy()

    # Assertion logic to ensure errors are within bounds
    mae_orig_hp = np.max(np.abs((map_orig_hp - map_true_hp) / map_true_hp))
    mae_hp = np.max(np.abs((map_hp - map_true_hp) / map_true_hp))
    mae_td = np.max(np.abs((map_td - map_true_td) / map_true_td))
    
    assert mae_orig_hp < 1e-1, f"Orig HEALPix error exceeds threshold: {mae_orig_hp}"
    assert mae_hp < 1e-1, f"Custom HEALPix error exceeds threshold: {mae_hp}"
    assert mae_td < 1e-1, f"Custom TDesign error exceeds threshold: {mae_td}"

    plt.figure(figsize=(8, 5))
    plt.plot(np.abs((map_orig_hp - map_true_hp)/map_true_hp), linewidth=5, label=r"$\frac{|f^{OrigHP} - f^{true}|}{f^{true}}$", color='C0')
    plt.plot(np.abs((map_hp - map_true_hp)/map_true_hp), linewidth=2, linestyle="--", label=r"$\frac{|f^{CustHP} - f^{true}|}{f^{true}}$", color='C1')
    plt.plot(np.abs((map_td - map_true_td)/map_true_td), linewidth=2, linestyle="--", label=r"$\frac{|f^{tdesign} - f^{true}|}{f^{true}}$", color='C2')
    
    plt.title("MAE of map reconstruction from alm true", fontsize=14)
    plt.xlabel("Pixel index", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_alm_true2map.png"), dpi=150)
    plt.close()
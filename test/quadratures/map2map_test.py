import os
import pytest
import numpy as np
import torch
import healpy as hp
import matplotlib.pyplot as plt

from lib.tools import map_HenyeyGreenstein
from sht import map2map_xn
from lib.grid.Quadrature.Chebishev.QuadratureHEALPix import QuadratureHEALPix
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign

OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'test', 'quadratures'))
os.makedirs(OUTPUT, exist_ok=True)

from lib.tools.func_Lanczos import get_lanczos_filter_hp, get_lanczos_filter_custom

# ─────────────────────── HEALPix simulation loop ────────────────────────────

def run_healpix_loop(initial_map, nside, lmax, iterations, method):
    """Round-trip Map->Alm->Map using native healpy, with optional Lanczos."""
    current_map = initial_map.copy()
    lanczos_filter = get_lanczos_filter_hp(lmax) if method == "Lanczos" else None
    initial_sum = np.sum(initial_map)
    metrics = {"mean_error": [], "energy_error": [], "min_value": []}

    for _ in range(iterations):
        alm = hp.map2alm(current_map, lmax=lmax, iter=1, pol=False)
        if method == "Lanczos":
            alm *= lanczos_filter
        current_map = hp.alm2map(alm, nside, lmax=lmax)

        metrics["mean_error"].append(np.mean(np.abs(current_map - initial_map)))
        metrics["energy_error"].append(1.0 - (np.sum(current_map) / initial_sum))
        metrics["min_value"].append(np.min(current_map))

    return metrics


# ─────────────────────── T-Design simulation loop ───────────────────────────

def run_tdesign_loop(initial_map_torch, quadrature, lmax, iterations, method, device):
    """Round-trip Map->Alm->Map using custom torch SHT, with optional Lanczos."""
    sh, sh_H = quadrature.get_spherical_harmonics(lmax, dtype=torch.complex128)
    
    # weights may be a plain float/np.float64 scalar (T-Design Chebyshev rule)
    raw_w = quadrature.get_weights()
    if isinstance(raw_w, (int, float, np.floating)):
        weights_t = torch.full((quadrature.num_bins,), float(raw_w), device=device, dtype=torch.complex128)
    else:
        weights_t = raw_w.to(device=device, dtype=torch.complex128)

    lanczos_filter = None
    if method == "Lanczos":
        lanczos_filter = get_lanczos_filter_custom(lmax).to(device=device, dtype=torch.complex128)

    current_map = initial_map_torch.to(device=device, dtype=torch.complex128)
    initial_sum = current_map.real.sum().item()
    initial_np = current_map.real.cpu().numpy()
    metrics = {"mean_error": [], "energy_error": [], "min_value": []}

    for _ in range(iterations):
        alm = torch.matmul(current_map * weights_t, sh_H)   # map -> alm
        if method == "Lanczos":
            alm = alm * lanczos_filter
        current_map = torch.einsum('p,qp->q', alm, sh).real  # alm -> map

        current_np = current_map.cpu().numpy()
        metrics["mean_error"].append(np.mean(np.abs(current_np - initial_np)))
        metrics["energy_error"].append(1.0 - (current_np.sum() / initial_sum))
        metrics["min_value"].append(current_np.min())

    return metrics


# ─────────────────────── shared plot helper ──────────────────────────────────

def _save_metrics_plot(results, iterations, title_prefix, filename):
    methods = ["Standard", "Lanczos"]
    sources = ["Delta", "HG"]
    metric_keys   = ["mean_error", "energy_error", "min_value"]
    metric_titles = ["Mean Absolute Error", "Energy Error (1 − sum/sum₀)", "Minimum Intensity"]
    colors = {"Standard": "red", "Lanczos": "blue"}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title_prefix, fontsize=15, fontweight="bold")

    for col, source in enumerate(sources):
        axes[0, col].set_title(f"Source: {source}", fontsize=13, fontweight="bold")
        for row, (metric, title) in enumerate(zip(metric_keys, metric_titles)):
            ax = axes[row, col]
            for method in methods:
                data = results[(source, method)][metric]
                ax.plot(range(1, iterations + 1), data,
                        color=colors[method], label=method, linewidth=2, alpha=0.8)
            ax.set_ylabel(title)
            if row == 0:
                ax.set_yscale('log')
            ax.set_xlabel("Projection cycles")
            ax.grid(True, linestyle=':', alpha=0.7)
            if row == 0 and col == 0:
                ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, filename), dpi=150)
    plt.close()


# ═════════════════════════════ tests ═════════════════════════════════════════

def test_map2map_healpix(tmp_path):
    """
    Round-trip SHT accuracy for HEALPix (nside=9, lmax=25).
    Standard vs Lanczos, Delta vs HG sources.
    """
    g = 0.9
    iterations = 10
    nside = 9
    lmax = 25

    npix = hp.nside2npix(nside)
    pixel_area = hp.nside2pixarea(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    I_delta = np.zeros(npix)
    I_delta[0] = 1.0 / pixel_area

    cos_theta = np.cos(theta)
    I_hg = (1.0 / (4 * np.pi)) * (1 - g**2) / (1 + g**2 - 2 * g * cos_theta)**1.5

    methods = ["Standard", "Lanczos"]
    results = {}
    for method in methods:
        results[("Delta", method)] = run_healpix_loop(I_delta, nside, lmax, iterations, method)
        results[("HG",    method)] = run_healpix_loop(I_hg,    nside, lmax, iterations, method)

        assert max(np.abs(results[("Delta", method)]["energy_error"])) < 0.5, \
            f"HEALPix energy diverged (Delta, {method})"
        assert max(np.abs(results[("HG", method)]["energy_error"])) < 0.5, \
            f"HEALPix energy diverged (HG, {method})"

    _save_metrics_plot(
        results, iterations,
        title_prefix=f"HEALPix nside={nside}, lmax={lmax}",
        filename="Mae_comparison_map2map_healpix.png"
    )


def test_map2map_tdesign():
    """
    Round-trip SHT accuracy for T-Design (n_size=1014, lmax=22).
    Standard vs Lanczos, Delta vs HG sources.
    """
    g = 0.9
    iterations = 10
    n_size = 1014
    lmax = 22
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    quadrature = QuadratureTdesign(n_size=n_size, device=device)
    theta, _ = quadrature.get_nodes_angles()

    # Delta source: all weight on a single pixel
    I_delta = torch.zeros(n_size, device=device)
    I_delta[0] = float(n_size) / (4.0 * np.pi)   # normalised to unit integral

    # HG source evaluated at each node
    I_hg = map_HenyeyGreenstein(g, theta).to(torch.float64)

    methods = ["Standard", "Lanczos"]
    results = {}
    for method in methods:
        results[("Delta", method)] = run_tdesign_loop(I_delta, quadrature, lmax, iterations, method, device)
        results[("HG",    method)] = run_tdesign_loop(I_hg,    quadrature, lmax, iterations, method, device)

        assert max(np.abs(results[("Delta", method)]["energy_error"])) < 0.5, \
            f"T-Design energy diverged (Delta, {method})"
        assert max(np.abs(results[("HG", method)]["energy_error"])) < 0.5, \
            f"T-Design energy diverged (HG, {method})"

    _save_metrics_plot(
        results, iterations,
        title_prefix=f"T-Design n_size={n_size}, lmax={lmax}",
        filename="Mae_comparison_map2map_tdesign.png"
    )
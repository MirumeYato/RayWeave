# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', '..'))
# print(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
#===============================#

import pytest
import numpy as np
import torch
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from matplotlib.colors import Normalize

# --- 2. Imports from your library ---
try:
    from lib.tools.sht import map_HenyeyGreenshtein, map2map_xn
    from lib.tools.sht.Chebishev.nodes_HEALpix import get_spherical_harmonics as sht_hp
    from lib.tools.sht.Chebishev.nodes_tdesign import get_spherical_harmonics as sht_td
except ImportError:
    pytest.fail("Could not import 'lib.tools'. Check your python path or directory structure.")

# --- 3. Configuration & Constants ---
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'test', 'sht_heatmap')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ITER = 1
NSIDE = 5
ERROR_THRESHOLD = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 4. Helper Functions ---

def compute_relative_error(map_calc, map_true):
    """Calculates the max relative error: max(|f_calc - f_true| / f_true)."""
    # Avoid division by zero if map_true has exact zeros (unlikely in HG but safe to handle)
    mask = map_true != 0
    diff = np.abs(map_calc[mask] - map_true[mask])
    rel_err = diff / np.abs(map_true[mask])
    return np.max(rel_err)

def run_orig_hp_method(map_true, n, lmax, nside):
    """Original Healpy method."""
    map_curr = map_true.copy()
    for _ in range(n):
        alm = hp.map2alm(map_curr, lmax=lmax, mmax=0, iter=0, pol=False)
        map_curr = hp.alm2map(alm, nside=nside, lmax=lmax, mmax=0, pol=False)
    return map_curr

def run_custom_hp_method(map_true, n, lmax, nside, device):
    """Custom HP method using Chebyshev nodes."""
    sh_hp, shH_hp, w_hp, theta_hp = sht_hp(nside, lmax, device)
    # Re-generate map on these nodes to ensure alignment
    map_input = map_HenyeyGreenshtein(0.0, theta_hp) # Placeholder g, overwritten in main loop usually
    # Note: The 'map_true' passed in is used, but we need the specific Y/weights
    return map2map_xn(map_true, n=n, device=device, Y_H=shH_hp, Y=sh_hp, weights=w_hp)

def run_tdesign_method(map_true, n, lmax, device):
    """T-Design method."""
    sh_td, shH_td, w_td, theta_td = sht_td(lmax, device)
    return map2map_xn(map_true, n=n, device=device, Y_H=shH_td, Y=sh_td, weights=w_td)


# --- 5. The Pytest Suite ---

def test_sht_accuracy_landscape():
    """
    Scans L_max and g parameters. 
    Stops calculating in 'g' direction if error exceeds threshold.
    Generates heatmaps for all 3 methods.
    """
    
    # Define Search Space
    # L_max: 2 to 20
    l_values = np.arange(2, 21, 1)
    # g: 0 to 0.95 (We assume symmetry and test positive g for simplicity in this heatmap)
    g_values = np.arange(0, 0.96, 0.05)
    
    methods = ["Original_HP", "Custom_HP", "T_Design"]
    
    # Storage for results: Dict[Method, Matrix(g, l)]
    # Initialize with NaNs to represent "skipped" areas
    heatmaps = {m: np.full((len(g_values), len(l_values)), np.nan) for m in methods}

    print(f"\nStarting Adaptive Grid Search (Threshold: {ERROR_THRESHOLD})...")

    # --- Main Loop ---
    for l_idx, lmax in enumerate(l_values):
        
        # Pre-compute SH bases for this Lmax to save time (if applicable)
        # Note: In strict unit testing we might re-init per call, but for speed we do it here
        sh_hp, shH_hp, w_hp, theta_hp = sht_hp(NSIDE, int(lmax), DEVICE)
        sh_td, shH_td, w_td, theta_td = sht_td(int(lmax), DEVICE)

        for method_name in methods:
            # For each method, iterate g starting from 0 (easiest) to 1 (hardest)
            for g_idx, g in enumerate(g_values):
                
                # 1. Generate Ground Truth
                if "T_Design" in method_name:
                    map_true = map_HenyeyGreenshtein(g, theta_td)
                else:
                    map_true = map_HenyeyGreenshtein(g, theta_hp)
                
                # 2. Run Round-Trip
                try:
                    if method_name == "Original_HP":
                        # Healpy requires numpy arrays, not torch
                        if isinstance(map_true, torch.Tensor): map_true = map_true.cpu().numpy()
                        map_final = run_orig_hp_method(map_true, N_ITER, int(lmax), NSIDE)
                        
                    elif method_name == "Custom_HP":
                        map_final = map2map_xn(map_true, n=N_ITER, device=DEVICE, Y_H=shH_hp, Y=sh_hp, weights=w_hp)
                        
                    elif method_name == "T_Design":
                        map_final = map2map_xn(map_true, n=N_ITER, device=DEVICE, Y_H=shH_td, Y=sh_td, weights=w_td)
                    
                    # 3. Calculate Error
                    # Ensure both are numpy for calc
                    if isinstance(map_final, torch.Tensor): map_final = map_final.cpu().numpy()
                    if isinstance(map_true, torch.Tensor): map_true = map_true.cpu().numpy()
                    
                    error = compute_relative_error(map_final, map_true)
                    heatmaps[method_name][g_idx, l_idx] = error
                    
                    # 4. Adaptive Pruning
                    # If error > 0.05, we assume higher g will also fail. Stop this g-loop.
                    if error > ERROR_THRESHOLD:
                        break 
                        
                except Exception as e:
                    print(f"Failed at {method_name} L={lmax}, g={g}: {e}")
                    heatmaps[method_name][g_idx, l_idx] = np.inf # Mark as failure
                    break

    # --- 6. Plotting ---
    for method_name, data in heatmaps.items():
        plt.figure(figsize=(10, 8))
        
        # Mask NaNs (skipped values)
        masked_data = np.ma.masked_invalid(data)
        
        # Define the minimum error value for the LogNorm. 
        # Using a very small positive number (e.g., 1e-10) is necessary 
        # because log(0) is undefined, and log(negative) is invalid.
        # Define maximum limit based on your threshold.
        vmin_log = 1e-10
        vmax_log = ERROR_THRESHOLD * 2.0 # Set max slightly higher than the threshold

        # 1. Use LogNorm for the color mapping
        norm = LogNorm(vmin=vmin_log, vmax=vmax_log)

        # 2. Plot using the new norm
        plt.imshow(masked_data, aspect='auto', origin='lower', 
                   cmap='viridis_r', # Reversed viridis: Yellow (bad) to Purple (good)
                   extent=[l_values[0], l_values[-1], g_values[0], g_values[-1]],
                   norm=norm) 
        
        # 3. Add a color bar that displays ticks in powers of 10
        cbar = plt.colorbar(label=r'Max Relative Error $\log_{10}(\frac{|f_{calc} - f_{true}|}{f_{true}})$')
        cbar.ax.yaxis.label.set_size(18)
        
        # Customize ticks to be visually appealing powers of 10
        ticks = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.2]
        # Filter ticks to only show those within the plotted range
        display_ticks = [t for t in ticks if vmin_log <= t <= vmax_log]
        cbar.set_ticks(display_ticks)
        cbar.set_ticklabels([f'{t:.0e}' for t in display_ticks])


        plt.title(f"Accuracy Heatmap: {method_name}\n(N={N_ITER}, nside={NSIDE})", fontsize=14)
        plt.xlabel(r"Orbital Momentum $L_{max}$", fontsize=12)
        plt.ylabel(r"Anisotropy $g$", fontsize=12)
        
        # Mark the "Safe Zone" (Error < 0.05)
        plt.contour(data, levels=[ERROR_THRESHOLD], 
                   extent=[l_values[0], l_values[-1], g_values[0], g_values[-1]],
                   colors='red', linestyles='dashed', linewidths=2)

        save_path = os.path.join(OUTPUT_DIR, f"heatmap_{method_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

    # --- 7. Assertions ---
    # Fail if the 'easiest' case (g=0, moderate L) fails for any method.
    # This ensures the pipeline isn't completely broken.
    mid_l_idx = len(l_values) // 2
    for method_name in methods:
        err_easy = heatmaps[method_name][0, mid_l_idx] # g=0
        assert err_easy < ERROR_THRESHOLD, \
            f"{method_name} failed basic accuracy test at g=0, L={l_values[mid_l_idx]}. Error: {err_easy}"

if __name__ == "__main__":
    # Allow running this file directly without typing 'pytest'
    sys.exit(pytest.main(["-v", __file__]))
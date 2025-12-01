# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#
# Retry with fixed-size frames (avoid bbox_inches='tight' which changed image sizes).

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import numpy as np
import torch
from math import pi
import healpy as hp

from lib.grid.Angle import Angle3D
from tqdm import trange, tqdm

OUTPUT = os.path.abspath(os.path.join(PATH, 'output'))

# ---- helpers ----

def _expand_f_l_to_lm(f_l: torch.Tensor, idx_l: torch.Tensor) -> torch.Tensor:
    """
    Map f_l (L_max+1,) to f_lm (P,), repeating each f_l[l] for (2l+1) entries.
    """
    # ensure on same device/dtype compat
    return f_l[idx_l]


@torch.no_grad()
def sph_filter_per_voxel_vectorized(
    map_full: torch.Tensor,         # (B,1,N,N,N) with B = hp.nside2npix(nside)
    f_lm: torch.Tensor,              # # (P,) real or complex (applied to alms)
    nside: int,
    L_max: int,
    Y: torch.Tensor ,  # optional precomputed (Q,P) complex
    Y_H: torch.Tensor ,  # optional precomputed (Q,P) complex
    use_real_output: bool = True,
) -> torch.Tensor:
    """
    Vectorized SHT filtering via Y-basis projection:
      A_p(v)   = sum_q I_q(v) * conj(Y_qp)
      A'_p(v)  = f_p * A_p(v)
      I*_q(v)  = sum_p A'_p(v) * Y_qp
    where v indexes voxels (i,j,k flattened).

    Returns: (B,1,N,N,N) on the same device/dtype as map_full (real if use_real_output).
    """
    assert map_full.ndim == 5 and map_full.size(1) == 1, "Expected map_full of shape (B,1,N,N,N)."
    device = map_full.device

    I = map_full.squeeze(1)                     # (Q, N, N, N)
    Q = I.size(0)
    norm = np.sqrt((4.0 * np.pi) / Q)

    # Ensure complex math for the harmonic projections
    # Use a complex dtype for Y and for the intermediate coefficients
    complex_dtype = torch.complex64 if map_full.dtype in (torch.float16, torch.float32) else torch.complex128
    Y = Y.to(device=device, dtype=complex_dtype)              # (Q, P)
    Y_H = Y_H.to(device=device, dtype=complex_dtype)              # (Q, P)
    I = I.to(device=device, dtype=complex_dtype)
    f_lm = f_lm.to(device=device, dtype=complex_dtype)

    # === Projection to a_lm across all voxels at once ===
    # A_pv = sum_q  I_qv * conj(Y_qp)
    # einsum shapes: (q) , (qv) , (qp) -> (pv)
    A_pv = torch.einsum('qijk,qp->pijk', I, Y_H) * norm        # (P, V), complex

    # === Spectral filtering: multiply by f_lm per (l,m) ===
    # A_pv = torch.einsum('pijk,p->pijk', A_pv, f_lm)  #A_pv * f_lm[:, None, None, None]                              # (P, V)

    # === Reconstruct ===
    # I*_qv = sum_p A_pv * Y_qp
    I_star_qv = torch.einsum('pijk,qp->qijk', A_pv, Y) / norm          # (Q, V), complex

    # Back to (Q,N,N,N)
    I_star = I_star_qv#I_star_qv.reshape(Q, N, N, N)

    # Output dtype/real part control
    if use_real_output:
        I_star = I_star.real
        I_star = I_star.to(dtype=map_full.dtype)
    else:
        # keep complex; if caller wants complex channel, they'll handle it
        pass

    return I_star.unsqueeze(1)                                # (B,1,N,N,N)

# ---- parameters ----
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = 10
    speed = 1.0
    dt = 1.
    n_steps = 5
    L_max = 2*3-2
    g = 0.6
    g_l = np.array([g**i / np.sqrt(4 * np.pi / (2 * i + 1)) for i in range(L_max + 1)])
    lambda_l_np = speed * (-1.0 + g_l)                        # shape [L_max+1]
    f_l_np = np.exp(lambda_l_np * (dt / 2.0))                          # shape [L_max+1]
    f_l = torch.from_numpy(f_l_np).to(device=device, dtype=torch.float32)
    nside = 3

    Angle = Angle3D(healpix_nside = nside)
    B = Angle.num_channels

    print("Precompute Shperical Harmonics")
    print("n pixels: ", B)
    Y, idx_l = _precompute_Ylm_healpix(nside, L_max, device=device)
    Y_H = torch.conj(Y)  
    f_lm = _expand_f_l_to_lm(f_l, idx_l) # # (P,)
    print("shape of shperical harmonics array: ", Y.shape)

    # Pre-calc grid shifting
    # Some gridy tools if needed.

    # Intencity (main grid)
    # should be replaces with Source class INI
    f_batch = torch.zeros((B, 1, N, N, N), dtype=torch.float32, device=device)
    c = N // 2
    c2 = B // 2 + 5
    f_batch[0, 0, c, c, c] = 1.0 # point-like source in the middl
    f_batch[c2, 0, c, c, c+3] = 1.0 # point-like source in the middl

    # Pre-cycle sub-functions
    # Needed refactoring into plotting tools
    frames = []

    # Pre-create a consistent figure/axes to keep identical canvas size
    fig, ax = plt.subplots(figsize=(5, 5))

    w0 = f_batch.sum().detach().cpu().numpy()

    # Collision test

    print("Start prop")
    frames = []
    os.makedirs(OUTPUT, exist_ok=True)

    for step in trange(0, n_steps + 1):
        # f_batch is torch tensor on CUDA: [B, 1, N, N, N]
        f_dirs = f_batch

        # Extract two angular maps at given voxels -> move to CPU NumPy 
        # (as example we want look only on them)
        map1_np = f_dirs[:, 0, c, c, c].detach().cpu().numpy()           # shape [B]
        map2_np = f_dirs[:, 0, c, c, c + 3].detach().cpu().numpy()      # shape [B]
        vol = f_dirs.sum().detach().cpu().numpy()

        # ---- two mollviews in one file ----
        fig = plt.figure(figsize=(10, 9))

        print("Min, Max")
        print(np.min(map1_np),np.max(map1_np))
        print(np.min(map2_np),np.max(map1_np))

        # Upper map
        hp.mollview(
            map1_np,
            fig=fig.number,
            sub=(2, 1, 1),
            title=f"Step {step} — voxel (c,c,c)",
            norm="hist",
            cbar=True,
            min=0,
            max=1
        )
        # Lower map
        hp.mollview(
            map2_np,
            fig=fig.number,
            sub=(2, 1, 2),
            title=f"Step {step} — voxel (c,c,c+10)",
            norm="hist",
            cbar=True,
            min=0,
            max=1
        )

        # Add global text annotation (use figure coordinates instead of axes)
        sum_ratio = float(vol.sum()) / float(w0)
        fig.text(
            0.02, 0.97,
            rf"$\sum_i \omega_i(t) / \sum_i \omega_i(0) = {sum_ratio:.2e}$",
            color="black",
            fontsize=11,
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3),
        )

        # Save frame
        frame_path = os.path.join(OUTPUT, f"flux_step_{step:02d}.png")
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

        # ---- alm update that depends only on l (no loops over m) ----

        # write back into torch on original device/dtype
        f_batch = sph_filter_per_voxel_vectorized(f_dirs, f_lm, nside, L_max, Y, Y_H) # shape (B,1,N,N,N) sph_filter_per_voxel

    # build GIF
    gif_path = os.path.join(OUTPUT, "photon_flux_propagation.gif")
    imageio.mimsave(gif_path, frames, duration=0.6)
    print("Finish")
    print(gif_path)

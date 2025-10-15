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
from matplotlib import cm

import numpy as np
import torch
import torch.nn.functional as F
from math import pi
import healpy as hp

from lib.physics.Angles import Angle3D
from lib.tools import ptm
from lib.physics.grid_tools import prepare_grid

OUTPUT = os.path.abspath(os.path.join(PATH, 'output'))

# ---- helpers ----

def make_delta_field(N: int, device):
    f = torch.zeros((1, 1, N, N, N), dtype=torch.float32, device=device)
    c = N // 2
    f[0, 0, c, c, c] = 1.0 # point-like source in the middle
    # f[0, 0, c, c, c+10] = 1.0 # point-like source in the middle
    return f

# ---- parameters ----
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = 100
    speed = 1.0
    dt = 1.0
    n_steps = 20

    Angle = Angle3D(healpix_nside = 3)
    B = Angle.channels

    # Pre-calc grid shifting
    # Some gridy tools.
    scale = 10.0 / (N - 1) * speed * dt
    dirs = Angle.get_all_vecs() # [B,3]
    shifts = torch.from_numpy(dirs.astype(np.float32)).to(device) * scale # [B,3]    
    grids = prepare_grid(shifts, N, device)

    # Intencity (main grid)
    # should be replaces with Source class INI
    # this example correspond to isotropic angle streaming.
    f0 = make_delta_field(N, device) # Intencity(1, 1, N, N, N) like defining sources in space
    f_batch = f0.expand(B, -1, -1, -1, -1) # Intencity (B, 1, N, N, N) just give to any sorce isotroopic angle distibution

    # Pre-cycle sub-functions
    # Needed refactoring into plotting tools
    frames = []
    vmax = None

    # Pre-create a consistent figure/axes to keep identical canvas size
    fig, ax = plt.subplots(figsize=(5, 5))

    w0 = f_batch.sum().detach().cpu().numpy()

    # 2D plotting
    
    # print("Start prop")
    # for step in range(0, n_steps + 1):
    #     f_dirs = f_batch

    #     f_step = (f_dirs.squeeze(1)).sum(dim=0)  # [N,N,N]
    #     proj = f_step.sum(dim=0).detach().cpu().numpy()  # [N,N]

    #     if vmax is None and not step == 0:
    #         vmax = proj.max() if proj.max() > 0 else 1.0

    #     ax.clear()    
    #     ax.imshow(proj, origin='lower', vmin=0, vmax=vmax)
    #     ax.set_title(f"Photon flux projection — step {step} (t={step*dt:.1f})")
    #     ax.set_axis_off()
    #     ax.text(
    #         0.05, 0.95,
    #         rf"$\sum_i \omega_i(t) / \sum_i \omega_i(0) = {proj.sum()/w0:.2e}$",
    #         transform=ax.transAxes,
    #         color='white',
    #         fontsize=15,
    #         verticalalignment='top',
    #     )

    #     frame_path = os.path.join(OUTPUT, f"flux_step_{step:02d}.png")
    #     fig.savefig(frame_path)  # fixed canvas size
    #     frames.append(imageio.imread(frame_path))

    #     f_batch = F.grid_sample( # actually this is  streaming step functionality
    #         f_dirs, grids, mode='bilinear', padding_mode='zeros', align_corners=True
    #     )  # [B,1,N,N,N]

    # plt.close(fig)

    # gif_path = os.path.join(OUTPUT, "photon_flux_propagation.gif")
    # imageio.mimsave(gif_path, frames, duration=0.6)

    # print('Finish')
    # print(gif_path)

# 3D plotting

    # ---- 3D viz knobs you can tweak ----
    THR_FRAC = 0.02         # show voxels with value >= THR_FRAC * f_step.max()
    MAX_POINTS = 150_000     # cap number of plotted voxels per frame
    POINT_SIZE = 2           # matplotlib scatter point size
    ROTATE_PER_STEP = 0     # degrees of azimuth change per frame
    ELEV = 20                # camera elevation
    # ------------------------------------

    frames = []
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    print("Start prop")
    azim0 = 45  # starting azimuth

    for step in range(0, n_steps + 1):
        f_dirs = f_batch                            # [B,1,N,N,N]
        f_step = (f_dirs.squeeze(1)).sum(dim=0)     # [N,N,N] torch tensor
        vol = f_step.detach().cpu().numpy()         # numpy [N,N,N]

        # Select voxels to plot (threshold + optional subsample)
        vmax = w0/200#float(vol.max())
        if vmax <= 0:
            coords = np.zeros((0, 3), dtype=int)
            vals = np.zeros((0,), dtype=float)
        else:
            thr = vmax * THR_FRAC
            coords = np.column_stack(np.nonzero(vol >= thr))  # (K,3) with (z,y,x) by default
            vals = vol[coords[:, 0], coords[:, 1], coords[:, 2]]

            # Subsample if too many points
            if coords.shape[0] > MAX_POINTS:
                idx = np.random.choice(coords.shape[0], size=MAX_POINTS, replace=False)
                coords = coords[idx]
                vals = vals[idx]

        # Color by value/w0; alpha by value within this frame
        if vals.size > 0:
            color_norm = np.clip(vals / w0, 0.0, 1.0)
            rgba = cm.viridis(color_norm)
            # emphasize stronger voxels with higher opacity (still readable in dense scenes)
            vrel = vals / (vals.max() if vals.max() > 0 else 1.0)
            rgba[:, 3] = np.clip(0.15 + 0.85 * (vrel ** 0.7), 0.15, 1.0)
        else:
            rgba = np.zeros((0, 4))

        # Plot
        ax.clear()
        if coords.size > 0:
            # coords from np.nonzero are (z, y, x); map to (x, y, z) for plotting
            xs, ys, zs = coords[:, 2], coords[:, 1], coords[:, 0]
            ax.scatter(xs, ys, zs, c=rgba, s=POINT_SIZE, depthshade=False)

        # Nice cube framing
        N = vol.shape[0]
        ax.set_xlim(0, N - 1)
        ax.set_ylim(0, N - 1)
        ax.set_zlim(0, N - 1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=ELEV, azim=azim0 + step * ROTATE_PER_STEP)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

        # Text: total weight vs w0 (now use full 3D sum)
        sum_ratio = float(vol.sum()) / (float(w0))
        ax.text2D(
            0.02, 0.98,
            rf"$\sum_i \omega_i(t) / \sum_i \omega_i(0) = {sum_ratio:.2e}$",
            transform=ax.transAxes,
            color="black",
            fontsize=11,
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3),
        )
        ax.set_title(f"Photon flux 3D — step {step} (t={step*dt:.1f})")

        # Save frame
        frame_path = os.path.join(OUTPUT, f"flux_step_{step:02d}.png")
        fig.savefig(frame_path, dpi=120)
        frames.append(imageio.imread(frame_path))

        # Advance simulation
        f_batch = F.grid_sample( # actually this is  streaming step functionality
            f_dirs, grids, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [B,1,N,N,N]

    plt.close(fig)

    gif_path = os.path.join(OUTPUT, "photon_flux_propagation_3d.gif")
    imageio.mimsave(gif_path, frames, duration=0.6)

    print("Finish")
    print(gif_path)
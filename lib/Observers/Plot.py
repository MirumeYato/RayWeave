from __future__ import annotations

import os

from lib import Observer, PATH
from lib.State import FieldState

import numpy as np

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm

OUTPUT = os.path.abspath(os.path.join(PATH, 'output'))

class Plot3D(Observer):
    """
    Plot 3D picture of field
    """
    # ---- 3D viz knobs you can tweak ----
    THR_FRAC = 0.02         # show voxels with value >= THR_FRAC * f_step.max()
    MAX_POINTS = 150_000     # cap number of plotted voxels per frame
    POINT_SIZE = 2           # matplotlib scatter point size
    ROTATE_PER_STEP = 0     # degrees of azimuth change per frame
    ELEV = 20                # camera elevation
    # ------------------------------------

    def __init__(self, every: int = 50, output_directory = OUTPUT):
        self.every = every
        self.initial_energy = 0.

        # Plot values
        self.frames = []
        self.fig = None
        self.ax = None

        self.output_dir = output_directory

    def on_setup(self, initial_state: FieldState) -> None:
        self.initial_energy = float((initial_state.field.squeeze(1)).sum().detach().cpu().numpy())
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")

        print("Start prop")
        self.azim0 = 45  # starting azimuth

        # Initial step
        self.on_step_end(-1, initial_state)

    def on_step_end(self, step_idx: int, state: FieldState) -> None:
        step_idx +=1 # We had zero step on setup. 
        
        vol = (state.field.squeeze(1)).sum(dim=0).detach().cpu().numpy()         # numpy [N,N,N]

        # Select voxels to plot (threshold + optional subsample)
        vmax = self.initial_energy/200#float(vol.max())
        if vmax <= 0:
            coords = np.zeros((0, 3), dtype=int)
            vals = np.zeros((0,), dtype=float)
        else:
            thr = vmax * self.THR_FRAC
            coords = np.column_stack(np.nonzero(vol >= thr))  # (K,3) with (z,y,x) by default
            vals = vol[coords[:, 0], coords[:, 1], coords[:, 2]]

            # Subsample if too many points
            if coords.shape[0] > self.MAX_POINTS:
                idx = np.random.choice(coords.shape[0], size=self.MAX_POINTS, replace=False)
                coords = coords[idx]
                vals = vals[idx]

        # Color by value/initial_energy; alpha by value within this frame
        if vals.size > 0:
            color_norm = np.clip(vals / self.initial_energy, 0.0, 1.0)
            rgba = cm.viridis(color_norm)
            # emphasize stronger voxels with higher opacity (still readable in dense scenes)
            vrel = vals / (vals.max() if vals.max() > 0 else 1.0)
            rgba[:, 3] = np.clip(0.15 + 0.85 * (vrel ** 0.7), 0.15, 1.0)
        else:
            rgba = np.zeros((0, 4))

        # Plot
        self.ax.clear()
        if coords.size > 0:
            # coords from np.nonzero are (z, y, x); map to (x, y, z) for plotting
            xs, ys, zs = coords[:, 2], coords[:, 1], coords[:, 0]
            self.ax.scatter(xs, ys, zs, c=rgba, s=self.POINT_SIZE, depthshade=False)

        # Nice cube framing
        N = vol.shape[0]
        self.ax.set_xlim(0, N - 1)
        self.ax.set_ylim(0, N - 1)
        self.ax.set_zlim(0, N - 1)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.view_init(elev=self.ELEV, azim=self.azim0 + step_idx * self.ROTATE_PER_STEP)
        self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_zticks([])
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("z")

        # Text: total weight vs initial_energy (now use full 3D sum)
        # sum_ratio = float(vol.sum())
        # if self.initial_energy > 1.e-12:
        #     sum_ratio = sum_ratio / (float(self.initial_energy))
        # else: print("[WARNING]: initial energy is zero!")
        sum_ratio = float(vol.sum()) / (self.initial_energy)
        self.ax.text2D(
            0.02, 0.98,
            rf"$\sum_i \omega_i(t) / \sum_i \omega_i(0) = {sum_ratio:.2e}$",
            transform=self.ax.transAxes,
            color="black",
            fontsize=11,
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3),
        )
        self.ax.set_title(f"Photon flux 3D — step {step_idx} (t={step_idx*state.dt:.1f})") # TODO: time of picture functionality can be expanded (id dt is not constant)

        # Save frame
        frame_path = os.path.join(self.output_dir, f"flux_step_{step_idx:02d}.png")
        self.fig.savefig(frame_path, dpi=120)
        self.frames.append(imageio.imread(frame_path))

    def on_teardown(self) -> None: 
        plt.close(self.fig)

        gif_path = os.path.join(self.output_dir, "photon_flux_propagation_3d.gif")
        imageio.mimsave(gif_path, self.frames, duration=0.6)

        print("Finish")
        print(gif_path)


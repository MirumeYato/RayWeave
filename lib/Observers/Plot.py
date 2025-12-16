from __future__ import annotations

from typing import List
import os

from lib import Observer, PATH
from lib.State import FieldState
from lib.grid.Angle import Angle

import torch
import numpy as np
import healpy as hp

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
        self.initial_energy = float((initial_state.field).sum().detach().cpu().numpy())
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")

        print("Start prop")
        self.azim0 = 45  # starting azimuth

        # Initial step
        self.on_step_end(-1, initial_state)

    def on_step_end(self, step_idx: int, state: FieldState) -> None:
        step_idx +=1 # We had zero step on setup. 
        
        vol = (state.field).sum(dim=0).detach().cpu().numpy()         # numpy [N,N,N]

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
        self.ax.set_title(f"Photon flux 3D — step {step_idx} (t={step_idx*state.dt:.1f})") 
        # TODO: what if dt is not constant? (look more same todo by "dt!=const")
        #       We need to predefine array of dt's and just call needed element by 'step_idx'

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


class PlotMollviewInPoint(Observer):
    """
    Plot mollview projection in one certain space bin (point)
    """

    def __init__(self, space_point_idxs: List[int], Angle, every: int = 50, output_directory = OUTPUT):
        self.every = every
        self.initial_energy = 0.

        self.x_id = space_point_idxs[0]
        self.y_id = space_point_idxs[1]
        self.z_id = space_point_idxs[2]
        self.Angle: Angle = Angle

        # Plot values
        self.frames = []
        self.fig = None
        self.ax = None

        self.output_dir = output_directory

    def __mollview(self, values: torch.Tensor):
        # Accumulate values into pixels
        # mode='sum' adds the 'values' into the indices specified by 'pixel_indices'
        # index_add_ is generally deterministic and fast
        self.healpix_map_sum = self.healpix_map_sum.to(device=values.device)
        self.healpix_map_sum.index_add_(0, self.pixel_indices.to(device=values.device), values.to(dtype=self.healpix_map_sum.dtype))
        
        # Accumulate counts (add 1.0 for every hit)
        self.healpix_map_count = self.healpix_map_count.to(device=values.device)
        self.healpix_map_count.index_add_(0, self.pixel_indices.to(device=values.device), self.ones.to(device=values.device))

        # ---------------------------------------------------------
        # 4. Normalize (Average) and Handle Empty Pixels
        # ---------------------------------------------------------
        
        # Avoid division by zero
        mask_nonzero = self.healpix_map_count > 0
        
        # Create final map initialized to UNSEEN (standard healpy convention for empty)
        final_map = torch.full((self.npix,), 1e-10, dtype=torch.float32, device=values.device)
        
        # Compute average where we have data
        final_map[mask_nonzero] = self.healpix_map_sum[mask_nonzero] / self.healpix_map_count[mask_nonzero]

        # Convert to numpy for healpy visualization
        final_map_np = final_map.detach().cpu().numpy()

        # Optional: Fill holes (simple neighbor averaging) if N was small compared to Npix
        # final_map_np = hp.sphtfunc.smoothing(final_map_np, fwhm=0.0) # or custom interpolation

        return final_map_np

    def on_setup(self, initial_state: FieldState) -> None:
        self.initial_energy = float((initial_state.field.real).sum().detach().cpu().numpy())
        N = initial_state.field.shape[0]

        # mollview preparations
        target_nside = int(np.sqrt(N / 12))    
        # NSIDE must be a power of 2 for many healpy features (optional but recommended)
        nside = 2**round(np.log2(target_nside))
        self.npix = hp.nside2npix(nside)        
        print(f"Selected NSIDE: {nside} (Total pixels: {self.npix})")

        thetas, phis = self.Angle.get_nodes_angles()
        pixel_indices = hp.ang2pix(nside, thetas.detach().cpu().numpy(), phis.detach().cpu().numpy())
        self.pixel_indices = torch.from_numpy(pixel_indices).long()

        # Prepare tensors for accumulation
        self.healpix_map_sum = torch.zeros(self.npix, dtype=torch.float32)
        self.healpix_map_count = torch.zeros(self.npix, dtype=torch.float32)

        self.ones = torch.ones(N, dtype=torch.float32)

        # Plots
        self.fig = plt.figure(figsize=(10, 9))

        print("Start prop")

        # Initial step
        self.on_step_end(-1, initial_state, True)

    def on_step_end(self, step_idx: int, state: FieldState, initial_flag = False) -> None:
        if step_idx % self.every == 0 or initial_flag:
            step_idx +=1 # We had zero step on setup. 
            
            vol = (state.field.real)[:, self.x_id, self.y_id, self.z_id]          # shape [B]

            mapped_vol = self.__mollview(vol)

            # Upper map
            hp.mollview(
                mapped_vol,
                fig=self.fig.number,
                sub=(2, 1, 1),
                title=f"Step {step_idx} — voxel (c,c,c)",
                norm=None,
                cbar=True,
                # min=0,
                max=1
            )

            # Add global text annotation (use figure coordinates instead of axes)
            sum_ratio = np.abs(1 - float(vol.sum().detach().cpu().numpy()) / float(self.initial_energy))
            self.fig.text(
                0.02, 0.97,
                r"$|1 - \sum_i \omega_i(t) / \sum_i \omega_i(0)| =$"+f"\n= {sum_ratio:.2e}",
                color="black",
                fontsize=11,
                ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3),
            )

            # Save frame
            frame_path = os.path.join(OUTPUT, f"flux_step_{step_idx:02d}.png")
            self.fig.savefig(frame_path, dpi=150, bbox_inches="tight")
            self.fig.clear()
            self.frames.append(imageio.imread(frame_path))

    def on_teardown(self) -> None: 
        plt.close(self.fig)

        gif_path = os.path.join(self.output_dir, "photon_flux_propagation_3d.gif")
        imageio.mimsave(gif_path, self.frames, duration=0.6)

        print("Finish")
        print(gif_path)

class EnergyPlotter(Observer):
    def __init__(self, n_steps: int, every: int = 50, output_directory = OUTPUT):
        self.every = every
        self.output_dir = output_directory
        self.total = np.zeros(n_steps // every)
        
    def on_step_end(self, step_idx: int, state: FieldState) -> None:
        if step_idx % self.every == 0:
            # cheap summary; no sync if possible
            self.total[step_idx // self.every] = torch.relu(state.field.real).sum().detach().item()

    def on_teardown(self) -> None: 
        plt.figure(figsize=(8, 5))
    
        plt.plot(np.abs(1 - self.total / self.total[0]), color='C0')
        plt.title("Conservation of energy", fontsize=14)
        plt.ylabel(r"$1 - \sum_i \omega_i(t) / \sum_i \omega_i(0)$", fontsize=12)
        plt.xlabel("Time step", fontsize=12)
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.5)
        # plt.legend(fontsize=14)
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, "Energy_t.png"), dpi=150)   
        plt.close()      
        
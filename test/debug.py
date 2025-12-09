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
        self.on_step_end(-1, initial_state)

    def on_step_end(self, step_idx: int, state: FieldState) -> None:
        if step_idx % self.every == 0:
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
            sum_ratio = float(vol.sum().detach().cpu().numpy()) / float(self.initial_energy)
            self.fig.text(
                0.02, 0.97,
                rf"$\sum_i \omega_i(t) / \sum_i \omega_i(0) = {sum_ratio:.2e}$",
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
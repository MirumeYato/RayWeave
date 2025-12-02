import torch
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def plot_scattered_data_on_mollview():
    # ---------------------------------------------------------
    # 1. Setup Mock Data (N scattered points)
    # ---------------------------------------------------------
    N = 100_000
    
    # Generate random points on a sphere (uniformly distributed)
    # Theta: [0, pi], Phi: [0, 2pi]
    # Using acos for uniform sphere coverage logic for theta
    u = torch.rand(N)
    v = torch.rand(N)
    
    thetas = torch.acos(2 * u - 1)  # Co-latitude [0, PI]
    phis = 2 * np.pi * v            # Longitude [0, 2PI]
    
    # Create some signal to visualize (e.g., a dipole or gradient)
    # Signal = cos(theta) + sin(phi)
    values = torch.cos(thetas) + torch.sin(phis)

    print(f"Generated {N} scattered points.")

    # ---------------------------------------------------------
    # 2. Determine HEALPix Resolution (NSIDE)
    # ---------------------------------------------------------
    # We want N_pixels approx equal to N to maintain resolution, 
    # or slightly less to avoid holes.
    # N_pix = 12 * nside^2  =>  nside = sqrt(N / 12)
    
    target_nside = int(np.sqrt(N / 12))
    
    # NSIDE must be a power of 2 for many healpy features (optional but recommended)
    nside = 2**round(np.log2(target_nside)-3)
    npix = hp.nside2npix(nside)
    
    print(f"Selected NSIDE: {nside} (Total pixels: {npix})")

    # ---------------------------------------------------------
    # 3. Binning / Interpolation using PyTorch
    # ---------------------------------------------------------
    
    # Convert angles to HEALPix pixel indices (requires CPU numpy typically)
    # Note: healpy inputs are (theta, phi) in radians
    # We move to numpy for ang2pix, then back to torch or stay in numpy depending on pipeline
    pixel_indices = hp.ang2pix(nside, thetas.numpy(), phis.numpy())
    pixel_indices = torch.from_numpy(pixel_indices).long()

    # Prepare tensors for accumulation
    # We need a map for the sum of values and a map for the count of hits per pixel
    healpix_map_sum = torch.zeros(npix, dtype=torch.float32)
    healpix_map_count = torch.zeros(npix, dtype=torch.float32)

    # Accumulate values into pixels
    # mode='sum' adds the 'values' into the indices specified by 'pixel_indices'
    # index_add_ is generally deterministic and fast
    healpix_map_sum.index_add_(0, pixel_indices, values.float())
    
    # Accumulate counts (add 1.0 for every hit)
    ones = torch.ones(N, dtype=torch.float32)
    healpix_map_count.index_add_(0, pixel_indices, ones)

    # ---------------------------------------------------------
    # 4. Normalize (Average) and Handle Empty Pixels
    # ---------------------------------------------------------
    
    # Avoid division by zero
    mask_nonzero = healpix_map_count > 0
    
    # Create final map initialized to UNSEEN (standard healpy convention for empty)
    final_map = torch.full((npix,), hp.UNSEEN, dtype=torch.float32)
    
    # Compute average where we have data
    final_map[mask_nonzero] = healpix_map_sum[mask_nonzero] / healpix_map_count[mask_nonzero]

    # Convert to numpy for healpy visualization
    final_map_np = final_map.numpy()

    # Optional: Fill holes (simple neighbor averaging) if N was small compared to Npix
    # final_map_np = hp.sphtfunc.smoothing(final_map_np, fwhm=0.0) # or custom interpolation

    # ---------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    hp.mollview(
        final_map_np, 
        title=f"Mollview from {N} Scattered Points (NSIDE={nside})",
        unit="Signal Intensity",
        cmap="jet"
    )
    plt.savefig("mollview_output.png")
    print("Plot saved to mollview_output.png")

if __name__ == "__main__":
    plot_scattered_data_on_mollview()
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

import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy.special import sph_harm_y
from math import pi
import healpy as hp

OUTPUT = os.path.abspath(os.path.join(PATH, 'output', 'test', 'SHT'))
os.makedirs(OUTPUT, exist_ok=True)

def fHenyeyGreenshtein(g, theta):
    for theta_i in theta:
        if 1+g*g+2*g*np.cos(theta_i) <= 0:
            print("Inf")
            return 
    return 1/(4*np.pi)*(1-g*g)/np.power(1+g*g-2*g*np.cos(theta), 3/2)

def _precompute_Ylm_healpix(nside: int, L_max: int, device=None, dtype=torch.complex64):
    """
    Fast vectorized precomputation of spherical harmonics for HEALPix grid.
    Returns:
      Y : torch.complex tensor of shape (Q, P) with P=(L_max+1)^2
      idx_l : torch.long tensor of shape (P,) — l index for each (l,m)

    Can be changed for more fuster solution later
    """
    Q = hp.nside2npix(nside)
    print(Q)
    theta, phi = hp.pix2ang(nside, np.arange(Q))  # theta ∈ [0,π], phi ∈ [0,2π)

    # --- vectorized (l,m) construction ---
    l_vals = np.arange(L_max + 1)
    m_vals = np.concatenate([np.arange(-l, l + 1) for l in l_vals])
    l_index_np = np.repeat(l_vals, [2 * l + 1 for l in l_vals])

    P = len(m_vals)  # total number of (l,m)

    # --- vectorized computation of all Y_lm ---
    # Broadcast theta, phi for all pixels
    theta_2d = theta[:, None]
    phi_2d = phi[:, None]
    # Broadcast l and m
    l_2d = l_index_np[None, :]
    m_2d = m_vals[None, :]

    # sph_harm_y supports vectorized l,m,phi,theta inputs
    print("Do not turn off process it is not stuck. Wait from 1 to 5 minets")
    Y_np = sph_harm_y(l_2d, m_2d, theta_2d, phi_2d).astype(np.complex64)
    Y_np *= np.sqrt((4.0 * np.pi) / Q)  # normalization
    print ("SH calculated successfully")

    # --- torch conversion ---
    Y = torch.from_numpy(Y_np).to(device=device, dtype=dtype)
    idx_l = torch.from_numpy(l_index_np).to(device=device, dtype=torch.long)
    return Y, idx_l


def test_HenyeyGreenshtein(nSIDE = 25, L_max = 25*3-3, g = 0.5, custom = True):
    """
    We want to check accuracy of map to alm and alm to map functions on fHenyeyGreenshtein example
    We have exact alm formula for fHenyeyGreenshtein: 
    
    >>> alm = g**l / np.sqrt(4*np.pi/(2*l+1)) 

    we normalize it via 1 / sqrt(4*np.pi/(2*l+1)) for simplicity
    """
    # Difine agles grid via HEALPix
    theta, phi = hp.pix2ang(nSIDE, np.arange(hp.nside2npix(nSIDE))) 

    # Calculate fHenyeyGreenshtein in each point of grid with certain g 
    map_true = fHenyeyGreenshtein(g, theta) 
    # Calculate HenyeyGreenshtein alm. But normalize it via 1 / np.sqrt(4*np.pi/(2*i+1))
    alm_true  = np.array([g**i / np.sqrt(4*np.pi/(2*i+1)) for i in range(L_max+1)])

    # Precompute custom Spherical Hatrmonics for custom method check
    if custom:
        # save torch versions of true map and alm
        map_true_torch = torch.from_numpy(map_true).to(device=device, dtype=torch.complex64)

        print("Starting precomputate Spherical Harmonics...")
        mY, idx_l = _precompute_Ylm_healpix(nSIDE, L_max, device=device)
        mY_H = torch.conj(mY)  

        # Check normalization's correctness
        # Via \sum_{angles} (Y*Y_H) = 1
        Q = mY.shape[0]
        mYY_H = torch.einsum('qp,qp->qp', mY, mY_H)
        # YYY2 = torch.abs(Y*Y)
        test1 = torch.sum(torch.real(mYY_H), axis = 0)
        
        print(f"Max in \sum_angles (Y*Y_H): {torch.max(test1)}, Min: {torch.min(test1)}")

        # Now we want to get alm true, but in shape correct for custom method
        alm_true_np = np.zeros(mY.shape[1])
        i=0
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                # print(l,m)
                if m == 0: alm_true_np[i] = np.real(alm_true[l])
                else : alm_true_np[i] = 0
                i+=1
        alm_ture_torch = torch.from_numpy(alm_true_np).to(device=device, dtype=torch.complex64)

        # get alm via custom SHs from true
        alm_custom_torch = torch.einsum('q,qp->p', map_true_torch, mY_H)
        # Convert torch tensor to numpy array
        alm_custom_long = alm_custom_torch.detach().cpu().numpy()
        # This method gives extended array for each m. But we not really need them
        alm_custom = np.zeros(L_max+1)
        i=0
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                # print(l,m) # DEBUG
                # Simply save only m = 0 items
                if m == 0: alm_custom[l] = np.real(alm_custom_long[i])
                i+=1
        
        # get map via custom SHs from true
        map_custom = torch.einsum('p,qp->q', alm_ture_torch, mY).detach().cpu().numpy()
        
        # Make double procedure from true map -> get alm_custom -> and back again to map_custom
        map_custom_rec_torch = torch.einsum('p,qp->q', alm_custom_torch, mY)
        map_custom_rec = map_custom_rec_torch.detach().cpu().numpy()

        # Make double procedure from true map -> get alm_custom -> and back again to map_custom 10 times
        map_custom_rec_torch_10 = map_true_torch
        for i in range(10):
            alm_custom_torch_10 = torch.einsum('q,qp->p', map_custom_rec_torch_10, mY_H)
            map_custom_rec_torch_10 = torch.einsum('p,qp->q', alm_custom_torch_10, mY)
        
        map_custom_rec_10 = map_custom_rec_torch_10.detach().cpu().numpy()

    # Get HEALPix alm and map from true:
    # we choose mmax = 0 for simplicity (there is no m dependence in formula of HenyeyGreenshtein)
    map_hp = hp.alm2map(alm_true.astype(np.complex128), nSIDE, lmax=L_max, mmax=0, pol = False)
    alm_hp = hp.map2alm(map_true, lmax=L_max, mmax=0, iter = 0, pol = False)
    # Make double procedure from true map -> get alm_hp -> and back again to map_hp
    map_hp_rec = hp.alm2map(alm_hp, nSIDE, lmax=L_max, mmax=0, pol = False)
    
    map_hp_rec_10 = map_true
    for i in range(10):
        alm_hp_10 = hp.map2alm(map_hp_rec_10, lmax=L_max, mmax=0, iter = 0, pol = False)
        map_hp_rec_10 = hp.alm2map(alm_hp_10.astype(np.complex128), nSIDE, lmax=L_max, mmax=0, pol = False)


    ### Get alm from true map (fHenyeyGreenshtein values)    

    plt.figure(figsize=(8, 5))
    plt.plot(np.abs(alm_hp - alm_true), linewidth=3, label=r"$|a_{\ell m}^{hp} - a_{\ell m}^{true}|$", color='C0')
    
    # If we want to check custom method (But we should normalize on)
    if custom:
        plt.plot(np.abs(alm_custom * np.sqrt((4.0 * np.pi) / Q) - alm_true),
                linewidth=2, linestyle="--", label=r"$|a_{\ell m}^{custom} - a_{\ell m}^{true}|$", color='C1')
    plt.title("MAE of alm reconstruction from map true", fontsize=14)
    plt.xlabel(r"Harmonic index $i$", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_map_true2alm.png"), dpi=150)
    plt.close()

    ### Get map (fHenyeyGreenshtein values) from true alm (g**l)

    plt.figure(figsize=(8, 5))
    plt.plot(np.abs(map_hp - map_true), linewidth=3, label=r"$|f^{hp} - f^{true}|$", color='C0')
    
    # If we want to check custom method (But we should normalize on)
    if custom:
        plt.plot(np.abs(map_custom / np.sqrt((4.0 * np.pi) / Q) - map_true),
                linewidth=2, linestyle="--", label=r"$|f^{custom} - f^{true}|$", color='C1')
    plt.title("MAE of map reconstruction from alm true", fontsize=14)
    plt.xlabel("Pixel index", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_alm_true2map.png"), dpi=150)
    plt.close()

    ### Get map (fHenyeyGreenshtein values) from alm (g**l)
    # in our collision step we a planning to  get alm form map, 
    # do some collision operation on alm for getting a^{star}_lm (alm on moment dt/2 later)
    # than we will get it back into map. 
    # And we will repeat it many many times. So here inportant to check
    # commutative error of each such tasport from map to alm and vice versa

    plt.figure(figsize=(8, 5))
    plt.plot(np.abs(map_hp_rec - map_true), linewidth=3, label=r"$|f^{rec}_{hp} - f^{true}|$", color='C0')

    # If we want to check custom method
    if custom:
        plt.plot(np.abs(map_custom_rec - map_true),
                linewidth=1, alpha = 0.4, label=r"$|f^{rec}_{custom} - f^{true}|$", color='C1')
    plt.title("Round-trip consistency: map → alm → map", fontsize=14)
    plt.xlabel("Pixel index", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_map2alm2map.png"), dpi=150)
    plt.close()

    ### Same as previous, but 10 times

    plt.figure(figsize=(8, 5))
    plt.plot(np.abs(map_hp_rec_10 - map_true), linewidth=3, label=r"$|f^{rec}_{hp} - f^{true}|$", color='C0')

    # If we want to check custom method
    if custom:
        plt.plot(np.abs(map_custom_rec_10 - map_true),
                linewidth=1, alpha = 0.4, label=r"$|f^{rec}_{custom} - f^{true}|$", color='C1')
    plt.title("Round-trip consistency: map → alm → map x10", fontsize=14)
    plt.xlabel("Pixel index", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "Mae_comparison_map2alm2map_x10.png"), dpi=150)
    plt.close()

    print("Plots saved in the same dir as script")

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test fHenyeyGreenshtein function
    print(f"Test fHenyeyGreenshtein is working (g = 0.9, theta = [0, pi/2, pi-0.00001, pi]): {fHenyeyGreenshtein(0.9, np.array([0, np.pi/2, np.pi-0.00001, np.pi]))}")

    test_HenyeyGreenshtein(nSIDE = 25, L_max = 25*3-3, g = 0.5, custom = True)

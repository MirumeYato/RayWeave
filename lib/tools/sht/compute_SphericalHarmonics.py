import numpy as np
import torch

from scipy.special import sph_harm_y
# from scipy.special import sph_harm

def compute_SH_scipy(
    theta: np.ndarray | torch.Tensor,
    phi: np.ndarray | torch.Tensor,
    L_max: int,
    device=None,
    dtype=torch.complex128
):
    """
    Fast, correct, memory-efficient spherical harmonics using SciPy + proper broadcasting.
    
    Returns:
        Y: torch.Tensor of shape (n_points, (L_max+1)^2), complex
    """

    theta = theta.astype(np.float64, copy=False)
    phi = phi.astype(np.float64, copy=False)

    # --- vectorized (l,m) construction ---
    l_vals = np.arange(L_max + 1)
    m_vals = np.concatenate([np.arange(-l, l + 1) for l in l_vals])
    m_counts = 2 * l_vals + 1
    l_index_np = np.repeat(l_vals, m_counts)

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
    # Y_np = sph_harm(m_2d, l_2d, phi_2d, theta_2d).astype(np.complex64)
    print ("SH calculated successfully\n")

    # --- torch conversion ---
    Y = torch.from_numpy(Y_np).to(device=device, dtype=dtype)
    return Y


def compute_SH_torch(
    theta: torch.Tensor,
    phi: torch.Tensor,
    L_max: int,
    device=None,
    dtype=torch.complex128
):
    """
    Computes Spherical Harmonics Y_lm(theta, phi) fully in PyTorch.
    Matches scipy.special.sph_harm_y conventions.
    
    Args:
        theta: Azimuthal (longitudinal) coordinate [0, 2*pi]. Shape (N,)
        phi: Polar (colatitudinal) coordinate [0, pi]. Shape (N,)
        L_max: Maximum spherical harmonic degree.
        
    Returns:
        Y: Tensor of shape (N, (L_max+1)^2)
    """
    # Ensure inputs are on the correct device/dtype
    # Invert angles because of mistake in this function definition
    theta_t = phi.to(device=device, dtype=torch.float64) # Angles must be real for calc
    phi_t = theta.to(device=device, dtype=torch.float64)
    
    N = theta_t.shape[0]
    n_harmonics = (L_max + 1) ** 2
    
    # Pre-allocate output tensor (complex)
    Y = torch.zeros((N, n_harmonics), device=device, dtype=dtype)
    
    # Cartesian components for Legendre recursion (x = cos(phi) corresponds to polar axis)
    # Note: Scipy's phi is the polar angle (0 to pi), so x = cos(phi)
    x = torch.cos(phi_t)
    s = torch.sin(phi_t) # sin(phi)
    
    # We will compute P_lm (Associated Legendre Polynomials)
    # We only need to compute for m >= 0, then derive m < 0 using symmetry.
    
    # --- 1. Compute P_mm (Sectorial) terms ---
    # P_mm(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2) 
    #         = (-1)^m * (2m-1)!! * sin(phi)^m
    # We compute these iteratively: P_mm = - (2m-1) * s * P_{m-1, m-1}
    
    # Dictionary to store P_lm for the current and previous l steps to handle recurrence
    # We flatten the storage to efficiently update Y later. 
    # Since we loop m then l, we can write directly to Y.
    
    # We need to handle the factorial prefactors and normalization constants.
    # To maintain precision, we compute the Normalized Legendre Polynomials directly if possible,
    # but for clarity and matching Scipy, we'll compute P_lm and multiply by Normalization.
    
    # It's faster to precompute Normalization Constants (N_lm) on CPU once.
    N_lm = _precompute_normalization(L_max).to(device=device, dtype=torch.float64)
    
    # P_lm storage: We only need current and previous rows of the recursion
    # But simpler strategy for PyTorch: Compute one m-column at a time.
    
    for m in range(0, L_max + 1):
        
        # --- A. Compute P_m^m ---
        if m == 0:
            P_mm = torch.ones_like(x)
        else:
            # P_m^m = -(2m-1) * sin(phi) * P_{m-1}^{m-1}
            # The (-1)^m is often included in the definition. Scipy includes Condon-Shortley phase.
            # Scipy definition involves (-1)^m.
            P_mm = -(2 * m - 1) * s * P_mm_prev
        
        P_mm_prev = P_mm # Save for next m iteration
        
        # Write P_m^m contribution
        # Index for (l=m, m=m)
        idx_pos = _get_idx(m, m, L_max)
        
        # Y_mm = N_mm * P_mm * exp(i*m*theta)
        phase = torch.exp(1j * m * theta_t)
        Y[:, idx_pos] = N_lm[idx_pos] * P_mm * phase
        
        # Handle m < 0 via symmetry: Y_{l, -m} = (-1)^m * conj(Y_{l, m})
        if m > 0:
            idx_neg = _get_idx(m, -m, L_max)
            # Y_{m, -m}
            Y[:, idx_neg] = (-1)**m * torch.conj(Y[:, idx_pos])

        # --- B. Compute P_l^m for l > m (Vertical Recurrence) ---
        # (l-m) P_l^m = x(2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
        
        if m < L_max:
            # First step: l = m + 1
            l = m + 1
            P_lm = x * (2 * l - 1) * P_mm # (l-m)=1 here, so divide by 1
            
            idx_pos = _get_idx(l, m, L_max)
            Y[:, idx_pos] = N_lm[idx_pos] * P_lm * phase
            
            if m > 0:
                idx_neg = _get_idx(l, -m, L_max)
                Y[:, idx_neg] = (-1)**m * torch.conj(Y[:, idx_pos])
            
            P_prev = P_mm
            P_curr = P_lm
            
            # Remaining steps: l = m+2 to L_max
            for l in range(m + 2, L_max + 1):
                # Recurrence: P_new = ( x(2l-1)P_curr - (l+m-1)P_prev ) / (l-m)
                P_next = (x * (2 * l - 1) * P_curr - (l + m - 1) * P_prev) / (l - m)
                
                idx_pos = _get_idx(l, m, L_max)
                Y[:, idx_pos] = N_lm[idx_pos] * P_next * phase
                
                if m > 0:
                    idx_neg = _get_idx(l, -m, L_max)
                    Y[:, idx_neg] = (-1)**m * torch.conj(Y[:, idx_pos])
                
                P_prev = P_curr
                P_curr = P_next

    return Y

def _get_idx(l, m, L_max):
    """Helper to flatten (l, m) into the user's expected 1D index."""
    # User's code implies a specific flattening. 
    # Usually scipy flattens as l=0(m=0), l=1(m=-1,0,1)...
    # That is index = l^2 + (m + l) = l^2 + l + m
    return l * (l + 1) + m

def _precompute_normalization(L_max):
    """
    Precompute normalization constants N_lm compatible with Scipy.
    N_lm = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
    """
    import math
    n_elems = (L_max + 1) ** 2
    N_lm = torch.zeros(n_elems, dtype=torch.float64)
    
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # Scipy's sph_harm normalization
            # Note: We only use this for m >= 0 in the main loop, 
            # but we fill all for safety or if logic changes.
            if m >= 0:
                pre = (2 * l + 1) / (4 * math.pi)
                fact = math.factorial(l - m) / math.factorial(l + m)
                val = math.sqrt(pre * fact)
                idx = _get_idx(l, m, L_max)
                N_lm[idx] = val
            
    return N_lm

# TODO: Needs fixes:
# from e3nn import o3

# def compute_SH_e3nn(theta, phi, L_max, device='cuda', dtype=torch.float64):
#     """
#     theta: Azimuth (long) [0, 2pi]
#     phi:   Polar (colat)  [0, pi]
#     """
#     theta_t = torch.from_numpy(phi).to(device=device, dtype=dtype)
#     phi_t = torch.from_numpy(theta).to(device=device, dtype=dtype)

#     # e3nn expects inputs as Cartesian coordinates (x, y, z)
#     # Convert Spherical (Scipy convention) -> Cartesian
#     # Scipy: theta=azimuth, phi=polar
#     x = torch.sin(phi_t) * torch.cos(theta_t)
#     y = torch.sin(phi_t) * torch.sin(theta_t)
#     z = torch.cos(phi_t)
#     vectors = torch.stack([x, y, z], dim=-1).to(device=device, dtype=dtype)
    
#     # Create the SH object (computes coefficients for all L up to L_max)
#     # normalization='integral' matches the standard orthonormal basis often used in physics
#     print("Do not turn off process it is not stuck. Wait from 1 to 5 minets")
#     sh_real = o3.spherical_harmonics(
#         l=range(L_max + 1),
#         x=vectors,
#         normalize=True,
#         normalization='integral' 
#     )
#     print ("SH calculated successfully\n")
#     # 3. Convert Real SH to Complex SH (Scipy/Physics Convention)
#     # e3nn output index j corresponds to (l, m) flattened.
#     # We need to map Real -> Complex:
#     # Y_l^0 (complex) = Y_l^0 (real)
#     # Y_l^m (complex) = 1/sqrt(2) * (Y_l^m(real) + i * Y_l^-m(real))  [for m > 0]
#     # Y_l^-m(complex) = 1/sqrt(2) * (Y_l^m(real) - i * Y_l^-m(real)) * (-1)^m
    
#     N = theta_t.shape[0]
#     n_harmonics = (L_max + 1) ** 2
#     Y_complex = torch.zeros((N, n_harmonics), device=device, dtype=torch.complex128)
    
#     # e3nn flattens indices as: l=0(m=0), l=1(m=-1,0,1), l=2(m=-2,-1,0,1,2)...
#     # This differs from standard loop order, but follows index = l^2 + l + m
    
#     for l in range(L_max + 1):
#         # Index of m=0 for this l
#         center_idx = l**2 + l 
        
#         # m = 0
#         Y_complex[:, center_idx] = sh_real[:, center_idx] + 0j
        
#         for m in range(1, l + 1):
#             idx_pos_m = center_idx + m # e3nn index for +m
#             idx_neg_m = center_idx - m # e3nn index for -m
            
#             # Get Real parts
#             # In e3nn (usually): index (l, m) stores Y_l^m (real)
#             # But e3nn basis is often rotated. 
#             # Standard relation for Real Y_lm:
#             # Y_real_pos = 1/sqrt(2) * (Y_complex + Y_complex_conj)
#             # Y_real_neg = 1/i*sqrt(2) * (Y_complex - Y_complex_conj)
            
#             # Reconstructing Complex from e3nn Real:
#             real_part = sh_real[:, idx_pos_m]
#             imag_part = sh_real[:, idx_neg_m]
            
#             # The standard conversion:
#             # Y_complex(+m) = (1/sqrt(2)) * (real_part + 1j * imag_part)
#             # Note: Signs might flip depending on e3nn version (Condon-Shortley phase)
#             # e3nn includes Condon-Shortley in the real basis construction.
            
#             val_pos = (1.0 / np.sqrt(2)) * (real_part + 1j * imag_part)
            
#             # Y_complex(-m) = (-1)^m * conj(Y_complex(+m))
#             val_neg = ((-1)**m) * torch.conj(val_pos)

#             Y_complex[:, idx_pos_m] = val_pos
#             Y_complex[:, idx_neg_m] = val_neg

#     return Y_complex


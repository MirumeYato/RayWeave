import numpy as np
import torch
from torch import Tensor
from typing import Union

def map_HenyeyGreenstein(g: float, theta: Tensor) -> Tensor:
    """
    Henyey-Greenstein phase function (scalar g, vectorized over theta).

    Parameters
    ----------
    g : float
        Asymmetry parameter, |g| < 1 (typically in [-0.99, 0.99] to avoid singularities).
    theta : Tensor
        Scattering angles in radians (any shape).

    Returns
    -------
    Tensor
        Phase function values with the same shape as theta.
    """
    # Avoid division by zero or negative inside the root at theta = π when |g| → 1
    cos_theta = torch.cos(theta)
    g_tensor = torch.tensor(g, device=theta.device, dtype=theta.dtype)
    denominator = 1 + g_tensor**2 - 2 * g_tensor * cos_theta
    # Clamp tiny/small negative values due to floating-point precision
    denominator = torch.clamp(denominator, min=1e-30)

    map = (1 - g_tensor**2) / (4 * torch.pi * torch.pow(denominator, 1.5))
    return map


def alm_HenyeyGreenstein(g: float, L_max: int, device = "cpu") -> Tensor:
    """
    Exact expansion coefficients a_lm = 0 for m≠0 and
    a_l0 = sqrt((2l+1)/(4π)) * g^l   (normalized real spherical harmonics).

    Parameters
    ----------
    g : float
        Asymmetry parameter (|g| < 1).
    L_max : int
        Maximum degree l (inclusive).

    Returns
    -------
    Tensor
        1D tensor of shape (L_max+1,) containing a_00, a_10, ..., a_{L_max}0
    """
    l = torch.arange(L_max + 1, device = device, dtype=torch.float64)        # degree l = 0,1,...,L_max
    alm = g ** l / torch.sqrt(4 * torch.pi / (2 * l + 1))
    return alm.to(device, torch.float64)

######################

# Some alm of HenyeyGreenstein function transformators
# (Just for simplicity of comparison)

def alm_transform_torch(alm_no_m: Tensor, L_max: int, length: int) -> Tensor:
    """
    Transforms a 1D tensor of l-dependent coefficients (a_l0) into a 
    full spherical harmonic coefficient tensor (a_lm) where a_lm = 0 for m != 0.

    Parameters
    ----------
    alm_no_m : Tensor
        1D tensor of shape (L_max + 1,) containing the coefficients a_00, a_10, ..., a_{L_max}0.
        (Assumed to be real, or only the real part is desired).
    L_max : int
        Maximum degree l (inclusive).
    length : int
        The total expected length of the full alm vector. 
        Must equal (L_max + 1)^2. (Used for array size check/initialization).

    Returns
    -------
    Tensor
        1D tensor of shape (length,) containing the full alm vector 
        in standard ordering (a_00, a_1-1, a_10, a_11, a_2-2, ...).
    """
    # 1. Initialize the full alm vector with zeros.
    # We enforce the dtype of the input for consistency.
    alm_full = torch.zeros(length, dtype=alm_no_m.dtype, device=alm_no_m.device)
    
    # Check if the input size is correct
    if alm_no_m.shape[0] != L_max + 1:
        raise ValueError(
            f"Input alm_no_m size ({alm_no_m.shape[0]}) must equal L_max + 1 ({L_max + 1})."
        )

    # 2. Determine the indices corresponding to m=0 for all l.
    # The standard ordering for alm coefficients is:
    # l=0: m=0 (1 coefficient) -> index 0
    # l=1: m=-1, 0, 1 (3 coefficients) -> index 1, 2, 3. The m=0 coefficient is at index 2.
    # l=2: m=-2, -1, 0, 1, 2 (5 coefficients) -> index 4, 5, 6, 7, 8. The m=0 coefficient is at index 6.
    
    # The index of the m=0 coefficient for degree l is: 
    # Index_l0 = sum_{j=0}^{l-1} (2j + 1) + l = l^2 + l
    
    l_range = torch.arange(L_max + 1, device=alm_no_m.device)
    # Calculate the index for m=0 for each degree l
    m_zero_indices = l_range * (l_range + 1)  # l^2 + l

    # 3. Use advanced indexing to place the a_l0 coefficients into the m=0 slots.
    alm_full[m_zero_indices] = alm_no_m
    
    # If the input was complex and only the real part was desired (as in the NumPy code), 
    # we already handled this by ensuring the output is the same (potentially real) dtype.
    
    return alm_full.real # Use .real for robustness if dtype is complex, otherwise it's a no-op

def alm_transform_np(alm_no_m, L_max, lenth)-> np.ndarray:
    alm = np.zeros(lenth)
    i=0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # print(l,m)
            if m == 0: alm[i] = np.real(alm_no_m[l])
            else : alm[i] = 0
            i+=1
    return alm

##########

def alm_diagonalize(alm_with_m, L_max)-> np.ndarray:
    # Pack into healpy-compatible format
    alm = np.zeros((L_max + 1))
    idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # print(l,m) # DEBUG
            # Simply save only m = 0 items
            if m == 0: alm[l] = np.real(alm_with_m[idx])
            idx += 1

    return alm

#############

def expand_repeating_al_to_alm(al: torch.Tensor, L_max: int) -> torch.Tensor:
    """
    Expands spherical harmonic coefficients al to alm.

    Maps a tensor of shape (L_max+1,) containing values for each degree l
    to a tensor of shape (P,), repeating each al[l] for (2l+1) entries
    corresponding to orders m = -l, ..., l.

    Args:
        al (torch.Tensor): Tensor of shape (L_max + 1,) containing coefficients per degree l.
        L_max (int): The maximum spherical harmonic degree.

    Returns:
        torch.Tensor: Expanded tensor where the value for degree l is repeated (2l+1) times.
                      The resulting shape is (sum_{l=0}^{L_max} (2l+1),) = ((L_max+1)^2,).
    
    Raises:
        AssertionError: If al shape does not match L_max + 1.
    """
    # PEP Optimization: Validate inputs
    assert al.shape[0] == L_max + 1, \
        f"Expected al to have size {L_max + 1}, got {al.shape[0]}"

    # Functional Optimization: 
    # Instead of Python loops, we use vector operations.
    # 1. Create a tensor of counts for each l: [1, 3, 5, ..., 2*L_max + 1]
    # We use the device of al to ensure we don't cause CPU/GPU mismatch errors.
    repeats = torch.arange(0, L_max + 1, device=al.device) * 2 + 1

    # 2. Use repeat_interleave which is highly optimized in C++
    # This avoids creating the explicit index tensor entirely, saving memory and time.
    return torch.repeat_interleave(al, repeats)

def expand_zeros_al_to_alm(al: torch.Tensor, L_max: int) -> torch.Tensor:
    """
    Expands spherical harmonic coefficients al to alm, storing al only for m=0
    and setting all other m!=0 coefficients to zero.

    The order of the resulting tensor follows the standard (l, m) indexing:
    l=0: m=0
    l=1: m=-1, m=0, m=1
    l=2: m=-2, m=-1, m=0, m=1, m=2
    ...

    Args:
        al (torch.Tensor): Tensor of shape (L_max + 1,) containing coefficients per degree l.
        L_max (int): The maximum spherical harmonic degree.

    Returns:
        torch.Tensor: Expanded tensor of shape ((L_max+1)^2,) with al[l] at m=0 positions, 
                      and zeros elsewhere.
    
    Raises:
        AssertionError: If al shape does not match L_max + 1.
    """
    # 1. Input Validation
    assert al.shape[0] == L_max + 1, \
        f"Expected al to have size {L_max + 1}, got {al.shape[0]}"

    # 2. Determine the total size P of the alm tensor
    # P = sum_{l=0}^{L_max} (2l+1) = (L_max + 1)^2
    P = (L_max + 1) ** 2

    # 3. Initialize the result tensor alm with zeros
    # Use the same dtype and device as the input al for consistency and performance
    alm = torch.zeros(P, dtype=al.dtype, device=al.device)

    # 4. Calculate the indices where m = 0 occurs
    # The m=0 index for degree l is: (2l + 1) * l
    # A more robust way is to find the cumulative sum of (2l+1) up to l-1,
    # and then add 'l' (the shift to the m=0 position).
    
    # The length of the alm tensor up to degree l-1 is l^2.
    # The m=0 position is then at index l^2 + l.
    
    # We want indices: 0^2+0, 1^2+1, 2^2+2, ..., L_max^2 + L_max
    
    # Create the tensor of degrees l: [0, 1, 2, ..., L_max]
    l_range = torch.arange(L_max + 1, device=al.device)
    
    # Calculate the m=0 index for each l
    m_zero_indices = l_range ** 2 + l_range

    # 5. Assign the al values to these m=0 indices in the alm tensor
    alm[m_zero_indices] = al

    return alm

# --- Legacy Function (For Comparison) ---
def _legacy_expand_al_to_lm(al: torch.Tensor, L_max: int) -> torch.Tensor:
    l_index_np = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            l_index_np.append(l)
    idx_l = torch.tensor(l_index_np, dtype=torch.long, device=al.device)
    return al[idx_l]

# ==============================
# Debug / test
# ==============================
if __name__ == "__main__":
    theta_test = torch.tensor([0.0, torch.pi/2, torch.pi - 1e-6, torch.pi])
    g_test = 0.9

    phase = map_HenyeyGreenstein(g_test, theta_test)
    print("Phase function values:")
    print(phase)

    alm_coeffs = alm_HenyeyGreenstein(g_test, L_max=10)
    print("\nFirst few alm coefficients (l=0 to 10):")
    print(alm_coeffs)
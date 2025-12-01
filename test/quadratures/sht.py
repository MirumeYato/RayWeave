import torch
import numpy as np

def map2alm(
    maps: np.ndarray,
    Y_H: torch.Tensor,       # (n_pix, (L_max+1)^2), Y_lm(pixel) in order l=0..Lmax, m=-l..l
    weights: torch.Tensor | float,
    dtype=torch.complex128
) -> np.ndarray:
    
    map_torch = maps.to(device=Y_H.device, dtype=dtype)
    
    if isinstance(weights, (int, float)):
        weights = torch.full_like(map_torch, float(weights))
    else:
        weights = weights.to(device=map_torch.device, dtype=map_torch.dtype)

    # Core transform: a_lm = sum_p map(p) * Y_lm(p) * w(p)
    return torch.matmul(map_torch * weights, Y_H)

def alm2map(alm_torch: torch.Tensor, Y: torch.Tensor, dtype = torch.complex128):
    alm_torch = alm_torch.to(dtype=dtype)
    Y = Y.to(dtype=dtype)
    # just summation for lm. No need quadrature
    return torch.einsum('p,qp->q', alm_torch, Y)

def map2map_xn(
    map_torch: np.ndarray,
    n: int,
    Y: torch.Tensor,
    Y_H: torch.Tensor,       # (n_pix, (L_max+1)^2), Y_lm(pixel) in order l=0..Lmax, m=-l..l
    weights: torch.Tensor | float,
) -> np.ndarray:
    
    if isinstance(weights, (int, float)):
        weights = torch.full_like(map_torch, float(weights))
    else:
        weights = weights.to(device=map_torch.device, dtype=map_torch.dtype)

    for i in range(1, n+1):
        # Core transform: a_lm = sum_p map(p) * Y_lm(p) * w(p)
        alm_torch = torch.matmul(map_torch * weights, Y_H)
        # just summation for lm. No need quadrature
        map_torch = torch.einsum('p,qp->q', alm_torch, Y)
    
    # print(f"Final i = {i}")

    return map_torch
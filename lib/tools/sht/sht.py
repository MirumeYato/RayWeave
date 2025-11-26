import torch
import numpy as np

def map2alm(
    maps: np.ndarray,
    L_max: int,
    device,
    Y_H: torch.Tensor,       # (n_pix, (L_max+1)^2), Y_lm(pixel) in order l=0..Lmax, m=-l..l
    weights: torch.Tensor | float,
    dtype=torch.complex128
) -> np.ndarray:
    
    map_torch = torch.from_numpy(maps).to(device=device, dtype=dtype)
    
    if isinstance(weights, (int, float)):
        weights = torch.full_like(map_torch, float(weights))
    else:
        weights = weights.to(device=device, dtype=map_torch.dtype)

    # Core transform: a_lm = sum_p map(p) * Y_lm(p) * w(p)
    alm_torch = torch.matmul(map_torch * weights, Y_H)  # Fast!

    alm_complex = alm_torch.cpu().numpy().astype(np.complex128)

    # Pack into healpy-compatible format
    alm = np.zeros((L_max + 1))
    idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # print(l,m) # DEBUG
            # Simply save only m = 0 items
            if m == 0: alm[l] = np.real(alm_complex[idx])
            idx += 1

    return alm

def alm2map(alm, Y, device, dtype = torch.complex128):
    alm_torch = torch.from_numpy(alm).to(device=device, dtype=dtype)
    # just summation for lm. No need quadrature
    return torch.einsum('p,qp->q', alm_torch, Y).detach().cpu().numpy()

def map2map_xn(
    maps: np.ndarray,
    n: int,
    device,
    Y: torch.Tensor,
    Y_H: torch.Tensor,       # (n_pix, (L_max+1)^2), Y_lm(pixel) in order l=0..Lmax, m=-l..l
    weights: torch.Tensor | float,
    dtype=torch.complex128
) -> np.ndarray:
    
    map_torch = torch.from_numpy(maps).to(device=device, dtype=dtype)
    
    if isinstance(weights, (int, float)):
        weights = torch.full_like(map_torch, float(weights))
    else:
        weights = weights.to(device=device, dtype=map_torch.dtype)

    for i in range(1, n+1):
        # Core transform: a_lm = sum_p map(p) * Y_lm(p) * w(p)
        alm_torch = torch.matmul(map_torch * weights, Y_H)
        # just summation for lm. No need quadrature
        map_torch = torch.einsum('p,qp->q', alm_torch, Y)
    
    # print(f"Final i = {i}")

    return map_torch.detach().cpu().numpy()
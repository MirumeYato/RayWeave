import torch

def prepare_grid(shifts: torch.Tensor, N: int, device=None):
    """
    shifts: (B, 3) on CUDA/CPU
    returns: grids (B, N, N, N, 3)
    """
    device = shifts.device if device is None else device
    B = shifts.shape[0]
    dtype = shifts.dtype

    with torch.no_grad():
        # 1) coordinate rails (4D, broadcast-friendly)
        lin = torch.linspace(-1.0, 1.0, N, device=device, dtype=dtype)
        x = lin.view(1, 1, 1, N)
        y = lin.view(1, 1, N, 1)
        z = lin.view(1, N, 1, 1)

        # 2) tiny base grid (1, N, N, N, 3); expands are views
        base = torch.stack((
            x.expand(1, N, N, N),
            y.expand(1, N, N, N),
            z.expand(1, N, N, N),
        ), dim=-1)  # ~12 MB when N=100 (float32)

        # 3) subtract shifts with a single broadcasted op into a preallocated output
        grids = torch.empty((B, N, N, N, 3), device=device, dtype=dtype)
        torch.sub(base, shifts.view(B, 1, 1, 1, 3), out=grids)  # no extra allocation

    return grids

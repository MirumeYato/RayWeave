import torch

def prepare_grid(shifts: torch.Tensor, spatial_size: int, device=None):
    """
    shifts: (angular_size, 3) on CUDA/CPU
    returns: grids (angular_size, spatial_size, spatial_size, spatial_size, 3)
    """
    device = shifts.device if device is None else device
    angular_size = shifts.shape[0]
    dtype = shifts.dtype

    with torch.no_grad():
        # 1) coordinate rails (4D, broadcast-friendly)
        lin = torch.linspace(-1.0, 1.0, spatial_size, device=device, dtype=dtype)
        x = lin.view(1, 1, 1, spatial_size)
        y = lin.view(1, 1, spatial_size, 1)
        z = lin.view(1, spatial_size, 1, 1)

        # 2) tiny base grid (1, spatial_size, spatial_size, spatial_size, 3); expands are views
        base = torch.stack((
            x.expand(1, spatial_size, spatial_size, spatial_size),
            y.expand(1, spatial_size, spatial_size, spatial_size),
            z.expand(1, spatial_size, spatial_size, spatial_size),
        ), dim=-1)  # ~12 MB when spatial_size=100 (float32)

        # 3) subtract shifts with a single broadcasted op into a preallocated output
        grids = torch.empty((angular_size, spatial_size, spatial_size, spatial_size, 3), device=device, dtype=dtype)
        torch.sub(base, shifts.view(angular_size, 1, 1, 1, 3), out=grids)  # no extra allocation

    return grids

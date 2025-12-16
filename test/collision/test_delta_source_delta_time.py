import os
import sys
import pytest
import torch

# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..', '..'))
sys.path.insert(0, PATH)
#===============================#

# =============================== #
# Project Imports
# =============================== #
from lib.State import FieldState
from lib.grid.Quadrature.Chebishev.QuadratureTdesign import QuadratureTdesign as Angle3D
from lib.Sources.Source import make_point_source
from collision_model import make_collision_model, get_analytical_solution as solution

# =============================== #
# Test Configuration & Fixtures
# =============================== #

# Parameters
PARAMS_G_FIN = [-0.1, 0.1, 0.5, 0.9]
PARAMS_MU_A = [0, 0.1, 1, 10]
PARAMS_MU_S = [0, 0.1, 1, 10]
PARAMS_DT = [1e-7, 1e-5, 1e-3]
ACCURACY_THRESHOLD = 1e-2

@pytest.fixture(scope="module")
def setup_environment():
    """
    Sets up the heavy objects (Device, Angle) once per module execution
    to save time.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Angle (Quadrature)
    Angle = Angle3D(n_size=240, device=device)
    
    point_r0 = 0
    point_s0 = Angle.num_bins // 2 + 5
    
    # Create the source tensor once, as it is reused
    field_tensor = make_point_source(Angle, device, c=point_r0, c2=point_s0)
    
    return {
        "device": device,
        "Angle": Angle,
        "field_tensor": field_tensor,
        "point_r0": point_r0,
        "point_s0": point_s0,
        "Lmax": 10,
        "speed": 1.0,
        "n_time_steps": 1
    }

# =============================== #
# The Test Function
# =============================== #

@pytest.mark.parametrize("g_fin", PARAMS_G_FIN)
@pytest.mark.parametrize("mu_a", PARAMS_MU_A)
@pytest.mark.parametrize("mu_s", PARAMS_MU_S)
@pytest.mark.parametrize("dt", PARAMS_DT)
def test_collision_model_accuracy(setup_environment, g_fin, mu_a, mu_s, dt):
    """
    Test the numerical solution against the theoretical solution for
    various physical parameters.
    """
    env = setup_environment
    device = env["device"]
    Angle = env["Angle"]
    point_r0 = env["point_r0"]
    point_s0 = env["point_s0"]

    # 1. Initialize FieldState
    # We clone the field tensor to ensure previous tests don't dirty the state
    state = FieldState(
        field=env["field_tensor"].clone(),
        dt=dt,
        meta={"L_max": env["Lmax"]}
    )

    # 2. Run Numerical Solution
    propagator = make_collision_model(
        mu_a, mu_s, g_fin, state.dt, 
        env["n_time_steps"], Angle, point_r0, device=device
    )
    
    # We use torch.no_grad to save memory during testing
    with torch.no_grad():
        final_state = propagator.run(state)
        final_field_slice = final_state.field[:, point_r0, point_r0, point_r0].real

        # 3. Run Theoretical Solution
        theory_field = solution(
            state.field[:, point_r0, point_r0, point_r0], 
            mu_a, mu_s, g_fin, env["speed"], dt, env["Lmax"], point_s0,
            Angle, device=device
        )

        # 4. Calculate RMAE
        # Check for zeros in theory to avoid division by zero
        # (Adding a tiny epsilon or masking might be necessary if theory can be 0)
        epsilon = 1e-15
        diff = torch.abs(final_field_slice - theory_field)
        rmae_tensor = diff #/ (torch.abs(theory_field) + epsilon)
        
        # We take the maximum error across the tensor to be strict
        max_rmae = rmae_tensor.max().item()

    # 5. Assertions
    error_msg = (
        f"Accuracy failed for params: g={g_fin}, mua={mu_a}, mus={mu_s}, dt={dt}. "
        f"Max RMAE was {max_rmae:.2e}"
    )
    
    # Check for NaN (divergence)
    assert not torch.isnan(rmae_tensor).any(), f"Simulation diverged (NaN values) for {error_msg}"
    
    # Check Threshold
    assert max_rmae <= ACCURACY_THRESHOLD, error_msg
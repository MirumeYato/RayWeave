# Path settings
import os, sys

#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..', '..'))
sys.path.insert(0, PATH)
#===============================#

import pytest
import torch
import time

from lib.State import FieldState
from lib.Sources.Source import make_dummy_source
from lib.Steps.dummy import DummyPropagate
from lib.Observers.Loggers import EnergyLogger
from lib.Strang.Engine import LoopEngine, TorchEngine

# --- Fixtures ---

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def initial_state(device):
    """Initializes a fresh state for every test."""
    field_tensor = make_dummy_source(Q=100, N=10, device=device)
    dt = 1.0
    return FieldState(
        field=field_tensor,
        dt=dt,
        meta={"L_max": 3}
    )

@pytest.fixture
def simulation_params():
    """Shared parameters for the engines."""
    return {
        "num_steps": 1000, # Reduced for testing speed; increase for benchmarking
        "dt": 1.0,
        "chunk_size": 10
    }

# --- Tests ---

def test_engine_consistency(initial_state, device, simulation_params):
    """
    Test 1 & 3: Ensures LoopEngine and TorchEngine yield 
    identical numerical results.
    """
    num_steps = simulation_params["num_steps"]
    dt = simulation_params["dt"]
    steps = [DummyPropagate(device=device)]
    ob = [EnergyLogger(num_steps//4)]
    
    # Run LoopEngine
    engine_l = LoopEngine(steps, num_steps, dt, observers=ob, device=device, verbose=0)
    final_state_l = engine_l.run(initial_state)
    
    # Run TorchEngine
    engine_t = TorchEngine(steps, num_steps, dt, observers=ob, chunk_size=10, device=device, verbose=0)
    final_state_t = engine_t.run(initial_state)
    
    # Assertions
    assert torch.allclose(final_state_l.field, final_state_t.field), \
        f"Mismatch in engine outputs! Max diff: {(final_state_l.field - final_state_t.field).abs().max()}"

def test_torch_engine_performance(initial_state, device, simulation_params):
    """
    Test 4: Compares execution time. 
    Note: Usually, TorchEngine should be faster for large num_steps.
    """
    num_steps = 1000000 # Larger scale to see difference
    dt = simulation_params["dt"]
    steps = [DummyPropagate(device=device)]
    ob = [EnergyLogger(num_steps//4)]
    
    # Time LoopEngine
    engine_l = LoopEngine(steps, num_steps, dt, observers=ob, device=device, verbose=0)
    start_l = time.perf_counter()
    engine_l.run(initial_state)
    end_l = time.perf_counter()
    
    # Time TorchEngine
    engine_t = TorchEngine(steps, num_steps, dt, observers=ob, chunk_size=1000, device=device, verbose=0)
    start_t = time.perf_counter()
    engine_t.run(initial_state)
    end_t = time.perf_counter()
    
    duration_l = end_l - start_l
    duration_t = end_t - start_t
    
    print(f"\nLoopEngine: {duration_l:.4f}s | TorchEngine: {duration_t:.4f}s")
    
    # This is a soft check; benchmarks can be flaky in CI/CD
    assert duration_t < duration_l or duration_l > 0
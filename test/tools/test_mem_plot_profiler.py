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
import numpy as np
import torch
import time

from lib.tools.mem_plot_profiler import profile_memory_usage

def calculate_theoretical_mb(shape, dtype):
    """Helper to calculate expected MB size of an array"""
    element_size = np.dtype(dtype).itemsize
    total_bytes = np.prod(shape) * element_size
    return total_bytes / (1024**2)

class TestMemoryProfiler:
    
    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_memory_agreement(self, use_gpu):
        """
        Verifies that profiling data agrees with theoretical calculations 
        for NumPy (CPU) and PyTorch (GPU).
        """
        if use_gpu and not torch.cuda.is_available():
            pytest.skip("Skipping GPU test: CUDA not available")

        # 1. Define Theoretical Sizes
        # shape (4000, 4000) float32 -> 64,000,000 bytes ~ 61.03 MB
        cpu_shape = (4000, 4000)
        cpu_dtype = np.float32
        expected_cpu_mb = calculate_theoretical_mb(cpu_shape, cpu_dtype)
        
        # 2. Define Test Simulation
        # We disable plotting to prevent GUI windows during tests
        @profile_memory_usage(interval=0.01, plot=False)
        def simulation():
            # Baseline (Sleep to establish stable start)
            time.sleep(0.2)
            
            # --- Allocation 1: CPU (NumPy) ---
            arr = np.ones(cpu_shape, dtype=cpu_dtype)
            time.sleep(0.5) # Hold memory so sampler catches it
            
            # --- Allocation 2: GPU (PyTorch) ---
            if use_gpu:
                # Move to GPU
                t = torch.tensor(arr).cuda()
                time.sleep(0.5) # Hold memory
                del t
                torch.cuda.empty_cache()

            # Cleanup CPU
            del arr
            time.sleep(0.2)
        
        # 3. Run Simulation
        simulation()
        
        # 4. Retrieve Profiler Data
        data = simulation.profiler_data
        cpu_trace = np.array(data['cpu'])
        gpu_trace = np.array(data['gpu'])
        
        # --- Assertions ---
        
        # CHECK 1: CPU Usage
        # CPU RSS is noisy (Python overhead, OS paging), so we check if the 
        # increase is *at least* the array size (minus a small tolerance).
        baseline_cpu = np.min(cpu_trace[:5]) # Approx start usage
        peak_cpu = np.max(cpu_trace)
        measured_cpu_increase = peak_cpu - baseline_cpu
        
        print(f"\n[CPU] Theoretical: {expected_cpu_mb:.2f} MB | Measured Increase: {measured_cpu_increase:.2f} MB")
        
        # We allow a 10% margin of error downwards (OS compression) 
        # and unlimited upwards (overhead is normal)
        assert measured_cpu_increase >= expected_cpu_mb * 0.9, \
            f"CPU memory did not increase enough. Expected ~{expected_cpu_mb}MB, got {measured_cpu_increase}MB"

        # CHECK 2: GPU Usage
        if use_gpu:
            peak_gpu = np.max(gpu_trace)
            
            print(f"[GPU] Theoretical: {expected_cpu_mb:.2f} MB | Measured Peak: {peak_gpu:.2f} MB")
            
            # GPU memory is usually exact. We check strictly within 1MB.
            # (PyTorch allocates in blocks, so it might be slightly higher, usually 2MB alignment)
            assert abs(peak_gpu - expected_cpu_mb) < 2.0, \
                f"GPU memory mismatch. Expected {expected_cpu_mb}MB, got {peak_gpu}MB"

    def test_scipy_style_allocation(self):
        """Checks if SciPy-style (float64) allocations are tracked."""
        
        # shape (2000, 2000) float64 -> 32MB
        shape = (2000, 2000)
        expected_mb = calculate_theoretical_mb(shape, np.float64)
        
        @profile_memory_usage(interval=0.01, plot=False)
        def scipy_sim():
            time.sleep(0.1)
            # np.random.rand creates float64 by default (standard for SciPy/NumPy)
            arr = np.random.rand(*shape) 
            time.sleep(0.3)
            del arr
            
        scipy_sim()
        
        data = scipy_sim.profiler_data
        increase = np.max(data['cpu']) - np.min(data['cpu'][:5])
        
        print(f"\n[SciPy/Float64] Theoretical: {expected_mb:.2f} MB | Measured: {increase:.2f} MB")
        
        assert increase >= expected_mb * 0.9
# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..', '..'))
sys.path.insert(0, PATH)
#===============================#

import torch
import numpy as np
import time
# Assuming you have the helper function `log_event` defined after the decorator
from lib.tools.mem_plot_profiler import profile_memory_usage, log_event

# Simulate a differential equation solver using Torch and NumPy
@profile_memory_usage(interval=0.01)
def run_simulation(**kwargs): # Must accept kwargs to receive the logger
    log_event("Simulation Start", **kwargs)
    
    # 1. Allocate some CPU memory (NumPy)
    arr_cpu = np.ones((4000, 4000), dtype=np.float32)  # ~61MB
    log_event("NumPy 61MB Allocated", **kwargs)
    time.sleep(0.5)
    
    # 2. Allocate GPU memory (PyTorch)
    if torch.cuda.is_available():
        log_event("Start GPU Transfer", **kwargs)
        # The memory doubles here (61MB on CPU + 61MB on GPU)
        tensor_gpu = torch.tensor(arr_cpu).cuda()
        log_event("GPU Tensor Created", **kwargs)
        time.sleep(0.5)
        
        # 2a. Intermediate GPU operation
        temp = torch.matmul(tensor_gpu, tensor_gpu)
        log_event("Heavy MatMul Op Done", **kwargs)
        time.sleep(0.2)
        
        del temp # Free intermediate GPU memory, watch the line drop!
        log_event("Intermediate GPU Freed", **kwargs)
        time.sleep(0.2)
        
        del tensor_gpu
        log_event("Final GPU Freed", **kwargs)
        torch.cuda.empty_cache()

    # 3. SciPy-style CPU work
    # Simulating post-processing on CPU
    arr_cpu_2 = np.random.rand(5000, 5000) # Another large allocation (~200MB)
    log_event("Float64 Array Allocated (SciPy-style)", **kwargs)
    time.sleep(0.5)
    
    del arr_cpu
    del arr_cpu_2
    log_event("All Memory Cleared", **kwargs)
    
    print("Simulation finished.")

if __name__ == "__main__":
    # Note: If running this in a Jupyter/Colab environment, you may need 
    # to use %matplotlib inline or similar setup for the plot to display.
    run_simulation()
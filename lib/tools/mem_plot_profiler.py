import torch
import psutil
import time
import threading
import os
import matplotlib.pyplot as plt
from functools import wraps

from lib import PATH

def profile_memory_usage(interval=0.1, plot=True, title_suffix=""):
    """
    Decorator to trace CPU and GPU memory usage.
    
    Args:
        interval (float): Sampling interval in seconds.
        plot (bool): Whether to show the plot after execution.
    
    Attaches a .profiler_data attribute to the decorated function containing:
    { 'timestamps': [], 'cpu': [], 'gpu': [] }
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Data containers
            stats = {
                'timestamps': [],
                'cpu': [], # RSS in MB
                'gpu': []  # Allocated in MB
            }
            
            stop_event = threading.Event()
            process = psutil.Process(os.getpid())
            has_gpu = torch.cuda.is_available()
            
            if has_gpu:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            def monitor():
                start_time = time.time()
                while not stop_event.is_set():
                    current_time = time.time() - start_time
                    stats['timestamps'].append(current_time)
                    
                    # Record CPU (RSS in MB)
                    mem_info = process.memory_info()
                    stats['cpu'].append(mem_info.rss / 1024**2)
                    
                    # Record GPU (Allocated in MB)
                    if has_gpu:
                        gpu_mem = torch.cuda.memory_allocated() / 1024**2
                        stats['gpu'].append(gpu_mem)
                    else:
                        stats['gpu'].append(0)
                    
                    time.sleep(interval)

            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()
            
            try:
                result = func(*args, **kwargs)
            finally:
                stop_event.set()
                monitor_thread.join()
                
                # Attach data to the wrapper for inspection by tests
                wrapper.profiler_data = stats
                
                if plot:
                    _plot_results(stats, func.__name__, has_gpu)

            return result

        def _plot_results(stats, func_name, has_gpu):
            plt.figure(figsize=(10, 8))
            
            # CPU Plot
            plt.subplot(2, 1, 1)
            plt.plot(stats['timestamps'], stats['cpu'], label='CPU (RSS)', color='blue')
            plt.ylabel('Memory (MB)')
            plt.title(f'Memory Profiling: {func_name}')
            plt.legend()
            plt.grid(True)
            
            # GPU Plot
            plt.subplot(2, 1, 2)
            if has_gpu:
                plt.plot(stats['timestamps'], stats['gpu'], label='GPU (Allocated)', color='green')
                peak = max(stats['gpu']) if stats['gpu'] else 0
                plt.axhline(y=peak, color='red', linestyle='--', alpha=0.5, label=f'Peak: {peak:.1f} MB')
            else:
                plt.text(0.5, 0.5, 'No GPU detected', ha='center')
                
            plt.ylabel('Memory (MB)')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(PATH, "output", "mem_prof.png"))
            plt.close()

        return wrapper
    return decorator
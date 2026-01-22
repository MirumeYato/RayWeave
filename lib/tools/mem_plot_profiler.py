import torch
import psutil
import time
import threading
import os
import matplotlib.pyplot as plt
from functools import wraps

from lib import PATH

# Define a global list to hold the event data logged from the main function.
# This makes it easy to pass data out of the monitoring scope for plotting.
# Using a dict to store both time and label.
class EventLogger:
    def __init__(self):
        self.events = []
        self.start_time = None
        
    def log(self, label):
        """Log an event with its time relative to the function start."""
        if self.start_time is not None:
            relative_time = time.time() - self.start_time
            self.events.append({'time': relative_time, 'label': label})
            print(f"--- Event Logged at {relative_time:.3f}s: {label} ---")
        else:
            # Should not happen if called correctly within the wrapper
            print("Warning: EventLogger not initialized. Event ignored.")

# The core decorator function (replacing the previous one)
def profile_memory_usage(interval=0.1, plot=True, title_suffix="", verbose = 0):
    """
    Decorator to trace CPU and GPU memory usage.
    
    Args:
        interval (float): Sampling interval in seconds.
        plot (bool): Whether to show the plot after execution.
    
    Attaches a .profiler_data attribute to the decorated function containing:
    { 'timestamps': [], 'cpu': [], 'gpu': [] }
    """
    if not verbose: return lambda func: func

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Initialize Event Logger and Profiler Data
            event_logger = EventLogger() # Initialize our logger
            stats = {
                'timestamps': [],
                'cpu': [], 
                'gpu': []
            }
            
            # ... (Rest of setup remains the same: stop_event, process, has_gpu, CUDA reset) ...
            stop_event = threading.Event()
            process = psutil.Process(os.getpid())
            has_gpu = torch.cuda.is_available()
            
            if has_gpu:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            # -------------------------------------------------------------

            def monitor():
                event_logger.start_time = time.time() # Sync monitor start time with main thread
                while not stop_event.is_set():
                    current_time = time.time() - event_logger.start_time # Use event_logger's start time
                    # ... (Monitoring logic remains the same) ...
                    stats['timestamps'].append(current_time)
                    mem_info = process.memory_info()
                    stats['cpu'].append(mem_info.rss / 1024**2)
                    
                    if has_gpu:
                        stats['gpu'].append(torch.cuda.memory_allocated() / 1024**2)
                    else:
                        stats['gpu'].append(0)
                    
                    time.sleep(interval)

            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()
            
            try:
                # 2. Call the function, passing the logger instance
                # We inject the logger as the first argument, or a specific keyword argument.
                # For simplicity, let's assume the function is modified to accept it.
                # For non-intrusive injection, we use a global/thread-local, but for robustness:
                kwargs['mem_event_logger'] = event_logger # Inject as keyword arg
                result = func(*args, **kwargs)
                
            finally:
                stop_event.set()
                monitor_thread.join()
                
                wrapper.profiler_data = stats
                wrapper.event_data = event_logger.events # Store events for plotting/testing
                
                if plot:
                    _plot_results(stats, event_logger.events, func.__name__, has_gpu, title_suffix)

            return result

        def _plot_results(stats, events, func_name, has_gpu, title_suffix):
            plt.figure(figsize=(10, 8))
            
            # --- Plot CPU ---
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(stats['timestamps'], stats['cpu'], label='CPU (RSS)', color='blue')
            ax1.set_ylabel('CPU Memory (MB)')
            ax1.set_title(f'Memory Profiling: {func_name} {title_suffix}')
            ax1.grid(True)
            
            # --- Plot GPU ---
            ax2 = plt.subplot(2, 1, 2, sharex=ax1) # Share X-axis for alignment
            if has_gpu:
                ax2.plot(stats['timestamps'], stats['gpu'], label='GPU (Allocated)', color='green')
            else:
                ax2.text(0.5, 0.5, 'No GPU detected', ha='center', transform=ax2.transAxes)
                
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.set_xlabel('Time (s)')
            ax2.grid(True)

            # --- Add Vertical Event Lines to Both Plots ---
            for event in events:
                time_val = event['time']
                label = event['label']
                
                # Draw on CPU plot
                ax1.axvline(x=time_val, color='r', linestyle='--', linewidth=1, alpha=0.7)
                
                # Draw on GPU plot
                if has_gpu:
                    ax2.axvline(x=time_val, color='r', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')

                # Add Text Annotation (only once, above the top plot)
                # Use a small offset for readability
                ax1.annotate(label, (time_val, ax1.get_ylim()[1] * 0.95), 
                             xytext=(3, 0), textcoords='offset points', 
                             rotation=90, va='top', ha='left', fontsize=8, color='red')

            # Ensure all legends are drawn
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            plt.tight_layout()
            if title_suffix: plt.savefig(os.path.join(PATH, "output", f"mem_prof_{func_name}_{title_suffix}.png"))
            else: plt.savefig(os.path.join(PATH, "output", f"mem_prof_{func_name}.png"))
            plt.close()

        return wrapper
    return decorator

# Helper function for easier logging in your main code
def log_event(label, **kwargs):
    """
    Looks for the injected event logger in the kwargs of the decorated function.
    """
    logger = kwargs.get('mem_event_logger')
    if logger:
        logger.log(label)
    # If the logger isn't found, the event is silently ignored (useful outside the decorator)
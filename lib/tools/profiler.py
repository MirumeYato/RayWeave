import time
import torch
import psutil
import functools
import gc
from typing import Optional

def _find_cuda_device_in_args(args, kwargs) -> Optional[int]:
    """
    Try to find a CUDA device index from any Tensor/Module/torch.device in args/kwargs.
    Falls back to torch.cuda.current_device() if CUDA is initialized.
    """
    def _check_obj(obj) -> Optional[int]:
        if isinstance(obj, torch.device) and obj.type == "cuda":
            return obj.index if obj.index is not None else (
                torch.cuda.current_device() if torch.cuda.is_initialized() else None
            )
        if torch.is_tensor(obj) and obj.is_cuda:
            return obj.device.index
        # nn.Module-like
        if hasattr(obj, "parameters"):
            for p in obj.parameters():
                if torch.is_tensor(p) and p.is_cuda:
                    return p.device.index
        return None

    # scan positional args
    for obj in args:
        idx = _check_obj(obj)
        if idx is not None:
            return idx

    # scan keyword args (values)
    for obj in kwargs.values():
        idx = _check_obj(obj)
        if idx is not None:
            return idx

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.cuda.current_device()
    return None

def profile_time_and_memory(func):
    """
    Decorator: measure wall time, CPU RSS delta, and detailed CUDA memory usage.
    Reports allocated vs reserved and peak deltas relative to start-of-call.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()

        # CPU before
        cpu_before = process.memory_info().rss / (1024 ** 2)  # MB

        # Which CUDA device (if any)
        dev_index = _find_cuda_device_in_args(args, kwargs)
        use_cuda = dev_index is not None

        # GPU before
        if use_cuda:
            torch.cuda.synchronize(dev_index)
            torch.cuda.reset_peak_memory_stats(dev_index)
            alloc_before = torch.cuda.memory_allocated(dev_index)
            reserv_before = torch.cuda.memory_reserved(dev_index)
        else:
            alloc_before = reserv_before = 0

        # Time start
        if use_cuda:
            torch.cuda.synchronize(dev_index)
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        # Time end
        if use_cuda:
            torch.cuda.synchronize(dev_index)
        t1 = time.perf_counter()

        # CPU after (post-GC to reduce noise)
        gc.collect()
        cpu_after = process.memory_info().rss / (1024 ** 2)
        cpu_delta = cpu_after - cpu_before

        # GPU after + peaks
        if use_cuda:
            alloc_after = torch.cuda.memory_allocated(dev_index)
            reserv_after = torch.cuda.memory_reserved(dev_index)

            peak_alloc_abs = torch.cuda.max_memory_allocated(dev_index)
            get_max_reserved = getattr(torch.cuda, "max_memory_reserved", None)
            peak_reserv_abs = get_max_reserved(dev_index) if get_max_reserved else reserv_after

            peak_alloc_delta = max(0, peak_alloc_abs - alloc_before)
            peak_reserv_delta = max(0, peak_reserv_abs - reserv_before)

            alloc_delta = alloc_after - alloc_before
            reserv_delta = reserv_after - reserv_before

        # Print
        print(f"\n🧠 Profiling report for `{func.__name__}`:")
        print(f"⏱️  Time elapsed: {t1 - t0:.4f} sec")
        print(f"💻 CPU RSS change: {cpu_delta:+.2f} MB (current {cpu_after:.2f} MB)")
        if use_cuda:
            to_mb = lambda b: b / (1024 ** 2)
            print(
                "🧩 CUDA memory (device {}):\n"
                "   • Allocated: {:+.2f} MB (before {:+.2f}, after {:+.2f})\n"
                "   • Reserved : {:+.2f} MB (before {:+.2f}, after {:+.2f})\n"
                "   • Peak during call — allocated: {:+.2f} MB over start (abs {:+.2f} MB)\n"
                "                       reserved : {:+.2f} MB over start (abs {:+.2f} MB)".format(
                    dev_index,
                    to_mb(alloc_delta), to_mb(alloc_before), to_mb(alloc_after),
                    to_mb(reserv_delta), to_mb(reserv_before), to_mb(reserv_after),
                    to_mb(peak_alloc_delta), to_mb(peak_alloc_abs),
                    to_mb(peak_reserv_delta), to_mb(peak_reserv_abs),
                )
            )
        print()
        return result
    return wrapper

from __future__ import annotations

from lib import PATH
import os

OUTPUT = os.path.abspath(os.path.join(PATH, 'output'))

# from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

def _maybe_plot(tracks: np.ndarray) -> None:
    if not _HAS_MPL:
        print("matplotlib not available; skipping plot")
        return
    # One figure per demo; lines for each particle
    plt.figure()
    for pid in range(tracks.shape[0]):
        xy = tracks[pid]
        plt.plot(xy[:, 0], xy[:, 1], marker="o", label=f"particle {pid}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D straight-line tracks (demo)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def maps_rmae(mae: np.ndarray, rtt_iter_num: int, output_path = OUTPUT, rmae_flag = True):
    plt.figure(figsize=(8, 5))
    
    if rmae_flag: plt.plot(mae, 
                label=r"$\frac{|f^{OrigHP} - f^{true}|}{f^{true}}$", color='C0')
    else: plt.plot(mae, 
                label=r"$|f^{OrigHP} - f^{true}|$", color='C0')
    plt.title(f"Round-trip consistency: map → alm → map ({rtt_iter_num} times)", fontsize=14)
    plt.xlabel("Pixel index", fontsize=12)
    plt.ylabel("Absolute error", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"Mae_comparison_map2map_x{rtt_iter_num}.png"), dpi=150)
    plt.close()
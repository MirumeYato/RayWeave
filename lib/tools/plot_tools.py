from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
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
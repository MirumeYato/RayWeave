"""
Minimal, extensible framework for 2D photon propagation with pluggable Grid,
Step, Source, and BackupData abstractions.

This file implements:
- Base abstract classes: Grid, Step, Source, Propagator, BackupData
- Concrete simple case for 2 particles, straight-line motion at speed c
- A TrackBackup that stores positions per timestep for plotting

Run this file directly to execute a tiny demo (2 particles, 3 steps) and plot tracks.
"""
from __future__ import annotations

# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from lib.physics.Grids import ParticleDict2D
from lib.physics.Sources import TwoPointSource
from lib.physics.Steps import StraightLineStep
from lib.physics.Propogators import SimplePropagator
from lib.results.BackupData import TrackBackup
from lib.results.plot_tools import _maybe_plot

# -----------------------------
# Demo / Example Usage
# -----------------------------

def run_demo() -> Tuple[np.ndarray, List[float]]:
    """Run a tiny 2-particle, 5-step straight-line demo.

    Returns
    -------
    tracks : np.ndarray
        Array of shape (2, 4, 2) with positions per particle per time index (t0..t3).
    times : list[float]
        List of time stamps [t0, t1, t2, t3].
    """
    # Units: set c=1.0 for simplicity
    c = 1.0
    dt = 1.0
    n_steps = 5
    t0 = 0.0

    grid = ParticleDict2D()
    source = TwoPointSource(
        pos1=(0.0, 0.0), dir1=(1.0, 0.0), speed1=c,
        pos2=(0.0, 1.0), dir2=(0.0, -1.0), speed2=c,
    )
    steps = [StraightLineStep()]
    backup = TrackBackup(n_expected_steps=n_steps)

    prop = SimplePropagator()
    prop.run(grid=grid, sources=[source], steps=steps, backup=backup, t_start=t0, dt=dt, n_steps=n_steps)

    tracks = backup.get_tracks()
    times = backup.get_times()
    return tracks, times

if __name__ == "__main__":
    trk, tt = run_demo()
    print("times:", tt)
    print("tracks shape:", trk.shape)
    print("tracks (particle, time, xy):\n", trk)
    _maybe_plot(trk)

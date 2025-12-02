from __future__ import annotations

from lib import Observer
from lib.State import FieldState

# import numpy as np

# -----------------------------
# List of BackupData classes:
# -----------------------------

class EnergyLogger(Observer):
    def __init__(self, every: int = 50):
        self.every = every
    def on_step_end(self, step_idx: int, state: FieldState) -> None:
        if step_idx % self.every == 0:
            # cheap summary; no sync if possible
            total = state.field.real.sum().detach().item()
            print(f"[{step_idx}] total_energy={total:.6e}")


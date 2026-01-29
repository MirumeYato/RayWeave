from __future__ import annotations

from lib import Observer
from lib.State import Field

# import numpy as np

# -----------------------------
# List of BackupData classes:
# -----------------------------

class EnergyLogger(Observer):
    """Prints total energy each 'every'"""
    def __init__(self, every: int = 50):
        super().__init__(every=every)

    def on_step_end(self, step_idx: int, field: Field) -> None:
        if step_idx % self.every == 0:
            # cheap summary; no sync if possible
            total = field.real.sum().detach().item()
            print(f"[{step_idx}] total_energy={total:.6e}")


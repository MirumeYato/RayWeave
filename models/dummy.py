from __future__ import annotations

# from dataclasses import dataclass, field
from typing import List

from lib import Step, Model, Sequential
from lib.Strang.Engine import StrangEngine
from lib.Observers.temprorary_test import EnergyLogger
from lib.Steps.dummy import DummyPropagate

def make_dev_dummy_model(dt: float, n_steps: int, device) -> StrangEngine:
    """
    Simpliest propogater pipeline example. Do nothing, just pushes same FieldState further
    """
    steps: List[Step] = [
        DummyPropagate(device),
        # here can be any steps you want
    ]
    observers = [EnergyLogger(every=1)]
    return StrangEngine(steps, n_steps, dt, observers,
                      device=device, compile_fused=True, use_cuda_graph=True)

from lib.State import FieldState

class Dummy_Model(Model):
    def __init__(self, T: Step, num_time_steps:int, dt:float, **kw):
        super().__init__(steps=[], num_time_steps=num_time_steps, dt=dt, **kw)
        self.test = T
    def forward(self, state: FieldState) -> FieldState:
        # symmetric composition per step (A half, B full, A half)
        # Here we assume Steps read `state.dt` and internally use a factor.
        state = self.test(state)
        return state
    
device = "cpu"

model = Dummy_Model(DummyPropagate(device), observers = [EnergyLogger()], num_time_steps=2, dt=0.01, use_cuda_graph = True)

layers = [DummyPropagate(device),DummyPropagate(device)]
obs = [EnergyLogger()]
model_seq = Sequential(layers, num_time_steps=100, dt=0.01, observers=obs, device="cuda", use_cuda_graph = True)
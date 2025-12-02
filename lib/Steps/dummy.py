from .Step import Step
from lib.State import FieldState

class DummyPropagate(Step):
    """
    Do nothing. Just push forward
    """
    def __init__(self, device = "cpu"):
        self.device = device # do not forget to initialize device here if you want to use specific one

    def forward(self, state: FieldState) -> FieldState:
        return FieldState(state.field, state.dt, state.meta)
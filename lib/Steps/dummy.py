from lib import Step
from lib.State import FieldState

class DummyPropagate(Step):
    """
    Do nothing. Just push forward
    """
    def __init__(self):
        pass

    def forward(self, state: FieldState) -> FieldState:
        return FieldState(state.field, state.t + state.dt, state.dt, state.meta)
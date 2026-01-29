from .Step import Step
from lib.State import FieldState, Field

class DummyPropagate(Step):
    """
    Do nothing. Just push forward
    """
    def __init__(self, device = "cpu", verbose = 0):        
        super().__init__(device, verbose)

    def forward(self, field: Field) -> Field:
        return field+1. # field.add_(1.)
import numpy as np
from numpy.typing import NDArray

class OutputParameters:
    start: float  # start altitude
    stop: float  # stop altitude
    step: float  # step altitude
    altitudes: NDArray[np.float64]  # Output altitudes
    Nz: int  # number of output altitudes
    Cheb_lower: NDArray[np.complex128]  # at lower altitudes
    Cheb_upper: NDArray[np.complex128]  # at upper altitudes

    def __init__(self, start: float, stop: float, step: float): ...

    def validate(self) -> int: ...

    def set_altitudes(self): ...

    def ChebMats(self, N: int, H: float): ...
        
    def describe(self): ...

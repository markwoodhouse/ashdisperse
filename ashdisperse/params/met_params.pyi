import numpy as np
from numpy.typing import NDArray

class MetParameters:
    U_scale: np.float64
    Ws_scale: NDArray[np.float64]

    def __init__(self, U_scale: np.float64, Ws_scale: NDArray[np.float64]): ...


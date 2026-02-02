import numpy as np
from params import Parameters

class ModelParameters:
    SettlingScale: list[np.float64]
    Velocity_ratio: list[np.float64]
    xScale: list[np.float64]
    yScale: list[np.float64]
    Lx: list[np.float64]
    Ly: list[np.float64]
    cScale: list[np.float64]
    QScale: list[np.float64]
    Peclet_number: np.float64
    Diffusion_ratio: np.float64
    sigma_hat: list[np.float64]
    sigma_hat_scale: list[np.float64]


    def __init__(self): ...

    def from_params(self, params: Parameters, xScale: list[np.float64], yScale: list[np.float64]): ...

    def from_values(
        self,
        SettlingScale: np.float64,
        Velocity_ratio: np.float64,
        xScale: np.float64,
        yScale: np.float64,
        Lx: np.float64,
        Ly: np.float64,
        cScale: np.float64,
        QScale: np.float64,
        Peclet_number: np.float64,
        Diffusion_ratio: np.float64,
        sigma_hat: np.float64,
        sigma_hat_scale: np.float64,
    ): ...

    def from_lists(self,
        SettlingScale: list[np.float64], 
        Velocity_ratio: list[np.float64], 
        xScale: list[np.float64], 
        yScale: list[np.float64], 
        Lx: list[np.float64], 
        Ly: list[np.float64], 
        cScale: list[np.float64], 
        QScale: list[np.float64], 
        Peclet_number: np.float64, 
        Diffusion_ratio: np.float64, 
        sigma_hat: list[np.float64], 
        sigma_hat_scale: list[np.float64], 
    ): ...

    def _add_to_list(self,
        SettlingScale: np.float64,
        Velocity_ratio: np.float64,
        xScale: np.float64,
        yScale: np.float64,
        Lx: np.float64,
        Ly: np.float64,
        cScale: np.float64,
        QScale: np.float64,
        sigma_hat: np.float64,
        sigma_hat_scale: np.float64,
    ): ...

    def _empty_lists(self, N: int): ...
    
    def describe(self): ...
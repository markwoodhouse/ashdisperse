
from collections import OrderedDict

import numpy as np
from numba import boolean, float64, int64
from numba.experimental import jitclass

solver_spec = OrderedDict()
solver_spec["domX"] = float64  # dimensionless domain length in x
solver_spec["domY"] = float64  # dimensionless domain length in y
solver_spec["minN_log2"] = int64  # log2 minimum number of Chebyshev points
solver_spec["maxN_log2"] = int64  # log2 maximum number of Chebyshev points
solver_spec["Nx_log2"] = int64  # log2 of number of points in x
solver_spec["Ny_log2"] = int64  # log2 of number of points in y
solver_spec["epsilon"] = float64  # tolerance for converged spectral series
solver_spec["plateau_factor"] = float64  # factor above noise plateau to treat as meaningful (e.g. 10 for conservative, 3 for aggressive)
solver_spec["fft_tol"] = float64  # tolerance for fft terms
solver_spec["meps"] = float64  # Machine epsilon


@jitclass(solver_spec)
class SolverParameters:
    def __init__(
        self,
        domX=1.5,
        domY=1.5,
        minN_log2=4,
        maxN_log2=8,
        Nx_log2=8,
        Ny_log2=8,
        epsilon=1e-8,
        plateau_factor=10.0,
        fft_tol=1e-10,
    ):
        self.meps = np.finfo(np.float64).eps

        self.domX = np.float64(domX)  # Dimensionless domain size in x
        self.domY = np.float64(domY)  # Dimensionless domain size in y

        self.minN_log2 = np.int64(minN_log2)  # Minimum z-resolution (log2)
        self.maxN_log2 = np.int64(maxN_log2)  # Maximum z-resolution (log2)

        self.Nx_log2 = np.int64(Nx_log2)  # x-resolution (log2)
        self.Ny_log2 = np.int64(Ny_log2)  # y-resolution (log2)

        self.epsilon = np.float64(epsilon)
        self.plateau_factor = np.float64(plateau_factor)
        self.fft_tol = np.float64(fft_tol)

    def validate(self):
        if self.domX < 0:
            raise ValueError("In SolverParameters, must have domX>0")
        if self.domY < 0:
            raise ValueError("In SolverParameters, must have domY>0")
        if self.minN_log2 < 0:
            raise ValueError("In SolverParameters, must have minN_log2>0")
        if self.maxN_log2 < 0:
            raise ValueError("In SolverParameters, must have maxN_log2>0")
        if self.minN_log2 > self.maxN_log2:
            raise ValueError("In SolverParameters, must have minN_log2 < maxN_log2")
        if self.Nx_log2 < 0:
            raise ValueError("In SolverParameters, must have Nx_log2>0")
        if self.Ny_log2 < 0:
            raise ValueError("In SolverParameters, must have Ny_log2>0")
        if self.epsilon < 0:
            raise ValueError("In SolverParameters, must have epsilon>0")
        if self.plateau_factor < 1:
            raise ValueError("In SolverParameters, must have plateau_factor>=1")
        if self.epsilon < self.meps:
            print(
                f"In SolverParameters, must have epsilon >= machine epsilon = {self.meps}\n \
                Setting epsilon = {self.meps}")
            self.epsilon = self.meps
        if self.fft_tol < 0:
            raise ValueError("In SolverParameters, must have fft_tol>0")
        # if self.fft_tol < 10 * self.meps:
        #     raise ValueError(
        #         f"In SolverParameters, must have fft_tol >= 10*machine epsilon = {10 * self.meps}"
        #     )
        return 1        

    @property
    def Nx(self):
        return 2**self.Nx_log2
    
    @property
    def Ny(self):
        return 2**self.Ny_log2

    @property
    def minN(self):
        return 2**self.minN_log2
    
    @property
    def maxN(self):
        return 2**self.maxN_log2
    
    @property
    def chebIts(self):
        return self.maxN_log2 - self.minN_log2 + 1

    def describe(self):
        print("Solver parameters for AshDisperse")
        print("  Dimensionless domain size in x, domX = ", self.domX)
        print("  Dimensionless domain size in y, domY = ", self.domY)
        print(
            "  Minimum resolution in z, minN = ",
            self.minN,
            " (minN_log2 = ",
            self.minN_log2,
            ")",
        )
        print(
            "  Maximum resolution in z, maxN = ",
            self.maxN,
            " (maxN_log2 = ",
            self.maxN_log2,
            ")",
        )
        print("  Number of Chebyshev iterates = ", self.chebIts)
        print("  Tolerance for Chebyshev series, epsilon = ", self.epsilon)
        print("  Noise plateau factor for Chebyshev series, plateau_factor = ", self.plateau_factor)
        print("  Tolerance for FFT terms, fft_tol = ", self.fft_tol)
        print("  Resolution in x, Nx = ", self.Nx, " (Nx_log2 = ", self.Nx_log2, ")")
        print("  Resolution in y, Ny = ", self.Ny, " (Ny_log2 = ", self.Ny_log2, ")")
        print("********************")


# pylint: disable=E1101
SolverParameters_type = SolverParameters.class_type.instance_type

def _solver_dict(p):
    return {
        'domX': float(p.domX),
        'domY': float(p.domY),
        'minN_log2': int(p.minN_log2),
        'maxN_log2': int(p.maxN_log2),
        'Nx_log2': int(p.Nx_log2),
        'Ny_log2': int(p.Ny_log2),
        'epsilon': float(p.epsilon),
        'plateau_factor': float(p.plateau_factor),
        'fft_tol': float(p.fft_tol),
    }

def _solver_params_equal(p1: SolverParameters, p2: SolverParameters) -> bool:
    test = (
        (p1.domX == p2.domX) and
        (p1.domY == p2.domY) and
        (p1.minN_log2 == p2.minN_log2) and
        (p1.maxN_log2 == p2.maxN_log2) and
        (p1.Nx_log2 == p2.Nx_log2) and
        (p1.Ny_log2 == p2.Ny_log2) and
        (p1.epsilon == p2.epsilon) and
        (p1.plateau_factor == p2.plateau_factor) and
        (p1.fft_tol == p2.fft_tol)
    )
    return test
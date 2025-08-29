from collections import OrderedDict

import numpy as np
from numba import complex128, float64, int64, boolean
from numba.experimental import jitclass

from ..spectral.cheb import ChebMat

output_spec = OrderedDict()
output_spec["start"] = float64  # start altitude
output_spec["stop"] = float64  # stop altitude
output_spec["step"] = float64  # step altitude
output_spec["altitudes"] = float64[::1]  # Output altitudes
output_spec["Nz"] = int64  # number of output altitudes
# Chebyshev matrices
output_spec["Cheb_lower"] = complex128[:, ::1]  # at lower altitudes
output_spec["Cheb_upper"] = complex128[:, ::1]  # at upper altitudes


@jitclass(output_spec)
class OutputParameters:
    def __init__(self, start=0, stop=40e3, step=1e3):

        self.start = np.float64(start)
        self.stop = np.float64(stop)
        self.step = np.float64(step)

        self.set_altitudes()

    def validate(self):
        if self.start<0:
            raise ValueError("In OuputParameters, start must be positive")
        if self.stop<self.start:
            raise ValueError("In OuputParameters, stop must be larger than start")
        if self.step<0:
            raise ValueError("In OuputParameters, step must be >= 0")
        return 1
        

    def set_altitudes(self):
        if self.stop == self.start:
            self.altitudes = np.asarray([0], dtype=np.float64)
        else:
            self.altitudes = np.arange(
                self.start, self.stop + self.step, self.step, dtype=np.float64
            )
        self.Nz = len(self.altitudes)

    def ChebMats(self, N, H):
        z = self.altitudes
        z_lower = z[z <= H]
        z_upper = z[z > H]

        x_lower = 2 * z_lower / H - 1
        x_upper = 1 - 2 * H / z_upper

        self.Cheb_lower = ChebMat(N, x_lower)
        self.Cheb_upper = ChebMat(N, x_upper)

    def describe(self):
        print("Output parameters for AshDisperse")
        print("  Altitudes (m): ")
        for z in self.altitudes:
            print("    ", z)
        print("********************")


# pylint: disable=E1101
OutputParameters_type = OutputParameters.class_type.instance_type

def _output_dict(p):
        return {
            "start": float(p.start),
            "stop": float(p.stop),
            "step": float(p.step),
        }

def _output_params_equal(p1: OutputParameters, p2: OutputParameters) -> bool:
    test = (
        (p1.start == p2.start) and
        (p1.stop == p2.stop) and
        (p1.step == p2.step)
    )
    return test
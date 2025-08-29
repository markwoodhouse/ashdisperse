from collections import OrderedDict

import numpy as np
from numba import float64
from numba.experimental import jitclass

phys_spec = OrderedDict()
phys_spec["Kappa_h"] = float64  # Horizontal diffusion coefficient
phys_spec["Kappa_v"] = float64  # Vertical diffusion coefficient
phys_spec["g"] = float64  # Gravitational acceleration
phys_spec["mu"] = float64  # Viscosity of air


@jitclass(phys_spec)
class PhysicalParameters:
    def __init__(self, Kappa_h=100, Kappa_v=10, g=9.81, mu=18.5e-6):
        self.Kappa_h = np.float64(Kappa_h)
        self.Kappa_v = np.float64(Kappa_v)
        self.g = np.float64(g)
        self.mu = np.float64(mu)

    def validate(self):
        if self.Kappa_h < 0:
            raise ValueError("In PhysicalParameters, Kappa_h must be positive")
        if self.Kappa_v < 0:
            raise ValueError("In PhysicalParameters, Kappa_v must be positive")
        if self.g <= 0:
            raise ValueError("In PhysicalParameters, g must be positive")
        if self.mu <= 0:
            raise ValueError("In PhysicalParameters, mu must be positive")
        return 1

    def describe(self):
        print("Physical parameters for AshDisperse")
        print("  Horizontal diffusion coefficient Kappa_h = ", self.Kappa_h, " m^2/s")
        print("  Vertical diffusion coefficient Kappa_v = ", self.Kappa_v, " m^2/s")
        print("  Gravitational acceleration g = ", self.g, " m/s^2")
        print("  Viscosity of air mu = ", self.mu, " kg/m/s")
        print("********************")


# pylint: disable=E1101
PhysicalParameters_type = PhysicalParameters.class_type.instance_type

def _physical_dict(self):
        return {
            "Kappa_h": float(self.Kappa_h),
            "Kappa_v": float(self.Kappa_v),
            "g": float(self.g),
            "mu": float(self.mu),
        }


def _physical_params_equal(p1: PhysicalParameters, p2: PhysicalParameters) -> bool:
    test = (
        (p1.Kappa_h == p2.Kappa_h) and
        (p1.Kappa_v == p2.Kappa_v) and
        (p1.g == p2.g) and
        (p1.mu == p2.mu)
    )
    return test
# -*- coding: utf-8 -*-
"""Grain parameters for AshDisperse.

Defines the GrainParameters class, containing attributes specifying grains
classes and methods for amending these, and a printer for this object.

Example:
    grains = GrainParameters()
    grains.add_grains(1e-4,1400,0.5)
    grains.add_grains(1e-3,1400,0.5)

    print_grain_parameters(grains)
"""

from collections import OrderedDict

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba.typed import List
from numba.types import ListType

grain_spec = OrderedDict()
grain_spec["bins"] = int64  # Number of grain classes
grain_spec["diameter"] = ListType(float64)  # Grain sizes
grain_spec["density"] = ListType(float64)  # Grain densities
grain_spec["proportion"] = ListType(float64)  # Grain proportions
"""OrderedDict: GrainParameters attribute type specification.

This specification is required for numba jitting.
"""


@jitclass(grain_spec)
class GrainParameters:
    """Grain parameters required for AshDisperse.

    Defines the GrainParameters class, containing attributes specifying grains
    classes and methods for amending these.

    Attributes:
        bins: An integer count of the number of grain classes.
        diameter: A list of floats containing the diameter of each grain class.
        density: A list of floats containing the density of each grain class.
        proportion: A list of floats containing the proportion of each grain
                    class.
    """

    def __init__(self):
        """Initialize of GrainParameters to empty lists.

        Use add_grain to add grains.
        """
        self.bins = 0
        self.diameter = List.empty_list(np.float64)
        self.density = List.empty_list(np.float64)
        self.proportion = List.empty_list(np.float64)

    def len(self):
        return len(self.diameter)

    def from_values(self, diameter, density, proportion):
        self.add_grain(diameter, density, proportion)
        return
    
    def from_lists(self, diameters, densities, proportions):
        """Initialize grain parameters from lists"""

        if len(diameters) != len(densities):
            raise ValueError("Size of diameters and densities not equal")
        if len(diameters) != len(proportions):
            raise ValueError("Size of diameters and proportions not equal")
        if len(proportions) != len(densities):
            raise ValueError("Size of densities and proportions not equal")
        
        self.clear()

        for j, (d, rho, p) in enumerate(zip(diameters, densities, proportions)):
            self.add_grain(d, rho, p)

        return
    
    def clear(self):
        self.diameter = List.empty_list(np.float64)
        self.density = List.empty_list(np.float64)
        self.proportion = List.empty_list(np.float64)
        self.bins = 0

    def add_grain(self, diameter, density, proportion):
        """Add a grain class to the GrainParameters object.

        A grain class requires a diameters, density and proportion.
        The proportions must sum to one.
        Adding a grain increments the number of classes through the bins
        attribute.

        Args:
            diameter: [float64] The diameter of the grain class;
                must be positive.
            density: [float64] The density of the grain class;
                must be positive.
            proportion: [float64] The proportion of the total mass in this
                grain class; must be in the range (0,1].
        """
        if diameter < 0:
            raise ValueError("Grain diameter must be positive")
        if density < 0:
            raise ValueError("Grain density must be positive")
        if proportion < 0:
            raise ValueError("Grain proportion must be in [0,1]")
        if proportion > 1:
            raise ValueError("Grain proportion must be in [0,1]")
        self.diameter.append(np.float64(diameter))
        self.density.append(np.float64(density))
        self.proportion.append(np.float64(proportion))
        self.bins += 1

    def remove_grain(self, grain_i):
        """Remove a grain class to the GrainParameters object.

        The grain class with a given index is removed from the GrainParameters
        object.
        Removing a grain decrements the number of classes through the bins
        attribute.

        Args:
            grain_i: [int] The index of the grain class.  Must be less than the
                           number of grain classes.
        """
        if grain_i < 0:
            raise ValueError("index to grain class must be positive")
        if grain_i >= self.bins:
            raise ValueError(f"index must in range [0,{self.bins-1}]")
        self.diameter.pop(grain_i)
        self.density.pop(grain_i)
        self.proportion.pop(grain_i)
        self.bins -= 1

    def validate(self):
        """Validate grain classes in GrainParameters object."""
        if self.bins==0:
            return 0
        if len(self.diameter) != self.bins:
            raise ValueError(
                f"In GrainParameters: number of grain diameter values ({len(self.diameter)}) must equal number of grain classes ({self.bins})")
        if len(self.density) != self.bins:
            raise ValueError(
                "In GrainParameters: number of grain density values must equal number of grain classes"
            )
        if len(self.proportion) != self.bins:
            raise ValueError(
                "In GrainParameters: number of grain proportion values must equal number of grain classes"
            )
        for j in range(self.bins):
            if self.diameter[j]<=0:
                raise ValueError("In GrainParameters: diameter values must be positive")
            if self.density[j]<=0:
                raise ValueError("In GrainParameters: density values must be positive")
            if self.proportion[j]<=0:
                raise ValueError("In GrainParameters: proportion values must be positive")
            if self.proportion[j]>1:
                raise ValueError("In GrainParameters: proportion values must be <=1")
        sum_prop = np.sum(np.array(list(self.proportion), dtype=np.float64), dtype=np.float64)
        if sum_prop > 1:
            raise ValueError("In GrainParameters: the sum of the proportions for the grain classes must be <= 1")
        return 1

    def describe(self):
        """Describe the grain classes in GrainParameters object."""
        print("Grain parameters for AshDisperse")
        print("  Number of grain classes, N_grains = ", self.bins)
        for j in range(self.bins):
            print("  Grain class ", j + 1)
            print("    Grain diameter = ", self.diameter[j], " m")
            print("    Grain density = ", self.density[j], " kg/m^3")
            print("    proportion = ", self.proportion[j])
        print("********************")

# pylint: disable=E1101
GrainParameters_type = GrainParameters.class_type.instance_type

def _grains_dict(p):
    return {
        'bins': int(p.bins),
        'diameter': list(p.diameter),
        'density': list(p.density),
        'proportion': list(p.proportion),
    }

def _grain_params_equal(p1: GrainParameters, p2: GrainParameters) -> bool:
    if not (p1.bins == p2.bins):
        return False
    for i in range(p1.bins):
        if (p1.diameter[i] != p2.diameter[i]):
            return False
        if (p1.density[i] != p2.density[i]):
            return False
        if (p1.proportion[i] != p2.proportion[i]):
            return False
    return True
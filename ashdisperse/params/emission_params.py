# -*- coding: utf-8 -*-
"""Emission profile parameters for AshDisperse.

Defines the EmissionParameters class, containing attributes specifying emission
classes and methods for amending these, and a printer for this object.

There can be a single emission profile, used for all grain classes,
or a different emission profile for each grain class

Example:
    emission = EmissionParameters()
    
    print_emission_parameters(emission)
"""

from collections import OrderedDict

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba.typed import List
from numba.types import ListType

emission_spec = OrderedDict()
emission_spec["lower"] = ListType(float64)  # Minimum altitude for release
emission_spec["upper"] = ListType(float64)  # Maximum altitude for release
emission_spec["profile"] = ListType(int64)  # profile type enumerator: =0 for suzuki, =1 for uniform
emission_spec["Suzuki_k"] = ListType(float64)  # Suzuki_k parameter
"""OrderedDict: EmissionParameters attribute type specification.

This specification is required for numba jitting.
"""

@jitclass(emission_spec)
class EmissionParameters:
    """Emission profile parameters required for AshDisperse.

    Defines the EmissionParameters class, containing attributes specifying emission
    classes and methods for amending these.

    Attributes:
        lower: A list of floats containing the minimum altitude for release of each grain class.
        Suzuki_k: A list of floats containing the Suzuki k parameter of each grain class.
    """

    def __init__(self):
        """Initialize of EmissionParameters to empty lists.

        Use add_profile to add profile for grain class.
        """
        self.lower = List.empty_list(np.float64)
        self.upper = List.empty_list(np.float64)
        self.profile = List.empty_list(np.int64)
        self.Suzuki_k = List.empty_list(np.float64)
    
    def len(self):
        return len(self.lower)
    
    def validate(self):
        """Validate grain classes in GrainParameters object."""
        if (self.len()==0):
            return 0
        if len(self.upper) != self.len():
            raise ValueError(
                "In EmissionParameters: number of upper values must equal number of lower values"
            )
        if len(self.profile) != self.len():
            raise ValueError(
                "In EmissionParameters: number of profile values must equal number of lower values"
            )
        if len(self.Suzuki_k) != self.len():
            raise ValueError(
                "In EmissionParameters: number of Suzuki_k values must equal number of lower values"
            )
        
        for j in range(self.len()):
            if (self.lower[j]<0):
                raise ValueError("In EmissionParameters: lower values must be non-negative")
            if (self.upper[j]<0):
                raise ValueError("In EmissionParameters: upper values must be non-negative")
            if (self.upper[j] - self.lower[j]<0):
                raise ValueError("In EmissionParameters: upper values must be larger than lower values")
            if (self.Suzuki_k[j]<=0):
                raise ValueError("In EmissionParameters: Suzuki_k values must be positive")
        return 1

    def from_values(self, lower, upper, profile, Suzuki_k):
        self.add_profile(lower, upper, profile, Suzuki_k)
        return

    def from_lists(self, lower, upper, profile, Suzuki_k):
        """Initialize emission parameters from lists"""

        if len(lower) != len(profile):
            raise ValueError("Size of lower and profile not equal")
        
        if len(lower) != len(upper):
            raise ValueError("Size of lower and upper not equal")
        
        if len(lower) != len(Suzuki_k):
            raise ValueError("Size of lower and Suzuki_k not equal")

        self.clear()

        for j, (l, u, p, k) in enumerate(zip(lower, upper, profile, Suzuki_k)):
            self.add_profile(l, u, p, k)

        return
    
    def clear(self):
        self.lower = List.empty_list(np.float64)
        self.upper = List.empty_list(np.float64)
        self.profile = List.empty_list(np.int64)
        self.Suzuki_k = List.empty_list(np.float64)

    def add_profile(self, lower, upper, profile, Suzuki_k):
        """Add a emission class to the EmissionParameters object.

        An emission profile requires a lower altitude and a Suzuki k parameter.

        Args:
            lower: [float64] The lower altitude of the emission profile;
                must be positive.
            upper: [float64] The upper altitude of the emission profile;
                must be positive and > lower.
            profile: [int64] The type of emission profile: =0 for Suzuki, =1 for uniform
                must be either 0 or 1
            Suzuki_k: [float64] The Suzuki k parameter of the emission profile;
                must be positive.
        """
        if lower < 0:
            raise ValueError("Emission profile lower altitude must be positive")
        if upper < 0:
            raise ValueError("Emission profile upper altitude must be positive")
        if upper < lower:
            raise ValueError("Emission profile must have upper altitude > lower altitude")
        if profile not in [0,1]:
            raise ValueError("Emission profile must have profile type of either 0 or 1")
        if Suzuki_k < 0:
            raise ValueError("Emission profile Suzukik parameter must be positive")
        self.lower.append(np.float64(lower))
        self.upper.append(np.float64(upper))
        self.profile.append(np.int64(profile))
        self.Suzuki_k.append(np.float64(Suzuki_k))

    def remove_profile(self, profile_i):
        """Remove a emission profile from the EmissionParameters object.

        The emission profile with a given index is removed from the EmissionParameters
        object.
        
        Args:
            profile_i: [int] The index of the emission profile.  Must be less than the
                           number of profiles.
        """
        if profile_i < 0:
            raise ValueError("index to profile class must be positive")
        if profile_i >= len(self.lower):
            raise ValueError(f"index must in range [0,{self.len()-1}]")
        self.lower.pop(profile_i)
        self.upper.pop(profile_i)
        self.profile.pop(profile_i)
        self.Suzuki_k.pop(profile_i)

    def describe(self):
        """Describe the emission profiles in EmissionParameters object."""
        print("Emission profile parameters for AshDisperse")
        for j in range(len(self.lower)):
            print(f"  Emission profile for grain class {j+1}")
            print("    Lower altitude =", self.lower[j], " m")
            print("    Upper altitude =", self.upper[j], " m")
            if self.profile[j]==0:
                print(f"    Profile type = 0 (Suzuki)")
            else:
                print(f"    Profile type = 1 (Uniform)")
            print("    Suzuki k = ", self.Suzuki_k[j])
            # print("    Suzuki peak = ", Suzuki_peak_from_k(self.Suzuki_k[j], self.lower[j], self.upper[j]), " m")
        print("********************")

# pylint: disable=E1101
EmissionParameters_type = EmissionParameters.class_type.instance_type

def Suzuki_k_from_peak(Suzuki_peak, lower, upper):
    Suzuki_k = (upper-lower)/(upper-Suzuki_peak)
    return Suzuki_k

def Suzuki_peak_from_k(Suzuki_k, lower, upper):
    Suzuki_peak = upper - (upper-lower)/Suzuki_k
    return Suzuki_peak

def _emission_dict(p):
    return {
        'lower': list(p.lower),
        'upper': list(p.upper),
        'profile': list(p.profile),
        'Suzuki_k': list(p.Suzuki_k),
    }

def _emission_params_equal(p1: EmissionParameters, p2: EmissionParameters) -> bool:
    n = len(p1.lower)
    if not (n == len(p2.lower)):
        return False
    for i in range(n):
        if (p1.lower[i] != p2.lower[i]):
            return False
        if (p1.upper[i] != p2.upper[i]):
            return False
        if (p1.profile[i] != p2.profile[i]):
            return False
        if (p1.Suzuki_k[i] != p2.Suzuki_k[i]):
            return False
    return True
from collections import OrderedDict

import numpy as np
from numba import float64
from numba.experimental import jitclass

met_spec = OrderedDict()
met_spec['U_scale'] = float64      # scale for wind velocity
met_spec['Ws_scale'] = float64[:]  # settling scale for each grain class


@jitclass(met_spec)
class MetParameters():
    def __init__(self, U_scale, Ws_scale):
        self.U_scale = U_scale
        self.Ws_scale = Ws_scale

    def validate(self):
        if self.U_scale <= 0:
            raise ValueError(
                "In MetParameters, U_scale must be positive"
            )
        if (np.any(self.Ws_scale <= 0)):
            raise ValueError(
                "In MetParameters, all Ws_scale values must be positive"
            )
        return 1


# pylint: disable=E1101
MetParameters_type = MetParameters.class_type.instance_type

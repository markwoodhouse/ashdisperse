from collections import OrderedDict
from typing import Optional

import numpy as np
from numba import float64, int64, optional
from numba.experimental import jitclass
from numba.types import unicode_type

from ashdisperse.utilities import latlon_point_to_utm_code

source_spec = OrderedDict()
source_spec["latitude"] = float64  # latitude of source
source_spec["longitude"] = float64  # longitude of source
source_spec["utmcode"] = int64  # EPSG utm code of source
source_spec["radius"] = float64  # radius of the source
source_spec["PlumeHeight"] = float64  # height of the source plume
source_spec["MER"] = float64  # mass eruption rate
source_spec["duration"] = float64  # duration of eruption
source_spec["name"] = unicode_type  # name of source


@jitclass(source_spec)
class SourceParameters:
    def __init__(self,
                 latitude:np.float64=np.nan,
                 longitude:np.float64=np.nan,
                 utmcode:int=0,
                 radius:np.float64=np.nan,
                 PlumeHeight:np.float64=np.nan,
                 MER:np.float64=np.nan,
                 duration:np.float64=np.nan,
                 name:str=""):
        self.latitude = latitude
        self.longitude = longitude
        self.utmcode = utmcode
        self.radius = radius
        self.PlumeHeight = PlumeHeight
        self.MER = MER
        self.duration = duration
        self.name = name

    def validate(self):
        if np.isnan(self.latitude) or np.abs(self.latitude)>90:
            raise ValueError("In SourceParameters, latitude must be in the range [-90, 90]")
        if np.isnan(self.longitude) or np.abs(self.longitude)>180:
            raise ValueError("In SourceParameters, longitude must be in the range [-180, 180]")
        if np.isnan(self.radius) or self.radius < 0:
            raise ValueError("In SourceParameters, radius must be positive")
        if np.isnan(self.PlumeHeight) or self.PlumeHeight < 0:
            raise ValueError("In SourceParameters, PlumeHeight must be positive")
        if np.isnan(self.MER) or self.MER < 0:
            raise ValueError("In SourceParameters, MER must be positive")
        if np.isnan(self.duration) or self.duration < 0:
            raise ValueError("In SourceParameters, duration must be positive")
        return 1

    def utm_from_latlon(self):
        offset = int(np.round((183+self.longitude)/6.0))
        self.utmcode = int(32600+offset) if (self.latitude > 0) else int(32700+offset)
        return

    def from_values(self,
        lat,
        lon,
        utmcode=None,
        radius=10e3,
        PlumeHeight=10e3,
        MER=1e6,
        duration=18000,
        name="",
    ):

        self.latitude = np.float64(lat)
        self.longitude = np.float64(lon)
        if utmcode is None:
            self.utm_from_latlon()
        else:
            self.utmcode = utmcode

        self.radius = np.float64(radius)

        self.PlumeHeight = np.float64(PlumeHeight)

        self.MER = np.float64(MER)

        self.duration = np.float64(duration)

        self.name = name

        self.validate()

    def describe(self):
        print("Source parameters for AshDisperse")
        print("  Mass eruption rate MER = ", self.MER, " kg/s")
        print("  Eruption duration = ", self.duration, " s")
        print("  Plume height H = ", self.PlumeHeight, " m")
        print("  Gaussian source radius = ", self.radius, " m")
        print("********************")

# pylint: disable=E1101
SourceParameters_type = SourceParameters.class_type.instance_type

def _source_dict(p: SourceParameters):
    return {
        "name": p.name,
        "latitude": p.latitude,
        "longitude": p.longitude,
        "utmcode": p.utmcode,
        "radius": p.radius,
        "PlumeHeight": p.PlumeHeight,
        "MER": p.MER,
        "duration": p.duration,
    }


def _source_params_equal(p1: SourceParameters, p2: SourceParameters) -> bool:

    test = (
        (p1.name == p2.name) and
        (p1.latitude == p2.latitude) and
        (p1.longitude == p2.longitude) and
        (p1.utmcode == p2.utmcode) and
        (p1.radius == p2.radius) and
        (p1.PlumeHeight == p2.PlumeHeight) and
        (p1.MER == p2.MER) and
        (p1.duration == p2.duration)
    )
    return test
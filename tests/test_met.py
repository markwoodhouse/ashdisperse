import os

import pytest

from ashdisperse.interface import set_met
from ashdisperse.params.params import parameters_from_dict

ERA5_HEKLA = "/home/mw9428/Documents/AshModel/MetData/Example.nc"


def _hekla(lat=63.983, lon=-19.666):
    d = {
        "source": dict(name="Hekla", latitude=lat, longitude=lon, utmcode=32627,
                       radius=10000.0, PlumeHeight=10000.0, MER=1e6, duration=18000.0),
        "grains": dict(bins=1, diameter=[1e-4], density=[2000.0], proportion=[1.0]),
        "emission": dict(lower=[0.0], upper=[10000.0], profile=[0], Suzuki_k=[10.0]),
        "solver": dict(domX=1.5, domY=1.5, minN_log2=4, maxN_log2=8, Nx_log2=5,
                       Ny_log2=5, epsilon=1e-8, plateau_factor=10.0, fft_tol=1e-10),
        "physical": dict(Kappa_h=100.0, Kappa_v=10.0, g=9.81, mu=1.85e-5),
        "output": dict(start=0.0, stop=11000.0, step=1100.0),
    }
    return parameters_from_dict(d)


@pytest.mark.skipif(not os.path.isfile(ERA5_HEKLA), reason="local ERA5 file not available")
def test_set_met_local_netcdf():
    met = set_met(_hekla(), source="local", met_filename=ERA5_HEKLA, datetime="2026-01-01 12:00")
    assert met.source == "netCDF"
    assert met.z_data.size == 37
    assert met.z_data.max() > 40e3


@pytest.mark.skipif(not os.path.isfile(ERA5_HEKLA), reason="local ERA5 file not available")
def test_set_met_local_rejects_point_outside_file():
    with pytest.raises(ValueError):
        set_met(_hekla(lat=61.3, lon=-152.3), source="local",
                met_filename=ERA5_HEKLA, datetime="2026-01-01 12:00")
    with pytest.raises(ValueError):
        set_met(_hekla(), source="local", met_filename=ERA5_HEKLA, datetime="2024-01-01 12:00")


def test_set_met_local_requires_datetime():
    with pytest.raises(RuntimeError):
        set_met(_hekla(), source="local", met_filename=ERA5_HEKLA)

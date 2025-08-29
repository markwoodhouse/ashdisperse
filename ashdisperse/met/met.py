import os
from collections import OrderedDict
from datetime import datetime
from math import ceil, floor
from typing import Optional, Literal

import matplotlib.pyplot as plt
import metpy.calc as metcalc
import metpy.units as units
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from matplotlib import ticker
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from numba import float64, int64, jit, njit
from numba.experimental import jitclass
from numba.types import Tuple, unicode_type
from scipy.integrate import solve_ivp

from ashdisperse.met.met_netcdf import NetcdfMet
from ashdisperse.met.met_gfs import GFSarchive, GFSforecast
from ..params.params import Parameters_type
from ..queryreport import print_text
from ..utilities.utilities import (SA_Density_array, SA_Density_value,
                                   SA_Pressure_array, SA_Pressure_value,
                                   SA_Temperature_array, SA_Temperature_value,
                                   interp_ex_array, interp_ex_value)

MetData_spec = OrderedDict()
MetData_spec["z_data"] = float64[::1]
MetData_spec["temperature_data"] = float64[::1]
MetData_spec["pressure_data"] = float64[::1]
MetData_spec["density_data"] = float64[::1]
MetData_spec["wind_U_data"] = float64[::1]
MetData_spec["wind_V_data"] = float64[::1]
MetData_spec["source"] = unicode_type
MetData_spec["surface_temperature"] = float64  # surface temperature
MetData_spec["surface_pressure"] = float64  # surface pressure
MetData_spec["lapse_tropos"] = float64  # lapse rate in troposphere
MetData_spec["lapse_stratos"] = float64  # lapse rate stratosphere
MetData_spec["height_tropos"] = float64  # height of the troposphere
MetData_spec["height_stratos"] = float64  # height of the stratosphere
MetData_spec["Ra"] = float64  # gas constant of dry air
MetData_spec["g"] = float64  # gravitational acceleration


@jitclass(MetData_spec)
class MetData:
    def __init__(
        self,
        source: str ="standardAtmos",
        surface_temperature: float=293,
        surface_pressure: float=101325,
        lapse_tropos: float=6.5e-3,
        lapse_stratos: float=2.0e-3,
        height_tropos: float=11e3,
        height_stratos: float=20e3,
        Ra: float=285,
        g: float=9.81,
    ):
        self.surface_temperature = np.float64(surface_temperature)
        self.surface_pressure = np.float64(surface_pressure)
        self.lapse_tropos = np.float64(lapse_tropos)
        self.lapse_stratos = np.float64(lapse_stratos)
        self.height_tropos = np.float64(height_tropos)
        self.height_stratos = np.float64(height_stratos)
        self.Ra = np.float64(Ra)
        self.g = np.float64(g)
        self.source = source

        self.z_data = np.empty((0), dtype=np.float64)
        self.temperature_data = np.empty((0), dtype=np.float64)
        self.pressure_data = np.empty((0), dtype=np.float64)
        self.density_data = np.empty((0), dtype=np.float64)
        self.wind_U_data = np.empty((0), dtype=np.float64)
        self.wind_V_data = np.empty((0), dtype=np.float64)

    def set_zuv_data(self, z, u, v):
        self.z_data = z
        self.wind_U_data = u
        self.wind_V_data = v
            

    def temperature_array(self, z):
        if self.source in ["netCDF", "GFS", "GFS Archive"]:
            temperature = interp_ex_array(z, self.z_data, self.temperature_data)
        if self.source == "standardAtmos":
            temperature = SA_Temperature_array(
                z,
                self.surface_temperature,
                self.lapse_tropos,
                self.lapse_stratos,
                self.height_tropos,
                self.height_stratos,
            )
        return temperature

    def temperature_value(self, z):
        if self.source in ["netCDF", "GFS", "GFS Archive"]:
            temperature = interp_ex_value(z, self.z_data, self.temperature_data)
        if self.source == "standardAtmos":
            temperature = SA_Temperature_value(
                z,
                self.surface_temperature,
                self.lapse_tropos,
                self.lapse_stratos,
                self.height_tropos,
                self.height_stratos,
            )
        return temperature

    def pressure_array(self, z):
        if self.source in ["netCDF", "GFS", "GFS Archive"]:
            pressure = interp_ex_array(z, self.z_data, self.pressure_data)
        elif self.source == "standardAtmos":
            pressure = SA_Pressure_array(
                z,
                self.surface_temperature,
                self.surface_pressure,
                self.lapse_tropos,
                self.lapse_stratos,
                self.height_tropos,
                self.height_stratos,
                self.g,
                self.Ra,
            )
        return pressure

    def pressure_value(self, z):
        if self.source in ["netCDF", "GFS", "GFS Archive"]:
            pressure = interp_ex_value(z, self.z_data, self.pressure_data)
        elif self.source == "standardAtmos":
            pressure = SA_Pressure_value(
                z,
                self.surface_temperature,
                self.surface_pressure,
                self.lapse_tropos,
                self.lapse_stratos,
                self.height_tropos,
                self.height_stratos,
                self.g,
                self.Ra,
            )
        return pressure

    def density_array(self, z):
        if self.source in ["netCDF", "GFS", "GFS Archive"]:
            density = interp_ex_array(z, self.z_data, self.density_data)
        elif self.source == "standardAtmos":
            density = SA_Density_array(
                z,
                self.surface_temperature,
                self.surface_pressure,
                self.lapse_tropos,
                self.lapse_stratos,
                self.height_tropos,
                self.height_stratos,
                self.g,
                self.Ra,
            )
        return density

    def density_value(self, z):
        if self.source in ["netCDF", "GFS", "GFS Archive"]:
            density = interp_ex_value(z, self.z_data, self.density_data)
        elif self.source == "standardAtmos":
            density = SA_Density_value(
                z,
                self.surface_temperature,
                self.surface_pressure,
                self.lapse_tropos,
                self.lapse_stratos,
                self.height_tropos,
                self.height_stratos,
                self.g,
                self.Ra,
            )
        return density

    def wind_U_array(self, z, scale=None):
        u = interp_ex_array(z, self.z_data, self.wind_U_data)
        if scale is None:
            ret = u
        else:
            ret = u / scale
        return ret

    def wind_U_value(self, z, scale=None):
        u = interp_ex_value(z, self.z_data, self.wind_U_data)
        if scale is None:
            ret = u
        else:
            ret = u / scale
        return ret

    def wind_V_array(self, z, scale=None):
        v = interp_ex_array(z, self.z_data, self.wind_V_data)
        if scale is None:
            ret = v
        else:
            ret = v / scale
        return ret

    def wind_V_value(self, z, scale=None):
        v = interp_ex_value(z, self.z_data, self.wind_V_data)
        if scale is None:
            ret = v
        else:
            ret = v / scale
        return ret

    def wind_speed_array(self, z):
        u = self.wind_U_array(z)
        v = self.wind_V_array(z)
        spd = np.sqrt(u * u + v * v)
        return spd

    def wind_speed_value(self, z):
        u = self.wind_U_value(z)
        v = self.wind_V_value(z)
        spd = np.sqrt(u * u + v * v)
        return spd

    def max_wind_speed(self, H, num=100):
        z = np.arange(0.0, H + 1.0, H / (num - 1), dtype=np.float64)
        spd = self.wind_speed_array(z)
        return np.amax(spd)

    def settling_speed_array(self, params, z, scale=None):
        Ws = self.calculate_settling_speed_array(params, z)
        if scale is None:
            ret = Ws
        else:
            ret = Ws / scale
        return ret

    def settling_speed_value(self, params, z, scale=None):
        Ws = self.calculate_settling_speed_value(params, z)
        if scale is None:
            ret = Ws
        else:
            ret = Ws / scale
        return ret

    def settling_speed_for_grain_class_array(self, params, grain_i, z, scale=None):
        Ws = self.calculate_settling_speed_for_grain_class_array(params, grain_i, z)
        if scale is None:
            return Ws
        else:
            return Ws / scale

    def settling_speed_for_grain_class_value(self, params, grain_i, z, scale=None):
        Ws = self.calculate_settling_speed_for_grain_class_value(params, grain_i, z)
        if scale is None:
            return Ws
        else:
            return Ws / scale

    @staticmethod
    def _solve_settling_function(diam, g, rho_p, rho_a, mu, max_iter=50):

        tolerance = 1e-8

        x0 = np.float64(1e-6)
        x1 = np.float64(10000)

        def _settling_func_white(Re, d, g, rho_p, rho_a, mu):
            gp = (rho_p - rho_a) * g / rho_a
            C1 = 0.25
            C2 = 6.0
            Cd = C1 + 24.0 / Re + C2 / (1 + np.sqrt(Re))
            d3 = d**3
            f = Cd * Re * Re - 4.0 / 3.0 * gp * d3 * (rho_a / mu) ** 2
            return f

        fx0 = _settling_func_white(x0, diam, g, rho_p, rho_a, mu)
        fx1 = _settling_func_white(x1, diam, g, rho_p, rho_a, mu)

        if np.abs(fx0) < np.abs(fx1):
            x0, x1 = x1, x0
            fx0, fx1 = fx1, fx0

        x2, fx2 = x0, fx0
        d = x2

        mflag = True
        steps_taken = 0

        while steps_taken < max_iter and np.abs(x1 - x0) > tolerance:
            fx0 = _settling_func_white(x0, diam, g, rho_p, rho_a, mu)
            fx1 = _settling_func_white(x1, diam, g, rho_p, rho_a, mu)
            fx2 = _settling_func_white(x2, diam, g, rho_p, rho_a, mu)

            if fx0 != fx2 and fx1 != fx2:
                L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
                L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
                L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
                new = L0 + L1 + L2
            else:
                new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

            if (
                (new < 0.25 * (3 * x0 + x1) or new > x1)
                or (mflag and np.abs(new - x1) >= 0.5 * np.abs(x1 - x2))
                or ((not mflag) and np.abs(new - x1) >= 0.5 * np.abs(x2 - d))
                or (mflag and np.abs(x1 - x2) < tolerance)
                or ((not mflag) and np.abs(x2 - d) < tolerance)
            ):
                new = 0.5 * (x0 + x1)
                mflag = True
            else:
                mflag = False

            fnew = _settling_func_white(new, diam, g, rho_p, rho_a, mu)
            d, x2 = x2, x1

            if fx0 * fnew < 0:
                x1 = new
            else:
                x0 = new

            if np.abs(fx0) < np.abs(fx1):
                x0, x1 = x1, x0

            steps_taken += 1

        return x1, steps_taken

    def calculate_settling_speed_array(self, params, z):

        ws = np.empty((z.size, params.grains.bins), dtype=np.float64)

        rho_a = np.empty((z.size, 1), dtype=np.float64)
        rho_a = self.density_array(z)

        for iz, rho_az in enumerate(rho_a):

            for j, (d, rho_p) in enumerate(
                zip(params.grains.diameter, params.grains.density)
            ):

                Re, steps_taken = self._solve_settling_function(
                    d, params.physical.g, rho_p, rho_az, params.physical.mu
                )
                if steps_taken > 50:
                    raise RuntimeError(
                        "In MetData: _solve_settling_function" + " failed to converge"
                    )
                ws[iz, j] = params.physical.mu * Re / rho_az / d

        return ws

    def calculate_settling_speed_value(self, params, z):

        ws = np.empty((params.grains.bins), dtype=np.float64)

        rho_a = self.density_value(z)

        for j, (d, rho_p) in enumerate(
            zip(params.grains.diameter, params.grains.density)
        ):

            Re, steps_taken = self._solve_settling_function(
                d, params.physical.g, rho_p, rho_a, params.physical.mu
            )
            if steps_taken > 50:
                raise RuntimeError(
                    "In MetData: _solve_settling_function" + " failed to converge"
                )
            ws[j] = params.physical.mu * Re / rho_a / d

        return ws

    def calculate_settling_speed_for_grain_class_array(self, params, grain_i, z):

        ws = np.empty((z.size, 1), dtype=np.float64)

        rho_a = np.empty((z.size, 1), dtype=np.float64)
        rho_a = self.density_array(z)

        for iz, rho_az in enumerate(rho_a):

            d = params.grains.diameter[grain_i]
            rho_p = params.grains.density[grain_i]

            Re, steps_taken = self._solve_settling_function(
                d, params.physical.g, rho_p, rho_az, params.physical.mu
            )
            if steps_taken > 50:
                raise RuntimeError(
                    "In MetData: _solve_settling_function" + " failed to converge"
                )
            ws[iz, 0] = params.physical.mu * Re / rho_az / d

        return ws

    def calculate_settling_speed_for_grain_class_value(self, params, grain_i, z):

        rho_a = self.density_value(z)

        d = params.grains.diameter[grain_i]
        rho_p = params.grains.density[grain_i]

        Re, steps_taken = self._solve_settling_function(
            d, params.physical.g, rho_p, rho_a, params.physical.mu
        )
        if steps_taken > 50:
            raise RuntimeError(
                "In MetData: _solve_settling_function" + " failed to converge"
            )
        ws = params.physical.mu * Re / rho_a / d

        return ws


# pylint: disable=E1101
MetData_type = MetData.class_type.instance_type


def save_met(met: MetData, file="meteorology.npz") -> None:
    if os.path.exists(file):
        print(
            "WARNING: {outname} ".format(outname=file)
            + "already exists and will be replaced"
        )

    if met.source == "standardAtmos":
        np.savez(
            file,
            source=met.source,
            surface_temperature=met.surface_temperature,
            surface_pressure=met.surface_pressure,
            lapse_tropos=met.lapse_tropos,
            lapse_stratos=met.lapse_stratos,
            height_tropos=met.height_tropos,
            height_stratos=met.height_stratos,
            Ra=met.Ra,
            g=met.g,
            z_data=met.z_data,
            wind_U_data=met.wind_U_data,
            wind_V_data=met.wind_V_data,
        )
    elif met.source in ["netCDF", "GFS"]:
        np.savez(
            file,
            source=met.source,
            Ra=met.Ra,
            g=met.g,
            z_data=met.z_data,
            wind_U_data=met.wind_U_data,
            wind_V_data=met.wind_V_data,
            temperature_data=met.temperature_data,
            pressure_data=met.pressure_data,
            density_data=met.density_data,
        )


def load_met(met_file: str) -> MetData:

    if not os.path.exists(met_file):
        raise IOError("AshDisperse meteorological file {} not found".format(met_file))

    met = MetData()
    data = np.load(met_file)

    if str(data["source"]) == "standardAtmos":
        met = MetData(
            source="standardAtmos",
            surface_temperature=np.float64(data["surface_temperature"]),
            surface_pressure=np.float64(data["surface_pressure"]),
            lapse_tropos=np.float64(data["lapse_tropos"]),
            lapse_stratos=np.float64(data["lapse_stratos"]),
            height_tropos=np.float64(data["height_tropos"]),
            height_stratos=np.float64(data["height_stratos"]),
            Ra=np.float64(data["Ra"]),
            g=np.float64(data["g"]),
        )
        met.set_zuv_data(data["Z"], data["U"], data["V"])
    elif str(data["source"]) in ["netCDF", "GFS"]:
        met = MetData(
            source="netCDF", Ra=np.float64(data["Ra"]), g=np.float64(data["g"])
        )
        met.z_data = data["z_data"]
        met.wind_U_data = data["wind_U_data"]
        met.wind_V_data = data["wind_V_data"]
        met.temperature_data = data["temperature_data"]
        met.pressure_data = data["pressure_data"]
        met.density_data = data["density_data"]

    return met

def load_met_from_repodf(met_dataset: pd.DataFrame, indx: int) -> MetData:

    if indx not in met_dataset.index:
        raise ValueError(f"index must be in met_dataset")
    
    met_data = met_dataset.iloc[indx]

    met = MetData()
    met.z_data = met_data['altitude']
    met.wind_U_data = met_data["wind_u"]
    met.wind_V_data = met_data["wind_v"]
    met.temperature_data = met_data["temperature"]
    met.pressure_data = met_data["pressure"]
    met.density_data = met_data["density"]

    return met


def netCDF_to_Met(met: MetData, netcdf_data: NetcdfMet) -> MetData:
    N = len(netcdf_data.altitude)
    met.z_data = np.empty((N), dtype=np.float64)
    met.z_data[:N] = netcdf_data.altitude[:N]
    met.wind_U_data = np.empty((N), dtype=np.float64)
    met.wind_U_data[:N] = netcdf_data.wind_U[:N]
    met.wind_V_data = np.empty((N), dtype=np.float64)
    met.wind_V_data[:N] = netcdf_data.wind_V[:N]
    met.temperature_data = np.empty((N), dtype=np.float64)
    met.temperature_data[:N] = netcdf_data.temperature[:N]
    met.pressure_data = np.empty((N), dtype=np.float64)
    met.pressure_data[:N] = netcdf_data.pressure[:N]
    met.density_data = np.empty((N), dtype=np.float64)
    met.density_data[:N] = netcdf_data.density[:N]
    met.source = "netCDF"
    return met


def gfs_to_Met(met: MetData, gfs_data: pd.DataFrame) -> MetData:
    gp = gfs_data["Geopotential_height_isobaric"]
    temp = gfs_data["Temperature_isobaric"]
    rh = gfs_data["Relative_humidity_isobaric"]
    u = gfs_data["ucomponent_of_wind_isobaric"]
    v = gfs_data["vcomponent_of_wind_isobaric"]
    pres = gfs_data["alt"]
    
    N = len(pres)

    # z = metcalc.

    met.z_data = np.empty((N), dtype=np.float64)
    met.z_data[:N] = np.flipud(z[:])
    # met.z_data[:N] = np.flipud(z[:].data.flatten())
    
    met.wind_U_data = np.empty((N), dtype=np.float64)
    met.wind_U_data[:N] = np.flipud(u[:])
    # met.wind_U_data[:N] = np.flipud(u[:].data.flatten())
    
    met.wind_V_data = np.empty((N), dtype=np.float64)
    met.wind_V_data[:N] = np.flipud(v[:])
    # met.wind_V_data[:N] = np.flipud(v[:].data.flatten())
    
    met.temperature_data = np.empty((N), dtype=np.float64)
    met.temperature_data[:N] = np.flipud(temp[:])
    # met.temperature_data[:N] = np.flipud(temp[:].data.flatten())
    
    met.pressure_data = np.empty((N), dtype=np.float64)
    met.pressure_data[:N] = np.flipud(pres[:])
    # met.pressure_data[:N] = np.flipud(pres[:].data.flatten())
    
    met.density_data = np.empty((N), dtype=np.float64)

    met.density_data[:N] = met.pressure_data[:N] / Ra / met.temperature_data[:N]
    # met.density_data[:N] = met.pressure_data[:N] / Ra / met.temperature_data[:N]
    
    met.source = "GFS"
    return met




def gfs_archive_to_Met(met: MetData, gfs_data: pd.DataFrame):
    z = gfs_data.altitude.values
    temp = gfs_data.temperature.values
    u = gfs_data.wind_U.values
    v = gfs_data.wind_V.values
    pres = gfs_data.pressure.values

    N = gfs_data.pressure.size

    met.z_data = np.empty((N), dtype=np.float64)
    met.z_data[:N] = z[:]

    met.wind_U_data = np.empty((N), dtype=np.float64)
    met.wind_U_data[:N] = u[:]

    met.wind_V_data = np.empty((N), dtype=np.float64)
    met.wind_V_data[:N] = v[:]

    met.temperature_data = np.empty((N), dtype=np.float64)
    met.temperature_data[:N] = temp[:]

    met.pressure_data = np.empty((N), dtype=np.float64)
    met.pressure_data[:N] = pres[:]

    met.density_data = np.empty((N), dtype=np.float64)
    met.density_data[:N] = met.pressure_data[:N] / Ra / met.temperature_data[:N]

    met.source = "GFS Archive"
    return met


def _near_lat_lon(target_lat: float, target_lon, lats, lons):

    lat_i_near = np.abs(lats - target_lat).argmin()
    lon_i_near = np.abs(lons - target_lon).argmin()

    if target_lat in lats:
        lat_i0 = lat_i_near
        lat_i1 = lat_i_near  # won't need this
    else:
        lat_i0 = lat_i_near if lat_i_near < target_lat else lat_i_near + 1
        lat_i1 = lat_i0 - 1
    if target_lon in lons:
        lon_i0 = lon_i_near
        lon_i1 = lon_i_near  # won't need this
    else:
        lon_i0 = lon_i_near if lon_i_near < target_lon else lon_i_near - 1
        lon_i1 = lon_i0 + 1

    return lat_i0, lat_i1, lon_i0, lon_i1


def _interp_latlon(target_lat, target_lon, data, lats, lons):

    lat_i0, lat_i1, lon_i0, lon_i1 = _near_lat_lon(target_lat, target_lon, lats, lons)

    d_lat = target_lat - lats[lat_i0]
    d_lon = target_lon - lons[lon_i0]

    data_00 = data[0, :, lat_i0, lon_i0]
    data_01 = data[0, :, lat_i1, lon_i0]
    data_10 = data[0, :, lat_i0, lon_i1]
    data_11 = data[0, :, lat_i1, lon_i1]

    data_i0 = data_00 + d_lon * (data_10 - data_00)
    data_i1 = data_01 + d_lon * (data_11 - data_01)
    data_ij = data_i0 + d_lat * (data_i1 - data_i0)

    return data_ij


def wind_scale(max_speed):

    max_log10 = np.log10(max_speed)
    major = []
    minor = []
    for j in range(np.int64(max_log10) + 1):
        major.append(10**j)
        for k in range(2, 10):
            if k * 10**j > max_speed:
                break
            minor.append(k * 10**j)

    return major, minor


def wind_plot(met, z, show=True, savename=None):

    fig, ax = plt.subplots()

    U = met.wind_U_array(z)
    V = met.wind_V_array(z)
    speed = met.wind_speed_array(z)

    max_speed = speed.max()
    major, minor = wind_scale(max_speed)
    major.append(minor[-1])
    minor.pop(-1)

    ax.set_xlim(-max_speed, max_speed)
    ax.set_ylim(-max_speed, max_speed)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=z[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=z[-1]))

    ax.quiver(
        0,
        -max_speed,
        0,
        2 * max_speed,
        color="darkgray",
        headwidth=6,
        headlength=10,
        scale=1,
        scale_units="xy",
        angles="xy",
    )
    ax.quiver(
        -max_speed / 2,
        0,
        max_speed,
        0,
        color="darkgray",
        headwidth=1,
        headlength=0,
        scale=1,
        scale_units="xy",
        angles="xy",
    )
    ax.text(0, 1.1 * max_speed, "N")

    for m in major:
        ax.add_artist(plt.Circle((0, 0), m, ec="darkgray", lw=0.5, fill=False))
        ax.text(m * np.cos(45 * np.pi / 180), m * np.sin(45 * np.pi / 180), str(m))
    for m in minor:
        ax.add_artist(plt.Circle((0, 0), m, ec="darkgray", lw=0.25, fill=False))

    for (this_z, this_u, this_v) in zip(z, U, V):
        ax.quiver(
            0,
            0,
            this_u,
            this_v,
            color=cmap(norm(this_z)),
            scale=1,
            scale_units="xy",
            angles="xy",
        )

    fmt = ticker.FuncFormatter(lambda z, pos: "{:g}".format(z * 1e-3))
    cbar = fig.colorbar(sm, format=fmt, ax=ax)
    cbar.ax.set_title("Altitude (km)")

    if savename is not None:
        plt.savefig(savename)

    if show:
        plt.show()
    else:
        plt.close()

    return


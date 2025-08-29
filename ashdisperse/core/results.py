import os
import warnings
from math import ceil, floor

import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj as Proj
import rasterio as rio
import rioxarray as rxa
import utm
import xarray as xa
from matplotlib import ticker
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import concat
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import get_data_window
from scipy.fft import ifft2, irfft2
from scipy.interpolate import RectBivariateSpline, interp1d
from shapely.geometry import Polygon
from skimage.measure import find_contours, marching_cubes

from ashdisperse.params.emission_params import EmissionParameters
from ashdisperse.params.grain_params import GrainParameters
from ashdisperse.params.met_params import MetParameters
from ashdisperse.params.model_params import ModelParameters
from ashdisperse.params.output_params import OutputParameters
from ashdisperse.params.params import Parameters, copy_parameters
from ashdisperse.params.physical_params import PhysicalParameters
from ashdisperse.params.solver_params import SolverParameters
from ashdisperse.params.source_params import SourceParameters
from ashdisperse.spectral import grid_freq
from ashdisperse.version import __version__

from ..mapping import (BindColormap, add_north_arrow, add_opentopo_basemap,
                       add_scale_bar, ax_ticks, latlon_to_utm_epsg,
                       set_figure_size, set_min_axes, stamen_zoom_level,
                       webmerc, wgs84)
from ..utilities import (lin_levels, log_levels, log_steps, nice_round_down,
                         nice_round_up, pad_window, plot_rowscols)


def compat_warning(message, category, filename, lineno, file=None, line=None):
    return f"{category.__name__}: {message}"

def _clip_to_window(raster, x, y, vmin=1e-5, pad=1):
    height, width = raster.shape
    raster_tmp = np.copy(raster)
    raster_tmp[raster < vmin] = -1.0
    window = get_data_window(raster_tmp, nodata=-1.0)
    winpad = pad_window(window, height, width, pad=pad)

    row_start = winpad["row_start"]
    row_stop = winpad["row_stop"]
    col_start = winpad["col_start"]
    col_stop = winpad["col_stop"]
    raster_out = raster[row_start:row_stop, col_start:col_stop]
    x_out = x[col_start:col_stop]
    y_out = y[row_start:row_stop]
    return raster_out, x_out, y_out


def _interpolate(raster, x, y, resolution, extent=None, nodata=-1, vmin=1e-6):

    if extent is not None:
        min_x = extent[0]
        min_y = extent[1]
        max_x = extent[2]
        max_y = extent[3]
    else:
        raster_tmp = np.copy(raster)
        raster_tmp[raster < vmin] = nodata
        window = get_data_window(raster_tmp, nodata=nodata)
        min_x = x[window.col_off]
        min_y = y[window.row_off]
        max_x = x[window.col_off + window.width - 1]
        max_y = y[window.row_off + window.height - 1]

    print("resolution = ", resolution)

    left = floor(min_x / resolution) * resolution
    right = ceil(max_x / resolution) * resolution
    bottom = floor(min_y / resolution) * resolution
    top = ceil(max_y / resolution) * resolution

    width = int((right - left) / resolution) + 1
    height = int((top - bottom) / resolution) + 1
    print("Output raster shape is ({h},{w})".format(h=height, w=width))

    x_out = np.linspace(left, right, num=width, endpoint=True)
    y_out = np.linspace(bottom, top, num=height, endpoint=True)

    raster_tmp, x_tmp, y_tmp = _clip_to_window(raster, x, y, vmin=vmin / 10, pad=12)
    raster_tmp[raster_tmp < vmin / 10] = 0.0

    intrp = RectBivariateSpline(y_tmp, x_tmp, raster_tmp)

    raster_out = intrp(y_out, x_out)

    return raster_out, x_out, y_out


class AshDisperseResult:
    """Results of an AshDisperse simulation"""
    def __init__(self, params, C0_FT, Cz_FT):
        """Create AshDisperse result class"""
        self.grain_classes = params.grains.bins
        self.params = copy_parameters(params)

        self._interpolated = False
        self._changed_MER = False
        self._changed_gsd = False
        self._changed_duration = False

        self.utm = utm.from_latlon(
            self.params.source.latitude, self.params.source.longitude
        )

        self.utmepsg = latlon_to_utm_epsg(
            self.params.source.latitude, self.params.source.longitude
        )

        self.C0_FT = C0_FT
        self.Cz_FT = Cz_FT

        Nx = self.params.solver.Nx
        Ny = self.params.solver.Ny
        Ng = self.params.grains.bins
        Nz = self.params.output.Nz

        x, kx = grid_freq(Nx)
        y, ky = grid_freq(Ny)

        self.x_dimless = x
        self.y_dimless = y

        self.kx = kx
        self.ky = ky

        res_x = x[1] - x[0]
        res_y = y[1] - y[0]

        C0 = np.zeros((Ny, Nx, Ng), dtype=np.float64)
        Cz = np.zeros((Ny, Nx, Nz, Ng), dtype=np.float64)
        
        # for igrain in range(Ng):
        #     C0[:, :, igrain] = np.real(ifft2(C0_FT[:, :, igrain], s=(Ny,Nx))) / (res_x * res_y)
        #     for k in range(Nz):
        #         Cz[:, :, k, igrain] = np.real(ifft2(Cz_FT[:, :, k, igrain], s=(Ny,Nx))) / (res_x * res_y)
        for igrain in range(Ng):
            C0[:, :, igrain] = irfft2(C0_FT[:, :, igrain], s=(Ny,Nx)) / (res_x * res_y)
            for k in range(Nz):
                Cz[:, :, k, igrain] = irfft2(Cz_FT[:, :, k, igrain], s=(Ny,Nx)) / (res_x * res_y)

        self.C0_dimless = C0
        self.Cz_dimless = Cz

        self.C0 = np.zeros_like(C0)
        self.Cz = np.zeros_like(Cz)
        self.SettlingFlux = np.zeros_like(C0)

        self.x = np.zeros((len(x), params.grains.bins))
        self.y = np.zeros((len(y), params.grains.bins))

        for j in range(params.grains.bins):
            # self.x[:, j] = x * params.model.xyScale[j] * params.model.Lx[j] / np.pi
            # self.y[:, j] = y * params.model.xyScale[j] * params.model.Ly[j] / np.pi
            self.x[:, j] = x * params.model.xScale[j] * params.model.Lx[j] / np.pi
            self.y[:, j] = y * params.model.yScale[j] * params.model.Ly[j] / np.pi
            self.C0[:, :, j] = C0[:, :, j] * params.model.cScale[j]
            self.Cz[:, :, :, j] = Cz[:, :, :, j] * params.model.cScale[j]
            self.SettlingFlux[:, :, j] = (
                params.met.Ws_scale[j] * C0[:, :, j] * params.model.cScale[j]
            )

        df = pd.DataFrame(columns=["Name", "Latitude", "Longitude"])
        df.loc[0] = [
            self.params.source.name,
            self.params.source.latitude,
            self.params.source.longitude,
        ]
        self.source_marker = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            crs=wgs84["init"],
        )

    def linear_interp(self, Nx_log2: int, Ny_log2: int):
        "Do a bilinear interpolation by zero-padding the FFTs"
        
        Nx_old = self.params.solver.Nx
        Ny_old = self.params.solver.Ny
        Ng = self.params.grains.bins
        Nz = self.params.output.Nz

        Nx = 2**Nx_log2
        Ny = 2**Ny_log2

        half_Nx_old = Nx_old//2 + 1
        half_Nx = Nx//2 + 1

        half_Ny_old = Ny_old//2

        C0_FT = np.zeros((Ny, half_Nx, Ng), dtype=np.complex128)
        Cz_FT = np.zeros((Ny, half_Nx, Nz, Ng), dtype=np.complex128)

        C0_FT[:half_Ny_old,:half_Nx_old,:] = self.C0_FT[:half_Ny_old,:,:]
        C0_FT[Ny-half_Ny_old+1:,:half_Nx_old, :] = self.C0_FT[half_Ny_old+1:,:, :]

        Cz_FT[:half_Ny_old,:half_Nx_old,:, :] = self.Cz_FT[:half_Ny_old,:,:, :]
        Cz_FT[Ny-half_Ny_old+1:,:half_Nx_old, :, :] = self.Cz_FT[half_Ny_old+1:,:, :, :]

        params = copy_parameters(self.params)
        params.solver.Nx_log2 = Nx_log2
        params.solver.Ny_log2 = Ny_log2

        r = AshDisperseResult(params, C0_FT, Cz_FT)
        r._interpolated = True

        return r


    def change_MER(self,MER):
        params = self.params
        params.source.MER = MER
        self.params.model.from_params(
            params,
            params.model.xScale,
            params.model.yScale
        )

        self.__init__(params, self.C0_FT, self.Cz_FT)
        self._changed_MER = True

    def change_duration(self,duration):
        self.params.source.duration = duration
        self._changed_duration = True

    def change_grain_proportions(self,props):
        if len(props)!=self.params.grains.bins:
            raise RuntimeError(f'length of props must equal number of grain size classes ({self.params.grains.bins})')

        params = self.params
        for j, p in enumerate(props):
            params.grains.proportion[j] = np.float64(p)

        self.params.model.from_params(
            params,
            params.model.xScale,
            params.model.yScale
        )

        self.__init__(params, self.C0_FT, self.Cz_FT)
        self._changed_gsd = True

    @classmethod
    def from_netcdf(cls, filename):
        """Initialize results from a netcdf file"""
        da = xa.load_dataset(filename)

        if da.version != __version__:
            warnings.formatwarning = compat_warning
            warnings.warn(f"Loading AshDisperse results from version {da.version}. Current version is {__version__}. \n There may be incompatibilities.\n")

        p = Parameters()
        p.solver = SolverParameters(
            domX=da.domX,
            domY=da.domY,
            minN_log2=da.minN_log2,
            maxN_log2=da.maxN_log2,
            Nx_log2=da.Nx_log2,
            Ny_log2=da.Ny_log2,
            epsilon=da.epsilon,
        )
        p.grains = GrainParameters()
        # TODO: do this properly
        if isinstance(da.diameter, np.float64):
            p.grains.from_values(da.diameter, da.density, da.proportion)
        else:
            p.grains.from_lists(da.diameter, da.density, da.proportion)

        p.source = SourceParameters(
            latitude=da.latitude,
            longitude=da.longitude,
            utmcode=da.utmcode,
            radius=da.radius,
            PlumeHeight=da.PlumeHeight,
            MER=da.MER,
            duration=da.duration,
            name=da.name,
        )

        p.emission = EmissionParameters()
        # TODO: do this properly
        if isinstance(da.emission_lower, np.float64):
            p.emission.from_values(da.emission_lower,
                da.emission_upper,
                da.emission_profile,
                da.Suzuki_k)
        else:
            p.emission.from_lists(
                da.emission_lower,
                da.emission_upper, 
                da.emission_profile, 
                da.Suzuki_k)

        p.physical = PhysicalParameters(
            Kappa_h=da.Kappa_h,
            Kappa_v=da.Kappa_v,
            g=da.g,
            mu=da.mu,
        )

        p.met = MetParameters(da.U_scale, np.atleast_1d(da.Ws_scale))

        p.output = OutputParameters(start=da.start, stop=da.stop, step=da.step)
        p.output.ChebMats(p.solver.maxN, p.source.PlumeHeight)

        p.model = ModelParameters()
        if isinstance(da.SettlingScale, np.float64):
            p.model.from_values(
                da.SettlingScale,
                da.Velocity_ratio,
                da.xScale,
                da.yScale,
                da.Lx,
                da.Ly,
                da.cScale,
                da.QScale,
                da.Peclet_number,
                da.Diffusion_ratio,
                da.sigma_hat,
                da.sigma_hat_scale,
            )
        else:
            p.model.from_lists(
                da.SettlingScale,
                da.Velocity_ratio,
                da.xScale,
                da.yScale,
                da.Lx,
                da.Ly,
                da.cScale,
                da.QScale,
                da.Peclet_number,
                da.Diffusion_ratio,
                da.sigma_hat,
                da.sigma_hat_scale,
            )

        C0_FT = da.C0_FT_r.values + 1j*da.C0_FT_i.values
        Cz_FT = da.Cz_FT_r.values + 1j*da.Cz_FT_i.values

        cls = AshDisperseResult(p, C0_FT, Cz_FT)

        return cls

    def to_netcdf(self, filename="AshDisperse.nc", compress=True, complevel=5):
        """Save AshDisperse results to a netCDF file"""
        Nx = self.params.solver.Nx
        da = xa.Dataset(
            data_vars=dict(
                C0_FT_r=(["ky", "kx", "grains"], np.real(self.C0_FT)),
                C0_FT_i=(["ky", "kx", "grains"], np.imag(self.C0_FT)),
                Cz_FT_r=(["ky", "kx", "z", "grains"], np.real(self.Cz_FT)),
                Cz_FT_i=(["ky", "kx", "z", "grains"], np.imag(self.Cz_FT)),
            ),
            coords=dict(
                grains=np.arange(self.params.grains.bins),
                kx=self.kx[:Nx//2+1],
                ky=self.ky,
                z=self.params.output.altitudes,
            ),
            attrs=dict(
                description="AshDisperse results",
                version=__version__,
                domX=self.params.solver.domX,
                domY=self.params.solver.domY,
                minN_log2=self.params.solver.minN_log2,
                maxN_log2=self.params.solver.maxN_log2,
                Nx_log2=self.params.solver.Nx_log2,
                Ny_log2=self.params.solver.Ny_log2,
                epsilon=self.params.solver.epsilon,
                meps=self.params.solver.meps,
                bins=self.params.grains.bins,
                diameter=list(self.params.grains.diameter),
                density=list(self.params.grains.density),
                proportion=list(self.params.grains.proportion),
                name=self.params.source.name,
                latitude=self.params.source.latitude,
                longitude=self.params.source.longitude,
                utmcode=self.params.source.utmcode,
                radius=self.params.source.radius,
                PlumeHeight=self.params.source.PlumeHeight,
                MER=self.params.source.MER,
                duration=self.params.source.duration,
                emission_lower=list(self.params.emission.lower),
                emission_upper=list(self.params.emission.upper),
                emission_profile=list(self.params.emission.profile),
                Suzuki_k=list(self.params.emission.Suzuki_k),
                Kappa_h=self.params.physical.Kappa_h,
                Kappa_v=self.params.physical.Kappa_v,
                g=self.params.physical.g,
                mu=self.params.physical.mu,
                U_scale=self.params.met.U_scale,
                Ws_scale=list(self.params.met.Ws_scale),
                SettlingScale=list(self.params.model.SettlingScale),
                Velocity_ratio=list(self.params.model.Velocity_ratio),
                xScale=list(self.params.model.xScale),
                yScale=list(self.params.model.yScale),
                Lx=list(self.params.model.Lx),
                Ly=list(self.params.model.Ly),
                cScale=list(self.params.model.cScale),
                QScale=list(self.params.model.QScale),
                Peclet_number=self.params.model.Peclet_number,
                Diffusion_ratio=self.params.model.Diffusion_ratio,
                sigma_hat=list(self.params.model.sigma_hat),
                sigma_hat_scale=list(self.params.model.sigma_hat_scale),
                start=self.params.output.start,
                stop=self.params.output.stop,
                step=self.params.output.step,
                altitudes=self.params.output.altitudes,
                Nz=self.params.output.Nz,
            ),
        )

        if compress:
            comp = dict(zlib=True, complevel=complevel)
            #comp = {"compression": "gzip", "compression_opts": complevel}
            encoding = {var: comp for var in da.data_vars}
        else:
            encoding = {}

        da.to_netcdf(
            path=filename,
            mode="w",
            format="NETCDF4",
            engine="h5netcdf",
            encoding=encoding,
        )

        return

    def _check_valid_grain(self, grain_i):
        if grain_i > self.params.grains.bins:
            raise ValueError(
                "Requested grain class not found; given {}".format(grain_i)
                + " in results with"
                + " {} grain classes.".format(self.params.grains.bins)
            )
        if grain_i < 0:
            raise ValueError("Grain class index must be positive")


    def get_groundconc_for_grain_class(self, grain_i, vmin=1e-9, masked=False):
        self._check_valid_grain(grain_i)

        x = self.x[:, grain_i]
        y = self.y[:, grain_i]

        conc = self.C0[:, :, grain_i]

        conc, x, y = _clip_to_window(conc, x, y, vmin=vmin)

        if conc.size == 0:
            return None

        if np.nanmax(conc) < vmin:
            return None

        if masked:
            conc = np.ma.masked_less(conc, vmin)

        return x, y, conc


    def get_settlingflux_for_grain_class(self, grain_i, vmin=1e-7, masked=False):
        self._check_valid_grain(grain_i)

        x = self.x[:, grain_i]
        y = self.y[:, grain_i]

        flux = self.SettlingFlux[:, :, grain_i]

        flux, x, y = _clip_to_window(flux, x, y, vmin=vmin)

        if flux.size == 0:
            return None

        if np.nanmax(flux) < vmin:
            return None

        if masked:
            flux = np.ma.masked_less(flux, vmin)

        return x, y, flux

    def get_ashload_for_grain_class(self, grain_i, vmin=1e-5, masked=False, clipped=True):
        self._check_valid_grain(grain_i)

        x = self.x[:, grain_i]
        y = self.y[:, grain_i]

        ashload = self.SettlingFlux[:, :, grain_i] * self.params.source.duration

        if clipped:
            ashload, x, y = _clip_to_window(ashload, x, y, vmin=vmin)

        if ashload.size == 0:
            return None

        if np.nanmax(ashload) < vmin:
            return None

        if masked:
            ashload = np.ma.masked_less(ashload, vmin)

        return x, y, ashload

    def POI_ashload_for_grain_class(
        self, grain_i: int, lat: float, lon: float, vmin: float = 1e-3
    ):
        self._check_valid_grain(grain_i)

        df = pd.DataFrame(columns=["Name", "Latitude", "Longitude"])
        df.loc[0] = ["POI", lat, lon]
        POI = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            crs=wgs84["init"],
        )

        POI = POI.to_crs(self.utmepsg)

        try:
            ashload = self.raster_ashload_for_grain_class(
                grain_i, vmin=vmin, nodata=np.nan, masked=False, crs=None
            )

            POI_ashload = ashload.interp(
                x=POI.geometry.x, y=POI.geometry.y, method="cubic"
            )

            load = POI_ashload["ash_load"].values[0][0]

            if np.isnan(load):
                return 0
            elif load < 0:
                return 0
            else:
                return load
        except:
            return 0

    def POI_ashload(self, lat: float, lon: float, vmin: float = 1e-3):

        loads = []
        total_load = 0
        for j in range(self.grain_classes):
            this_ashload = self.POI_ashload_for_grain_class(j, lat, lon, vmin)
            loads.append(
                {
                    "diameter": self.params.grains.diameter[j],
                    "density": self.params.grains.density[j],
                    "proportion": self.params.grains.proportion[j],
                    "load": this_ashload,
                }
            )
            total_load += this_ashload

        return total_load, loads

    def POI_groundconc_for_grain_class(
        self, grain_i: int, lat: float, lon: float, vmin: float = 1e-3
    ):
        self._check_valid_grain(grain_i)

        df = pd.DataFrame(columns=["Name", "Latitude", "Longitude"])
        df.loc[0] = ["POI", lat, lon]
        POI = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            crs=wgs84["init"],
        )

        POI = POI.to_crs(self.utmepsg)

        try:
            
            conc = self.raster_groundconc_for_grain_class(

                grain_i, vmin=vmin, nodata=np.nan, masked=False, crs=None
            )

            POI_ashload = conc.interp(
                x=POI.geometry.x, y=POI.geometry.y, method="cubic"
            )

            load = POI_ashload["ground_concentration"].values[0][0]

            if np.isnan(load):
                return 0
            elif load < 0:
                return 0
            else:
                return load
        except:
            return 0

    def POI_groundconc(self, lat: float, lon: float, vmin: float = 1e-3):

        concs = []
        total_conc = 0
        for j in range(self.grain_classes):
            this_conc = self.POI_groundconc_for_grain_class(j, lat, lon, vmin)
            concs.append(
                {
                    "diameter": self.params.grains.diameter[j],
                    "density": self.params.grains.density[j],
                    "proportion": self.params.grains.proportion[j],
                    "ground_concentration": this_conc,
                }
            )
            total_conc += this_conc

        return total_conc, concs
    
    def POI_settlingflux_for_grain_class(
        self, grain_i: int, lat: float, lon: float, vmin: float = 1e-3
    ):
        self._check_valid_grain(grain_i)

        df = pd.DataFrame(columns=["Name", "Latitude", "Longitude"])
        df.loc[0] = ["POI", lat, lon]
        POI = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            crs=wgs84["init"],
        )

        POI = POI.to_crs(self.utmepsg)

        try:
            
            flux = self.raster_settlingflux_for_grain_class(

                grain_i, vmin=vmin, nodata=np.nan, masked=False, crs=None
            )

            POI_flux = flux.interp(
                x=POI.geometry.x, y=POI.geometry.y, method="cubic"
            )

            flux = POI_flux["settling_flux"].values[0][0]

            if np.isnan(flux):
                return 0
            elif flux < 0:
                return 0
            else:
                return flux
        except:
            return 0

    def POI_settlingflux(self, lat: float, lon: float, vmin: float = 1e-3):

        fluxes = []
        total_flux = 0
        for j in range(self.grain_classes):
            this_flux = self.POI_settlingflux_for_grain_class(j, lat, lon, vmin)
            fluxes.append(
                {
                    "diameter": self.params.grains.diameter[j],
                    "density": self.params.grains.density[j],
                    "proportion": self.params.grains.proportion[j],
                    "settling_flux": this_flux,
                }
            )
            total_flux += this_flux

        return total_flux, fluxes

    def max_ashload_for_grain_class(self, grain_i):
        self._check_valid_grain(grain_i)
        return np.nanmax(self.SettlingFlux[:, :, grain_i]) * self.params.source.duration


    def raster_groundconc_for_grain_class(
        self, grain_i, vmin=1e-9, nodata=np.nan, masked=False, crs=None
    ):

        x, y, conc = self.get_groundconc_for_grain_class(
            grain_i, vmin=vmin, masked=masked
        )

        ds = xa.Dataset()
        ds.coords["x"] = x + self.utm[0]
        ds.coords["y"] = y + self.utm[1]

        ds["ground_concentration"] = (("y", "x"), conc)
        ds["ground_concentration"].attrs["short_name"] = "ground_concentration_{}".format(grain_i)
        ds["ground_concentration"].attrs[
            "long_name"
        ] = "concentration at ground level for grain size {} m".format(
            self.params.grains.diameter[grain_i]
        )
        ds["ground_concentration"].attrs["units"] = "kg/m**3"
        ds["ground_concentration"].rio.write_nodata(nodata, inplace=True)

        ds.x.attrs["units"] = "metres"
        ds.y.attrs["units"] = "metres"

        ds = ds.rio.write_crs(self.utmepsg)

        if crs is not None:
            ds = ds.rio.reproject(crs, resampling=Resampling.cubic)

        return ds


    def raster_settlingflux_for_grain_class(
        self, grain_i, vmin=1e-6, nodata=np.nan, masked=False, crs=None
    ):

        x, y, flux = self.get_settlingflux_for_grain_class(
            grain_i, vmin=vmin, masked=masked
        )

        ds = xa.Dataset()
        ds.coords["x"] = x + self.utm[0]
        ds.coords["y"] = y + self.utm[1]

        ds["settling_flux"] = (("y", "x"), flux)
        ds["settling_flux"].attrs["short_name"] = "settling_flux_{}".format(grain_i)
        ds["settling_flux"].attrs[
            "long_name"
        ] = "settling flux at ground level for grain size {} m".format(
            self.params.grains.diameter[grain_i]
        )
        ds["settling_flux"].attrs["units"] = "kg/m**2/s"
        ds["settling_flux"].rio.write_nodata(nodata, inplace=True)

        ds.x.attrs["units"] = "metres"
        ds.y.attrs["units"] = "metres"

        ds = ds.rio.write_crs(self.utmepsg)

        if crs is not None:
            ds = ds.rio.reproject(crs, resampling=Resampling.cubic)

        return ds

    def raster_ashload_for_grain_class(
        self, grain_i, vmin=1e-2, nodata=np.nan, masked=False, clipped=True, crs=None
    ):

        x, y, ashload = self.get_ashload_for_grain_class(
            grain_i, vmin=vmin, masked=masked, clipped=clipped,
        )

        ds = xa.Dataset()
        ds.coords["x"] = x + self.utm[0]
        ds.coords["y"] = y + self.utm[1]

        ds["ash_load"] = (("y", "x"), ashload)
        ds["ash_load"].attrs["short_name"] = "ash_load_{}".format(grain_i)
        ds["ash_load"].attrs[
            "long_name"
        ] = "ash load at ground level for grain size {} m".format(
            self.params.grains.diameter[grain_i]
        )
        ds["ash_load"].attrs["units"] = "kg/m**2"
        ds["ash_load"].rio.write_nodata(nodata, inplace=True)

        ds.x.attrs["units"] = "metres"
        ds.y.attrs["units"] = "metres"

        ds = ds.rio.write_crs(self.utmepsg)

        if crs is not None:
            ds = ds.rio.reproject(crs, resampling=Resampling.cubic, nodata=nodata)

        return ds
    

    def _raster_contour(self, raster, name, cntrs):

        data = raster.data
        x = raster.x.data
        y = raster.y.data
        crs = raster.rio.crs

        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_max = nice_round_up(data_max, mag=10 ** np.floor(np.log10(data_max)))

        fx = interp1d(np.arange(0, len(x)), x)
        fy = interp1d(np.arange(0, len(y)), y)

        cntrs = cntrs[cntrs > data_min]
        cntrs = cntrs[cntrs < data_max]

        for kk, this_cntr in enumerate(cntrs):

            C = find_contours(data, this_cntr)

            if len(C) == 0:
                pass
            for jj, p in enumerate(C):
                p[:, 0] = fy(p[:, 0])
                p[:, 1] = fx(p[:, 1])

                p[:, [0, 1]] = p[:, [1, 0]]

                if len(p[:, 1]) > 2:
                    thisPoly = Polygon(p).buffer(0)
                    if not thisPoly.is_empty:
                        if jj == 0:
                            geom = thisPoly
                            g_tmp = gpd.GeoDataFrame(
                                columns=["contour", "name", "geometry"], crs=crs
                            )
                            g_tmp.loc[0, "contour"] = this_cntr
                            g_tmp.loc[[0], "geometry"] = gpd.GeoSeries([geom])
                            g_tmp.loc[0, "name"] = name
                        else:
                            if g_tmp.loc[0, "geometry"].contains(thisPoly):
                                geom = g_tmp.loc[0, "geometry"].difference(thisPoly)
                            else:
                                try:
                                    geom = g_tmp.loc[0, "geometry"].union(thisPoly)
                                except:
                                    print(
                                        "Error processing polygon contour -- skipping.  Better check the result!"
                                    )
                            g_tmp.loc[[0], "geometry"] = gpd.GeoSeries([geom])

            if kk == 0:
                g = gpd.GeoDataFrame(g_tmp).set_geometry("geometry")
            else:
                g = gpd.GeoDataFrame(pd.concat([g, g_tmp], ignore_index=True))

        g["contour"] = g["contour"].astype("float64")
        g.crs = crs

        return g

    def write_gtiff(self, raster, x, y, outname, nodata=-1, vmin=1e-6, resolution=None):

        print("Writing data to {outname}".format(outname=outname))
        if os.path.isfile(outname):
            print(
                "WARNING: {outname} ".format(outname=outname)
                + "already exists and will be replaced"
            )

        utmcode = self.params.source.utmcode

        if resolution is not None:
            raster, x, y = _interpolate(
                raster, x, y, resolution, nodata=nodata, vmin=vmin
            )

        height, width = raster.shape
        raster[raster < vmin] = nodata

        transform = rio.transform.from_bounds(
            x[0] + self.utm[0],
            y[0] + self.utm[1],
            x[-1] + self.utm[0],
            y[-1] + self.utm[1],
            width,
            height,
        )

        with rio.open(
            outname,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=str(raster.dtype),
            crs=utmcode,
            nodata=nodata,
            transform=transform,
            resampling=Resampling.cubic,
            compress="lzw",
        ) as dst:
            dst.write(np.flipud(raster), 1)

    def write_settling_flux_for_grain_class(
        self, grain_i, nodata=-1, vmin=1e-6, resolution=None
    ):
        self._check_valid_grain(grain_i)
        self.write_gtiff(
            self.SettlingFlux[:, :, grain_i],
            self.x[:, grain_i],
            self.y[:, grain_i],
            "SettlingFlux_{}.tif".format(grain_i),
            nodata=nodata,
            vmin=vmin,
            resolution=resolution,
        )

    def write_ashload_for_grain_class(
        self, grain_i, vmin=1e-3, resolution=None, outname=None, compress='LZW',
    ):
        self._check_valid_grain(grain_i)

        ashload = self.raster_ashload_for_grain_class(grain_i, vmin=vmin, masked=False)

        res_x, res_y = ashload.rio.resolution()
        height = ashload.rio.height
        width = ashload.rio.width
        if resolution is not None:
            new_width = round(width * res_x / resolution)
            new_height = round(height * res_y / resolution)
            ashload = ashload.rio.reproject(
                ashload.rio.crs,
                shape=(new_height, new_width),
                resampling=Resampling.cubic,
            )

        if outname is None:
            outname = "AshLoad_{}.tif".format(grain_i)

        print("Writing data to {outname}".format(outname=outname))
        if os.path.isfile(outname):
            print(
                "WARNING: {outname} ".format(outname=outname)
                + "already exists and will be replaced"
            )

        ashload.rio.to_raster(outname, compress=compress)
        return

    def get_ashload(
        self,
        resolution=300.0,
        vmin=1e-3,
        nodata=-1,
        export_gtiff=True,
        export_name="AshLoad.tif",
    ):

        ash_load_grains = self.SettlingFlux * self.params.source.duration
        Ngrains = self.params.grains.bins

        row_off = []
        col_off = []
        h = []
        w = []
        x_min = []
        y_min = []
        x_max = []
        y_max = []

        for igrain in range(Ngrains):
            thisAshLoad = ash_load_grains[:, :, igrain]
            thisAshLoad[thisAshLoad < vmin / Ngrains / 10] = nodata
            window = get_data_window(thisAshLoad, nodata=nodata)
            row_off.append(window.row_off)
            col_off.append(window.col_off)
            h.append(window.height)
            w.append(window.width)
            x_min.append(self.x[window.col_off, igrain])
            x_max.append(self.x[window.col_off + window.width - 1, igrain])
            y_min.append(self.y[window.row_off, igrain])
            y_max.append(self.y[window.row_off + window.height - 1, igrain])

        min_x = min(x_min)
        min_y = min(y_min)
        max_x = max(x_max)
        max_y = max(y_max)

        left = floor(min_x / resolution) * resolution
        right = ceil(max_x / resolution) * resolution
        bottom = floor(min_y / resolution) * resolution
        top = ceil(max_y / resolution) * resolution

        width = int((right - left) / resolution) + 1
        height = int((top - bottom) / resolution) + 1
        print("Output raster shape is ({h},{w})".format(h=height, w=width))

        x_out = np.linspace(left, right, num=width, endpoint=True)
        y_out = np.linspace(bottom, top, num=height, endpoint=True)

        totalAshLoad = np.zeros((height, width), dtype=np.float64)

        for igrain in range(Ngrains):
            thisAshLoad = ash_load_grains[
                row_off[igrain] : row_off[igrain] + h[igrain] - 1,
                col_off[igrain] : col_off[igrain] + w[igrain] - 1,
                igrain,
            ]
            thisAshLoad[thisAshLoad < vmin / Ngrains / 10] = 0
            x = self.x[col_off[igrain] : col_off[igrain] + w[igrain] - 1, igrain]
            y = self.y[row_off[igrain] : row_off[igrain] + h[igrain] - 1, igrain]
            intrp = RectBivariateSpline(y, x, thisAshLoad)

            totalAshLoad += intrp(y_out, x_out)

        totalAshLoad[totalAshLoad < vmin] = nodata
        window = get_data_window(totalAshLoad, nodata=nodata)
        winpad = pad_window(window, height, width)
        totalAshLoad = totalAshLoad[
            winpad["row_start"] : winpad["row_stop"],
            winpad["col_start"] : winpad["col_stop"],
        ]
        x_out = x_out[winpad["col_start"] : winpad["col_stop"]]
        y_out = y_out[winpad["row_start"] : winpad["row_stop"]]

        if export_gtiff:
            self.write_gtiff(
                totalAshLoad,
                x_out,
                y_out,
                export_name,
                nodata=nodata,
                vmin=vmin,
                resolution=None,
            )
        totalAshLoad = np.ma.masked_where(totalAshLoad < vmin, totalAshLoad)

        totalAshLoad, x_out, y_out = _clip_to_window(totalAshLoad, x_out, y_out, vmin=vmin)
        return x_out, y_out, totalAshLoad
    
    def raster_ashload(self,
        resolution=300.0,
        vmin=1e-3,
        nodata=-1,
        export_gtiff=True,
        export_name="AshLoad.tif",
        crs=None):

        Ngrains = self.params.grains.bins

        if Ngrains==1:
            ds = self.raster_ashload_for_grain_class(0, vmin=vmin, nodata=0.0, clipped=True)
        else:
            # Find largest extent, will be smallest grain size
            # imax = np.argmax(self.params.model.xyScale)
            imax = np.argmax(np.maximum(self.params.model.xScale, self.params.model.yScale))

            # Initialize with this ash load
            ds = self.raster_ashload_for_grain_class(imax, vmin= vmin/Ngrains/10, nodata=0.0, clipped=False)

            ds = ds.rio.reproject(ds.rio.crs, resolution=resolution, resampling=Resampling.cubic, nodata=0.0)

            # Loop through remaining, reproject to match
            for igrain in range(Ngrains):
                # skip over the already used result
                if igrain==imax:
                    continue

                ds_tmp = self.raster_ashload_for_grain_class(igrain, vmin= vmin/Ngrains/100, nodata=0.0, clipped=False)

                ds_tmp = ds_tmp.rio.reproject_match(ds, nodata=0.0, resampling=Resampling.cubic)

                ds += ds_tmp

        ds = ds.where(ds.ash_load >= vmin, nodata)

        ds["ash_load"].attrs["short_name"] = "ash_load"
        ds["ash_load"].attrs[
            "long_name"
        ] = "total ash load at ground level for all grain classes"
        ds["ash_load"].attrs["units"] = "kg/m**2"
        ds["ash_load"].rio.write_nodata(nodata, inplace=True)

        ds.x.attrs["units"] = "metres"
        ds.y.attrs["units"] = "metres"

        if crs is not None:
            ds = ds.rio.reproject(crs, resampling=Resampling.cubic)

        return ds

    def contour_settling_flux(self, grain_i, logscale=True, vmin=1e-6):

        self._check_valid_grain(grain_i)
        data = self.SettlingFlux[:, :, grain_i]
        vmax = np.nanmax(data)
        mag = np.log10(vmax)
        if mag > 0:
            mag = ceil(mag)
        else:
            mag = floor(mag)
        vmax = nice_round_up(vmax, mag=10**mag)

        fx = interp1d(np.arange(0, len(self.x)), self.x[:, grain_i] + self.utm[0])
        fy = interp1d(np.arange(0, len(self.y)), self.y[:, grain_i] + self.utm[1])

        if logscale:
            levels = log_levels(vmin, vmax)
        else:
            levels = lin_levels(vmin, vmax)

        g = None
        for level in levels:
            C = find_contours(data, level)

            if len(C) > 0:
                for jj, p in enumerate(C):
                    pxy = np.zeros_like(p)
                    px = fx(p[:, 1])
                    py = fy(p[:, 0])
                    pxy[:, 0] = px
                    pxy[:, 1] = py

                    this_poly = Polygon(pxy)
                    if not this_poly.is_empty:
                        if jj == 0:
                            geom = this_poly
                            g_tmp = gpd.GeoDataFrame(
                                columns=["SettlingFlux", "GrainSize", "geometry"],
                                crs=self.params.source.utmcode,
                            )
                            g_tmp.loc[0, "GrainClass"] = grain_i
                            g_tmp.loc[0, "GrainSize"] = self.params.grains.diameter[
                                grain_i
                            ]
                            g_tmp.loc[0, "SettlingFlux"] = level
                            g_tmp.loc[0, "geometry"] = geom
                        else:
                            if g_tmp.loc[0, "geometry"].contains(this_poly):
                                geom = g_tmp.loc[0, "geometry"].difference(this_poly)
                            else:
                                geom = g_tmp.loc[0, "geometry"].union(this_poly)
                            g_tmp.loc[[0], "geometry"] = gpd.GeoSeries([geom])

            if g is None:
                g = gpd.GeoDataFrame(g_tmp).set_geometry("geometry")
            else:
                g = gpd.GeoDataFrame(concat([g, g_tmp], ignore_index=True))

        g["SettlingFlux"] = g["SettlingFlux"].astype("float64")
        g["GrainClass"] = g["GrainClass"].astype("int64")
        return g

    def plot_settling_flux_for_grain_class(
        self,
        grain_i,
        logscale=True,
        vmin=1e-6,
        vmax=None,
        cmap=plt.cm.Purples,
        basemap=True,
        alpha=0.5,
        max_zoom=None,
        show=True,
        save=False,
        save_name="ashdisperse_result.png",
        min_ax_width=None,
        min_ax_height=None,
    ):
        self._check_valid_grain(grain_i)

        ds = self.raster_settlingflux_for_grain_class(
            grain_i, vmin=vmin, crs=webmerc["init"]
        )
        if ds is None:
            raise RuntimeError("No data in plot_settling_flux_for_grain_class()")

        x = ds.x
        y = ds.y
        data = ds["settling_flux"].values

        x_intrp = interp1d(np.arange(len(x)), x)
        y_intrp = interp1d(np.arange(len(y)), y)

        if vmax is None:
            maxflux = np.nanmax(data)
            mag = np.log10(maxflux)
            if mag > 0:
                mag = ceil(mag)
            else:
                mag = floor(mag)
            vmax = nice_round_up(maxflux, mag=10**mag)
        else:
            mag = np.log10(vmax)
            if mag > 0:
                mag = ceil(mag)
            else:
                mag = floor(mag)

        if vmin is None:
            vmin = 10 ** (mag - 3)

        cbar_fig, cbar_ax = plt.subplots()
        if logscale:
            levels = log_levels(vmin, vmax)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=LogNorm()
            )
        else:
            levels = lin_levels(vmin, vmax, num=20)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=Normalize()
            )

        fig, ax = plt.subplots()
        for j, l in enumerate(levels):
            cntrs = find_contours(data, l)
            for c in cntrs:
                ax.fill(
                    x_intrp(c[:, 1]),
                    y_intrp(c[:, 0]),
                    color=cmap(j / len(levels)),
                    alpha=alpha,
                    zorder=1,
                )

        source = self.source_marker.to_crs(webmerc["init"])
        source.plot(ax=ax, marker="^", color="k", markersize=20, zorder=2)

        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        x_width = xlim[1] - xlim[0]
        y_height = ylim[1] - ylim[0]

        min_ax_width = int(1.5 * x_width)
        min_ax_height = int(1.5 * y_height)

        set_min_axes(ax, min_width=min_ax_width, min_height=min_ax_height)

        fig = set_figure_size(fig, ax)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = plt.colorbar(tmp, ax=ax, cax=cax)
        cbar.minorticks_off()
        cbar.set_label("Settling flux (kg/m\u00B2/s)")
        plt.close(cbar_fig)

        ax = ax_ticks(ax, source.geometry[0].x, source.geometry[0].y)
        if basemap:
            ax = add_opentopo_basemap(ax, zorder=0)
            (Narrow, ntext) = add_north_arrow(ax, zorder=11, fontsize=16)
            (scalebar, sbframe) = add_scale_bar(ax, segments=1)

        plt.draw()
        if save:
            plt.savefig(save_name, dpi=300, transparent=True, format="png")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return (fig, ax, cbar)

    def plot_ashload_for_grain_class(
        self,
        grain_i,
        logscale=True,
        vmin=None,
        vmax=10,
        cmap=plt.cm.plasma,
        basemap=True,
        alpha=0.5,
        max_zoom=None,
        show=True,
        save=False,
        save_name="ashdisperse_result.png",
        min_ax_width=None,
        min_ax_height=None,
    ):
        self._check_valid_grain(grain_i)

        ds = self.raster_ashload_for_grain_class(
            grain_i, vmin=1e-6, crs=webmerc["init"]
        )
        if ds is None:
            raise RuntimeError("No data in plot_ashload_for_grain_class()")

        x = ds.x
        y = ds.y
        data = ds["ash_load"].values

        x_intrp = interp1d(np.arange(len(x)), x)
        y_intrp = interp1d(np.arange(len(y)), y)

        if vmax is None:
            maxload = np.nanmax(data)
            mag = np.log10(maxload)
            if mag > 0:
                mag = ceil(mag)
            else:
                mag = floor(mag)
            vmax = nice_round_up(maxload, mag=10**mag)
        else:
            mag = np.log10(vmax)
            if mag > 0:
                mag = ceil(mag)
            else:
                mag = floor(mag)

        if vmin is None:
            vmin = 10 ** (mag - 3)

        cbar_fig, cbar_ax = plt.subplots()
        if logscale:
            levels = log_levels(vmin, vmax)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=LogNorm()
            )
        else:
            levels = lin_levels(vmin, vmax, num=20)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=Normalize()
            )

        fig, ax = plt.subplots()
        for j, l in enumerate(levels):
            cntrs = find_contours(data, l)
            for c in cntrs:
                ax.fill(
                    x_intrp(c[:, 1]),
                    y_intrp(c[:, 0]),
                    color=cmap(j / len(levels)),
                    alpha=alpha,
                    zorder=1,
                )

        source = self.source_marker.to_crs(webmerc["init"])
        source.plot(ax=ax, marker="^", color="k", markersize=20, zorder=2)

        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        x_width = xlim[1] - xlim[0]
        y_height = ylim[1] - ylim[0]

        min_ax_width = int(1.5 * x_width)
        min_ax_height = int(1.5 * y_height)

        set_min_axes(ax, min_width=min_ax_width, min_height=min_ax_height)

        fig = set_figure_size(fig, ax)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = plt.colorbar(tmp, ax=ax, cax=cax)
        cbar.minorticks_off()
        # cbar.set_ticks([levels])
        cbar.set_label("Ash load (kg/m\u00B2)")
        plt.close(cbar_fig)

        ax = ax_ticks(ax, source.geometry[0].x, source.geometry[0].y)

        if basemap:
            ax = add_opentopo_basemap(ax, zorder=0)
            (Narrow, ntext) = add_north_arrow(ax, zorder=11, fontsize=16)
            (scalebar, sbframe) = add_scale_bar(ax, segments=1)

        plt.draw()
        if save:
            plt.savefig(save_name, dpi=300, transparent=True, format="png")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return (fig, ax, cbar)

    def contour_ashload_for_grain_class(self, grain_i, cntrs_levels="log", vmin=1e-2):

        self._check_valid_grain(grain_i)

        ashload = self.raster_ashload_for_grain_class(
            grain_i,
            vmin=1e-6,
            nodata=np.nan,
            masked=False,
            crs=webmerc["init"],
        )

        maxload = np.nanmax(ashload["ash_load"])
        mag = np.log10(maxload)
        if mag > 0:
            mag = ceil(mag)
        else:
            mag = floor(mag)
        vmax = nice_round_up(maxload, mag=10**mag)

        if vmin is None:
            vmin = 10 ** (mag - 3)

        if cntrs_levels == "log":
            levels = log_levels(vmin, vmax)
        else:
            levels = lin_levels(vmin, vmax)

        g = self._raster_contour(
            ashload["ash_load"], str(self.params.grains.diameter[grain_i]), levels
        )
        g.rename(columns={"contour": "load", "name": "grain_size"}, inplace=True)
        return g

    def plot_conc_for_grain_class(
        self, grain_i, logscale=True, vmin=1e-6, cmap=plt.cm.bone, basemap=False
    ):
        self._check_valid_grain(grain_i)
        Nz = self.params.output.Nz
        rows, cols = plot_rowscols(Nz + 1)
        fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
        maxc = np.nanmax(self.Cz[:, :, :, grain_i])
        mag = np.log10(maxc)
        if mag > 0:
            mag = ceil(mag)
        else:
            mag = floor(mag)
        vmax = nice_round_up(maxc, mag=10**mag)

        cbar_fig, cbar_ax = plt.subplots()
        if logscale:
            levels = log_levels(vmin, vmax)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=LogNorm()
            )
        else:
            levels = lin_levels(vmin, vmax, num=20)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=Normalize()
            )

        x = self.x[:, grain_i] / 1e3
        y = self.y[:, grain_i] / 1e3
        for j, ax in enumerate(axes.reshape(-1)):
            if j < Nz:
                data = self.Cz[:, :, Nz - j - 1, grain_i]
                if np.nanmax(data) > vmin:
                    # # data = np.ma.masked_where(data <= vmin, data)
                    data[data <= vmin] = np.nan
                    if logscale:
                        CS = ax.contourf(
                            x,
                            y,
                            data,
                            levels,
                            locator=ticker.LogLocator(),
                            cmap=cmap,
                            origin="lower",
                        )
                    else:
                        CS = ax.contourf(
                            x,
                            y,
                            data,
                            levels,
                            cmap=cmap,
                            origin="lower",
                            vmin=vmin,
                            vmax=vmax,
                        )
                ax.set_title(
                    f"z = {self.params.output.altitudes[Nz - j - 1]} m", fontsize=10
                )
            elif j == Nz:
                data = self.C0[:, :, grain_i]
                data[data <= 1e-6] = np.nan
                if np.nanmax(data) > vmin:
                    if logscale:
                        CS = ax.contourf(
                            x,
                            y,
                            data,
                            levels,
                            locator=ticker.LogLocator(),
                            cmap=cmap,
                            origin="lower",
                        )
                    else:
                        CS = ax.contourf(
                            x,
                            y,
                            data,
                            levels,
                            cmap=cmap,
                            origin="lower",
                            vmax=vmax,
                        )
                ax.set_title("z = 0 m")
            else:
                ax.axis("off")

        fig.subplots_adjust(right=0.75)
        cbar_ax = fig.add_axes([0.8, 0.2, 0.05, 0.7])

        cbar = fig.colorbar(tmp, cax=cbar_ax)
        cbar.minorticks_off()
        cbar.set_label("Ash concentration (kg/m\u00B3)")
        plt.close(cbar_fig)

        if basemap:
            ax = add_opentopo_basemap(ax, zorder=0)
            (Narrow, ntext) = add_north_arrow(ax, zorder=11, fontsize=16)
            (scalebar, sbframe) = add_scale_bar(ax, segments=1)

        return (fig, ax, cbar)

    def plot_iso_conc_for_grain_class(self, grain_i, conc):
        self._check_valid_grain(grain_i)
        C = self.Cz[:, :, :, grain_i]

        if np.nanmax(C) < conc:
            message = (
                "In plotIsoConc, no concentration values in excess of"
                + " {}".format(conc)
            )
            print(message)
            return

        # xyScale = self.params.model.xyScale[grain_i]
        xScale = self.params.model.xScale[grain_i]
        yScale = self.params.model.yScale[grain_i]

        z = self.params.output.altitudes

        verts, faces, _, _ = marching_cubes(
            np.where(C > conc, np.ones_like(C), np.zeros_like(C)),
            0.5,
            spacing=(1, 1, 1),
        )

        dx = self.x_dimless[1] - self.x_dimless[0]
        dy = self.y_dimless[1] - self.y_dimless[0]
        dz = z[1] - z[0]

        yv = (
            (verts[:, 0] * dx - np.pi)
            * xScale #xyScale
            * self.params.model.Lx[grain_i]
            / np.pi
            / 1e3
        )
        xv = (
            (verts[:, 1] * dy - np.pi)
            * yScale #xyScale
            * self.params.model.Ly[grain_i]
            / np.pi
            / 1e3
        )
        zv = verts[:, 2] * dz

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_trisurf(xv, yv, faces, zv, cmap="Spectral", lw=1)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.set_zlabel("Altitude (m)")

    def plot_ashload(
        self,
        resolution=300.0,
        logscale=True,
        cmap=plt.cm.viridis,
        vmin=1e-3,
        nodata=-1,
        alpha=0.5,
        basemap=False,
        export_gtiff=False,
        export_name="AshLoad.tif",
        show=True,
    ):

        ds = self.raster_ashload(
            resolution=resolution,
            vmin=vmin / 10,
            nodata=nodata,
            export_gtiff=export_gtiff,
            export_name=export_name,
            crs=webmerc["init"]
        )
        if ds is None:
            raise RuntimeError("No data in plot_ashload()")

        x = ds.x
        y = ds.y
        data = ds["ash_load"].values

        x_intrp = interp1d(np.arange(len(x)), x)
        y_intrp = interp1d(np.arange(len(y)), y)

        maxc = np.nanmax(data)
        mag = np.log10(maxc)
        if mag > 0:
            mag = ceil(mag)
        else:
            mag = floor(mag)
        vmax = nice_round_up(maxc, mag=10**mag)
        data = np.ma.masked_where(data <= vmin, data)

        cbar_fig, cbar_ax = plt.subplots()
        if logscale:
            levels = log_levels(vmin, vmax)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=LogNorm()
            )
            # CS = ax.contourf(
            #     x,
            #     y,
            #     data,
            #     levels,
            #     locator=ticker.LogLocator(),
            #     cmap=cmap,
            #     origin="lower",
            # )
        else:
            levels = lin_levels(vmin, vmax, num=20)
            tmp = cbar_ax.scatter(
                levels, np.ones_like(levels), c=levels, cmap=cmap, norm=Normalize()
            )
            # CS = ax.contourf(x, y, data, levels, cmap=cmap, origin="lower")

        fig, ax = plt.subplots()
        for j, l in enumerate(levels):
            cntrs = find_contours(data, l)
            for c in cntrs:
                ax.fill(
                    x_intrp(c[:, 1]),
                    y_intrp(c[:, 0]),
                    color=cmap(j / len(levels)),
                    alpha=alpha,
                    zorder=1,
                )

        source = self.source_marker.to_crs(webmerc["init"])
        source.plot(ax=ax, marker="^", color="k", markersize=20, zorder=2)

        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        x_width = xlim[1] - xlim[0]
        y_height = ylim[1] - ylim[0]

        min_ax_width = int(1.5 * x_width)
        min_ax_height = int(1.5 * y_height)

        set_min_axes(ax, min_width=min_ax_width, min_height=min_ax_height)

        fig = set_figure_size(fig, ax)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = plt.colorbar(tmp, ax=ax, cax=cax)
        cbar.minorticks_off()
        # cbar.set_ticks([levels])
        cbar.set_label("Ash load (kg/m\u00B2)")
        plt.close(cbar_fig)

        ax = ax_ticks(ax, source.geometry[0].x, source.geometry[0].y)

        if basemap:
            ax = add_opentopo_basemap(ax, zorder=0)
            (Narrow, ntext) = add_north_arrow(ax, zorder=11, fontsize=16)
            (scalebar, sbframe) = add_scale_bar(ax, segments=1)

        plt.draw()

        if show:
            plt.show()
        else:
            plt.close(fig)

        return (fig, ax, cbar)
    
    def plot_spectrum_for_grainsize(self, grain_i, vmin=None, vmax=None):
        fig, ax = plt.subplots()
        im = ax.imshow(np.log10(np.absolute(self.C0_FT[:,:,grain_i])), vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        plt.show()

    def folium_ashloads(self, savename, vmin=1e-3):

        m = folium.Map(
            location=[self.params.source.latitude, self.params.source.longitude],
            zoom_start=13,
            control_scale=True,
            prefer_canvas=True,
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="OpenTopoMap"
        )

        style_func = lambda x: {
            "weight": 0.1,
            "color": "black",
            "fillColor": cmap(x["properties"]["load"]),
            "fillOpacity": 0.5,
        }

        for grain_i in range(self.params.grains.bins):
            g = self.contour_ashload_for_grain_class(grain_i, vmin=1e-4)
            g = g.to_crs(webmerc["init"])
            g["geoid"] = g.index.astype(str)
            loads = g[["geoid", "load", "grain_size", "geometry"]]
            loads["grain_size"] = loads["grain_size"].astype(float) * 1e6
            loads["grain_size"] = loads["grain_size"].map("{0:.2f}".format)

            geo_str = loads.to_json()

            if vmin is None:
                vmin = loads.load.min()
            vmax = loads.load.max()

            levels = log_steps(
                nice_round_down(vmin, mag=10 ** np.floor(np.log10(vmin))),
                nice_round_up(vmax),
                step=10,
            )

            cmap = cm.linear.viridis.to_step(
                data=loads["load"], index=levels, method="log", round_method="log10"
            )
            cmap.caption = "ash load (kg/m^2) for {0:.2f} micron grains".format(
                self.params.grains.diameter[grain_i] * 1e6
            )

            lm = folium.features.GeoJson(
                loads,
                style_function=style_func,
                control=True,
                name="Grain size = {0:.2f} microns".format(
                    self.params.grains.diameter[grain_i] * 1e6
                ),
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["grain_size", "load"],
                    aliases=["Grain size (microns)", "Ash load (kg/m^2)"],
                    style=(
                        "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
                    ),
                    sticky=True,
                ),
            )

            m.add_child(lm)
            m.add_child(cmap)
            m.add_child(BindColormap(lm, cmap))

        folium.LayerControl().add_to(m)

        m.save(savename)

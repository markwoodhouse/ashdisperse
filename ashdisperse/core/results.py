from __future__ import annotations
import os
import warnings
from math import ceil, floor
from typing import Self, Optional, TypeAlias

import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
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

from ashdisperse.params import Parameters
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

Point: TypeAlias = tuple[float, float]
PointList: TypeAlias = list[Point]

def compat_warning(message: str, category: str, filename:str, lineno, file=None, line=None) -> str:
    return f"{category.__name__}: {message}"

def _clip_to_window(
        raster: np.ndarray, 
        x: np.ndarray, 
        y: np.ndarray, 
        vmin: float, 
        pad: int=1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clip a raster to values >= vmin

    Parameters
    ----------
    raster : np.ndarray
        2D array to clip
    x : np.ndarray
        1D array of x-coordinates
    y : np.ndarray
        1D array of y-coordinates
    vmin : float
        threshold for clipping
    pad : int, optional
        Include extra cells around clipped values, by default 1

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        raster_clip, x_clip, y_clip
        The clipped raster, and clipped x and y coordinates
    """
    
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


def _interpolate(
        raster: np.ndarray, 
        x: np.ndarray, 
        y: np.ndarray, 
        resolution: float, 
        extent: Optional[list[float]]=None, 
        vmin: float=1e-6,
        nodata: float=-1, 
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a raster to different grid resolution

    Parameters
    ----------
    raster : np.ndarray
        2D array to interpolate
    x : np.ndarray
        1D array of x coordinates
    y : np.ndarray
        1D array of y coordinates
    resolution : float
        grid cell size to interpolate on to
    extent : Optional[list[float]], optional
        bounding box to limit the extent, ordered as [min_x, min_y, max_x, max_y], by default None
    vmin : float, optional
        lower thresholding value.  Values in input raster < vmin are ignored, by default 1e-6
    nodata : float, optional
        no data value.  Values in input raster < vmin are set to nodata , by default -1
    

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (x_out, y_out, raster_out)
        The interpolated raster (raster_out) and new coordinates (x_out, y_out) as arrays.
    """


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

    left = floor(min_x / resolution) * resolution
    right = ceil(max_x / resolution) * resolution
    bottom = floor(min_y / resolution) * resolution
    top = ceil(max_y / resolution) * resolution

    width = int((right - left) / resolution) + 1
    height = int((top - bottom) / resolution) + 1

    x_out = np.linspace(left, right, num=width, endpoint=True)
    y_out = np.linspace(bottom, top, num=height, endpoint=True)

    raster_tmp, x_tmp, y_tmp = _clip_to_window(raster, x, y, vmin=vmin / 10, pad=12)
    raster_tmp[raster_tmp < vmin / 10] = 0.0

    intrp = RectBivariateSpline(y_tmp, x_tmp, raster_tmp)

    raster_out = intrp(y_out, x_out)

    return x_out, y_out, raster_out



def _sample_point_from_dataset(
    ds: xa.Dataset,
    target_crs: rio.crs.CRS,
    latlon: Optional[Point | PointList] = None,
    xy: Optional[Point | PointList] = None,
    var: str = "ash_load",
    fallback: float = 0.0,
    method: str = "linear",
) -> float | np.ndarray:
    """
    Sample one or more points from an xarray Dataset using projected or geographic coordinates.

    This utility interpolates values from a gridded dataset at specified locations.
    It supports input as projected coordinates (`xy`) or geographic coordinates (`latlon`),
    automatically transforming WGS84 coordinates to the target CRS when necessary.

    Behavior
    --------
    - If `xy` is provided, it is used directly.
    - If `xy` is not provided, `latlon` (in WGS84) is transformed to `target_crs`.
    - If neither `xy` nor `latlon` is provided, returns `fallback`.
    - Returns `fallback` for missing variables, interpolation errors, or no valid data.
    - Handles both scalar and vector inputs; returns a float for a single point or a NumPy array for multiple points.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to sample and `x`, `y` coordinates (projected CRS preferred).
    target_crs : rasterio.crs.CRS
        Coordinate reference system to project into when `latlon` is provided.
    latlon : Point or list of Point, optional
        Geographic coordinates (latitude, longitude) in WGS84. Used if `xy` is not provided.
    xy : Point or list of Point, optional
        Projected coordinates in the same CRS as `ds`. Preferred over `latlon` if both are given.
    var : str, optional
        Name of the variable in `ds` to sample. Default is `"ash_load"`.
    fallback : float, optional
        Value returned when sampling fails, variable is missing, or no valid data exists. Default is `0.0`.
    method : str, optional
        Interpolation method passed to `xarray.DataArray.interp`. Common options: `"linear"`, `"nearest"`.
        Default is `"linear"`.

    Returns
    -------
    float or ndarray
        Interpolated value(s) for the requested variable:
        - A single float if one point is sampled.
        - A NumPy array if multiple points are sampled.
        Returns `fallback` or an array filled with `fallback` if sampling fails.

    Raises
    ------
    UserWarning
        - If `var` is not found in the dataset.
        - If neither `xy` nor `latlon` is provided.
        - If an error occurs during interpolation or CRS transformation.

    Notes
    -----
    - Negative or NaN values are replaced with `fallback`.
    - CRS transformation uses `pyproj.Transformer` with `always_xy=True` (longitude, latitude order).
    - For multiple points, input can be a list of tuples or a NumPy array with shape `(n, 2)`.

    Examples
    --------
    >>> # Sample using projected coordinates
    >>> val = _sample_point_from_dataset(ds, target_crs, xy=(500000, 5200000), var="ash_load")
    >>> print(val)
    0.0021

    >>> # Sample multiple points using    >>> # Sample multiple points using lat/lon
    >>> points = [(45.1, -122.3), (45.2, -122.4)]
    >>> vals = _sample_point_from_dataset(ds, target_crs, latlon=points, var="ash_load")
    >>> vals
    array([0.0018, 0.0020])
    """
    if ds is None:
        return fallback
    
    if xy is not None:
        coords = xy
    elif latlon is not None:
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

        if isinstance(latlon, list):
            latlon = np.asarray(latlon)
            lat = latlon[:,0] # type: ignore
            lon = latlon[:,1] # type: ignore
        else:
            lat = latlon[0]
            lon = latlon[1]
        x, y = transformer.transform(lon, lat)
        coords = (x, y)
    else:
        warnings.warn("Need either xy or latlon to sample")
        return fallback
    
    if isinstance(coords, list) or (isinstance(coords, np.ndarray) and coords.ndim == 2):
        x = np.asarray(coords)[:, 0]
        y = np.asarray(coords)[:, 1]
    else:
        x = coords[0]
        y = coords[1]

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    try:
        
        target_x = xa.DataArray(np.atleast_1d(x), dims="locations")
        target_y = xa.DataArray(np.atleast_1d(y), dims="locations")

        sampled = ds[var].interp(x=target_x, y=target_y, method=method)
        vals = np.array(sampled.values)

        if vals.size == 0:
            return fallback

        vals[vals<=0] = fallback
        vals[np.isnan(vals)] = fallback

        return vals
    except Exception as exc:
        warnings.warn(f"Error sampling point from dataset: {exc}")
        return fallback*np.ones_like(x)
        

class AshDisperseResult:
    """Results of an AshDisperse simulation"""
    params: Parameters
    utm: tuple[float,float,int,str]
    utmepsg: int
    C0_FT: np.ndarray
    Cz_FT: np.ndarray
    x_dimless: np.ndarray
    y_dimless: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    C0_dimless: np.ndarray
    Cz_dimless: np.ndarray
    C0: np.ndarray
    Cz: np.ndarray
    SettlingFlux: np.ndarray
    x: np.ndarray
    y: np.ndarray
    
    _interpolated: bool

    def __init__(
            self, params: Parameters, C0_FT: np.ndarray, Cz_FT: np.ndarray
        ):
        """
        Create a AshDisperseResult instance from parameters and arrays of Fourier coefficients

        Parameters
        ----------
        params : Parameters
            Parameter settings for this result
        C0_FT : np.ndarray
            Array of Fourier coefficients for ground-level concentration ordered as [y,x,grain]
        Cz_FT : np.ndarray
            Array of Fourier coefficients for z-level concentration ordered as [y,x,z,grain]

        Notes
        -----
        params contains essential data for reconstructing the solution from the Fourier coefficient arrays.
        It is expected that params is created using ashdisperse.set_parameters()
        or loaded from a file saved after this construction
        """
        
        self.params = copy_parameters(params)

        self._interpolated = False

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
        
        for igrain in range(Ng):
            C0[:, :, igrain] = irfft2(C0_FT[:, :, igrain], s=(Ny,Nx)) / (res_x * res_y)
            for k in range(Nz):
                Cz[:, :, k, igrain] = irfft2(Cz_FT[:, :, k, igrain], s=(Ny,Nx)) / (res_x * res_y)

        self.C0_dimless = C0
        self.Cz_dimless = Cz

        self.C0 = np.zeros_like(C0)
        self.Cz = np.zeros_like(Cz)
        self.SettlingFlux = np.zeros_like(C0)

        self.x = np.zeros((len(x), Ng))
        self.y = np.zeros((len(y), Ng))

        for j in range(Ng):
            self.x[:, j] = x * params.model.xScale[j] * params.model.Lx[j] / np.pi
            self.y[:, j] = y * params.model.yScale[j] * params.model.Ly[j] / np.pi
            self.C0[:, :, j] = C0[:, :, j] * params.model.cScale[j]
            self.Cz[:, :, :, j] = Cz[:, :, :, j] * params.model.cScale[j]
            self.SettlingFlux[:, :, j] = (
                params.met.Ws_scale[j] * C0[:, :, j] * params.model.cScale[j]
            )


    @property
    def grain_classes(self) -> int:
        """
        Number of grain classes

        Returns
        -------
        int
            Number of grain classes
        """
        return self.params.grains.bins
    
    @property
    def interpolated(self) -> bool:
        """
        Check if this AshDisperseResult is produced by interpolation

        Returns
        -------
        bool
            True if this AshDisperseResult is produced by zero-padding interpolation from a computed AshDisperseResult,
            False otherwise
        """
        return self._interpolated

    @property
    def source_marker(self) -> gpd.GeoDataFrame:
        """
        GeoDataFrame containing the source location

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the source location in WGS84 coordinates
        """

        df = pd.DataFrame(columns=["Name", "Latitude", "Longitude"])
        df.loc[0] = [
            self.params.source.name,
            self.params.source.latitude,
            self.params.source.longitude,
        ]
        source_marker = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            crs=wgs84["init"],
        )
        return source_marker


    def linear_interp(self, Nx_log2: int, Ny_log2: int) -> AshDisperseResult:
        """
        Return a new AshDisperseResult with bilinear interpolation by zero-padding the FFTs.

        Parameters
        ----------
        Nx_log2 : int
            log2 of the number of Fourier modes in x
        Ny_log2 : int
            log2 of the number of Fourier modes in y

        Returns
        -------
        AshDisperseResult
            A new AshDisperseResult.  The original object is unchanged.

        Notes
        -----
        The internal stored parameters are updated for the new sizes.
        A class private variable _interpolated is set to True to indicate this AshDisperseResult is generated by zero-padding interpolation.
        """
        
        Nx_old = self.params.solver.Nx
        Ny_old = self.params.solver.Ny
        Ng = self.params.grains.bins
        Nz = self.params.output.Nz

        Nx = 2**Nx_log2
        Ny = 2**Ny_log2

        half_Nx_old = Nx_old // 2 + 1
        half_Nx = Nx // 2 + 1

        # For the ky dimension we must preserve the Nyquist/mid row when present.
        # Copy the low-frequency block (including mid if Ny_old is even) and the
        # high-frequency tail into the end of the new array so the spectral
        # layout is preserved when transforming back to physical space.
        half_Ny_old = Ny_old // 2 + 1

        C0_FT = np.zeros((Ny, half_Nx, Ng), dtype=np.complex128)
        Cz_FT = np.zeros((Ny, half_Nx, Nz, Ng), dtype=np.complex128)

        # copy low-frequency block (rows 0 .. half_Ny_old-1)
        C0_FT[:half_Ny_old, :half_Nx_old, :] = self.C0_FT[:half_Ny_old, :, :]
        # copy high-frequency tail to the end of the new array
        start_row = Ny - (Ny_old - half_Ny_old)
        C0_FT[start_row:, :half_Nx_old, :] = self.C0_FT[half_Ny_old:, :, :]

        Cz_FT[:half_Ny_old, :half_Nx_old, :, :] = self.Cz_FT[:half_Ny_old, :, :, :]
        Cz_FT[start_row:, :half_Nx_old, :, :] = self.Cz_FT[half_Ny_old:, :, :, :]

        params = copy_parameters(self.params)
        params.solver.Nx_log2 = Nx_log2
        params.solver.Ny_log2 = Ny_log2

        r = AshDisperseResult(params, C0_FT, Cz_FT)
        r._interpolated = True

        return r


    def change_MER(self, MER: float) -> AshDisperseResult:
        """
        Create a new AshDisperseResult with updated mass eruption rate (MER).

        This method does not modify the original object; instead, it returns a new
        instance with the specified MER applied.

        Parameters
        ----------
        MER : float
            A float representing the new MER.  The MER must be positive.

        Returns
        -------
        AshDisperseResult
            A new AshDisperseResult instance with updated MER.

        Raises
        ------
        ValueError
            If MER is not positive.

        Examples
        --------
        Change original_result with new MER of 1e7
            >>> result = original_result.change_duration(1e6)
            >>> result.params.source.MER
            1000000.0
        """

        if MER<=0:
            raise ValueError(f"MER must be positive, received {MER}")
        
        params = copy_parameters(self.params)
        params.source.MER = MER
        params.model.from_params(
            params,
            params.model.xScale,
            params.model.yScale
        )
        return AshDisperseResult(params, self.C0_FT, self.Cz_FT)

    def change_duration(self, duration: float) -> AshDisperseResult:
        """
        Create a new AshDisperseResult with updated duration.

        This method does not modify the original object; instead, it returns a new
        instance with the specified duration applied.

        Parameters
        ----------
        duration : float
            A float representing the new duration.  The duration must be positive.

        Returns
        -------
        AshDisperseResult
            A new AshDisperseResult instance with updated duration.

        Raises
        ------
        ValueError
            If duration is not positive.

        Examples
        --------
        Change original_result with new duration of one hour
            >>> result = original_result.change_duration(3600)
            >>> result.params.source.duration
            3600.0
        """
        if duration<=0:
            raise ValueError(f"duration must be positive, received {duration}")

        params = copy_parameters(self.params)
        params.source.duration = duration
        return AshDisperseResult(params, self.C0_FT, self.Cz_FT)

    def change_grain_proportions(self, props: list[float]) -> AshDisperseResult:
        """
        Create a new AshDisperseResult with updated grain size proportions.

        This method does not modify the original object; instead, it returns a new
        instance with the specified grain size proportions applied.

        Parameters
        ----------
        props : list of float
            A list of floats representing the new proportions for each grain size class.
            The length of this list must equal the number of grain size classes defined
            in `self.params.grains.bins`. Each value must be in the range [0, 1].

        Returns
        -------
        AshDisperseResult
            A new AshDisperseResult instance with updated grain proportions.

        Raises
        ------
        RuntimeError
            If the length of `props` does not match the number of grain size classes
            (`self.params.grains.bins`).
        ValueError
            If any element in `props` is outside the valid range [0, 1]. Each proportion
            must be a float between 0 and 1 inclusive, representing the fraction of material
            in that grain size class.

        Examples
        --------
        Change original_result with three grain classes
            >>> result = original_result.change_grain_proportions([0.2, 0.3, 0.5])
            >>> result.params.grains.proportion
            array([0.2, 0.3, 0.5])
        """
        
        if len(props) != self.params.grains.bins:
            raise RuntimeError(f'length of props must equal number of grain size classes ({self.params.grains.bins})')

        params = copy_parameters(self.params)
        for j, p in enumerate(props):
            if p<0 or p>1:
                raise ValueError(f"proportion must be in range [0,1], recieved {p}")
            params.grains.proportion[j] = np.float64(p)

        params.model.from_params(
            params,
            params.model.xScale,
            params.model.yScale
        )
        return AshDisperseResult(params, self.C0_FT, self.Cz_FT)


    @classmethod
    def from_netcdf(cls, filename: str) -> AshDisperseResult:
        """
        Create an :class:`AshDisperseResult` from a NetCDF file.

        This classmethod loads a NetCDF dataset, reconstructs the full set of model
        parameters (solver, grains, source, emission, physical, meteorology, output,
        and model parameters), and returns a new result object populated with the
        spectral fields `C0_FT` and `Cz_FT`.

        The method attempts to handle both scalar and array encodings for several
        parameter groups (e.g., grains, emission, model), choosing between
        ``from_values`` and ``from_lists`` depending on the underlying type.

        Parameters
        ----------
        filename : str
            Path to the NetCDF file containing AshDisperse output. This file is
            expected to include fields such as ``version``, solver domain and grid
            settings, grain properties (diameter, density, proportion), source
            parameters, emission profile parameters, physical constants, meteorology,
            output scheduling, model scaling parameters, and the real/imaginary parts
            of the spectral fields:
            ``C0_FT_r``, ``C0_FT_i``, ``Cz_FT_r``, and ``Cz_FT_i``.

        Returns
        -------
        Self
            A new instance of :class:`AshDisperseResult` initialized from the
            contents of the NetCDF file.

        Raises
        ------
        FileNotFoundError
            If ``filename`` does not exist.
        OSError
            If the file cannot be opened or read as a NetCDF dataset.
        KeyError
            If required variables are missing from the dataset (e.g., grid settings,
            grain properties, spectral components).
        ValueError
            If variable shapes or dtypes are incompatible with the expected parameter
            constructors (e.g., mismatched array lengths across diameter/density/
            proportion or emission profiles).

        Warnings
        --------
        UserWarning
            If the dataset version (``da.version``) differs from the current library
            version (``__version__``). The method continues but warns about possible
            incompatibilities.

        Notes
        -----
        - Grain parameters:
        Switches between ``GrainParameters.from_values`` and
        ``GrainParameters.from_lists`` depending on whether ``da.diameter`` is
        loaded as a scalar (``np.float64``) or an array-like.
        - Emission parameters:
        Similarly selects ``EmissionParameters.from_values`` vs ``from_lists``
        based on the type of ``da.emission_lower``.
        - Model parameters:
        Chooses ``ModelParameters.from_values`` vs ``from_lists`` using the type
        of ``da.SettlingScale``.
        - Spectral fields:
        The complex arrays are reconstructed from their real and imaginary parts as
        ``C0_FT = C0_FT_r + 1j * C0_FT_i`` and ``Cz_FT = Cz_FT_r + 1j * Cz_FT_i``.

        Examples
        --------
        >>> result = AshDisperseResult.from_netcdf("outputs/ashdisperse_run.nc")
        >>> result.params.solver.Nx_log2, result.params.solver.Ny_log2
        (10, 10)
        >>> result.C0_FT.shape, result.Cz_FT.shape
        ((256, 256), (256, 256))
        """
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

        return AshDisperseResult(p, C0_FT, Cz_FT)

    
    def to_netcdf(self, filename: str = "AshDisperse.nc", compress: bool = True, complevel: int = 5) -> None:
        """
        Save the current :class:`AshDisperseResult` to a NetCDF file.

        This method writes the complex spectral fields and all associated model
        parameters to a NetCDF file using the ``h5netcdf`` engine. Complex arrays
        are stored as separate real and imaginary components.

        Parameters
        ----------
        filename : str, optional
            Output file path. If the file exists, it will be overwritten.
            Default is ``"AshDisperse.nc"``.
        compress : bool, optional
            Whether to apply zlib compression to data variables. Default is ``True``.
        complevel : int, optional
            Compression level passed to zlib (typically in the range 0-9).
            Ignored if ``compress=False``. Default is ``5``.

        Returns
        -------
        None

        Raises
        ------
        PermissionError
            If the file cannot be created or overwritten due to insufficient
            permissions or the path is not writable.
        OSError
            If there is a low-level I/O or HDF5/NetCDF issue while writing.
        ValueError
            If an invalid ``complevel`` is provided (e.g., out of the supported
            range for the underlying zlib implementation) or if array shapes are
            inconsistent with the declared dimensions.

        Notes
        -----
        **Data variables** (stored as float arrays):
        - ``C0_FT_r``: Real part of ``C0_FT`` with dims ``("ky", "kx", "grains")``  
        - ``C0_FT_i``: Imag part of ``C0_FT`` with dims ``("ky", "kx", "grains")``  
        - ``Cz_FT_r``: Real part of ``Cz_FT`` with dims ``("ky", "kx", "z", "grains")``  
        - ``Cz_FT_i``: Imag part of ``Cz_FT`` with dims ``("ky", "kx", "z", "grains")``

        **Coordinates**:
        - ``grains``: ``range(self.params.grains.bins)``
        - ``kx``: ``self.kx[: Nx // 2 + 1]`` (positive and zero wavenumbers)
        - ``ky``: ``self.ky``
        - ``z``: ``self.params.output.altitudes``

        **Global attributes** include solver/domain configuration, grain parameters
        (``diameter``, ``density``, ``proportion``), source parameters (location,
        plume, MER, duration), emission profile, physical constants, meteorology,
        model scaling, and output schedule. The current package ``version`` is also
        recorded.

        **Compression and encoding**:
        If ``compress=True``, all data variables are written with
        ``zlib=True`` and ``complevel=complevel``. If ``compress=False``,
        the dataset is written without compression.

        **File format and engine**:
        The file is written with ``format="NETCDF4"`` using the
        ``engine="h5netcdf"`` backend.

        Examples
        --------
        >>> result.to_netcdf("outputs/ashdisperse_run.nc")
        >>> result.to_netcdf("outputs/run_compressed.nc", compress=True, complevel=9)
        >>> result.to_netcdf("outputs/run_uncompressed.nc    >>> result.to_netcdf("outputs/run_uncompressed.nc", compress=False)
        """
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


    def _check_valid_grain(self, grain_i: int) -> None:
        """
        Utility to check if requested grain index is within result set

        Parameters
        ----------
        grain_i : int
            index for the grain class

        Raises
        ------
        IndexError
            If requested index, grain_i, is not in the grain class
        ValueError
            If grain_i is negative
        """
        if grain_i >= self.params.grains.bins:
            raise IndexError(
                "Requested grain class not found; given {}".format(grain_i)
                + " in results with"
                + " {} grain classes.".format(self.params.grains.bins)
            )
        if grain_i < 0:
            raise ValueError("Grain class index must be non-negative")
        return


    
    def get_groundconc_for_grain_class(
        self,
        grain_i: int,
        vmin: float = 1e-9,
        masked: bool = False,
        clipped: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Retrieve the ground-level concentration grid for a specific grain size class.

        This method extracts the concentration field for the given grain class at
        ground level, optionally masking low values and/or clipping the grid to the
        minimal bounding region containing valid data.

        Parameters
        ----------
        grain_i : int
            Zero-based index of the grain size class to extract.
        vmin : float, optional
            Minimum concentration threshold. Values below this threshold are treated
            as absent. Default is ``1e-9``.
        masked : bool, optional
            If ``True``, return a masked array where values below ``vmin`` are masked.
            Default is ``False``.
        clipped : bool, optional
            If ``True``, clip the returned grid to the smallest bounding box that
            contains all values above ``vmin``. If ``False``, return the full model
            grid for the grain class. Default is ``True``.

        Returns
        -------
        tuple of ndarray or None
            If valid data exists, returns a tuple ``(x, y, conc)``:
            - ``x`` : 1D ndarray of x-coordinates (model units)
            - ``y`` : 1D ndarray of y-coordinates (model units)
            - ``conc`` : 2D ndarray or MaskedArray of concentrations with shape
            ``(ny, nx)``
            If no valid data exists (e.g., all values below ``vmin`` or clipped
            region is empty), returns ``None``.

        Raises
        ------
        IndexError
            If ``grain_i`` is out of range for the available grain size classes.
        ValueError
            If ``grain_i`` is negative.

        Notes
        -----
        - Use ``masked=True`` to obtain a masked array suitable for plotting or
        further filtering.
        - When ``clipped=True``, the returned coordinates and grid are cropped to
        the minimal region containing concentrations above ``vmin``.
        - If all concentrations are below ``vmin``, the method returns ``None``.

        Examples
        --------
        >>> x, y, conc = result.get_groundconc_for_grain_class(0, vmin=1e-8, masked=True)
        """
        self._check_valid_grain(grain_i)

        x = self.x[:, grain_i]
        y = self.y[:, grain_i]

        conc = self.C0[:, :, grain_i]

        if clipped:
            conc, x, y = _clip_to_window(conc, x, y, vmin=vmin)

            # If the clipped window is empty, return None to indicate no data
            if conc.size == 0:
                return None

        # If all values are below vmin, treat as no-data for consistency
        if np.nanmax(conc) < vmin:
            return None

        if masked:
            conc = np.ma.masked_less(conc, vmin)

        return x, y, conc



    def get_settlingflux_for_grain_class(
        self,
        grain_i: int,
        vmin: float = 1e-7,
        masked: bool = False,
        clipped: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Retrieve the ground-level settling flux (mass flux to surface) for a specific grain size class.

        This method extracts the settling flux field for the given grain class at ground level,
        optionally masking low values and/or clipping the grid to the minimal bounding region
        containing valid flux data.

        Parameters
        ----------
        grain_i : int
            Zero-based index of the grain size class to extract.
        vmin : float, optional
            Minimum flux threshold (kg/m²/s). Values below this threshold are treated as absent.
            Default is ``1e-7``.
        masked : bool, optional
            If ``True``, return a masked array where values below ``vmin`` are masked.
            Default is ``False``.
        clipped : bool, optional
            If ``True``, clip the returned grid to the smallest bounding box that contains all
            values above ``vmin``. If ``False``, return the full model grid for the grain class.
            Default is ``True``.

        Returns
        -------
        tuple of ndarray or None
            If valid data exists, returns a tuple ``(x, y, flux)``:
            - ``x`` : 1D ndarray of x-coordinates (model units)
            - ``y`` : 1D ndarray of y-coordinates (model units)
            - ``flux`` : 2D ndarray or MaskedArray of settling flux values with shape ``(ny, nx)``
            If no valid data exists (e.g., all values below ``vmin`` or clipped region is empty),
            returns ``None``.

        Raises
        ------
        IndexError
            If ``grain_i`` is out of range for the available grain size classes.
        ValueError
            If ``grain_i`` is negative.

        Notes
        -----
        - Settling flux represents the mass flux to the ground surface for the specified grain class.
        - Use ``masked=True`` to obtain a masked array suitable for plotting or further filtering.
        - When ``clipped=True``, the returned coordinates and grid are cropped to the minimal region
        containing flux values above ``vmin``.
        - If all flux values are below ``vmin``, the method returns ``None``.

        Examples
        --------
        >>> x, y, flux = result.get_settlingflux_for_grain_class(1, vmin=1e-6, masked=True)
        """
        self._check_valid_grain(grain_i)

        x = self.x[:, grain_i]
        y = self.y[:, grain_i]

        flux = self.SettlingFlux[:, :, grain_i]

        if clipped:
            flux, x, y = _clip_to_window(flux, x, y, vmin=vmin)

        if flux.size == 0:
            return None

        if np.nanmax(flux) < vmin:
            return None

        if masked:
            flux = np.ma.masked_less(flux, vmin)

        return x, y, flux


    
    def get_ashload_for_grain_class(
        self,
        grain_i: int,
        vmin: float = 1e-5,
        masked: bool = False,
        clipped: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Retrieve the ground-level ash load grid for a specific grain size class.

        Ash load represents the total deposited mass per unit area (kg/m²) for the
        specified grain class, computed as:

        ``ashload = settling_flux * eruption_duration``

        Parameters
        ----------
        grain_i : int
            Zero-based index of the grain size class to extract.
        vmin : float, optional
            Minimum ash load threshold (kg/m²). Values below this threshold are treated
            as absent. Default is ``1e-5``.
        masked : bool, optional
            If ``True``, return a masked array where values below ``vmin`` are masked.
            Default is ``False``.
        clipped : bool, optional
            If ``True``, clip the returned grid to the smallest bounding box that contains
            all values above ``vmin``. If ``False``, return the full model grid for the
            grain class. Default is ``True``.

        Returns
        -------
        tuple of ndarray or None
            If valid data exists, returns a tuple ``(x, y, ashload)``:
            - ``x`` : 1D ndarray of x-coordinates (model units)
            - ``y`` : 1D ndarray of y-coordinates (model units)
            - ``ashload`` : 2D ndarray or MaskedArray of ash load values (kg/m²) with
            shape ``(ny, nx)``
            If no valid data exists (e.g., all values below ``vmin`` or clipped region
            is empty), returns ``None``.

        Raises
        ------
        IndexError
            If ``grain_i`` is out of range for the available grain size classes.
        ValueError
            If ``grain_i`` is negative.

        Notes
        -----
        - Ash load is derived from the precomputed settling flux multiplied by the
        eruption duration.
        - Use ``masked=True`` to obtain a masked array suitable for plotting or further
        filtering.
        - When ``clipped=True``, the returned coordinates and grid are cropped to the
        minimal region containing ash load values above ``vmin``.
        - If all ash load values are below ``vmin``, the method returns ``None``.

        See Also
        --------
        get_settlingflux_for_grain_class : Retrieve settling flux (kg/m²/s) for a grain class.
        raster_ashload_for_grain_class : Export ash load as an xarray.Dataset with geospatial metadata.

        Examples
        --------
        >>> x, y, ashload = result.get_ashload_for_grain_class(2, vmin=1e-4, masked=True)
        """
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
        self, grain_i: int, latlon: Point | PointList, vmin: float = 1e-3
    ) -> float | np.ndarray:
        self._check_valid_grain(grain_i)
        ds = self.raster_ashload_for_grain_class(
            grain_i, vmin=vmin, nodata=np.nan, masked=False, crs=None
        )

        return _sample_point_from_dataset(ds, ds.rio.crs, latlon=latlon, var="ash_load", fallback=0)

    def POI_ashload(self, latlon: Point | PointList, vmin: float = 1e-3) -> tuple[float, list[dict[str, float|np.ndarray]]]:

        loads = []
        total_load = 0
        for j in range(self.grain_classes):
            this_ashload = self.POI_ashload_for_grain_class(j, latlon, vmin)
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
        self, grain_i: int, latlon: Point | PointList, vmin: float = 1e-3
    ):
        self._check_valid_grain(grain_i)
        ds = self.raster_groundconc_for_grain_class(
            grain_i, vmin=vmin, nodata=np.nan, masked=False, crs=None
        )

        return _sample_point_from_dataset(ds, ds.rio.crs, latlon=latlon, var="ground_concentration", fallback=0)

    def POI_groundconc(self, latlon: Point | PointList, vmin: float = 1e-3):

        concs = []
        total_conc = 0
        for j in range(self.grain_classes):
            this_conc = self.POI_groundconc_for_grain_class(j, latlon, vmin)
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
        self, grain_i: int, latlon: Point | PointList, vmin: float = 1e-3
    ):
        self._check_valid_grain(grain_i)
        ds = self.raster_settlingflux_for_grain_class(
            grain_i, vmin=vmin, nodata=np.nan, masked=False, crs=None
        )

        return _sample_point_from_dataset(ds, ds.rio.crs, latlon=latlon, var="settling_flux", fallback=0)

    def POI_settlingflux(self, latlon: Point | PointList, vmin: float = 1e-3):

        fluxes = []
        total_flux = 0
        for j in range(self.grain_classes):
            this_flux = self.POI_settlingflux_for_grain_class(j, latlon, vmin)
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

    def max_ashload_for_grain_class(self, grain_i: int):
        self._check_valid_grain(grain_i)
        return np.nanmax(self.SettlingFlux[:, :, grain_i]) * self.params.source.duration

    def _array_to_dataset(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        arr: np.ndarray | None,
        units: str,
        short_name: Optional[str]=None,
        long_name: Optional[str]=None,
        nodata: Optional[float]=np.nan,
        crs: Optional[str]=None,
    ) -> xa.Dataset:
        """Helper: wrap x/y/array into an xarray.Dataset with rio metadata.

        Returns None if `arr` is None.
        """
        if arr is None:
            return None

        ds = xa.Dataset()
        ds.coords["x"] = x + self.utm[0]
        ds.coords["y"] = y + self.utm[1]

        ds[name] = (("y", "x"), arr)
        if short_name is not None:
            ds[name].attrs["short_name"] = short_name
        if long_name is not None:
            ds[name].attrs["long_name"] = long_name
        ds[name].attrs["units"] = units
        ds[name].rio.write_nodata(nodata, inplace=True)

        ds.x.attrs["units"] = "metres"
        ds.y.attrs["units"] = "metres"

        ds = ds.rio.write_crs(self.utmepsg)

        if crs is not None:
            ds = ds.rio.reproject(crs, resampling=Resampling.cubic)

        return ds


    def raster_groundconc_for_grain_class(
        self, grain_i: int, vmin: float=1e-9, nodata: float=np.nan, masked: bool=False, clipped: bool=True, crs: Optional[str]=None
    ) -> xa.Dataset:
        data = self.get_groundconc_for_grain_class(grain_i, vmin=vmin, masked=masked, clipped=clipped)
        if data is None:
            return None
        x, y, conc = data

        return self._array_to_dataset(
            name="ground_concentration",
            x=x,
            y=y,
            arr=conc,
            units="kg/m**3",
            short_name=f"ground_concentration_{grain_i}",
            long_name=(
                f"concentration at ground level for grain size {self.params.grains.diameter[grain_i]} m"
            ),
            nodata=nodata,
            crs=crs,
        )


    def raster_settlingflux_for_grain_class(
        self, grain_i: int, vmin: float=1e-6, nodata: float=np.nan, masked: bool=False, clipped: bool=True, crs: Optional[str]=None
    ) -> xa.Dataset:
        data = self.get_settlingflux_for_grain_class(grain_i, vmin=vmin, masked=masked, clipped=clipped)
        if data is None:
            return None
        x, y, flux = data

        return self._array_to_dataset(
            name="settling_flux",
            x=x,
            y=y,
            arr=flux,
            units="kg/m**2/s",
            short_name=f"settling_flux_{grain_i}",
            long_name=(
                f"settling flux at ground level for grain size {self.params.grains.diameter[grain_i]} m"
            ),
            nodata=nodata,
            crs=crs,
        )


    def raster_ashload_for_grain_class(
        self, grain_i: int, vmin: float=1e-2, nodata: float=np.nan, masked: bool=False, clipped: bool=True, crs: Optional[str]=None
    ) -> xa.Dataset:
        data = self.get_ashload_for_grain_class(
            grain_i, vmin=vmin, masked=masked, clipped=clipped,
        )
        if data is None:
            return None
        x, y, ashload = data

        return self._array_to_dataset(
            name="ash_load",
            x=x,
            y=y,
            arr=ashload,
            units="kg/m**2",
            short_name=f"ash_load_{grain_i}",
            long_name=(
                f"ash load at ground level for grain size {self.params.grains.diameter[grain_i]} m"
            ),
            nodata=nodata,
            crs=crs,
        )
    

    # def _raster_contour(self, raster, name, cntrs):

    #     data = raster.data
    #     x = raster.x.data
    #     y = raster.y.data
    #     crs = raster.rio.crs

    #     data_min = np.nanmin(data)
    #     data_max = np.nanmax(data)
    #     data_max = nice_round_up(data_max, mag=10 ** np.floor(np.log10(data_max)))

    #     fx = interp1d(np.arange(0, len(x)), x)
    #     fy = interp1d(np.arange(0, len(y)), y)

    #     cntrs = cntrs[cntrs > data_min]
    #     cntrs = cntrs[cntrs < data_max]

    #     for kk, this_cntr in enumerate(cntrs):

    #         C = find_contours(data, this_cntr)

    #         if len(C) == 0:
    #             pass
    #         for jj, p in enumerate(C):
    #             p[:, 0] = fy(p[:, 0])
    #             p[:, 1] = fx(p[:, 1])

    #             p[:, [0, 1]] = p[:, [1, 0]]

    #             if len(p[:, 1]) > 2:
    #                 thisPoly = Polygon(p).buffer(0)
    #                 if not thisPoly.is_empty:
    #                     if jj == 0:
    #                         geom = thisPoly
    #                         g_tmp = gpd.GeoDataFrame(
    #                             columns=["contour", "name", "geometry"], crs=crs
    #                         )
    #                         g_tmp.loc[0, "contour"] = this_cntr
    #                         g_tmp.loc[[0], "geometry"] = gpd.GeoSeries([geom])
    #                         g_tmp.loc[0, "name"] = name
    #                     else:
    #                         if g_tmp.loc[0, "geometry"].contains(thisPoly):
    #                             geom = g_tmp.loc[0, "geometry"].difference(thisPoly)
    #                         else:
    #                             try:
    #                                 geom = g_tmp.loc[0, "geometry"].union(thisPoly)
    #                             except:
    #                                 print(
    #                                     "Error processing polygon contour -- skipping.  Better check the result!"
    #                                 )
    #                         g_tmp.loc[[0], "geometry"] = gpd.GeoSeries([geom])

    #         if kk == 0:
    #             g = gpd.GeoDataFrame(g_tmp).set_geometry("geometry")
    #         else:
    #             g = gpd.GeoDataFrame(pd.concat([g, g_tmp], ignore_index=True))

    #     g["contour"] = g["contour"].astype("float64")
    #     g.crs = crs

    #     return g

    def write_gtiff(self, raster: np.ndarray, x: np.ndarray, y: np.ndarray, outname: str, 
                    nodata: float=-1, vmin: float=1e-6, resolution: Optional[float]=None) -> None:

        print("Writing data to {outname}".format(outname=outname))
        if os.path.isfile(outname):
            print(
                "WARNING: {outname} ".format(outname=outname)
                + "already exists and will be replaced"
            )

        utmcode = self.params.source.utmcode

        if resolution is not None:
            x, y, raster = _interpolate(
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
        self, grain_i: int, nodata: float=-1.0, vmin: float=1e-6, resolution: Optional[float]=None,
    ) -> None:
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
        self, grain_i: int, vmin: float=1e-3, resolution: Optional[float]=None, outname: Optional[str]=None, compress: str='LZW',
    ) -> None:
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
    

    def get_ashload(
        self,
        resolution: float=300.0,
        vmin: float=1e-3,
        nodata: float=-1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Delegate numeric aggregation to compute_total_ashload
        x_out, y_out, totalAshLoad = self.get_total_ashload(
            resolution=resolution, vmin=vmin, nodata=nodata
        )

        return x_out, y_out, totalAshLoad

    def get_total_ashload(self, 
                        resolution: float=300.0, 
                        vmin: float=1e-3, 
                        nodata: float=-1.0, 
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute aggregated total ash load raster from all grain-class settling fluxes.

        Parameters
        ----------
        resolution : float, optional
            Output grid resolution in model units (default: 300.0).
        vmin : float, optional
            Minimum value to keep (values below are masked/nodata, default: 1e-3).
        nodata : float, optional
            Value to use for nodata in output arrays (default: -1).

        Returns
        -------
        (x, y, totalAshLoad) : tuple of arrays,
            x, y are 1D arrays of coordinates; totalAshLoad is a masked 2D array.
                            
        """
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

        totalAshLoad = np.ma.masked_where(totalAshLoad < vmin, totalAshLoad)

        totalAshLoad, x_out, y_out = _clip_to_window(totalAshLoad, x_out, y_out, vmin=vmin)
        return x_out, y_out, totalAshLoad
    

    def raster_total_ashload(self,
                        resolution: float=300.0, 
                        vmin: float=1e-3, 
                        nodata: float=-1.0, 
                        crs: Optional[str]=None,
    ) -> xa.Dataset:
        """
        Compute aggregated total ash load raster from all grain-class settling fluxes

        Parameters
        ----------
        resolution : float, optional
            Output grid resolution in model units (default: 300.0).
        vmin : float, optional
            Minimum value to keep (values below are masked/nodata, default: 1e-3).
        nodata : float, optional
            Value to use for nodata in output arrays (default: -1).

        Returns
        -------
                                    
        """
        
        x, y, totalAshLoad = self.get_total_ashload(resolution=resolution, vmin=vmin, nodata=nodata)
    
        if totalAshLoad is None:
            return None
        
        ds = xa.Dataset()
        ds.coords["x"] = x + self.utm[0]
        ds.coords["y"] = y + self.utm[1]
        ds["ash_load"] = (("y", "x"), totalAshLoad)
        ds["ash_load"].attrs["short_name"] = "ash_load"
        ds["ash_load"].attrs["long_name"] = "total ash load at ground level for all grain classes"
        ds["ash_load"].attrs["units"] = "kg/m**2"
        ds["ash_load"].rio.write_nodata(nodata, inplace=True)
        ds.x.attrs["units"] = "metres"
        ds.y.attrs["units"] = "metres"
        ds = ds.rio.write_crs(self.utmepsg)
        if crs is not None:
            ds = ds.rio.reproject(crs, resampling=Resampling.cubic)
        return ds

    
    # def raster_ashload(self,
    #     resolution=300.0,
    #     vmin=1e-3,
    #     nodata=-1,
    #     export_gtiff=True,
    #     export_name="AshLoad.tif",
    #     crs=None):

    #     Ngrains = self.params.grains.bins

    #     if Ngrains==1:
    #         ds = self.raster_ashload_for_grain_class(0, vmin=vmin, nodata=0.0, clipped=True)
    #     else:
    #         # Find largest extent, will be smallest grain size
    #         # imax = np.argmax(self.params.model.xyScale)
    #         imax = np.argmax(np.maximum(self.params.model.xScale, self.params.model.yScale))

    #         # Initialize with this ash load
    #         ds = self.raster_ashload_for_grain_class(imax, vmin= vmin/Ngrains/10, nodata=0.0, clipped=False)

    #         ds = ds.rio.reproject(ds.rio.crs, resolution=resolution, resampling=Resampling.cubic, nodata=0.0)

    #         # Loop through remaining, reproject to match
    #         for igrain in range(Ngrains):
    #             # skip over the already used result
    #             if igrain==imax:
    #                 continue

    #             ds_tmp = self.raster_ashload_for_grain_class(igrain, vmin= vmin/Ngrains/100, nodata=0.0, clipped=False)

    #             ds_tmp = ds_tmp.rio.reproject_match(ds, nodata=0.0, resampling=Resampling.cubic)

    #             ds += ds_tmp

    #     ds = ds.where(ds.ash_load >= vmin, nodata)

    #     ds["ash_load"].attrs["short_name"] = "ash_load"
    #     ds["ash_load"].attrs[
    #         "long_name"
    #     ] = "total ash load at ground level for all grain classes"
    #     ds["ash_load"].attrs["units"] = "kg/m**2"
    #     ds["ash_load"].rio.write_nodata(nodata, inplace=True)

    #     ds.x.attrs["units"] = "metres"
    #     ds.y.attrs["units"] = "metres"

    #     if crs is not None:
    #         ds = ds.rio.reproject(crs, resampling=Resampling.cubic)

    #     return ds

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
        ds=None,
    ):
        """
        Plot total ash load as filled contours.

        Parameters
        ----------
        resolution : float, optional
            Output grid resolution (default: 300.0). Ignored if ds is provided.
        logscale : bool, optional
            Use logarithmic color scale (default: True).
        cmap : matplotlib colormap, optional
            Colormap to use (default: viridis).
        vmin : float, optional
            Minimum value to plot (default: 1e-3).
        nodata : float, optional
            Value to use for nodata (default: -1).
        alpha : float, optional
            Alpha blending for filled contours (default: 0.5).
        basemap : bool, optional
            If True, add a basemap (default: False).
        export_gtiff : bool, optional
            If True, write a GeoTIFF (default: False).
        export_name : str, optional
            Filename for GeoTIFF (default: 'AshLoad.tif').
        show : bool, optional
            If True, show the plot (default: True).
        ds : xarray.Dataset, optional
            If provided, use this dataset (must have 'ash_load'). Otherwise, compute from model.

        Returns
        -------
        fig, ax, cbar : tuple
            Matplotlib Figure, Axes, and Colorbar objects.

        Notes
        -----
        If ds is not provided, the function computes the total ash load using current model parameters.
        """
        if ds is None:
            ds = self.compute_total_ashload(resolution=resolution, vmin=vmin/10, nodata=nodata, to_dataset=True)
            if export_gtiff:
                arr = ds["ash_load"].values
                x = ds.x.values
                y = ds.y.values
                self.write_gtiff(arr, x, y, export_name, nodata=nodata, vmin=vmin, resolution=None)
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

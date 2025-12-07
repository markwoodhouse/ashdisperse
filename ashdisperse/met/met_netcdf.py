import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil, floor
from pathlib import Path
from typing import Literal, Optional

import cdsapi
import metpy.calc as metcalc
import netCDF4
import numpy as np
import requests
import xarray as xr
from netCDF4 import Dataset
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

from .met_adapters import get_adapter

# -------------------------------
# Small utility dataclass
# -------------------------------

@dataclass
class MetProfile:
    altitude: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    wind_speed: np.ndarray
    wind_dir: np.ndarray
    density: np.ndarray
    relhum: np.ndarray
    wind_U: np.ndarray
    wind_V: np.ndarray


# -------------------------------
# Dataset and extraction utilities
# -------------------------------

class MetDataset:
    """
    Wrapper around xarray for CF/MetPy parsing and interpolation.

    This class loads a meteorological dataset (NetCDF), applies CF conventions parsing via MetPy,
    optionally applies an adapter for variable relabeling/enrichment, ensures consistent coordinate naming,
    and derives geometric height from geopotential if available.

    Attributes:
        path (Path): Path to the dataset file.
        type (Optional[str]): Optional dataset type for adapter selection.
        _ds (Optional[xr.Dataset]): Cached xarray Dataset after processing.
    """

    def __init__(self, file: str, type: Optional[str] = None):
        """
        Initialize the MetDataset wrapper.

        Args:
            file (str): Path to the NetCDF file.
            type (Optional[str]): Optional type hint for adapter selection.
        """
        self.path = Path(file)
        self.type = type
        self._ds: Optional[xr.Dataset] = None

    @property
    def ds(self) -> xr.Dataset:
        """
        Load and process the dataset if not already cached.

        Steps:
            1. Load dataset using xarray with NetCDF4 engine.
            2. Parse CF conventions using MetPy.
            3. Apply adapter for relabeling/enrichment if type is provided.
            4. Ensure consistent vertical coordinate naming (`pressure_level`).
            5. Derive geometric height from geopotential if available.

        Returns:
            xr.Dataset: Processed dataset with CF parsing and optional enhancements.
        """
        if self._ds is None:
            dat = xr.load_dataset(self.path, engine="netcdf4")
            dat = dat.metpy.parse_cf()

            # --- Apply adapter based on explicit type ---
            try:
                adapter = get_adapter(dat, kind=self.type)
                dat = adapter.relabel(dat)
                dat = adapter.enrich(dat)
                print(f"🧩 Applied adapter: {adapter.__name__} (type={self.type})")
            except ValueError:
                print(f"⚠️ No adapter matched this dataset (type={self.type}) — using raw variable names.")


            # --- Ensure consistent vertical coordinate naming ---
            if "pressure_level" not in dat.coords:
                for alt_name in ["level", "isobaric", "isobaricInhPa"]:
                    if alt_name in dat.coords:
                        dat = dat.rename({alt_name: "pressure_level"})
                        break

            # --- Derive geometric height while preserving original geopotential ---
            geo_var = None
            for candidate in ["z", "geopotential", "Geopotential_height_isobaric"]:
                if candidate in dat.variables:
                    geo_var = candidate
                    break

            if geo_var is not None:
                try:
                    # Quantify and convert using MetPy
                    geopot = dat[geo_var]
                    if hasattr(geopot, "metpy"):
                        geopot_q = geopot.metpy.quantify()
                    else:
                        geopot_q = geopot * metcalc.units("m^2/s^2")

                    height = metcalc.geopotential_to_height(geopot_q)
                    dat["height"] = height.metpy.convert_units("m")
                    dat["height"].attrs.update({
                        "long_name": "Geometric height from geopotential",
                        "units": "m",
                        "source_variable": geo_var,
                    })
                    print(f"📏 Derived geometric height from {geo_var}")
                except Exception as e:
                    print(f"⚠️ Failed to derive geometric height from {geo_var}: {e}")
            else:
                print("⚠️ No geopotential variable found to derive geometric height")

            self._ds = dat
        return self._ds

    @property
    def latitude(self) -> np.ndarray:
        """Return latitude values as a NumPy array."""
        return self.ds.latitude.values

    @property
    def longitude(self) -> np.ndarray:
        """Return longitude values as a NumPy array."""
        return self.ds.longitude.values

    @property
    def time(self) -> np.ndarray:
        """
        Return time coordinate values.

        Checks for 'valid_time' or 'time' in dataset coordinates.

        Raises:
            KeyError: If no valid time coordinate is found.
        """
        for name in ("valid_time", "time"):
            if name in self.ds.coords:
                return self.ds[name].values
        raise KeyError("No valid time coordinate found.")

    def extent(self) -> list[float]:
        """
        Compute spatial extent of the dataset.

        Returns:
            list[float]: [min_lon, max_lon, min_lat, max_lat]
        """
        return [
            self.longitude.min(),
            self.longitude.max(),
            self.latitude.min(),
            self.latitude.max(),
        ]

    def get_point(self, lat: float, lon: float, time: datetime) -> xr.Dataset:
        """
        Interpolate dataset to a specific latitude, longitude, and time.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            time (datetime): Target time for interpolation.

        Returns:
            xr.Dataset: Interpolated dataset at the given point.
        """
        ds = self.ds

        # Ensure ascending latitude for interpolation
        if ds.latitude.values[0] > ds.latitude.values[-1]:
            ds = ds.reindex(latitude=list(reversed(ds.latitude)))

        # Clamp coordinates to valid ranges
        lat = np.clip(lat, ds.latitude.min().item(), ds.latitude.max().item())
        lon = np.clip(lon, ds.longitude.min().item(), ds.longitude.max().item())

        coord = "valid_time" if "valid_time" in ds.coords else "time"
        nearest_time = ds[coord].sel({coord: np.datetime64(time)}, method="nearest")

        return (
            ds.sel({coord: nearest_time})
            .interp(latitude=lat, longitude=lon, method="linear")
            .metpy.quantify()
        )



# -------------------------------
# Scientific extraction
# -------------------------------

class MetProfileExtractor:
    """
    Computes derived physical quantities from a meteorological dataset.

    This class extracts a vertical profile of atmospheric variables (temperature, wind, pressure, etc.)
    at a given latitude, longitude, and time, with optional temporal interpolation and derived quantities
    such as wind speed, wind direction, mixing ratio, and air density.

    Attributes:
        dataset (xr.Dataset): The meteorological dataset (CF-compliant, parsed with MetPy).
    """

    def __init__(self, dataset: xr.Dataset):
        """
        Initialize the profile extractor.

        Args:
            dataset (xr.Dataset): The meteorological dataset to extract profiles from.
        """
        self.dataset = dataset

    def extract(self, 
                lat: float, 
                lon: float, 
                time: datetime,
                convention: Literal["to", "from"] = "to",
                interp_time: Literal["nearest", "linear", "weighted"] = "nearest",
                weight: Optional[float] = None,
    ) -> MetProfile:
        """
        Extract a vertical profile at a given location and time, computing derived quantities.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            time (datetime): Target time for extraction.
            convention (Literal["to", "from"], optional): Wind direction convention.
                - "to": Direction wind is blowing toward.
                - "from": Direction wind is coming from.
                Default is "to".
            interp_time (Literal["nearest", "linear", "weighted"], optional): Temporal interpolation scheme.
                - "nearest": Use the closest time step.
                - "linear": Interpolate linearly between time steps.
                - "weighted": Weighted average between nearest two times.
                Default is "nearest".
            weight (Optional[float], optional): Weighting factor for "weighted" interpolation (0.0–1.0).
                Default is 0.5 if not provided.

        Returns:
            MetProfile: A profile object containing altitude, temperature, pressure, wind speed/direction,
                        density, relative humidity, and wind components.

        Raises:
            ValueError: If requested lat/lon or time is outside dataset bounds.
            KeyError: If required variables (height, geopotential, pressure) are missing.
        """
    
        data = self.dataset

        lon = lon%360

        # --- Validate spatial domain ---
        lat_min, lat_max = data.latitude.min().item(), data.latitude.max().item()
        lon_min, lon_max = data.longitude.min().item(), data.longitude.max().item()

        if not (lat_min <= lat <= lat_max) or not (lon_min <= lon <= lon_max):
            raise ValueError(f"Requested (lat, lon) {lat:.2f}, {lon:.2f} outside dataset range.")

        # --- Determine time coordinate ---
        time_coord = "valid_time" if "valid_time" in data.coords else "time"
        times = data[time_coord].values
        t_min, t_max = times.min(), times.max()

        if not (t_min <= np.datetime64(time) <= t_max):
            raise ValueError(f"Requested time {time} outside dataset range "
                            f"({np.datetime_as_string(t_min)}–{np.datetime_as_string(t_max)})")

        # --- Temporal interpolation ---
        if interp_time == "nearest":
            nearest_time = data[time_coord].sel({time_coord: np.datetime64(time)}, method="nearest")
            data_t = data.sel({time_coord: nearest_time})

        elif interp_time == "linear":
            data_t = data.interp({time_coord: np.datetime64(time)}, method="linear")

        elif interp_time == "weighted":
            # Custom weighted average between nearest two times
            before = data.sel({time_coord: np.datetime64(time)}, method="ffill")
            after = data.sel({time_coord: np.datetime64(time)}, method="bfill")
            if weight is None:
                weight = 0.5
            data_t = (1 - weight) * before + weight * after

        else:
            raise ValueError(f"Unknown interpolation scheme: {interp_time}")

        # --- Spatial interpolation ---
        data_interp = data_t.interp(latitude=lat, longitude=lon, method="linear").metpy.quantify()

        # --- Extract height or derive from geopotential ---
        if "height" in data_interp:
            Z = data_interp["height"].metpy.convert_units("m")
        elif "geopotential" in data_interp:
            Z = metcalc.geopotential_to_height(data_interp["geopotential"]).metpy.convert_units("m")
        elif "z" in data:
            Z = metcalc.geopotential_to_height(data_interp["z"]).metpy.convert_units("m")
        else:
            raise KeyError("No height or geopotential variable found in dataset")
        
        Z = Z.values * Z.metpy.units
        Z = Z.to("m")

        # --- Extract wind components ---
        U = data_interp["wind_u"]
        U = U.values * U.metpy.units
        U = U.to("m/s")

        V = data_interp["wind_v"]
        V = V.values * V.metpy.units
        V = V.to("m/s")

        # --- Extract temperature ---
        T = data_interp["temperature"]
        T = T.values * T.metpy.units
        T = T.to("K")

        # --- Extract relative humidity ---
        RH = data_interp["rel_humidity"]
        RH = RH.values * RH.metpy.units
        RH = RH.to("%")

        # --- Extract pressure ---
        if "pressure_level" in data_interp:
            P = data_interp["pressure_level"]
        elif "level" in data:
            P = data_interp["level"]
        else:
            raise KeyError("No pressure coordinate found in dataset")
        P = P.values * P.metpy.units
        P = P.to("Pa")
        
        # --- Compute derived quantities ---
        spd = metcalc.wind_speed(U, V)
        dirn = metcalc.wind_direction(U, V, convention=convention)
        mixr = metcalc.mixing_ratio_from_relative_humidity(P, T, RH, phase='auto')
        mixr[np.isnan(mixr)] = 0.0  # handle NaNs in mixing ratio
        rho = metcalc.density(P, T, mixr)

        # --- Return profile object ---
        return MetProfile(
            altitude=Z.magnitude.astype(np.float64),
            temperature=T.magnitude.astype(np.float64),
            pressure=P.magnitude.astype(np.float64),
            wind_speed=spd.magnitude.astype(np.float64),
            wind_dir=dirn.magnitude.astype(np.float64),
            density=rho.magnitude.astype(np.float64),
            relhum=RH.magnitude.astype(np.float64),
            wind_U=U.magnitude.astype(np.float64),
            wind_V=V.magnitude.astype(np.float64),
        )


# -------------------------------
# Base class coordinating workflow
# -------------------------------

class NetcdfMet:
    """
    Coordinates downloading and extraction logic for meteorological datasets.

    This class provides:
        - Downloading data for a single time or a time range.
        - Loading the dataset into a MetDataset wrapper.
        - Extracting a vertical profile at a given location and time.

    Attributes:
        file (Path): Path to the NetCDF file.
        dataset (Optional[MetDataset]): Loaded dataset wrapper.
        profile (Optional[MetProfile]): Last extracted profile.
        _kind (Optional[str]): Dataset type identifier (set by subclasses).
    """

    _kind: Optional[str] = None  # to be set by subclasses

    def __init__(self, file: str):
        """
        Initialize the NetcdfMet object.

        Args:
            file (str): Path to the NetCDF file.
        """
        self.file = Path(file)
        self.dataset: Optional[MetDataset] = None
        self.profile: Optional[MetProfile] = None

    # -------- Download --------
    def download(
        self,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        extent: Optional[list[float]] = None,
        padding: float = 0.25,
        datetime: Optional[datetime] = None,
        datetime_start: Optional[datetime] = None,
        datetime_end: Optional[datetime] = None,
        timestep_hours: int = 1,
        quiet: bool = False,
        **kwargs,
    ) -> None:
        """
        Download dataset for a single datetime or a range of datetimes.

        Args:
            lat (Optional[float]): Latitude for spatial subset.
            lon (Optional[float]): Longitude for spatial subset.
            extent (Optional[list[float]]): Spatial extent [lon_min, lon_max, lat_min, lat_max].
            padding (float): Padding (in degrees) around lat/lon if extent is not provided.
            datetime (Optional[datetime]): Single datetime for download.
            datetime_start (Optional[datetime]): Start datetime for range download.
            datetime_end (Optional[datetime]): End datetime for range download.
            timestep_hours (int): Time step in hours for multiple timesteps.
            quiet (bool): Suppress verbose output.
            **kwargs: Additional arguments passed to subclass-specific download logic.

        Raises:
            ValueError: If neither `datetime` nor (`datetime_start` and `datetime_end`) are provided.
        """
        self._check_file_overwrite()

        # --- Build time list ---
        if datetime is not None:
            times = [datetime]
        elif datetime_start and datetime_end:
            dt = timedelta(hours=timestep_hours)
            times = []
            t = datetime_start
            while t <= datetime_end:
                times.append(t)
                t += dt
        else:
            raise ValueError("Must specify either `datetime` or both `datetime_start` and `datetime_end`.")

        # Construct extent if lat/lon is provided and extent is None
        if extent is None and lat is not None and lon is not None:
            extent = [
                lon%360 - padding, lon%360 + padding,
                lat - padding, lat + padding,
            ]

        # Pass times to subclass download
        self._download_custom(lat=lat, lon=lon, extent=extent, datetimes=times, quiet=quiet, **kwargs)

        # Set dataset with type from subclass
        self.dataset = MetDataset(str(self.file), type=self._kind)

    
    # -------- Load --------
    def load(self):
        """
        Load the dataset from file into a MetDataset wrapper.
        """
        self.dataset = MetDataset(str(self.file), type=self._kind)


    def _download_custom(self, **kwargs) -> None:
        """
        Subclass-specific download logic.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _check_file_overwrite(self):
        """
        Warn if the target file already exists and will be overwritten.
        """
        if self.file.exists():
            print(f"⚠️ File {self.file} exists and will be overwritten.")

    # -------- Extract --------
    def extract(
        self,
        lat: float,
        lon: float,
        time: datetime,
        convention: Literal["to", "from"] = "to",
        interp_time: Literal["nearest", "linear", "weighted"] = "nearest",
        weight: Optional[float] = None,
    ) -> MetProfile:
        """
        Extract a vertical profile at a given location and time.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            time (datetime): Target time for extraction.
            convention (Literal["to", "from"], optional): Wind direction convention.
            interp_time (Literal["nearest", "linear", "weighted"], optional): Temporal interpolation scheme.
            weight (Optional[float], optional): Weighting factor for "weighted" interpolation.

        Returns:
            MetProfile: Extracted and processed profile.

        Raises:
            KeyError: If required variables are missing in the dataset.
        """
        if self.dataset is None:
            self.dataset = MetDataset(str(self.file), type=self._kind)

        ds = self.dataset.ds

        # --- Compute profile ---
        extractor = MetProfileExtractor(ds)
        profile = extractor.extract(lat, lon, time, convention, interp_time=interp_time, weight=weight)

        # Squeeze singleton dimensions
        self.profile = MetProfile(
            altitude=np.squeeze(profile.altitude),
            temperature=np.squeeze(profile.temperature),
            pressure=np.squeeze(profile.pressure),
            wind_speed=np.squeeze(profile.wind_speed),
            wind_dir=np.squeeze(profile.wind_dir),
            density=np.squeeze(profile.density),
            relhum=np.squeeze(profile.relhum),
            wind_U=np.squeeze(profile.wind_U),
            wind_V=np.squeeze(profile.wind_V),
        )
        return self.profile



# -------------------------------
# ERA5 downloader subclass
# -------------------------------

class ERA5(NetcdfMet):
    """
    Subclass of NetcdfMet for downloading ERA5 reanalysis data.

    This class implements the custom download logic for ERA5 pressure-level data
    using the Copernicus Climate Data Store (CDS) API.

    Attributes:
        _kind (str): Dataset type identifier ("ERA5").
    """
    
    _kind = "ERA5"

    def _download_custom(
        self,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        extent: Optional[list[float]] = None,
        datetimes: Optional[list[datetime]] = None,
        quiet: bool = False,
        **kwargs,
    ) -> None:
        """
        Download ERA5 pressure-level data for one or more datetimes.

        Args:
            lat (Optional[float]): Latitude for spatial subset (unused if extent provided).
            lon (Optional[float]): Longitude for spatial subset (unused if extent provided).
            extent (Optional[list[float]]): Spatial extent [west, east, south, north].
                Required for ERA5 downloads.
            datetimes (Optional[list[datetime]]): List of datetime objects for requested timesteps.
            quiet (bool): Suppress verbose output.
            **kwargs: Additional arguments for customization (currently unused).

        Raises:
            ValueError: If `datetimes` is empty or `extent` is missing.
            RuntimeError: If the downloaded file is empty or missing.
        """

        if datetimes is None or len(datetimes) == 0:
            raise ValueError("`datetimes` must be a non-empty list of datetime objects.")

        # --- Prepare time components for CDS API ---
        date_strs = sorted({dt.strftime("%Y-%m-%d") for dt in datetimes})
        hours = sorted({f"{dt.hour:02d}:00" for dt in datetimes})

        # --- Define bounding box ---
        if extent is None:
            raise ValueError("ERA5 requires `extent` (west, east, south, north).")

        north, west, south, east = extent[3], extent[0], extent[2], extent[1]

        if not quiet:
            print(f"⬇️  Starting ERA5 download for {len(datetimes)} timesteps")

        # --- Prepare CDS API request ---
        c = cdsapi.Client()
        request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "temperature",
                "geopotential",
                "relative_humidity",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": [
                # "1000", "925", "850", "700", "500", "300", "200", "100",
                1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 
                700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 
                200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
            ],
            "date": date_strs,
            "time": hours,
            "area": [north, west, south, east],  # N,W,S,E
        }

        # --- Download ---
        tmpfile = str(self.file)
        c.retrieve("reanalysis-era5-pressure-levels", request, tmpfile)

        # --- Validate download ---
        if not os.path.exists(tmpfile) or os.path.getsize(tmpfile) == 0:
            raise RuntimeError("❌ ERA5 file is empty or missing after download.")

        if not quiet:
            print(f"✅ ERA5 data saved to {self.file}")


# -------------------------------
# GFS forecast downloader subclass
# -------------------------------
class GFSForecast(NetcdfMet):
    """
    Subclass of NetcdfMet for downloading GFS forecast data via THREDDS NCSS.

    This class implements custom logic to:
        - Query available forecast times from the THREDDS metadata.
        - Download GFS subsets for specified spatial and temporal ranges.
    """

    _kind = "GFSForecast"
    
    @staticmethod
    def _get_available_times(dataset) -> list[datetime]:
        """
        Retrieve available forecast times from a THREDDS dataset.

        Args:
            dataset: THREDDS dataset object with NCSS metadata.

        Returns:
            list[datetime]: List of available forecast times.

        Raises:
            KeyError: If required metadata attributes are missing.
        """

        ncss = dataset.subset()

        # Reference time (udunits)
        udunits = ncss.metadata.axes['time']['attributes'][3]['udunits']  # "Hour since 2025-08-28T12:00:00Z"
        ref_time_str = udunits.split("since")[1].strip()
        ref_time = pd.to_datetime(ref_time_str)

        # Forecast hour offsets (from attribute 5)
        offsets = [float(v) for v in ncss.metadata.axes['time']['attributes'][5]['values']]

        # Available forecast times
        available_times = ref_time + pd.to_timedelta(offsets, unit='h')

        return available_times
    

    def _download_custom(
        self,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        extent: Optional[list[float]] = None,
        datetimes: Optional[list[datetime]] = None,
        padding: float = 0.25,
        quiet: bool = False,
    ) -> None:
        """
        Download GFS forecast data via THREDDS NCSS.

        Args:
            lat (Optional[float]): Latitude for spatial subset.
            lon (Optional[float]): Longitude for spatial subset.
            extent (Optional[list[float]]): Spatial extent [west, east, south, north].
            datetimes (Optional[list[datetime]]): List of datetime objects for requested timesteps.
            padding (float): Padding (in degrees) around lat/lon if extent is not provided.
            quiet (bool): Suppress verbose output.

        Raises:
            RuntimeError: If data retrieval or file saving fails.
        """
        if not quiet:
            print("⬇️  Starting GFS download via THREDDS NCSS...")

        # --- Resolve extent ---
        if extent is None and lat is not None and lon is not None:
            extent = [
                lon%360 - padding, lon%360 + padding,
                lat - padding, lat + padding,
            ]

        # --- Connect to THREDDS server ---
        cat_url = "http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best"
        if not quiet:
            print(f"🌐 Connecting to THREDDS GFS server: {cat_url}")

        catalog = TDSCatalog(cat_url)
        dataset = list(catalog.datasets.values())[0]
        if not quiet:
            print(f"📦 Using dataset: {dataset.name}")

        # --- Build NCSS query ---
        ncss = dataset.subset()
        query = ncss.query()
        query.lonlat_box(east=extent[1], west=extent[0], south=extent[2], north=extent[3])
        query.time_range(start=datetimes[0], end=datetimes[-1])
        query.accept("netcdf4")
        query.variables(
            "Temperature_isobaric",
            "u-component_of_wind_isobaric",
            "v-component_of_wind_isobaric",
            "Relative_humidity_isobaric",
            "Geopotential_height_isobaric",
        )

        if not quiet:
            print(f"📡 Requesting GFS subset for {datetimes[0]}--{datetimes[-1]} over extent {extent} ...")
        
        # --- Retrieve data ---
        try:
            ncss_data = ncss.get_data(query)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to retrieve GFS data: {e}")

        data = xr.open_dataset(xr.backends.NetCDF4DataStore(ncss_data))

        # --- Save to file ---
        if not quiet:
            print(f"💾 Writing netCDF4 Dataset to file {self.file}")
        if os.path.exists(self.file):
            if not quiet:
                print(f"⚠️ Warning: File '{self.file}' already exists and will be overwritten.")
            os.remove(self.file)

        data.to_netcdf(self.file)

        # --- Validate file ---
        if not os.path.exists(self.file) or os.path.getsize(self.file) == 0:
            raise RuntimeError("❌ Saved GFS file is empty or missing.")

        if not quiet:
            print(f"✅ GFS subset saved to {self.file} ({os.path.getsize(self.file)/1024:.1f} KB)")

        data.close()



def round_datetime(dt: datetime, resolution_hours: int, rounding: str = "nearest") -> datetime:
    """
    Round a datetime object to a specified interval in hours, with control over rounding direction.

    Args:
        dt (datetime): The original datetime to round.
        resolution_hours (int): The rounding resolution in hours (e.g., 3 for 3-hour intervals).
        rounding (str): Rounding mode. Options:
            - "nearest": Round to the nearest interval.
            - "floor": Round down to the previous interval.
            - "ceil": Round up to the next interval.
            Default is "nearest".

    Returns:
        datetime: A new datetime object rounded according to the specified resolution and mode.

    Raises:
        ValueError: If `rounding` is not one of "nearest", "floor", or "ceil".

    Example:
        >>> from datetime import datetime
        >>> dt = datetime(2025, 11, 19, 16, 44, 6)
        >>> round_datetime(dt, 3, "nearest")
        datetime(2025, 11, 19, 15, 0, 0)
    """
    resolution = resolution_hours * 3600  # Convert hours to seconds
    timestamp = dt.timestamp()
    ratio = timestamp / resolution

    if rounding == "nearest":
        rounded_ratio = round(ratio)
    elif rounding == "floor":
        rounded_ratio = floor(ratio)
    elif rounding == "ceil":
        rounded_ratio = ceil(ratio)
    else:
        raise ValueError("Invalid rounding mode. Choose 'nearest', 'floor', or 'ceil'.")

    rounded_timestamp = rounded_ratio * resolution
    return datetime.fromtimestamp(rounded_timestamp)

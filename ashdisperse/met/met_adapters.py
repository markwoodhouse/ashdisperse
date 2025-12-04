# =========================================
#  met_adapters.py
# =========================================
import warnings
from typing import Optional

import metpy.calc as metcalc
import numpy as np
import xarray as xr


# ----------------------------------------
# Base adapter
# ----------------------------------------
class BaseAdapter:
    """Base class for all dataset adapters."""
    _kind = "base"

    @staticmethod
    def detect(ds: xr.Dataset) -> bool:
        raise NotImplementedError

    @staticmethod
    def relabel(ds: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    @staticmethod
    def enrich(ds: xr.Dataset) -> xr.Dataset:
        return ds


# ----------------------------------------
# ERA5 Adapter
# ----------------------------------------
class ERA5_Adapter(BaseAdapter):
    _kind = "ERA5"

    @staticmethod
    def detect(ds: xr.Dataset) -> bool:
        return "geopotential" in ds.data_vars and "pressure_level" in ds.coords

    @staticmethod
    def relabel(ds: xr.Dataset) -> xr.Dataset:
        rename_map = {
            "geopotential": "geopotential",
            "t": "temperature",
            "u": "wind_u",
            "v": "wind_v",
            "r": "rel_humidity",
            "level": "pressure_level",
        }
        renamed = ds.rename({k: v for k, v in rename_map.items() if k in ds})
        for old, new in rename_map.items():
            if old in ds.data_vars:
                renamed[new].attrs.update(ds[old].attrs)
        return renamed

    @staticmethod
    def enrich(ds: xr.Dataset) -> xr.Dataset:
        if "geopotential" in ds and "geopotential_height" not in ds:
            units = ds["geopotential"].attrs.get("units", "").lower()
            if "m2" in units or "m**2" in units:
                print("🧮 Converting ERA5 geopotential to geometric height...")
                z = metcalc.geopotential_to_height(ds["geopotential"])
                z.attrs.update({"units": "m", "long_name": "geopotential_height"})
                ds["geopotential_height"] = z
        # if "density" not in ds:
        #     ds['mix_ratio'] = ds.apply(lambda row: metcalc.mixing_ratio_from_relative_humidity(row['pressure_level'], row['temperature'], row['rel_humidity']), axis=1)
        #     ds['density'] = ds.apply(lambda row: metcalc.density(row['pressure_level'], row['t'], row['mix_ratio']), axis=1)
        return ds


# ----------------------------------------
# GFS Forecast Adapter
# ----------------------------------------
class GFS_Forecast_Adapter(BaseAdapter):
    _kind = "GFSForecast"

    @staticmethod
    def detect(ds: xr.Dataset) -> bool:
        return any("isobaric" in str(k).lower() for k in list(ds.coords) + list(ds.data_vars))

    @staticmethod
    def relabel(ds: xr.Dataset) -> xr.Dataset:
        # --- Rename core variables ---
        rename_map = {
            "Temperature_isobaric": "temperature",
            "u-component_of_wind_isobaric": "wind_u",
            "v-component_of_wind_isobaric": "wind_v",
            "Relative_humidity_isobaric": "rel_humidity",
            "Geopotential_height_isobaric": "geopotential_height",
        }

        # --- Normalize coordinate names ---
        coord_map = {}
        if "time" not in ds.coords:
            # rename time if it is not in the coords
            if "validtime2" in ds.coords:
                coord_map["validtime2"] = "time"
            elif "valid_time" in ds.coords:
                coord_map["valid_time"] = "time"
            elif "reftime" in ds.coords:
                # fallback if no forecast-valid time exists
                coord_map["reftime"] = "time"

        ds = ds.rename({**rename_map, **coord_map})

        # Remove redundant time-like variable if present
        if "validtime2Forecast" in ds:
            ds = ds.drop_vars("validtime2Forecast")

        # Preserve attributes from renamed vars
        for old, new in rename_map.items():
            if old in ds.data_vars:
                ds[new].attrs.update(ds[old].attrs)

        return ds

    @staticmethod
    def enrich(ds: xr.Dataset) -> xr.Dataset:
        if "geopotential_height" in ds and "geopotential" not in ds:
            g0 = 9.80665
            print("🧮 Deriving geopotential from height (GFS)...")
            ds["geopotential"] = ds["geopotential_height"] * g0
            ds["geopotential"].attrs.update({
                "units": "m^2 s^-2",
                "long_name": "derived geopotential"
            })
        return ds


# ----------------------------------------
# Adapter registry
# ----------------------------------------
import warnings


def get_adapter(dataset: xr.Dataset, kind: Optional[str] = None) -> type:
    """Return an adapter class for a given dataset type or structure."""

    # Build dynamic registry of available adapters
    ADAPTERS = {cls._kind: cls for cls in BaseAdapter.__subclasses__()}

    # ---- Case 1: Explicit kind provided ----
    if kind is not None:
        if kind in ADAPTERS:
            return ADAPTERS[kind]
        else:
            warnings.warn(
                f"No adapter registered for dataset type '{kind}'. "
                "Attempting automatic detection instead.",
                UserWarning,
            )

    # ---- Case 2: Automatic detection fallback ----
    for adapter_cls in ADAPTERS.values():
        try:
            if adapter_cls.detect(dataset):
                print(f"🔍 Automatically detected adapter: {adapter_cls.__name__}")
                return adapter_cls
        except Exception as e:
            warnings.warn(f"Adapter {adapter_cls.__name__} failed detection: {e}")

    # ---- Case 3: No match ----
    raise ValueError("❌ No suitable adapter found for this dataset.")




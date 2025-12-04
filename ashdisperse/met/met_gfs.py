import os
from typing import Optional

import metpy.calc as metcalc
import numpy as np
import pandas as pd
import requests
import xarray as xr


class GFS:
    def __init__(self, cycle_datetime: str, forecast_hr: str, lat: float, lon: float):

        self.lat = lat
        self.lon = lon % 360

        self.forecast_hr = forecast_hr
        self.cycle_datetime = pd.to_datetime(cycle_datetime)


    def _setup(self) -> None:
        grib_exists = self._check_grib()
        if not grib_exists:
            raise ValueError(f"No GFS data for cycle {self.cycle_datetime:%Y%m%d %H:%M} forecast hour {self.forecast_hr}")

        idx_exists = self._check_idx()

        if grib_exists and idx_exists:
            self.idx = self._get_idx_as_dataframe()

            self.levels = self.idx.loc[
                (self.idx["level"].str.match(r"(\d+(?:\.\d+)?) mb"))
            ].level.unique()

    @property
    def url(self) -> str:
        return ''
    
    @property
    def idx_url(self) -> str:
        return f"{self.url}.idx"

    def download(self, outFile: str="./gfs_data.nc") -> None:

        gfs = TDSCatalog(self.url)
        gfs_ds = gfs.datasets[0]
        ncss = gfs_ds.subset()
        query = ncss.query()
        # Make box containing latitude, longitude
        query.lonlat_box(north=0.25*ceil(self.lat/0.25), south=0.25*floor(self.lat/0.25), east=0.25*ceil(self.lon/0.25), west=0.25*floor(self.lon/0.25))
        query.time(self.forecast_hr)

    def profiles(self) -> pd.DataFrame:

        data_P = np.zeros(self.levels.size)
        data_Z = np.zeros(self.levels.size)
        data_T = np.zeros(self.levels.size)
        data_RH = np.zeros(self.levels.size)
        data_U = np.zeros(self.levels.size)
        data_V = np.zeros(self.levels.size)

        outFile = "./gfs_grib_file.grib2"

        for j, l in enumerate(self.levels):
            data_P[j] = np.float64(l.replace(" mb", "")) * 100

            self.download_grib(f":HGT:{l}", outFile=outFile)

            gp_data = xr.load_dataset(outFile, engine="cfgrib")
            gp = np.float64(
                gp_data["gh"]
                .interp(latitude=self.lat, longitude=self.lon, method="cubic")
                .values
            ) * gp_data["gh"].metpy.units

            data_Z[j] = metcalc.geopotential_to_height(gp)

            self.download_grib(f":TMP:{l}", outFile="./gfs_grib_file.grib2")
            T_data = xr.load_dataset(outFile, engine="cfgrib")
            data_T[j] = np.float64(
                T_data["t"]
                .interp(latitude=self.lat, longitude=self.lon, method="cubic")
                .values
            ) * T_data["t"].metpy.units

            self.download_grib(f":RH:{l}", outFile="./gfs_grib_file.grib2")
            RH_data = xr.load_dataset(outFile, engine="cfgrib")
            data_RH[j] = np.float64(
                RH_data[""]
                .interp(latitude=self.lat, longitude=self.lon, method="cubic")
                .values
            ) * T_data["t"].metpy.units
            
            self.download_grib(f":(?:U|V)GRD:{l}", outFile="./gfs_grib_file.grib2")
            UV_data = xr.load_dataset(outFile, engine="cfgrib")
            uv_interp = UV_data.interp(latitude=self.lat, longitude=self.lon, method="cubic")
            data_U[j] = np.float64(uv_interp["u"].values) * UV_data['U'].metpy.units
            data_V[j] = np.float64(uv_interp["v"].values) * UV_data['U'].metpy.units

        df = pd.DataFrame(
            columns=["altitude", "temperature", "pressure", "wind_U", "wind_V"]
        )
        df["altitude"] = data_Z
        df["temperature"] = data_T
        df["pressure"] = data_P
        df["wind_U"] = data_U
        df["wind_V"] = data_V

        df = df.dropna()
        df = df.sort_values("altitude", ignore_index=True)

        return df

class GFSarchive(GFS):
    def __init__(self, cycle_datetime, forecast_hr, lat, lon):

        super().__init__(cycle_datetime, forecast_hr, lat, lon)

        self._setup()

    @property
    def url(self) -> str:
        cycle_date = f"{self.cycle_datetime:%Y%m%d}"
        cycle_hour = f"{self.cycle_datetime:%H}"
        forecast_hour = f"{self.forecast_hr:03d}"
        return f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{cycle_date}/{cycle_hour}/gfs.t{cycle_hour}z.pgrb2.0p25.f{forecast_hour}"



class GFSforecast(GFS):
    def __init__(self, cycle_datetime, forecast_hr, lat, lon):

        super().__init__(cycle_datetime, forecast_hr, lat, lon)

        self._setup()

    @property
    def url(self) -> str:
        return f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{self.cycle_datetime:%Y%m%d/%H}/atmos/gfs.t{self.cycle_datetime:%H}z.pgrb2.0p25.f{self.forecast_hr:03d}"


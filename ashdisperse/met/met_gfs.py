import metpy.calc as metcalc
import numpy as np
import os
import pandas as pd
import requests
from typing import Optional
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

    def _check_grib(self) -> bool:
        if self.url:
            head = requests.head(self.url)
            check_exists = head.ok
            if check_exists:
                check_content = int(head.raw.info()["Content-Length"]) > 1_000_000
                return check_exists and check_content
        return False

    def _check_idx(self) -> bool:
        if self.idx_url:
            idx_exists = requests.head(self.idx_url).ok
            return idx_exists
        return False

    def _get_idx_as_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.idx_url,
            sep=":",
            names=[
                "grib_message",
                "start_byte",
                "reference_time",
                "variable",
                "level",
                "forecast_time",
                "?",
                "??",
                "???",
            ],
        )

        # Format the DataFrame
        df["reference_time"] = pd.to_datetime(df.reference_time, format="d=%Y%m%d%H")
        df["valid_time"] = df["reference_time"] + pd.to_timedelta(
            f"{self.forecast_hr}h"
        )
        df["start_byte"] = df["start_byte"].astype(int)
        df["end_byte"] = df["start_byte"].shift(-1, fill_value="")
        # TODO: Check this works: Assign the ending byte for the last row...
        # TODO: df["end_byte"] = df["start_byte"].shift(-1, fill_value=requests.get(self.grib, stream=True).headers['Content-Length'])
        # TODO: Based on what Karl Schnieder did.
        df["range"] = df.start_byte.astype(str) + "-" + df.end_byte.astype(str)
        df = df.reindex(
            columns=[
                "grib_message",
                "start_byte",
                "end_byte",
                "range",
                "reference_time",
                "valid_time",
                "variable",
                "level",
                "forecast_time",
                "?",
                "??",
                "???",
            ]
        )

        df = df.dropna(how="all", axis=1)
        df = df.fillna("")

        df["search_this"] = (
            df.loc[:, "variable":]
            .astype(str)
            .apply(
                lambda x: ":" + ":".join(x).rstrip(":").replace(":nan:", ":"),
                axis=1,
            )
        )

        return df

    def read_idx(self, searchString: Optional[str]=None) -> pd.DataFrame | None:
        """
        Inspect the GRIB2 file contents by reading the index file.

        This reads index files created with the wgrib2 utility.

        Parameters
        ----------
        searchString : str
            Filter dataframe by a searchString regular expression.
            Searches for strings in the index file lines, specifically
            the variable, level, and forecast_time columns.
            Execute ``_searchString_help()`` for examples of a good
            searchString.

            .. include:: ../../user_guide/searchString.rst

        Returns
        -------
        A Pandas DataFrame of the index file.
        """

        # Filter DataFrame by searchString
        if searchString not in [None, ":"]:
            logic = self.idx.search_this.str.contains(searchString)
            if logic.sum() == 0:
                print(
                    f"No GRIB messages found. There might be something wrong with {searchString=}"
                )
            df = self.idx.loc[logic]
            return df
        else:
            return None

    def download_grib(self, searchString: str, outFile: str="./gfs_grib_file.grib2") -> None:

        grib_source = self.url

        # Download subsets of the file by byte range with cURL.
        # > Instead of using a single curl command for each row,
        # > group adjacent messages in the same curl command.

        # Find index groupings
        # TODO: Improve this for readability
        # https://stackoverflow.com/a/32199363/2383070
        idx_df = self.read_idx(searchString)
        li = idx_df.index
        inds = (
            [0]
            + [ind for ind, (i, j) in enumerate(zip(li, li[1:]), 1) if j - i > 1]
            + [len(li) + 1]
        )

        curl_groups = [li[i:j] for i, j in zip(inds, inds[1:])]
        curl_ranges = []
        group_dfs = []
        for i, group in enumerate(curl_groups):
            _df = idx_df.loc[group]
            curl_ranges.append(f"{_df.iloc[0].start_byte}-{_df.iloc[-1].end_byte}")
            group_dfs.append(_df)

            for i, (range, _df) in enumerate(zip(curl_ranges, group_dfs)):

                if i == 0:
                    # If we are working on the first item, overwrite the existing file...
                    curl = f"curl -s --range {range} {grib_source} > {outFile}"
                else:
                    # ...all other messages are appended to the subset file.
                    curl = f"curl -s --range {range} {grib_source} >> {outFile}"
                os.system(curl)

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


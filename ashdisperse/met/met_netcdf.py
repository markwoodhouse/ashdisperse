import cdsapi
from datetime import datetime
from math import ceil, floor
import metpy.units as metunits
import metpy.calc as metcalc
import numpy as np
import numpy.typing as npt
import os
from typing import Optional, Literal
import xarray as xr

from ..queryreport import print_text

class NetcdfMet:
    def __init__(self, file: str):
        self._file = file

        self._open = False

        self.altitude = np.empty((0), dtype=np.float64)
        self.wind_U = np.empty((0), dtype=np.float64)
        self.wind_V = np.empty((0), dtype=np.float64)
        self.wind_speed = np.empty((0), dtype=np.float64)
        self.wind_direction = np.empty((0), dtype=np.float64)
        self.temperature = np.empty((0), dtype=np.float64)
        self.pressure = np.empty((0), dtype=np.float64)
        self.density = np.empty((0), dtype=np.float64)

        self._time_coord = None


    def close(self):
        if self._open:
            self._data.close()
            self._open = False

    @property
    def file(self) -> str:
        return self._file
    
    @property
    def data(self) -> xr.Dataset:
        if not self._open:
            dat = xr.load_dataset(self._file, engine='netcdf4')
            dat = dat.metpy.parse_cf() # read metadata
            self._data = dat
            self._open = True
        return self._data

    def _get_time_coord(self):
        if 'valid_time' in self.data.coords:
            self._time_coord = 'valid_time'
        elif 'time' in self.data.coords:
            self._time_coord = 'time'
        else:
            self._time_coord = None
            Warning(f'time coordinate not recognized in {self._file}')
            

    @property
    def time_coord(self) -> str | None:
        if self._time_coord is None:
            self._get_time_coord()
        return self._time_coord

    @property
    def latitude(self) -> npt.NDArray:
        return self.data.latitude.values
    
    @property
    def longitude(self) -> npt.NDArray:
        return self.data.longitude.values
    
    @property
    def time(self) -> npt.NDArray:
        return self.data[self.time_coord].values
        
    @property
    def extent(self) -> list[float]:
        return [self.longitude.min(), self.longitude.max(), self.latitude.min(), self.latitude.max()]
    
    def _in_extent(self, lat: float, lon: float) -> bool:
        lon_min, lon_max, lat_min, lat_max = self.extent
        return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)
    
    def _in_timespan(self, datetime: datetime) -> bool:
        return np.datetime64(datetime) in self.time

    def extract(self, lat: float, lon: float, datetime: datetime, convention: Literal["to", "from"]="to"):

        if not self._in_extent(lat, lon):
            print_text(f"Location {lat}, {lon} not in extent of {self.file}")
            return
        
        if not self._in_timespan(datetime):
            print_text(f"Time {datetime} not in timespan of {self.file}")
            return
                
        data0 = self.data.sel(valid_time=datetime).interp(latitude=lat, longitude=lon, method='cubic').metpy.quantify()

        geopot = data0['z']
        Z = metcalc.geopotential_to_height(geopot)
        Z = Z.metpy.convert_units("m")

        self.altitude = np.float64(Z.values)

        U = data0['u'].metpy.convert_units("m/s")
        V = data0['v'].metpy.convert_units("m/s")
        T = data0['t'].metpy.convert_units("K")
        RH = data0['r'].metpy.convert_units("%")
        P = data0['pressure_level']
        spd = metcalc.wind_speed(U,V)
        dir = metcalc.wind_direction(U,V, convention=convention)

        self.wind_U = np.float64(U.values)
        self.wind_V = np.float64(V.values)
        self.wind_speed = np.float64(spd.values)
        self.wind_direction = np.float64(dir.values)
        self.temperature = np.float64(T.values)
        self.relhum = np.float64(RH.values)
        self.pressure = np.float64(P.metpy.convert_units("Pa").values)

        mixing_ratio = metcalc.mixing_ratio_from_relative_humidity(P,T,RH)
        rho = metcalc.density(P,T,mixing_ratio)
        rho = rho.metpy.convert_units("kg/m^3")
        self.density = rho.values

    def download_era5(self, *, 
                 lat: Optional[float]=None, 
                 lon: Optional[float]=None, 
                 datetime: Optional[datetime]=None,
                 extent: Optional[list[float]]=None,
                 year: Optional[list[int]]=None,
                 month: Optional[list[int]]=None,
                 day: Optional[list[int]]=None,
                 hour: Optional[list[int]]=None):
        if os.path.isfile(self.file):
            print_text(f"Met file {self.file} exists and will be overwritten")

        if (lat is None) and (lon is None):
            if extent is None:
                    raise ValueError('set either lat and lon, or extent')
        else:
            if (lat is None) or (lon is None):
                raise ValueError('set both lat and lon')
            else:
                extent = [0.25*floor(lon/0.25), 
                          0.25*ceil(lon/0.25), 
                          0.25*floor(lat/0.25),
                          0.25*ceil(lat/0.25)]
        
        if datetime is not None:
            year = [datetime.year]
            month = [datetime.month]
            day = [datetime.day]
            hour = [datetime.hour]
        else:
            if (year is None) or (month is None) or (day is None) or (hour is None):
                raise ValueError('set either datetime or all of year, month, day and hour')

        cds_dataset = "reanalysis-era5-pressure-levels"
        cds_request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "geopotential",
                "temperature",
                "relative_humidity",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": [
                "1","2","3","5","7","10","20","30","50","70",
                "100","125","150","175","200","225","250","300",
                "350","400","450","500","550","600","650","700","750",
                "775","800","825","850","875","900","925","950","975","1000",
            ],
            "year": [str(yr) for yr in year],
            "month": ["{:02d}".format(mnth) for mnth in month],
            "day": ["{:02d}".format(d) for d in day],
            "time": ["{:02d}:00".format(hr) for hr in hour],
            "area": [
                extent[3], # max lat
                extent[0], # min lon
                extent[2], # min lat
                extent[1], # max lon
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        cds = cdsapi.Client()
        cds.retrieve(cds_dataset, cds_request).download(self.file)
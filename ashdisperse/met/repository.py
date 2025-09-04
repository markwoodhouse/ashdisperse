
import os
import pathlib
import random
from collections.abc import Iterable
from datetime import datetime
from multiprocessing import Pool
from typing import Literal, Optional

import matplotlib.pyplot as plt
import matplotlib.typing as mplType
import numpy as np
import pandas as pd
import seaborn as sns
import windrose as wr
from metpy.calc import wind_direction, wind_speed

from ..queryreport import print_text
from ..utilities import nice_round_down, nice_round_up
from .met import NetcdfMet

iVector = Iterable[list[int]]
fVector = Iterable[list[float]]
sVector = Iterable[list[str]]

seasonDict = dict[str,iVector]
seasonList = list[seasonDict]

_ENSEMBLE_TYPES = Literal["randomize", "sequential"]

class MetRepo:
    """
    A class to access a repository of netcdf met data.
    The repository is assumed to be a directory containing netcdf files in the form
    YYYYMM.nc
    with each file containing met data on pressure levels over days, hours, latitude and longitude

    """
    _lat = float
    _lon = float

    _dirc = str
    _filetype = str
    _files = sVector
    _years = iVector
    _months = iVector
    _days = iVector
    _hours = iVector

    _seasons = seasonList
    _seasons_combine = str

    def __init__(self, dirc: str, lat: float, lon: float, filetype: Literal["nc","grib"]="nc"):
        """
        Initialize a MetRepo instance

        :param dirc: path to met data repository
        :type dirc: str
        :param lat: latitude 
        :type lat: float
        :param lon: _description_
        :type lon: float
        :param filetype: _description_, defaults to "nc"
        :type filetype: Literal[&quot;nc&quot;,&quot;grib&quot;], optional
        :raises RuntimeError: if dirc is not a directory or is not found
        """

        if not os.path.isdir(dirc):
            raise RuntimeError('{dirc} is not a directory or not found')
        
        self._lat = lat
        self._lon = lon

        self._dirc = dirc
        self._filetype = filetype

        self._files = [file.stem for file in sorted(pathlib.Path(self.dirc).glob(f"*.{filetype}"))]

        self._years = []
        self._months = []
        self._days = []
        self._hours = []

        self._seasons = []
        self._seasons_combine = 'first'

    @property
    def latitude(self):
        return self._lat
    
    @property
    def longitude(self):
        return self._lon
    
    @property
    def years_available(self) -> set[int]:
        return set(sorted([int(f[:4]) for f in self._files]))
    
    @property
    def months_available(self) -> set[int]:
        return set(sorted([int(f[4:6]) for f in self._files]))

    @property
    def dirc(self):
        return self._dirc
    
    @property
    def years(self):
        return self._years

    @years.setter
    def years(self, yr: list[int]):
        self._years = yr

    @property
    def months(self):
        return self._months
    
    @months.setter
    def months(self, mnths: list[int]):
        self._months = mnths

    @property
    def days(self):
        return self._days
    
    @days.setter
    def days(self, d):
        self._days = d

    @property
    def hours(self):
        return self._hours
    
    @hours.setter
    def hours(self, hrs: list[int]):
        self._hours = hrs

    @property
    def seasons(self):
        return self._seasons
    
    @seasons.setter
    def seasons(self, season_list: list[dict[str,list[int]]], combine='first'):
        self._seasons = season_list
        self._seasons_combine = combine

    def add_season(self, name: str, months: list[int]):
        self._seasons.append(
            {'name':name,
             'months': sorted(months)}
        )

    @property
    def season_names(self):
        return [s['name'] for s in self.seasons]

    @property
    def dataframe(self):
        return self._dataframe

    def retrieve(self, N: int, method: _ENSEMBLE_TYPES='randomize', processes=1):

        if not self.years:
            years = list(self.years_available)
        else:
            years = [y for y in self.years if y in self.years_available]
            skipped_years = [y for y in self.years if y not in years]
            if skipped_years:
                print_text(f"skipping years {[print(y) for y in skipped_years]}")
        
        if not self.months:
            months = list(self.months_available)
        else:
            months = [m for m in self.months if m in self.months_available]
            skipped_months = [m for m in self.months if m not in months]
            if skipped_months:
                print_text(f"skipping months {[print(m) for m in skipped_months]}")

        if not self.days: # No days are set
            self.days = [d for d in range(1,32)]

        if not self.hours:
            self.hours = [0]

        datetimes = []
        for j in range(N):
            match method:
                case 'randomize':
                    year = random.choice(years)
                    month = random.choice(months)
                    if month==2:
                        day = random.choice(self.days[:28])
                    elif month in [4,6,9,11]:
                        day = random.choice(self.days[:-2])
                    else:
                        day = random.choice(self.days)
                    hour = random.choice(self.hours)
                case 'sequential':
                    year = years
            datetimes.append(datetime.fromisoformat(f"{year:04d}{month:02d}{day:02d} {hour:02d}00"))

        out = []
        pool = Pool(processes=processes)
        out = pool.map(self._build_met_list, datetimes)
        out = [x for sublist in out for x in sublist]

        met_data = pd.concat(out, ignore_index=True)

        self._apply_seasons_to_dataframe(met_data)

        return met_data


    def _apply_seasons_to_dataframe(self, df):        
        if len(self._seasons)>0:
            df['season'] = df.apply(lambda x: self._get_seasons(x), axis=1)
        return df
    
    def _get_seasons(self, x: pd.core.series.Series, method: Literal['combine','first']='combine'):
        xx = [s['name'] for s in self.seasons if x['month'] in s['months']]
        if xx:
            if method=='combine':
                return ' & '.join(xx)
            else:
                return xx[0]
        else:
            return 'other'
    

    def _build_met_list(self, dt: datetime) -> list[pd.DataFrame]:
        out = []
        this_met = self._get_met_as_df(dt)
        out.append(this_met)
        return out


    def _get_met_as_df(self, dt: datetime) -> pd.DataFrame:
        filename = f"{dt.year:04d}{dt.month:02d}.{self._filetype}"
        met = NetcdfMet(os.path.join(self._dirc,filename))
        met.extract(self.latitude, self.longitude, dt, convention="to")
        met_df = pd.DataFrame(columns=(
            'year',
            'month',
            'day',
            'hour',
            'altitude',
            'pressure',
            'temperature',
            'relative_humidity',
            'density',
            'wind_u',
            'wind_v',
            'wind_speed',
            'wind_direction'))
        met_df.loc[0,'year'] = dt.year
        met_df.loc[0,'month'] = dt.month
        met_df.loc[0,'day'] = dt.day
        met_df.loc[0,'hour'] = dt.hour
        met_df.loc[0,'altitude'] = met.altitude
        met_df.loc[0,'pressure'] = met.pressure
        met_df.loc[0,'temperature'] = met.temperature
        met_df.loc[0,'relative_humidity'] = met.relhum
        met_df.loc[0,'density'] = met.density
        met_df.loc[0,'wind_u'] = met.wind_U
        met_df.loc[0,'wind_v'] = met.wind_V
        met_df.loc[0,'wind_speed'] = met.wind_speed
        met_df.loc[0,'wind_direction'] = met.wind_direction
        met.close()
        return met_df


def _plot_windrose_subplots(data, *, direction, var, color=None, **kwargs):
    """wrapper function to create subplots per axis"""
    ax = plt.gca()
    ax = wr.WindroseAxes.from_ax(ax=ax)
    wr.plot_windrose(direction_or_df=data[direction], var=data[var], ax=ax, **kwargs)


def _add_windrose_to_ax(ax, data, spd_bins,
                        normed=True,
                        calm_limit=0.1,
                        cmap='viridis',
                        kind="bar",
                        opening=1.0,
                        blowto=True):
    
    ax = wr.WindroseAxes.from_ax(ax=ax)
    wr.plot_windrose(data, ax=ax,
                     normed=normed,
                     calm_limit=calm_limit,
                     kind=kind,
                     cmap=cmap,
                     opening=opening,
                     bins=spd_bins,
                     blowto=blowto,
                     )
    return ax


def _get_spd_bins(max_speed: float, calm_limit: float=0.1, intervals: int=5) -> list[float]:
    spd_bins = [calm_limit]
    r = 1.0/intervals
    for k in range(1,intervals):
        spd_bins.append(nice_round_down(max_speed*r*k, 1))
    spd_bins.append(nice_round_up(max_speed, 1))

    spd_bins = list(dict.fromkeys(spd_bins))
    spd_bins = [s for s in spd_bins if s>=calm_limit]

    return spd_bins

def plot_windroses(data: MetRepo | pd.DataFrame, altitudes: list[float], 
                   N: int=1000, 
                   title: Optional[str]=None,
                   altitude_unit: Optional[str]='km',
                   row: Optional[str]='altitude',
                   col: Optional[int]=None,
                   col_order: Optional[list[str]]=None,
                   col_titles: Optional[list[str]]=None,
                   row_titles: Optional[list[str]]=None,
                   legend_share: Optional[str]='row',
                   freq_share: Optional[str]='col',
                   cmap: Optional[mplType.ColorType]='viridis',
                   include_all: bool=True,
                   method: _ENSEMBLE_TYPES='randomize', 
                   isblowto: bool=True,
                   processes=1):

    if isinstance(data, MetRepo):
        dataset = data.retrieve(N, method=method, processes=processes)
    elif isinstance(data, pd.DataFrame):
        dataset = data.copy()
    else:
        raise ValueError('plot_windroses data should be either MetRepo or pandas.DataFrame')
    
    if altitude_unit not in ['km', 'm']:
        raise ValueError('altitude_unit must be either km or m')
    
    share_allowed = ['row', 'col', 'none']
    if legend_share not in share_allowed:
        raise ValueError(f"legend_share must be one of {share_allowed}; received {legend_share}")
    
    if freq_share not in share_allowed:
        raise ValueError(f"freq_share must be one of {share_allowed}; received {freq_share}")
    
    if row == col:
        raise ValueError(f"row and col must be different, recieved row={row}, col={col}")
    
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    df = pd.DataFrame(columns=('altitude','speed','direction'))
    for alt in altitudes:

        dataset[f'wind speed at {alt} m'] = dataset.apply(lambda x: np.interp(alt,x['altitude'],x['wind_speed']), axis=1)
        dataset[f'wind direction at {alt} m'] = dataset.apply(lambda x: np.interp(alt,x['altitude'],x['wind_direction']), axis=1)

        this_df = pd.DataFrame(columns=('altitude','speed','direction'))
        this_df['speed'] = dataset[f'wind speed at {alt} m']
        this_df['direction'] = dataset[f'wind direction at {alt} m']
        this_df['altitude'] = alt
        if col is not None and col!="altitude":
            this_df[col] = dataset[col]
        if row is not None and row!="altitude":
            this_df[row] = dataset[row]

        if len(df)==0:
            df = this_df.copy()
        else:
            df = pd.concat([df, this_df], ignore_index=True)

    if include_all:
        df1 = df.copy()
        df1[col] = 'all'

        df = pd.concat([df,df1], ignore_index=True)

    if row:
        row_values = df[row].unique()
        Nrows = len(row_values)
    else:
        row_values = ''
        Nrows = 1

    if col:
        if col_order is not None:
            col_values = col_order
        else:
            col_values = df[col].unique()
        Ncols = len(col_values)
    else:
        col_values = ''
        Ncols = 1
    
    max_speeds = df.groupby([row, col]).speed.max()

    fig, axs = plt.subplots(ncols=Ncols, nrows=Nrows, subplot_kw={'projection':'windrose'})
    for i in range(Nrows):

        df_row = df.loc[df[row]==row_values[i]]

        if legend_share=='row':
            spd_bins = _get_spd_bins(max_speeds[row_values[i], :].max())
        
        for j in range(Ncols):
            df_row_col = df_row.loc[df_row[col]==col_values[j]]

            if legend_share=='col':
                spd_bins = _get_spd_bins(max_speeds[:, col_values[j]].max())

            if legend_share=='none':
                spd_bins = _get_spd_bins(max_speeds[row_values[i], col_values[j]].max())
            
            ax = _add_windrose_to_ax(axs[i,j], df_row_col, spd_bins[:-1],
                        normed=True,
                        calm_limit=0.1,
                        kind="bar",
                        cmap=cmap,
                        opening=1.0,
                        blowto=not isblowto)

            if legend_share=='row':
                if j==Ncols-1:
                    ax.set_legend(ncols=1, bbox_to_anchor=(1.15,0.25,0.1,0.5), title='Wind speed (m/s)')
                    for k, l in enumerate(ax.get_legend().texts):
                        lt = f"{spd_bins[k]}" + u'\u2014' + f"{spd_bins[k+1]}"
                        l.set_text(lt)
                else:
                    ax.legend().set_visible(False)

            ax.set_xticklabels([])

            if i==0:
                if col_titles is None:
                    c_title = f"{col_values[j]}"
                else:
                    c_title = col_titles[j]
                ax.annotate(c_title, xy=(0.5,1.02), 
                        xytext=(0,5), 
                        xycoords='axes fraction', 
                        textcoords='offset points', 
                        ha='center', 
                        va='baseline',
                        size='large')
                
        
        if row_titles is None:
            if row=='altitude':
                if altitude_unit=='km':
                    r_title = f"{row} = {row_values[i]/1e3} km"
                else:
                    r_title = f"{row} = {row_values[i]} m"
            else:
                r_title = f"{row}: {row_values[i]}"
        else:
            r_title = row_titles[i]
        axs[i,0].annotate(r_title, xy=(0,0.5), 
                    xytext=(-axs[i,0].yaxis.labelpad - 5,0), 
                    xycoords=axs[i,0].yaxis.label, 
                    textcoords='offset points', 
                    ha='right',
                    va='center',
                    size='large',
                    rotation=90)

    if freq_share=='row':
        for i in range(Nrows):
            max_wd_freq = 0 
            for ax in axs[i,:]:
                table = ax._info["table"]
                wd_freq = np.sum(table, axis=0)
                max_wd_freq = np.maximum(max_wd_freq, np.amax(wd_freq))

            freq_ticks = np.linspace(0,np.ceil(max_wd_freq/5)*5,6, dtype=int)
            freq_ticks = freq_ticks[1:]
            for ax in axs[i,:]:
                ax.set_rgrids(freq_ticks, freq_ticks)
                ax.set(rlabel_position=-90)
                ax.tick_params(axis='both', labelsize=8, pad=0)

    if freq_share=='col':
        for j in range(Ncols):
            max_wd_freq = 0 
            for ax in axs[:,j]:
                table = ax._info["table"]
                wd_freq = np.sum(table, axis=0)
                max_wd_freq = np.maximum(max_wd_freq, np.amax(wd_freq))

            freq_ticks = np.linspace(0,np.ceil(max_wd_freq/5)*5,6, dtype=int)
            freq_ticks = freq_ticks[1:]
            for ax in axs[:,j]:
                ax.set_rgrids(freq_ticks, freq_ticks)
                ax.set(rlabel_position=-90)
                ax.tick_params(axis='both', labelsize=8, pad=0)

    if freq_share=='none':
        for ax in axs.flatten():
            table = ax._info["table"]
            wd_freq = np.sum(table, axis=0)
            max_wd_freq = np.amax(wd_freq)

            freq_ticks = np.linspace(0,np.ceil(max_wd_freq/5)*5,6, dtype=int)
            freq_ticks = freq_ticks[1:]
            ax.set_rgrids(freq_ticks, freq_ticks)
            ax.set(rlabel_position=-90)
            ax.tick_params(axis='both', labelsize=8, pad=0)

    
    fig.subplots_adjust(right=0.8, wspace=0.1, hspace=0.1)

    return fig, axs




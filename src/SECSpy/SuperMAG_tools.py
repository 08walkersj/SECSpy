import datetime as _datetime
import importlib.util
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from importlib.resources import files

import pandas as pd


_STATION_COLUMNS = [
    "IAGA",
    "GEOLON",
    "GEOLAT",
    "AACGMLON",
    "AACGMLAT",
    "STATION-NAME",
    "OPERATOR-NUM",
    "OPERATORS",
    "OPERATOR2",
    "OPERATOR3",
    "OPERATOR4",
    "OPERATOR5",
]


@lru_cache(maxsize=1)
def _supermag_api():
    """Load the official SuperMAG API copy, whose filename is not importable."""
    api_path = Path(__file__).with_name("supermag-api.py")
    spec = importlib.util.spec_from_file_location("SECSpy._official_supermag_api", api_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_stations_path():
    return files("SECSpy").joinpath("assets", "supermag-stations.csv")


def _pop_stations_source(kwargs=None, stations=None):
    if kwargs is None:
        kwargs = {}
    if stations is None:
        stations = kwargs.pop("stations", None)

    if stations is not None:
        return stations
    station_info = kwargs.pop("station_info", None)
    if station_info is not None:
        return station_info
    stations_path = kwargs.pop("stations_path", False)
    if stations_path is not False:
        return stations_path
    station_path = kwargs.pop("station_path", False)
    if station_path is not False:
        return station_path
    return False


def _normalise_start(start):
    if isinstance(start, list):
        return start
    if isinstance(start, np.datetime64):
        return np.datetime_as_string(start.astype("datetime64[m]"))
    if isinstance(start, pd.Timestamp):
        return start.strftime("%Y-%m-%dT%H:%M")
    if isinstance(start, _datetime.datetime):
        return start.strftime("%Y-%m-%dT%H:%M")
    if isinstance(start, _datetime.date):
        return start.strftime("%Y-%m-%dT00:00")
    if isinstance(start, str):
        return start
    raise ValueError(f"Time format not understood: {type(start)!r}")


def _normalise_flagstring(flagstring):
    if isinstance(flagstring, (list, tuple, set)):
        return ",".join(str(flag) for flag in flagstring)
    return flagstring


def _looks_like_filter(value):
    if not isinstance(value, str):
        return False
    filter_tokens = (
        "(",
        ")",
        ">",
        "<",
        "=",
        "&",
        "|",
        "GEOLON",
        "GEOLAT",
        "GLON",
        "GLAT",
        "AACGMLON",
        "AACGMLAT",
        "MLON",
        "MLAT",
    )
    return any(token in value.upper() for token in filter_tokens)


def _read_stations(stations=False):
    if isinstance(stations, pd.DataFrame):
        stations = stations.copy()
    else:
        if not stations:
            stations = _default_stations_path()
        stations = pd.read_csv(stations, names=_STATION_COLUMNS, skiprows=1)

    if "GEOLON" in stations:
        stations["GEOLON"] %= 360
    if "GEOLAT" in stations and "GLAT" not in stations:
        stations["GLAT"] = stations["GEOLAT"]
    if "GEOLON" in stations and "GLON" not in stations:
        stations["GLON"] = stations["GEOLON"]
    if "AACGMLON" in stations and "MLON" not in stations:
        stations["MLON"] = stations["AACGMLON"]
    if "AACGMLAT" in stations and "MLAT" not in stations:
        stations["MLAT"] = stations["AACGMLAT"]
    return stations


def Filtered(*Filtering, stations=None, **kwargs):
    """
    Provides a list of station keys that satisfy criteria provided
    
    Parameters
    ----------
    *Filtering : string/ multiple strings (that will be joined by and)
        strings containing the column in the stations file and requirements to be satified by interacting with the column string.
        Brackets must be used as doing a normal boolean set up when using & (and) and | (or) statements. Mutiple strings will be assumed to be joined by &
    stations : pandas.DataFrame or path-like, optional
        Station metadata to filter. This can be a CSV path or an already
        prepared DataFrame with extra columns such as cube-sphere xi/eta
        coordinates. If omitted, the packaged station CSV is used.

    Returns
    -------
    stations
        List of station keys that satify the criteria provided.
        If no criteria is provided it will rerturn station keys.

    """
    stations = _read_stations(stations=_pop_stations_source(kwargs, stations=stations))
    if not len(Filtering):
        return stations.IAGA.values
    else:
        
        Filtering=['(stations.'.join(('('+filt).split('(')) +')' for filt in Filtering]
        Filtering=['('+') | (stations.'.join(filt.split(' | ')) for filt in Filtering]
        Filtering=[') & (stations.'.join(filt.split(' & ')) for filt in Filtering]
        Filtering=') & '.join(Filtering)+')'
        Filtering='('.join(Filtering.split('stations.('))
        return stations[eval(Filtering)].IAGA.values


def SuperMAGGetInventory(logon, start, extent, *Filtering, **filter_kwargs):
    stations_source = _pop_stations_source(filter_kwargs)
    if filter_kwargs:
        unsupported = ", ".join(sorted(filter_kwargs))
        raise TypeError(
            "SuperMAGGetInventory only accepts station metadata kwargs "
            f"for filtering; unsupported keyword(s): {unsupported}"
        )
    status, stations = _supermag_api().SuperMAGGetInventory(
        logon, _normalise_start(start), extent
    )
    if not status:
        return status, stations

    if Filtering:
        allowed = Filtered(*Filtering, stations=stations_source)
        stations = np.asarray(stations)
        stations = stations[np.isin(stations, allowed)]
    return status, stations


def SuperMAGGetIndices(logon, start, extent, flagstring="", **kwargs):
    status, indices = _supermag_api().SuperMAGGetIndices(
        logon, _normalise_start(start), extent, _normalise_flagstring(flagstring), **kwargs
    )
    if "tval" in indices:
        indices["Date_UTC"] = (
            np.datetime64("1970-01-01T00:00")
            + indices.tval.values.astype("timedelta64[s]")
        )
    return status, indices


def Dictionary2Column(dic):
    return dic["nez"], dic["geo"]


def tqdm_joblib(total=None, desc_prefix="Progress", unit="task", enabled=True):
    """Patch joblib so completed batches update a tqdm progress bar."""
    import joblib
    from tqdm import tqdm

    @contextmanager
    def _tqdm_joblib_context():
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                progress_bar.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        if not enabled:
            yield
            return

        progress_bar = tqdm(total=total, desc=desc_prefix, unit=unit)
        original_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield
        finally:
            joblib.parallel.BatchCompletionCallBack = original_callback
            progress_bar.close()

    return _tqdm_joblib_context()


def _download_station(args):
    logon, start, extent, flagstring, station, kwargs = args
    return _supermag_api().SuperMAGGetData(
        logon, start, extent, flagstring, station, **kwargs
    )


def _combine_station_results(results, format_type):
    status = sum(result[0] for result in results)
    payloads = [result[1] for result in results]
    # if format_type == "list":
    #     return status, [row for payload in payloads for row in payload]
    # if format_type == "raw":
    #     return status, payloads
    frames = [
        frame
        for frame in payloads
        if isinstance(frame, pd.DataFrame) and len(frame) and 'N' in frame.columns and 'E' in frame.columns and 'Z' in frame.columns
    ]
    return status, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _postprocess_data(data):
    if data is None or not len(data):
        return data
    data = data.copy()
    if "tval" in data:
        data["Date_UTC"] = (
            np.datetime64("1970-01-01T00:00")
            + data.tval.values.astype("timedelta64[s]")
        )
        sort_columns = [column for column in ["Date_UTC", "iaga"] if column in data]
        data.sort_values(sort_columns, inplace=True)
        data.index = range(len(data))
    if all(column in data for column in "NEZ"):
        for column in "NEZ":
            data[[f"{column}_nez", f"{column}_geo"]] = np.vstack(
                data[column].apply(Dictionary2Column)
            )
    preferred = [column for column in ["Date_UTC"] if column in data]
    preferred += [
        column
        for column in data.columns
        if column not in ["Date_UTC", "tval", "N", "E", "Z"]
        and not column.endswith(("_nez", "_geo"))
    ]
    preferred += [f"{column}_nez" for column in "NEZ" if f"{column}_nez" in data]
    preferred += [f"{column}_geo" for column in "NEZ" if f"{column}_geo" in data]
    return data[preferred] if preferred else data


def SuperMAGGetData(
    logon,
    start,
    extent,
    flagstring,
    *Filtering,
    stationlist=False,
    multi=True,
    processes=8,
    show_stations=True,
    progress=True,
    parallel_prefer="threads",
    stations=None,
    **kwargs,
):
    stations_source = _pop_stations_source(kwargs, stations=stations)
    start = _normalise_start(start)
    flagstring = _normalise_flagstring(flagstring)
    format_type = kwargs.get("FORMAT", "none").lower()
    if not stationlist and len(Filtering) == 1 and not _looks_like_filter(Filtering[0]):
        stationlist = Filtering[0]
        Filtering = ()

    if not stationlist:
        stationlist = Filtered(*Filtering, stations=stations_source)
        if show_stations:
            print("stations: ", stationlist)

    if isinstance(stationlist, str):
        status, data = _supermag_api().SuperMAGGetData(
            logon, start, extent, flagstring, stationlist, **kwargs
        )
    else:
        stationlist = np.asarray(stationlist)
        if not len(stationlist):
            raise ValueError("No sites have been provided or meet criteria imposed")
        args = [
            (logon, start, extent, flagstring, station, kwargs)
            for station in stationlist
        ]
        if multi and len(args) > 1:
            from joblib import Parallel, delayed

            n_jobs = min(processes, len(args))
            with tqdm_joblib(
                total=len(args),
                desc_prefix="Downloading SuperMAG stations",
                unit="station",
                enabled=progress,
            ):
                results = Parallel(n_jobs=n_jobs, prefer=parallel_prefer)(
                    delayed(_download_station)(arg) for arg in args
                )
        else:
            results = [_download_station(arg) for arg in args]
        status, data = _combine_station_results(results, format_type)

    if format_type in {"list", "raw"}:
        return status, data
    return status, _postprocess_data(data)
def cubedsphere_stations(grid, stations=None):
    """
    Add cube-sphere coordinates to a station DataFrame.

    Parameters
    ----------
    stations : pandas.DataFrame
        Station metadata, including 'GEOLON' and 'GEOLAT' columns.
    grid : SECSpy.Grid_tools.Grid
        Cube-sphere grid object with a 'projection' attribute.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added 'xi' and 'eta' columns for cube-sphere coordinates.
    """
    if stations is None:
        stations = pd.read_csv(_default_stations_path(), names=_STATION_COLUMNS, skiprows=1)
    stations['xi'], stations['eta'] = grid.projection.geo2cube(stations.GEOLON, stations.GEOLAT)
    return stations

def download_in_grid(username, event_time, stations, grid, grid_buffer=(.1, .1),
                    start= -np.timedelta64(1, 'h'), end= np.timedelta64(1, 'h'), auto_remove=False):
    """Download and prepare SuperMAG magnetometer and SML data.

    Parameters
    ----------
    username : str, optional, default: None
        SuperMAG username. If ``None``, the previously stored
        ``self.username`` is used.
    stations : pandas.DataFrame, optional, default: None
        Station metadata, including 'GEOLON' and 'GEOLAT' columns.
    grid_buffer : tuple of float, default: (0.1, 0.1)
        Fractional expansion of the xi and eta grid bounds used when
        selecting stations.
    start : numpy.timedelta64, default: numpy.timedelta64(-1, 'h')
        Start of the download window relative to ``self.event_time``.
    end : numpy.timedelta64, default: numpy.timedelta64(1, 'h')
        End of the download window relative to ``self.event_time``.
    save : str, bool, or None, default: ''
        Prefix/path control for optional CSV output through
        :meth:`_repo_path`. Use ``None`` or ``False`` to keep results only
        in memory.
    auto_remove : bool, default: False
        If ``True``, automatically remove stations with incomplete time
        coverage. If ``False``, ask before removing them.

    Attributes
    ----------
    SM_data : pandas.DataFrame or str
        SuperMAG magnetometer data with cubed-sphere coordinates and
        projected magnetic perturbations, or a CSV path if saved.
    SML_data : pandas.DataFrame or str
        SuperMAG SML index data, or a CSV path if saved.
    stations : pandas.DataFrame
        Station metadata with cubed-sphere coordinates.

    Raises
    ------
    ValueError
        If no username is available, or if station availability is
        inconsistent and removal is declined.
    """
    from SECSpy.SuperMAG_tools import SuperMAGGetData, _default_stations_path, _STATION_COLUMNS, SuperMAGGetIndices
    from SECSpy.utils import validinput
    import time
    import tempfile
    import os

    # SuperMAG returns minute samples from the requested start up to one
    # minute before start + extent. Because this query starts one minute
    # before the requested inversion window, include an extra minute so the
    # requested end time is present for run_inversion().
    extent= (end-start + np.timedelta64(2, 'm')).astype('timedelta64[s]').astype(int)
    start = event_time + start
    xi_min=grid.xi_min.copy()
    xi_max=grid.xi_max.copy()
    xi_min-=grid_buffer[0]*xi_min
    xi_max+=grid_buffer[0]*xi_max
    eta_min=grid.eta_min.copy()
    eta_max=grid.eta_max.copy()
    eta_min-=grid_buffer[1]*eta_min
    eta_max+=grid_buffer[1]*eta_max
    status, SM_data= SuperMAGGetData(username, start-np.timedelta64(1, 'm'), extent, ['GEO', 'decl'], f'xi>={xi_min} & xi<={xi_max} & eta>={eta_min} & eta<={eta_max}',
                                     stations=stations)
    if not status:
        raise ValueError('No data was downloaded check inputs e.g. username or internet connection')
    SM_data.sort_values(['Date_UTC', 'glat'], inplace=True)
    SM_data.index= list(range(len(SM_data)))
    decl_rad = np.deg2rad(SM_data['decl'])
    # SM_data['N_geo'] = SM_data['N_nez'] * np.cos(decl_rad) - SM_data['E_nez'] * np.sin(decl_rad)
    # SM_data['E_geo'] = SM_data['N_nez'] * np.sin(decl_rad) + SM_data['E_nez'] * np.cos(decl_rad)
    SM_data['xi'], SM_data['eta']= grid.projection.geo2cube(SM_data.glon, SM_data.glat)
    SM_data.replace(999999, np.nan, inplace=True)
    SM_data.dropna(inplace=True)
    SM_data.index= list(range(len(SM_data)))
    # Rotate measured east/north perturbations into the cubed-sphere basis
    # used by the SECS forward model and plotting routines.
    SM_data['Be_cs'], SM_data['Bn_cs']= grid.projection.vector_cube_projection(
        SM_data.E_geo*1e-9,
        SM_data.N_geo*1e-9,
        SM_data.glon,
        SM_data.glat,
        return_xi_eta=False,
    )
    vals, counts= np.unique(SM_data.iaga.values, return_counts=True)
    if len(np.unique(counts))>1:
        # Create temporary CSV file with magnetometer availability
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, 'magnetometer_availability.csv')
        availability_df = pd.DataFrame({'IAGA': vals, 'counts': counts})
        availability_df.to_csv(temp_file, index=False)
        
        if not auto_remove:
            # Print information with clickable file link
            print('Inconsistent availability of magnetometers')
            print(f'Details saved to: file://{temp_file}')
            time.sleep(1)
        if auto_remove or validinput('Remove Problem Site(s)? (y/n)', 'y', 'n'):
            if not auto_remove:
                sites= input('Enter site IAGA codes to remove, separated by commas. Leave blank to remove all sites with incomplete data: ').strip()
            else:
                sites= ''
            if sites=='':
                sites= vals[counts!=np.max(counts)]
            else:
                sites= [site.strip() for site in sites.split(',') if site.strip()]
            SM_data= SM_data.loc[~SM_data.iaga.isin(sites)]
        else:
            raise ValueError('inconsistent availability of magnetometers\n'+'\n'.join([f'{v}:{c}' for v, c in zip(vals, counts)]))
    return SM_data
def format_data_vector(data):
    """Format SuperMAG data as a time-indexed magnetometer data table.

    Parameters
    ----------
    data : pandas.DataFrame
        SuperMAG observations with ``Date_UTC``, ``iaga``, ``glat``,
        ``glon``, ``N_geo``, ``E_geo``, and ``Z_geo`` columns. Magnetic
        perturbation components are expected in nanotesla.

    Returns
    -------
    pandas.DataFrame
        Pivoted table indexed by ``Date_UTC`` with station IAGA codes as the
        second column level. Columns include ``glat``, ``glon``,
        ``East_ground``, ``South_ground``, and ``Up_ground``. Ground magnetic
        perturbations are converted to tesla, with south and up components
        sign-flipped from ``N_geo`` and ``Z_geo`` respectively. Stations are
        ordered by geographic latitude.
    """
    data= (
        data.assign(South_ground=-data['N_geo']*1e-9,
                    East_ground=data['E_geo']*1e-9,
                    Up_ground=-data['Z_geo']*1e-9)
        .assign(iaga=pd.Categorical(data['iaga'], categories=(
        data[['iaga', 'glat']]
        .drop_duplicates('iaga')
        .sort_values('glat')
        ['iaga']), ordered=True))
        .pivot_table(
            index='Date_UTC',
            columns='iaga',
            values=['glat', 'glon', 'East_ground', 'South_ground', 'Up_ground', 'xi', 'eta'],
            aggfunc='first',
            observed=True,
            sort=False,))

    return data

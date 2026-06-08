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
    status, stations = _supermag_api().SuperMAGGetInventory(
        logon, _normalise_start(start), extent, **filter_kwargs
    )
    if status and Filtering:
        allowed = Filtered(*Filtering, stations=stations_source)
        stations = np.asarray(stations)[np.isin(stations, allowed)]
    return status, stations


def SuperMAGGetIndices(logon, start, extent, flagstring="", **kwargs):
    return _supermag_api().SuperMAGGetIndices(
        logon, _normalise_start(start), extent, _normalise_flagstring(flagstring), **kwargs
    )


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

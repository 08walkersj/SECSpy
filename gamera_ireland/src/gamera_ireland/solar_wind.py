"""Solar-wind preparation for GAMERA boundary-condition files."""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from astropy.time import Time


REQUIRED_OMNI_COLUMNS = [
    "BX_GSE",
    "BY_GSE",
    "BZ_GSE",
    "Vx",
    "Vy",
    "Vz",
    "proton_density",
    "T",
    "AE_INDEX",
    "AL_INDEX",
    "AU_INDEX",
    "SYM_H",
    "BSN_x",
    "BSN_y",
    "BSN_z",
]

BAD_DATA_VALUES = [
    -999.900,
    99999.9,
    9999.99,
    999.990,
    1.00000e7,
    9999999.0,
    99999.0,
]


def ensure_runtime_config_dirs() -> None:
    """Provide writable config/cache directories for SpacePy and Matplotlib."""
    project_root = Path(__file__).resolve().parents[2]
    environment_dir = project_root / "environment"
    spacepy_parent = environment_dir / "spacepy"
    mpl_dir = environment_dir / "matplotlib"
    spacepy_parent.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SPACEPY", str(spacepy_parent))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def add_solar_wind_tools_to_path(repo: str | Path) -> Path:
    """Add the local Solar_Wind_Tools source tree to sys.path."""
    src = Path(repo).expanduser().resolve() / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Solar_Wind_Tools src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def download_omni_with_solar_wind_tools(
    repo: str | Path,
    output_hdf: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    """Download monthly OMNI 1-minute data using the local Solar_Wind_Tools repo."""
    add_solar_wind_tools_to_path(repo)
    from Solar_Wind_Tools.download import download_omni_1min

    output = Path(output_hdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    download_omni_1min(
        int(start.year),
        int(end.year),
        monthFirstYear=int(start.month),
        monthLastYear=int(end.month),
        path=str(output),
    )


def read_omni_subset(
    omni_hdf: str | Path,
    key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read and time-subset a Solar_Wind_Tools OMNI HDF store."""
    try:
        omni = pd.read_hdf(omni_hdf, key=key)
    except ImportError as exc:
        raise ImportError(
            "pandas HDF support is unavailable. Install PyTables in SECSpy, e.g. "
            "`conda install -n SECSpy -c conda-forge pytables`."
        ) from exc

    if not isinstance(omni.index, pd.DatetimeIndex):
        raise ValueError("Expected the OMNI HDF table index to be a DatetimeIndex.")

    omni = omni.sort_index()
    subset = omni.loc[(omni.index >= start) & (omni.index <= end)].copy()
    if subset.empty:
        raise ValueError(f"No OMNI rows found between {start} and {end}.")
    return subset


def validate_omni_columns(omni: pd.DataFrame) -> None:
    """Raise a clear error if columns required by KaiPy's OMNI reader are absent."""
    missing = [column for column in REQUIRED_OMNI_COLUMNS if column not in omni.columns]
    if missing:
        raise ValueError(
            "OMNI HDF table is missing required columns for bcwind conversion: "
            + ", ".join(missing)
        )


def dataframe_to_kaipy_omni_input(omni: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a Solar_Wind_Tools OMNI DataFrame into KaiPy OMNI reader input."""
    validate_omni_columns(omni)
    data = {column: omni[column].to_numpy() for column in REQUIRED_OMNI_COLUMNS}
    data["Epoch"] = omni.index.to_pydatetime()
    return data


def _interp_to_minutes(minutes: Iterable[int], source_minutes: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(minutes, source_minutes, np.asarray(values, dtype=float))


def _clean_series(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    raw_bad = values.isna().to_numpy().copy()
    for bad_value in BAD_DATA_VALUES:
        raw_bad |= values.to_numpy() == bad_value
    values = values.mask(raw_bad)
    cleaned = values.interpolate(method="time", limit_direction="both")
    return cleaned, raw_bad


def _prepare_clean_omni(omni: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = pd.DataFrame(index=omni.index)
    interped = pd.DataFrame(index=omni.index)
    for column in REQUIRED_OMNI_COLUMNS:
        cleaned[column], interped[column] = _clean_series(omni[column])
        if cleaned[column].isna().any():
            raise ValueError(f"Column {column} has no finite values after bad-data cleaning.")
    return cleaned, interped.astype(bool)


def _gse_vectors_to_gsm(vectors: np.ndarray, times: Iterable[dt.datetime]) -> np.ndarray:
    import kaipy.transform

    out = np.zeros_like(vectors, dtype=float)
    for i, timestamp in enumerate(times):
        out[i] = kaipy.transform.GSEtoGSM(
            vectors[i, 0],
            vectors[i, 1],
            vectors[i, 2],
            timestamp,
        )
    return out


def _gsm_vector_to_sm(x: float, y: float, z: float, timestamp: dt.datetime) -> tuple[float, float, float]:
    import kaipy.transform

    return kaipy.transform.GSMtoSM(x, y, z, timestamp)


def _tilt_angle(timestamp: dt.datetime) -> float:
    import kaipy.transform

    x, _, z = kaipy.transform.SMtoGSM(0, 0, 1, timestamp)
    return float(np.arctan2(x, z))


def _fallback_or_column(
    omni: pd.DataFrame,
    names: list[str],
    time_1minute: range,
    start_time: pd.Timestamp,
    fallback: float | None,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    for name in names:
        if name in omni.columns:
            valid = np.isfinite(omni[name].to_numpy(dtype=float))
            if valid.any():
                source_minutes = ((omni.index - start_time) / pd.Timedelta(minutes=1)).to_numpy()
                return scale * np.interp(
                    list(time_1minute),
                    source_minutes[valid],
                    omni[name].to_numpy(dtype=float)[valid],
                )

    if fallback is None:
        raise ValueError(
            f"None of {names} were present with finite data. Provide an explicit fallback."
        )
    return np.full(len(time_1minute), float(fallback))


def write_bcwind_from_omni_dataframe(
    omni: pd.DataFrame,
    output_hdf: str | Path,
    *,
    include_bx_fit: bool = False,
    coarse_filter: bool = False,
    sigma: float = 3.0,
    lfm_start_minutes: float = 0.0,
    gamera_start_minutes: float = 0.0,
    fallback_f107: float | None = None,
    fallback_kp: float | None = None,
) -> Path:
    """Write a GAMERA `bcwind.h5` file using KaiPy's verified data conventions."""
    ensure_runtime_config_dirs()

    output = Path(output_hdf)
    output.parent.mkdir(parents=True, exist_ok=True)

    if coarse_filter:
        raise NotImplementedError(
            "Coarse sigma filtering is only available through KaiPy's OMNI class, "
            "which is not importable in this environment because of pyspedas.kyoto."
        )

    validate_omni_columns(omni)
    cleaned, interped_source = _prepare_clean_omni(omni)
    timestamps = cleaned.index.to_pydatetime()
    start_datetime = timestamps[0]
    source_minutes = ((cleaned.index - cleaned.index[0]) / pd.Timedelta(minutes=1)).to_numpy()

    b_gsm = _gse_vectors_to_gsm(
        cleaned[["BX_GSE", "BY_GSE", "BZ_GSE"]].to_numpy(dtype=float),
        timestamps,
    )
    v_gsm = _gse_vectors_to_gsm(
        cleaned[["Vx", "Vy", "Vz"]].to_numpy(dtype=float),
        timestamps,
    )
    bs_gsm = _gse_vectors_to_gsm(
        cleaned[["BSN_x", "BSN_y", "BSN_z"]].to_numpy(dtype=float),
        timestamps,
    )

    n_source = cleaned["proton_density"].to_numpy(dtype=float)
    tp_source_kK = cleaned["T"].to_numpy(dtype=float) * 1.0e-3
    b_source = np.sqrt(np.sum(b_gsm**2, axis=1))
    v_source = np.sqrt(np.sum(v_gsm**2, axis=1))
    cs_source = np.sqrt(5.0 * 1e3 * tp_source_kK * 1.38e-23 / (3.0 * 1.67e-27) / 1.0e6)
    va_source = b_source * 1.0e-10 / np.sqrt(1.97e-24 * 4 * np.pi * n_source)

    if include_bx_fit:
        raise NotImplementedError(
            "Bx fitting is not enabled in the local converter while KaiPy's OMNI "
            "class is unavailable."
        )
    else:
        bcoef = [0.0, 0.0, 0.0]

    time_1minute = range(int(np.min(source_minutes)), int(np.max(source_minutes)))
    if len(time_1minute) < 2:
        raise ValueError("Need at least two one-minute samples to write bcwind.h5.")

    n = _interp_to_minutes(time_1minute, source_minutes, n_source)
    tp = _interp_to_minutes(time_1minute, source_minutes, tp_source_kK)
    vx = _interp_to_minutes(time_1minute, source_minutes, v_gsm[:, 0])
    vy = _interp_to_minutes(time_1minute, source_minutes, v_gsm[:, 1])
    vz = _interp_to_minutes(time_1minute, source_minutes, v_gsm[:, 2])
    cs = _interp_to_minutes(time_1minute, source_minutes, cs_source)
    va = _interp_to_minutes(time_1minute, source_minutes, va_source)
    bx = _interp_to_minutes(time_1minute, source_minutes, b_gsm[:, 0])
    by = _interp_to_minutes(time_1minute, source_minutes, b_gsm[:, 1])
    bz = _interp_to_minutes(time_1minute, source_minutes, b_gsm[:, 2])
    b = _interp_to_minutes(time_1minute, source_minutes, b_source)
    ae = _interp_to_minutes(time_1minute, source_minutes, cleaned["AE_INDEX"].to_numpy(dtype=float))
    al = _interp_to_minutes(time_1minute, source_minutes, cleaned["AL_INDEX"].to_numpy(dtype=float))
    au = _interp_to_minutes(time_1minute, source_minutes, cleaned["AU_INDEX"].to_numpy(dtype=float))
    symh = _interp_to_minutes(time_1minute, source_minutes, cleaned["SYM_H"].to_numpy(dtype=float))
    bsx = _interp_to_minutes(time_1minute, source_minutes, bs_gsm[:, 0])
    bsy = _interp_to_minutes(time_1minute, source_minutes, bs_gsm[:, 1])
    bsz = _interp_to_minutes(time_1minute, source_minutes, bs_gsm[:, 2])

    interped_columns = [
        "BX_GSE",
        "BY_GSE",
        "BZ_GSE",
        "Vx",
        "Vy",
        "Vz",
        "proton_density",
        "T",
        "BSN_x",
        "BSN_y",
        "BSN_z",
    ]
    interped = np.vstack([
        _interp_to_minutes(time_1minute, source_minutes, interped_source[column])
        for column in interped_columns
    ])
    is_interped = np.any(interped, axis=0)

    mfast = np.sqrt((vx**2 + vy**2 + vz**2) / (cs**2 + va**2))

    t_seconds = (np.asarray(list(time_1minute)) - lfm_start_minutes + gamera_start_minutes) * 60.0
    ut = np.asarray(
        [
            np.bytes_((start_datetime + dt.timedelta(seconds=float(seconds))).isoformat())
            for seconds in t_seconds
        ]
    )
    mjd_ref = Time(start_datetime).mjd
    mjd = np.asarray([mjd_ref + seconds / 86400.0 for seconds in t_seconds])

    v_sm = np.zeros((len(time_1minute), 3))
    b_sm = np.zeros((len(time_1minute), 3))
    bs_sm = np.zeros((len(time_1minute), 3))
    tilt = np.zeros(len(time_1minute))

    for i, minute in enumerate(time_1minute):
        timestamp = start_datetime + dt.timedelta(minutes=float(minute))
        v_sm[i] = _gsm_vector_to_sm(vx[i], vy[i], vz[i], timestamp)
        b_sm[i] = _gsm_vector_to_sm(bx[i], by[i], bz[i], timestamp)
        bs_sm[i] = _gsm_vector_to_sm(bsx[i], bsy[i], bsz[i], timestamp)
        tilt[i] = _tilt_angle(timestamp)

    start_timestamp = pd.Timestamp(start_datetime)
    f107 = _fallback_or_column(
        omni,
        ["F10_INDEX1800", "F10_INDEX", "f10.7", "F107"],
        time_1minute,
        start_timestamp,
        fallback_f107,
    )
    kp = _fallback_or_column(
        omni,
        ["KP1800", "Kp", "KP"],
        time_1minute,
        start_timestamp,
        fallback_kp,
        scale=0.1 if "KP1800" in omni.columns else 1.0,
    )

    with h5py.File(output, "w") as handle:
        handle.attrs["source"] = "Solar_Wind_Tools OMNI HDF converted with gamera_ireland"
        handle.attrs["start_utc"] = start_datetime.isoformat()
        handle.create_dataset("T", data=t_seconds)
        handle.create_dataset("UT", data=ut)
        handle.create_dataset("MJD", data=mjd)
        handle.create_dataset("D", data=n)
        handle.create_dataset("Temp", data=tp * 1.0e3)
        handle.create_dataset("Vx", data=v_sm[:, 0] * 1.0e3)
        handle.create_dataset("Vy", data=v_sm[:, 1] * 1.0e3)
        handle.create_dataset("Vz", data=v_sm[:, 2] * 1.0e3)
        handle.create_dataset("Bx", data=b_sm[:, 0])
        handle.create_dataset("By", data=b_sm[:, 1])
        handle.create_dataset("Bz", data=b_sm[:, 2])
        handle.create_dataset("tilt", data=tilt)
        handle.create_dataset("ae", data=ae)
        handle.create_dataset("al", data=al)
        handle.create_dataset("au", data=au)
        handle.create_dataset("symh", data=symh)
        handle.create_dataset("Interped", data=1 * is_interped)
        handle.create_dataset("f10.7", data=f107)
        handle.create_dataset("Kp", data=kp)
        handle.create_dataset("Bx0", data=bcoef[0])
        handle.create_dataset("ByC", data=bcoef[1])
        handle.create_dataset("BzC", data=bcoef[2])
        handle.create_dataset("Va", data=va * 1.0e3)
        handle.create_dataset("Cs", data=cs * 1.0e3)
        handle.create_dataset("xBS", data=bs_sm[:, 0])
        handle.create_dataset("yBS", data=bs_sm[:, 1])
        handle.create_dataset("zBS", data=bs_sm[:, 2])
        handle.create_dataset("Magnetosonic Mach", data=mfast)

    return output

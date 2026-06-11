from netCDF4 import Dataset
import numpy as np
from secsy.cubedsphere import CSgrid, CSprojection
from secsy import cubedsphere as cs
import pandas as pd
from SECSpy import SECS
from SECSpy.constants import RE
from apexpy import Apex
from SECSpy.utils import date2filename
import os
import time
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from polplot import Polarplot as polar
import json
import xarray as xr
from tqdm import tqdm
from SECSpy.Plotting_Tools import contour_labels, subplot_align, quiverkey_left_label

class inversion(object):
    def __init__(self, ezie_l2_file, event_time, inversion_repo= None, **sat_vector_kwargs) -> None:
        """Initialize an EZIE/SuperMAG SECS inversion workspace.

        Parameters
        ----------
        ezie_l2_file : str or path-like
            EZIE L2 netCDF file containing beam geolocation and spacecraft velocity.
        event_time : numpy.datetime64
            Central event time used to choose the EZIE footprint, data window, and
            Apex epoch.
        inversion_repo : str or path-like, optional, default: None
            Directory used for plots, saved state, and optional intermediate data.
            If ``None``, an event-time folder is created under ``.Inversions``.
        **sat_vector_kwargs
            Passed to :meth:`get_sat_vector`.

        Attributes
        ----------
        sat_heading : ndarray, shape (2,)
            Local north/east heading vector passed to ``CSprojection``.
        l1, l2 : float
            Default regularization weights used by :meth:`run_inversion`.
        SM_data, SML_data, amplitudes, estimated
            Data products populated by later workflow methods. These may be
            in-memory objects or file paths after saving/loading state.
        """
        self.ezie_l2_file = ezie_l2_file
        self.event_time = event_time
        if inversion_repo is None:
            self.inversion_repo= f'.Inversions/{date2filename(event_time)}/'
        else:
            self.inversion_repo= inversion_repo
        os.makedirs(self.inversion_repo, exist_ok=True)
        self.sat_heading= self.get_sat_vector(**sat_vector_kwargs)
        self.stations= None
        self.username= None
        self.l1= 3e3
        self.l2= 6e3
        self.poles= None
        self.A= Apex(pd.Timestamp(self.event_time.astype('datetime64[Y]')))
        self.SM_data= None
        self.SML_data= None
        self.amplitudes= None
        self.estimated= None
        self.SECS_kwargs= None
        
    def get_sat_vector(self, event_time_buffer=np.timedelta64(5, 'm')):
        """Estimate the cubed-sphere grid heading from EZIE spacecraft velocity.

        Parameters
        ----------
        event_time_buffer : numpy.timedelta64, default: numpy.timedelta64(5, 'm')
            Offset from ``self.event_time`` used when selecting the EZIE
            footprint and spacecraft velocity sample.

        Returns
        -------
        numpy.ndarray
            Two-element ``[North, East]`` unit vector for ``CSprojection``.

        Notes
        -----
        ``self.lon_center`` and ``self.lat_center`` are set from the beam 1
        footprint at the selected sample. The ECEF velocity is rotated into ENU
        coordinates before the horizontal heading is normalized.
        """
        ezie = Dataset(self.ezie_l2_file)
        print(event_time_buffer)
        try:
            time= ezie['Time']['time_utc'][:].astype('datetime64[ns]')
            ind= np.argmax(np.isfinite(ezie['Geolocation'][f'obs_lon{1}'][:].filled(np.nan))&\
                (np.arange(len(time))>=np.argmin(abs(time-(self.event_time+event_time_buffer)))))
            closest= {'time':time[ind]}
            closest.update({f'lon{i}':ezie['Geolocation'][f'obs_lon{i}'][:][ind] for i in range(1, 5)})
            closest.update({f'lat{i}':ezie['Geolocation'][f'obs_lat{i}'][:][ind] for i in range(1, 5)})
            closest.update({'sat_lat':ezie['Geolocation']['sat_lat'][:][ind],
                            'sat_lon':ezie['Geolocation']['sat_lon'][:][ind],
                            'sat_vel_ecef':ezie['Geolocation']['sat_vel_ecef'][:][ind]})
        finally:
            ezie.close()
        glon, glat= closest['lon1'], closest['lat1']

        vel_enu = self._ecef_to_enu(closest['sat_vel_ecef'], closest['sat_lat'], closest['sat_lon'])[:-1]
        East, North= vel_enu/np.linalg.norm(vel_enu)
        if North<0:
            North*=-1
        self.lon_center, self.lat_center= glon, glat
        # return np.rad2deg(np.arctan2(North, East))
        return np.array([North, East])

    def _ecef_to_enu(self, v_ecef, lat, lon):
        """Rotate a vector from ECEF coordinates into local east/north/up.

        Parameters
        ----------
        v_ecef : array-like, shape (3,)
            Vector in Earth-centered, Earth-fixed coordinates.
        lat, lon : float
            Geographic latitude and longitude in degrees.

        Returns
        -------
        numpy.ndarray
            Vector components in local east, north, and up coordinates.
        """
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        R = np.array([
            [-np.sin(lon),              np.cos(lon),             0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
            [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
        ])
        return R @ v_ecef

    def create_grid(self, magnetic_north=False, L=6000, W=6000, Lres=150., Wres=150., center_shift=(0, 0),
                    auto_center=False, auto_center_step=.25, auto_center_max=3., **grid_kwargs):
        """Create the SECS node grid and the staggered evaluation grid.

        Parameters
        ----------
        magnetic_north : bool, default: False
            If ``True``, align the grid with quasi-dipole magnetic north instead
            of the spacecraft track.
        L : float, default: 6000
            Grid length passed to ``CSgrid``.
        W : float, default: 6000
            Grid width passed to ``CSgrid``.
        Lres : float, default: 150.0
            Grid resolution along the length dimension.
        Wres : float, default: 150.0
            Grid resolution along the width dimension.
        center_shift : tuple of float, default: (0, 0)
            Longitude and latitude offset, in degrees, applied to the EZIE beam
            footprint selected by :meth:`get_sat_vector`.
        auto_center : bool, default: False
            If ``True``, search nearby center shifts when secsy cannot compute
            local-to-geographic vector rotations for the initial grid.
        auto_center_step : float, default: 0.25
            Center-shift search spacing in degrees.
        auto_center_max : float, default: 3.0
            Maximum absolute center shift searched in degrees.
        **grid_kwargs
            Additional options passed to ``CSgrid``.

        Attributes
        ----------
        node_grid : secsy.cubedsphere.CSgrid
            Grid used for SECS pole amplitudes.
        eval_grid : secsy.cubedsphere.CSgrid
            One-cell-larger grid used to evaluate modeled fields and currents.
        grid_lon_center, grid_lat_center : float
            Final grid center after applying ``center_shift`` or auto-centering.
        grid_center_shift : tuple of float
            Final longitude/latitude center shift.

        Raises
        ------
        ValueError
            If ``auto_center=True`` and no valid nearby center shift is found.
        """
        self.A= Apex(pd.Timestamp(self.event_time.astype('datetime64[Y]')))

        def build_grid(shift):
            """Build node and evaluation grids for a candidate center shift.

            Parameters
            ----------
            shift : tuple of float
                Longitude and latitude offsets, in degrees, from
                ``self.lon_center`` and ``self.lat_center``.

            Returns
            -------
            tuple
                ``(center_lon, center_lat, node_grid, eval_grid)``, where
                ``node_grid`` contains the SECS pole locations and ``eval_grid``
                is one cell larger for evaluating modeled fields and currents.
            """
            center_lon = self.lon_center + shift[0]
            center_lat = self.lat_center + shift[1]
            if magnetic_north:
                f1, f2= self.A.basevectors_qd(center_lat, center_lon%360, 0, coords='geo')
                qd_north = f2/np.linalg.norm(f2)
                East, North= qd_north[0], qd_north[1]
                grid_proj= CSprojection((center_lon, center_lat), (East, North))
            else:
                grid_proj= CSprojection((center_lon, center_lat), self.sat_heading)

            node_grid= CSgrid(grid_proj, L, W, Lres, Wres, **grid_kwargs)
            # The evaluation grid is one cell larger so fields can be evaluated
            # on cell edges while amplitudes remain on the SECS node grid.
            xi_e  = np.hstack((node_grid.xi_mesh[0]    , node_grid.xi_mesh [0 , - 1] + node_grid.dxi )) - node_grid.dxi /2
            eta_e = np.hstack((node_grid.eta_mesh[:, 0], node_grid.eta_mesh[-1,   0] + node_grid.deta)) - node_grid.deta/2
            eval_grid= CSgrid(grid_proj,
                                node_grid.L + node_grid.Lres, node_grid.W + node_grid.Wres,
                                node_grid.Lres, node_grid.Wres,
                                edges = (xi_e, eta_e), R = node_grid.R)
            return center_lon, center_lat, node_grid, eval_grid

        def valid_grid(node_grid):
            """Check whether secsy can rotate local grid vectors to ENU.

            Parameters
            ----------
            node_grid : secsy.cubedsphere.CSgrid
                Candidate node grid to validate.

            Returns
            -------
            bool
                ``True`` if ``local2geo_enu_rotation`` succeeds for all node
                locations; ``False`` if secsy raises an ``AssertionError``.
            """
            try:
                node_grid.projection.local2geo_enu_rotation(node_grid.local_lon.flatten(), node_grid.local_lat.flatten())
            except AssertionError:
                return False
            return True

        center_shift = tuple(center_shift)
        center_lon, center_lat, node_grid, eval_grid = build_grid(center_shift)
        if auto_center and not valid_grid(node_grid):
            # secsy can reject grids whose local ENU basis is numerically
            # inconsistent near projection singularities. Prefer the smallest
            # shift, with a slight bias away from the geographic pole.
            candidates = []
            steps = np.arange(-auto_center_max, auto_center_max + auto_center_step / 2, auto_center_step)
            for dlon in steps:
                for dlat in steps:
                    shift = (float(center_shift[0] + dlon), float(center_shift[1] + dlat))
                    if np.isclose(shift[0], center_shift[0]) and np.isclose(shift[1], center_shift[1]):
                        continue
                    distance = np.hypot(shift[0] - center_shift[0], shift[1] - center_shift[1])
                    pole_direction = -np.sign(self.lat_center) * (shift[1] - center_shift[1])
                    candidates.append((distance, -pole_direction, shift))
            candidates.sort(key=lambda item: (item[0], item[1], abs(item[2][0]), abs(item[2][1])))

            original_diagnostics = self._local2geo_rotation_diagnostics_for_grid(node_grid, center_lon, center_lat)
            for _, _, shift in candidates:
                candidate_lon, candidate_lat, candidate_node_grid, candidate_eval_grid = build_grid(shift)
                if valid_grid(candidate_node_grid):
                    print('Adjusted grid center to avoid secsy local2geo_enu_rotation assertion.')
                    print(original_diagnostics)
                    print(f'Using center_shift={shift} instead of {center_shift}.')
                    center_shift = shift
                    center_lon, center_lat = candidate_lon, candidate_lat
                    node_grid, eval_grid = candidate_node_grid, candidate_eval_grid
                    break
            else:
                self.grid_lon_center = center_lon
                self.grid_lat_center = center_lat
                self.grid_center_shift = center_shift
                self.node_grid = node_grid
                self.eval_grid = eval_grid
                raise ValueError(
                    original_diagnostics
                    + f'\nNo valid center shift found within +/-{auto_center_max} degrees at {auto_center_step} degree resolution.'
                )

        self.grid_lon_center = center_lon
        self.grid_lat_center = center_lat
        self.grid_center_shift = center_shift
        self.node_grid = node_grid
        self.eval_grid = eval_grid
    def create_SECS(self, **SECS_kwargs):
        """Create SECS basis functions for the current node and evaluation grids.

        Parameters
        ----------
        **SECS_kwargs
            Keyword arguments passed to ``SECSpy.SECS``. Missing defaults are
            ``mode='image'``, ``current_radius=RE + 110e3``, and
            ``image_current_radius=RE - 500e3``.

        Attributes
        ----------
        poles : SECSpy.SECS
            SECS object initialized with node-grid pole locations and eval-grid
            evaluation locations.
        SECS_kwargs : dict
            Copy of the keyword arguments used to create ``self.poles``.

        Notes
        -----
        Missing options are filled with the defaults used by the EZIE workflow:
        image currents at ``RE - 500 km`` and ionospheric currents at
        ``RE + 110 km``.
        """
        if not 'mode' in SECS_kwargs:
            SECS_kwargs['mode']= 'image'
        if not 'current_radius' in SECS_kwargs:
            SECS_kwargs['current_radius']= RE+110e3
        if not 'image_current_radius' in SECS_kwargs:
            SECS_kwargs['image_current_radius']= RE-500e3
        self.SECS_kwargs= SECS_kwargs.copy()
        self.poles= SECS(self.node_grid.lon.flatten(), self.node_grid.lat.flatten(), self.eval_grid.lon.flatten(), self.eval_grid.lat.flatten(), **SECS_kwargs)

    def _json_safe(self, value):
        """Convert numpy, pandas, and nested values to JSON-serializable types.

        Parameters
        ----------
        value : object
            Value to convert.

        Returns
        -------
        object
            JSON-compatible value with numpy arrays/scalars, datetimes, and
            timedeltas converted to builtin Python types.
        """
        if isinstance(value, dict):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(val) for val in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit='ns')
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.timedelta64):
            return int(value.astype('timedelta64[ns]').astype(int))
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _save_file_backed_state(self, value, filename, file_type):
        """Return a file path for state data, saving in-memory objects first.

        Parameters
        ----------
        value : pandas.DataFrame, xarray.Dataset, str, path-like, or None
            State object or existing file path.
        filename : str
            Filename to use inside ``self.inversion_repo`` when ``value`` is an
            in-memory object.
        file_type : {'csv', 'netcdf'}
            Serialization format expected for in-memory objects.

        Returns
        -------
        str or None
            Absolute file path, or ``None`` when ``value`` is ``None``.

        Raises
        ------
        TypeError
            If ``value`` cannot be serialized as ``file_type``.
        """
        if value is None:
            return None
        if isinstance(value, (str, os.PathLike)):
            return os.path.abspath(value)

        path= os.path.abspath(os.path.join(self.inversion_repo, filename))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if file_type == 'csv' and isinstance(value, pd.DataFrame):
            tmp_path = path + '.tmp'
            value.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
            return path
        if file_type == 'netcdf' and isinstance(value, xr.Dataset):
            source = value.encoding.get('source')
            if source is not None and os.path.abspath(os.fspath(source)) == path:
                return path
            tmp_path = path + '.tmp'
            value.to_netcdf(tmp_path)
            os.replace(tmp_path, path)
            return path

        raise TypeError(f'Cannot save {filename}: unsupported object type {type(value)}')

    def _repo_path(self, filename, prefix=None):
        """Resolve optional output paths inside ``self.inversion_repo``.

        Parameters
        ----------
        filename : str
            Base filename.
        prefix : str, bool, or None, default: None
            If ``None`` or ``False``, saving is disabled and ``None`` is
            returned. If ``True``, no prefix is added. Otherwise ``prefix`` is
            prepended to ``filename``.

        Returns
        -------
        str or None
            Repository path, or ``None`` when output is disabled.
        """
        if prefix is None or prefix is False:
            return None
        if prefix is True:
            prefix= ''
        return os.path.join(self.inversion_repo, f'{prefix}{filename}')

    def save_state(self, save_file=None):
        """Save enough metadata and file paths to recreate this inversion object.

        Parameters
        ----------
        save_file : str or path-like, optional, default: None
            JSON state path. If ``None``, ``inversion_state.json`` is written in
            ``self.inversion_repo``.

        Returns
        -------
        dict
            JSON-safe state dictionary written to disk.

        Notes
        -----
        Large tables and xarray datasets are stored as CSV/netCDF files in the
        inversion repository. The JSON state records their paths along with grid
        dictionaries, regularization values, and SECS keyword arguments.
        """
        if save_file is None:
            save_file= os.path.join(self.inversion_repo, 'inversion_state.json')
        save_file= os.path.abspath(save_file)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        SM_data_path= self._save_file_backed_state(self.SM_data, 'magnetometer_data.csv', 'csv')
        SML_data_path= self._save_file_backed_state(self.SML_data, 'SML_data.csv', 'csv')
        amplitudes_path= self._save_file_backed_state(self.amplitudes, 'amplitudes.ncdf4', 'netcdf')
        estimated_path= self._save_file_backed_state(self.estimated, 'estimated.ncdf4', 'netcdf')
        stations_path= self._save_file_backed_state(self.stations, 'stations.csv', 'csv')

        state= {
            'state_file': save_file,
            'ezie_l2_file': os.path.abspath(self.ezie_l2_file) if self.ezie_l2_file is not None else None,
            'event_time': self.event_time,
            'inversion_repo': os.path.abspath(self.inversion_repo),
            'sat_heading': self.sat_heading,
            'lon_center': getattr(self, 'lon_center', None),
            'lat_center': getattr(self, 'lat_center', None),
            'username': self.username,
            'l1': self.l1,
            'l2': self.l2,
            'SECS_kwargs': self.SECS_kwargs,
            'files': {
                'SM_data': SM_data_path,
                'SML_data': SML_data_path,
                'amplitudes': amplitudes_path,
                'estimated': estimated_path,
                'stations': stations_path,
            },
            'grids': {
                'node_grid': self.node_grid.to_dictionary() if hasattr(self, 'node_grid') else None,
                'eval_grid': self.eval_grid.to_dictionary() if hasattr(self, 'eval_grid') else None,
            },
        }
        state= self._json_safe(state)

        with open(save_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f'state saved: {save_file}')
        return state

    @staticmethod
    def _load_csv_state(path):
        """Load a saved CSV state table.

        Parameters
        ----------
        path : str, path-like, or None
            CSV file path.

        Returns
        -------
        pandas.DataFrame or None
            Loaded table, with ``Date_UTC`` converted to datetime if present.
        """
        if path is None:
            return None
        data= pd.read_csv(path)
        if 'Date_UTC' in data.columns:
            data['Date_UTC']= pd.to_datetime(data['Date_UTC']).values
        return data

    @staticmethod
    def _load_netcdf_state(path):
        """Load a netCDF state dataset into memory and close the file handle.

        Parameters
        ----------
        path : str, path-like, or None
            netCDF file path.

        Returns
        -------
        xarray.Dataset or None
            Loaded dataset, or ``None`` if ``path`` is ``None``.
        """
        if path is None:
            return None
        data= xr.open_dataset(path)
        loaded= data.load()
        data.close()
        return loaded
    @staticmethod
    def _load_file_path_state(path):
        """Return an absolute path for file-backed state entries.

        Parameters
        ----------
        path : str, path-like, or None
            File path stored in a state file.

        Returns
        -------
        str or None
            Absolute path, or ``None`` if no path was saved.
        """
        if path is None:
            return None
        return os.path.abspath(path)
    @staticmethod
    def _mlt_contour_fmt(mlt):
        """Format magnetic local time contour labels.

        Parameters
        ----------
        mlt : float
            Magnetic local time in hours.

        Returns
        -------
        str
            Two-line contour label, for example ``'03\\nMLT'``.
        """
        if mlt < 0:
            mlt+= 24
        return f'{int(mlt):02d}\nMLT'

    def _local2geo_rotation_diagnostics_for_grid(self, node_grid, center_lon, center_lat):
        """Build a detailed diagnostic message for secsy projection failures.

        Parameters
        ----------
        node_grid : secsy.cubedsphere.CSgrid
            Grid whose projection failed ``local2geo_enu_rotation``.
        center_lon, center_lat : float
            Geographic longitude and latitude of the grid center.

        Returns
        -------
        str
            Multi-line diagnostic message with grid ranges and worst residuals.
        """
        lon = node_grid.local_lon.flatten()
        lat = node_grid.local_lat.flatten()
        projection = node_grid.projection
        d2r = np.pi / 180

        th = (90 - np.array(lat).flatten()) * d2r
        ph = np.array(lon).flatten() * d2r
        e_R = np.vstack((-np.sin(ph), np.cos(ph), np.zeros_like(ph))).T
        n_R = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th))).T
        u_R = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).T
        R_enulocal2eceflocal = np.stack((e_R, n_R, u_R), axis=2)

        lon_G, lat_G = projection.local2geo(lon, lat)
        th = (90 - lat_G) * d2r
        ph = lon_G * d2r
        e_G = np.vstack((-np.sin(ph), np.cos(ph), np.zeros_like(ph))).T
        n_G = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th))).T
        u_G = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).T
        R_ecefgeo2enugeo = np.stack((e_G, n_G, u_G), axis=1)

        # Reproduce secsy's rotation consistency check so the error message can
        # identify where and how badly the local vertical leaks into east/north.
        R_enulocal2ecefgeo = np.einsum('ij , njk -> nik', projection.R_local2geo, R_enulocal2eceflocal)
        R = np.einsum('nij, njk -> nik', R_ecefgeo2enugeo, R_enulocal2ecefgeo)
        residual = np.abs(R[:, 2, [0, 1]]).max(axis=1)
        bad = np.where(residual > 1e-6)[0]
        worst = bad[np.argsort(residual[bad])[-5:]][::-1] if bad.size else np.array([], dtype=int)

        lines = [
            'secsy failed while converting the cubed-sphere grid derivatives to geographic east/north.',
            'The grid/projection is internally inconsistent for local2geo_enu_rotation().',
            f'grid shape: {node_grid.shape}, size: {node_grid.size}',
            f'EZIE center lon/lat: {getattr(self, "lon_center", None)}, {getattr(self, "lat_center", None)}',
            f'grid center lon/lat: {center_lon}, {center_lat}',
            f'satellite heading: {getattr(self, "sat_heading", None)}',
            f'geo lon range: {np.nanmin(node_grid.lon):.6g} to {np.nanmax(node_grid.lon):.6g}',
            f'geo lat range: {np.nanmin(node_grid.lat):.6g} to {np.nanmax(node_grid.lat):.6g}',
            f'local lon range: {np.nanmin(lon):.6g} to {np.nanmax(lon):.6g}',
            f'local lat range: {np.nanmin(lat):.6g} to {np.nanmax(lat):.6g}',
            f'max vertical/east-north residual: {residual.max():.6g}',
            f'failing grid points: {bad.size}',
        ]
        for idx in worst:
            lines.append(
                f'  index {idx}: local lon/lat=({lon[idx]:.6g}, {lat[idx]:.6g}), '
                f'geo lon/lat=({lon_G[idx]:.6g}, {lat_G[idx]:.6g}), residual={residual[idx]:.6g}'
            )
        lines.append('Try recreating the grid with auto_center=True, a manual center_shift, smaller L/W, or magnetic_north=True before run_inversion().')
        return '\n'.join(lines)

    def _local2geo_rotation_diagnostics(self):
        """Return diagnostics for the object's current node grid.

        Returns
        -------
        str
            Multi-line diagnostic message for ``self.node_grid``.
        """
        center_lon = getattr(self, "grid_lon_center", getattr(self, "lon_center", None))
        center_lat = getattr(self, "grid_lat_center", getattr(self, "lat_center", None))
        return self._local2geo_rotation_diagnostics_for_grid(self.node_grid, center_lon, center_lat)

    @classmethod
    def load_state(cls, save_file):
        """Recreate an inversion object from a JSON state file.

        Parameters
        ----------
        save_file : str or path-like
            JSON file produced by :meth:`save_state`.

        Returns
        -------
        inversion
            Reconstructed inversion object.

        Notes
        -----
        File-backed arrays are restored as paths rather than eagerly loaded so
        plotting and inversion methods can load only the data they need.
        """
        save_file= os.path.abspath(save_file)
        with open(save_file, 'r') as f:
            state= json.load(f)

        inv= cls.__new__(cls)
        inv.state_file= save_file
        inv.ezie_l2_file= state.get('ezie_l2_file')
        inv.event_time= np.datetime64(state['event_time']) if state.get('event_time') is not None else None
        inv.SML_data= None
        inv.inversion_repo= state.get('inversion_repo', os.path.dirname(save_file))
        os.makedirs(inv.inversion_repo, exist_ok=True)

        inv.sat_heading= state.get('sat_heading')
        inv.lon_center= state.get('lon_center')
        inv.lat_center= state.get('lat_center')
        inv.username= state.get('username')
        inv.l1= state.get('l1', 1e2)
        inv.l2= state.get('l2', 1e1)
        inv.SECS_kwargs= state.get('SECS_kwargs')
        inv.poles= None

        files= state.get('files', {})
        inv.SM_data= cls._load_file_path_state(files.get('SM_data'))
        inv.SML_data= cls._load_file_path_state(files.get('SML_data'))
        inv.stations= cls._load_file_path_state(files.get('stations'))
        inv.amplitudes= cls._load_file_path_state(files.get('amplitudes'))
        inv.estimated= cls._load_file_path_state(files.get('estimated'))

        grids= state.get('grids', {})
        inv.node_grid= cs.from_dictionary(grids['node_grid']) if grids.get('node_grid') is not None else None
        inv.eval_grid= cs.from_dictionary(grids['eval_grid']) if grids.get('eval_grid') is not None else None

        inv.A= Apex(pd.Timestamp(inv.event_time.astype('datetime64[Y]'))) if inv.event_time is not None else None
        if inv.node_grid is not None and inv.eval_grid is not None and inv.SECS_kwargs is not None:
            inv.poles= SECS(inv.node_grid.lon.flatten(), inv.node_grid.lat.flatten(),
                            inv.eval_grid.lon.flatten(), inv.eval_grid.lat.flatten(),
                            **inv.SECS_kwargs)

        return inv
    def download_data(self, username=None, grid_buffer=(.1, .1),
                      start= -np.timedelta64(1, 'h'), end= np.timedelta64(1, 'h'), save='', auto_remove=False):
        """Download and prepare SuperMAG magnetometer and SML data.

        Parameters
        ----------
        username : str, optional, default: None
            SuperMAG username. If ``None``, the previously stored
            ``self.username`` is used.
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
        if self.stations is None or 'xi' not in self.stations.columns:
            stations_path = _default_stations_path()
            stations = pd.read_csv(stations_path, names=_STATION_COLUMNS, skiprows=1)
            stations['xi'], stations['eta']= self.node_grid.projection.geo2cube(stations.GEOLON, stations.GEOLAT)
            self.stations= stations
        if username is not None:
            self.username= username
        elif self.username is None:
            raise ValueError('Username is required to download SuperMAG data.')
        else:
            username= self.username
        extent= (end-start + np.timedelta64(1, 'm')).astype('timedelta64[s]').astype(int)
        start = self.event_time + start
        xi_min=self.node_grid.xi_min.copy()
        xi_max=self.node_grid.xi_max.copy()
        xi_min-=grid_buffer[0]*xi_min
        xi_max+=grid_buffer[0]*xi_max
        eta_min=self.node_grid.eta_min.copy()
        eta_max=self.node_grid.eta_max.copy()
        eta_min-=grid_buffer[1]*eta_min
        eta_max+=grid_buffer[1]*eta_max
        self.SM_data= SuperMAGGetData(username, start-np.timedelta64(1, 'm'), extent, ['GEO', 'decl'], f'xi>={xi_min} & xi<={xi_max} & eta>={eta_min} & eta<={eta_max}', stations=self.stations)[1]
        self.SM_data.sort_values(['Date_UTC', 'glat'], inplace=True)
        self.SM_data.index= list(range(len(self.SM_data)))
        decl_rad = np.deg2rad(self.SM_data['decl'])
        # self.SM_data['N_geo'] = self.SM_data['N_nez'] * np.cos(decl_rad) - self.SM_data['E_nez'] * np.sin(decl_rad)
        # self.SM_data['E_geo'] = self.SM_data['N_nez'] * np.sin(decl_rad) + self.SM_data['E_nez'] * np.cos(decl_rad)
        self.SM_data['xi'], self.SM_data['eta']= self.node_grid.projection.geo2cube(self.SM_data.glon, self.SM_data.glat)
        self.SM_data.replace(999999, np.nan, inplace=True)
        self.SM_data.dropna(inplace=True)
        self.SM_data.index= list(range(len(self.SM_data)))
        # Rotate measured east/north perturbations into the cubed-sphere basis
        # used by the SECS forward model and plotting routines.
        self.SM_data['Be_cs'], self.SM_data['Bn_cs']= self.node_grid.projection.vector_cube_projection(
            self.SM_data.E_geo*1e-9,
            self.SM_data.N_geo*1e-9,
            self.SM_data.glon,
            self.SM_data.glat,
            return_xi_eta=False,
        )
        vals, counts= np.unique(self.SM_data.iaga.values, return_counts=True)
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
                self.SM_data= self.SM_data.loc[~self.SM_data.iaga.isin(sites)]
                # self.SM_data.to_csv(folder+'magnetometer_data.csv', index=False)
            else:
                raise ValueError('inconsistent availability of magnetometers\n'+'\n'.join([f'{v}:{c}' for v, c in zip(vals, counts)]))
        save_path= self._repo_path('magnetometer_data.csv', save)
        if save_path is not None:
            self.SM_data.to_csv(save_path, index=False)
            self.SM_data= save_path

        self.SML_data= SuperMAGGetIndices(username, start-np.timedelta64(1, 'm'), extent, ['SML'])[1].sort_values('Date_UTC')
        save_path= self._repo_path('SML_data.csv', save)
        if save_path is not None:
            self.SML_data.to_csv(save_path, index=False)
            self.SML_data= save_path


    def run_inversion(self, l1=None, l2=None, start=-np.timedelta64(30, 'm'), end=np.timedelta64(30, 'm'), save='',
                      block_rows=100):
        """Run the divergence-free SECS inversion over a minute cadence window.

        Parameters
        ----------
        l1 : float, optional, default: None
            Magnetic-east derivative regularization weight. If ``None``, use
            ``self.l1``.
        l2 : float, optional, default: None
            Cubed-sphere nearest-neighbor roughness regularization weight. If
            ``None``, use ``self.l2``.
        start : numpy.timedelta64, default: numpy.timedelta64(-30, 'm')
            Start of the inversion window relative to ``self.event_time``.
        end : numpy.timedelta64, default: numpy.timedelta64(30, 'm')
            End of the inversion window relative to ``self.event_time``.
        save : str, bool, or None, default: ''
            Prefix/path control for optional netCDF output through
            :meth:`_repo_path`. Use ``None`` or ``False`` to keep datasets in
            memory.
        block_rows : int, default: 100
            Row block size used while forming the magnetic-east regularization
            matrix to reduce peak memory use.

        Returns
        -------
        tuple
            ``(self.amplitudes, self.estimated)`` as xarray datasets or saved
            file paths, depending on ``save``.

        Attributes
        ----------
        amplitudes : xarray.Dataset or str
            SECS amplitudes and grid-cell areas on ``self.node_grid``.
        estimated : xarray.Dataset or str
            Modeled magnetic perturbations and equivalent currents on
            ``self.eval_grid``.

        Raises
        ------
        ValueError
            If secsy cannot compute grid derivative rotations for the current
            grid projection.
        """
        if l1 is not None:
            self.l1= l1
        if l2 is not None:
            self.l2= l2
        l1= self.l1
        l2= self.l2
        
        try:
            Le, Ln=self.node_grid.get_Le_Ln()
        except AssertionError as exc:
            raise ValueError(self._local2geo_rotation_diagnostics()) from exc
        node_f1, node_f2= self.A.basevectors_qd(self.node_grid.lat.flatten(), self.node_grid.lon.flatten(), 110, coords='geo')
        e= node_f1/np.linalg.norm(node_f1, axis=0)
        n= node_f2/np.linalg.norm(node_f2, axis=0)

        n_nodes= Le.shape[1]
        L_eTL_e= np.zeros((n_nodes, n_nodes), dtype=np.result_type(Le, Ln, e))
        # Build the magnetic-east derivative penalty in row blocks. This keeps
        # the dense normal matrix accumulation but avoids materializing the full
        # projected derivative operator at once.
        for row_start in tqdm(range(0, Le.shape[0], block_rows), desc='Building lambda2', unit='block'):
            rows= slice(row_start, row_start+block_rows)
            L_e_block= e[0, rows, None]*Le[rows] + e[1, rows, None]*Ln[rows]
            L_eTL_e+= L_e_block.T@L_e_block
        L_eTL_e*= l1/np.max(np.abs(L_eTL_e))
        λ2= L_eTL_e
        λ3= l2

        block_size, block_count= self.node_grid.shape[0], self.node_grid.shape[-1]
        n_nodes= block_size*block_count
        L_cube= np.zeros((n_nodes, n_nodes))
        # Graph-Laplacian roughness penalty over neighboring cubed-sphere nodes.
        # The first loop links rows within each block; the second links columns
        # between adjacent blocks.
        if block_size > 1:
            block_diag= np.full(block_size, 2.)
            block_diag[[0, -1]]= 1.
            for block in tqdm(range(block_count), desc='Building L_cube rows', unit='block'):
                offset= block*block_size
                idx= offset + np.arange(block_size)
                L_cube[idx, idx]+= block_diag
                L_cube[idx[:-1], idx[1:]]-= 1.
                L_cube[idx[1:], idx[:-1]]-= 1.
        stride= self.node_grid.shape[1]
        for i in tqdm(range(n_nodes-stride), desc='Building L_cube columns', unit='link'):
            j= i+stride
            L_cube[i, i]+= 1.
            L_cube[j, j]+= 1.
            L_cube[i, j]-= 1.
            L_cube[j, i]-= 1.
        L_cube*= λ3
        if isinstance(self.SM_data, str):
            SM_path= self.SM_data
            self.SM_data= pd.read_csv(self.SM_data, parse_dates=['Date_UTC'])
        else:
            SM_path= None


        MagLon, MagLat= self.SM_data.loc[self.SM_data.Date_UTC==self.SM_data.iloc[0].Date_UTC, ['glon', 'glat']].values.T
        G_df= self.poles.G_Matrix(MagLon, MagLat)
        GTG= G_df.T@G_df
        scale= np.median(GTG)
        matrix = GTG+ λ2*scale + L_cube*scale
        Inverse= self.poles.Inverse_Matrix(matrix, cond=0)
        # The station geometry is fixed for the event, so the inverse and
        # evaluation matrices can be reused for every minute in the window.
        self.poles.fitting_matrix['divergence-free']= np.dot(Inverse, G_df.T)
        GJE_df, GJN_df= self.poles.eval_G_Matrix_J()
        (GBR_df, GBE_df, GBN_df), (GBR2, GBE2, GBN2)= self.poles.eval_G_Matrix_B()
        GBR_df+=GBR2
        GBE_df+=GBE2
        GBN_df+=GBN2
        amps= []
        estimated= []

        for date in tqdm(np.arange(self.event_time+start, self.event_time+end+np.timedelta64(1, 'm'), np.timedelta64(1, 'm')), desc='Running inversion'):
            East_ground= self.SM_data[self.SM_data.Date_UTC==date].E_geo.values*1e-9
            South_ground=-self.SM_data[self.SM_data.Date_UTC==date].N_geo.values*1e-9
            Up_ground= -self.SM_data[self.SM_data.Date_UTC==date].Z_geo.values*1e-9

            d_df= np.concatenate((South_ground, East_ground, Up_ground))
            DF_amps = self.poles.fitting_matrix['divergence-free']@d_df

            # SECS amplitudes live on the node grid and are later divided by
            # cell area for equivalent current-density style plots.
            amps_= xr.Dataset(data_vars=dict(Amplitude_df=(['time', 'xi', 'eta'], [DF_amps.reshape(self.node_grid.shape).T], {'units': 'A'}),
                                            Area=(['xi', 'eta'], self.node_grid.A.T, {'units': 'm^2'})), 
                    coords= dict(xi=self.node_grid.xi[0,:],
                                    eta=self.node_grid.eta[:,0],
                                    lon=(['xi', 'eta'], self.node_grid.lon.T),
                                    lat=(['xi', 'eta'], self.node_grid.lat.T),
                                    time=[date.astype('datetime64[ns]')]),
                    attrs={'cs_grid': json.dumps(self.node_grid.to_dictionary())})

            BE, BN= self.node_grid.projection.vector_cube_projection(GBE_df@DF_amps, GBN_df@DF_amps, self.poles.eval_longitude, 
                                                                self.poles.eval_latitude, return_xi_eta=False)
            BE_df, BN_df, BU_df= BE.reshape(self.eval_grid.shape), BN.reshape(self.eval_grid.shape), (GBR_df@DF_amps).reshape(self.eval_grid.shape)



            JE, JN= self.node_grid.projection.vector_cube_projection(GJE_df@DF_amps, GJN_df@DF_amps, self.poles.eval_longitude, 
                                                                self.poles.eval_latitude, return_xi_eta=False)
            JE_df, JN_df= JE.reshape(self.eval_grid.shape), JN.reshape(self.eval_grid.shape)

            # Modeled magnetic perturbations and equivalent current are evaluated
            # on the larger staggered grid for plotting and comparison.
            estimated_= xr.Dataset(data_vars=dict(Je_df=(['time', 'xi', 'eta'], [JE_df.T], {'units':'A/m'}),
                                                Jn_df=(['time', 'xi', 'eta'], [JN_df.T], {'units':'A/m'}),
                                        Be_df=(['time', 'xi', 'eta'], [BE_df.T], {'units':'T'}),
                                        Bn_df=(['time', 'xi', 'eta'], [BN_df.T], {'units':'T'}),
                                        Bu_df=(['time', 'xi', 'eta'], [BU_df.T], {'units':'T'}),),
                            coords=dict(xi= self.eval_grid.xi[0, :],
                                        eta=self.eval_grid.eta[:,0],
                                    lon=(['xi', 'eta'], self.eval_grid.lon.T),
                                        lat=(['xi', 'eta'], self.eval_grid.lat.T),
                                        time=[date.astype('datetime64[ns]')]),
                    attrs={'cs_grid': json.dumps(self.eval_grid.to_dictionary())})
            amps.append(amps_)
            estimated.append(estimated_)
        self.amplitudes= xr.concat(amps, dim='time', data_vars='all')
        self.estimated= xr.concat(estimated, dim='time', data_vars='all')
        if SM_path is not None:
            self.SM_data= SM_path
        amps_path= self._repo_path('amps.ncdf4', save)
        estimated_path= self._repo_path('estimated.ncdf4', save)
        if amps_path is not None:
            self.amplitudes.to_netcdf(amps_path)
            self.amplitudes= amps_path
            self.estimated.to_netcdf(estimated_path)
            self.estimated= estimated_path
        return self.amplitudes, self.estimated
    def plot_pass(self, save='', show_site_code=True):
        """Plot the EZIE beam tracks, SECS grid, and SuperMAG stations in MLT.

        Parameters
        ----------
        save : str, bool, or None, default: ''
            Prefix/path control for optional ``EZIE_Pass.png`` output through
            :meth:`_repo_path`. Use ``None`` or ``False`` to skip saving.
        show_site_code : bool, default: True
            If ``True``, annotate SuperMAG station markers with IAGA codes.

        Returns
        -------
        matplotlib.figure.Figure
            Polar MLT figure.
        """
        fig= plt.figure(figsize=(20, 20))
        ax= polar(fig.add_subplot(), color='grey', linewidth=2)
        ezie = Dataset(self.ezie_l2_file)
        time= ezie['Time']['time_utc'][:].astype('datetime64[ns]')
        for beam, color in zip(range(1, 5), ['orange', 'green', 'red', 'blue']):
            time, lon, lat= ezie['Time']['time_utc'][:].astype('datetime64[ns]'), ezie['Geolocation'][f'obs_lon{beam}'][:], ezie['Geolocation'][f'obs_lat{beam}'][:]
            ind= abs(time-self.event_time)<np.timedelta64(10, 'm')
            alt= 80
            lon, lat= lon.filled(np.nan), lat.filled(np.nan)
            mlat, mlt= self.A.convert(lat, lon, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time), height=alt)
            p= ax.plot(mlat[ind], mlt[ind], label=f'beam {beam}', color=color, zorder=100, marker='d', lw=2)
        ezie.close()

        mlat, mlt= self.A.convert(self.node_grid.lat, self.node_grid.lon, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time))
        ax.scatter(mlat, mlt, color='red', marker='+', label='SECS Grid', zorder=50, alpha=.5)
        ax.coastlines(mag=self.A, color='black', zorder=-1, time=pd.Timestamp(self.event_time))
        ax.writeLTlabels(size=40)
        ax.writeLATlabels(lt=9, size=40, color='black', bbox= {'facecolor':'white', 'edgecolor':'none', 'alpha':.7, 'zorder':2})
        # ax.plot(*self.A.convert(self.node_grid.lat[:, -1], self.node_grid.lon[:, -1], source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time)),
        #         color='grey')
        ax.scatter(*self.A.convert(self.lat_center, self.lon_center, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time)), s=1000,
                   label='grid center')
        
        if self.SM_data is not None:
            if isinstance(self.SM_data, str):
                SM_data= pd.read_csv(self.SM_data, parse_dates=['Date_UTC'])
                ind= SM_data.Date_UTC==self.event_time
                lon, lat= SM_data.glon[ind], SM_data.glat[ind]
                if show_site_code:
                    for i, (lo, la, site) in SM_data.loc[ind, ['glon', 'glat', 'iaga']].iterrows():
                        lt, la= self.A.convert(la, lo, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time))
                        ax.text(lt, la, text=site, zorder=1000, size=20, color='white',
                                    path_effects=[pe.withStroke(linewidth=3, foreground='black')])
            elif isinstance(self.SM_data, pd.DataFrame):
                ind= self.SM_data.Date_UTC==self.event_time
                lon, lat= self.SM_data.glon[ind], self.SM_data.glat[ind]
                if show_site_code:
                    for i, (lo, la, site) in self.SM_data.loc[ind, ['glon', 'glat', 'iaga']].iterrows():
                        lt, la= self.A.convert(la, lo, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time))
                        ax.text(lt, la, text=site, zorder=1000, size=20, color='white',
                                    path_effects=[pe.withStroke(linewidth=3, foreground='black')])
            lon= np.asarray(lon, dtype=float)
            lat= np.asarray(lat, dtype=float)
            mlat, mlt= self.A.convert(lat, lon, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time))
            mlat= np.asarray(mlat, dtype=float)
            mlt= np.asarray(mlt, dtype=float)
            ax.scatter(mlat, mlt, color='darkorange', marker='h', label='SuperMAG Stations', zorder=100, s=100, edgecolor='black')
        else:
            print('No SuperMAG data')
        
        

        ax.ax.legend(fontsize=30, ncols=3)
        save_path= self._repo_path('EZIE_Pass.png', save)
        if save_path is not None:
            fig.savefig(save_path)
        return fig

    def create_CS_figure(self, figsize=(40, 20), Br_lim=500, secs_lim=2, **grid_spec_kwargs):
        """Create the reusable three-panel Cartesian event figure.

        Parameters
        ----------
        figsize : tuple of float, default: (40, 20)
            Matplotlib figure size in inches.
        Br_lim : float, default: 500
            Symmetric radial magnetic perturbation color limit in nT.
        secs_lim : float, default: 2
            Symmetric equivalent current color limit used by the current panel.
        **grid_spec_kwargs
            Extra keyword arguments passed to ``Figure.add_gridspec``. Defaults
            are ``hspace=0``, ``wspace=0.3``, and
            ``height_ratios=[1, 0.1]`` when not supplied.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with magnetic-field, current, SML, colorbar, and legend axes.
        """
        if 'hspace' not in grid_spec_kwargs:
            grid_spec_kwargs['hspace']= 0
        if 'wspace' not in grid_spec_kwargs:
            grid_spec_kwargs['wspace']= .3
        if 'height_ratios' not in grid_spec_kwargs:
            grid_spec_kwargs['height_ratios']= [1, .1]
        fig= plt.figure(figsize=figsize)
        gs= fig.add_gridspec(2, 3,**grid_spec_kwargs)

        mag_ax= fig.add_subplot(gs[0, 0])
        mag_ax.set_title('Magnetic Field', size=50)
        mag_ax.Q_B= None
        mag_ax.Q_B2= None
        mag_ax.Br= None
        mag_ax.Magqk1=None
        mag_ax.Mag_scatter= None
        mag_ax.text1= None
        mag_ax.text2= None
        mag_ax.coastlines= []
        mag_ax.clon= []
        mag_ax.clat= []
        mag_cax= fig.add_subplot(gs[1, 0])
        cbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-Br_lim, vmax=Br_lim), cmap='seismic'), cax=mag_cax, orientation='horizontal',
                     extend='both')
        cbar.set_label(label='Br [nT]', size=30)

        secs_ax= fig.add_subplot(gs[0, 1])
        secs_ax.set_title('Current', size=50)
        secs_ax.ezie_pass= []
        secs_ax.Q_J= None
        secs_ax.Curqk= None
        secs_ax.A= None
        secs_ax.coastlines= []
        secs_ax.clon= []
        secs_ax.clat= []
        secs_cax= fig.add_subplot(gs[1, 1])
        cbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-secs_lim, vmax=secs_lim), cmap='seismic'), cax=secs_cax,
                     orientation='horizontal', 
                     extend='both')
        cbar.set_label(label=r'Equivalent FAC [$\mu/m^{2}$]', size=30)

        sml_ax= fig.add_subplot(gs[0, 2])
        sml_ax.set_title('SML Index', size=50)
        sml_ax.set_ylabel('Index [nT]', size=40, labelpad=-15)
        sml_ax.set_xlabel('Universal Time', size=40, labelpad=40)
        ind= (self.SML_data.Date_UTC>self.event_time-np.timedelta64(1, 'h'))&\
            (self.SML_data.Date_UTC<self.event_time+np.timedelta64(1, 'h'))

        sml_ax.plot(self.SML_data.Date_UTC[ind], self.SML_data.SML[ind])
        sml_ax.axhline(0, color='black', lw=3)
        sml_ax.axvline(self.event_time, color='darkred', linestyle='-', lw=8, label='Event Time', zorder=10)



        leg_ax= fig.add_subplot(gs[1, 2])
        leg_ax.axis('off')
        leg_ax.leg= None
        for ax in [mag_ax, secs_ax]:
            ax.set_xlim(self.node_grid.xi_min, self.node_grid.xi_max)
            ax.set_ylim(self.node_grid.eta_min, self.node_grid.eta_max)
            ax.set_aspect('equal')

        subplot_align(sml_ax, secs_ax, dim='y')
        for ax, cax in zip([mag_ax, secs_ax, sml_ax], [mag_cax, secs_cax, leg_ax]):
            pos= ax.get_position()
            pos2= cax.get_position()
            cax.set_position([pos2.x0, pos.y0-.2, pos.width, pos2.height])
            subplot_align(cax, ax, dim='x')
        for ax in [mag_ax, secs_ax]:
            ax.coastlines= self.node_grid.add_natural_earth_feature(ax, 'coastline', resolution='10m')
            ax.axis('off')
        sml_ax.spines[['top', 'right']].set_visible(False)
        sml_ax.tick_params(labelsize=30)
        for ax in [mag_cax, secs_cax]:
            ax.tick_params(labelsize=30)
        import matplotlib.dates as mdates
        sml_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        for label in sml_ax.get_xticklabels():
            label.set_rotation(45)
        
        return fig
    def cartesian_plot(self, time= np.timedelta64(0, 'm'), fig=None, ezie_path_buffer=(-np.timedelta64(20, 'm'), np.timedelta64(20, 's')),
                       mag_quiver_kwargs={'scale': 6e-6, 'zorder': 100},
                       mag_quiver_key_kwargs= {'U':800e-9, 'zorder':500, 'fontproperties':{'size':30}},
                       mag_scatter_kwargs={'zorder': 120, 's': 600, 'marker':'o', 'edgecolors': 'black'}, 
                       current_quiver_kwargs={'scale': 7e0, 'zorder': 100}, 
                       current_quiver_key_kwargs={'U':  800e-3, 'zorder':500, 'fontproperties':{'size':30}}, **create_fig_kwargs):
        """Draw or update a Cartesian inversion frame for one relative time.

        Parameters
        ----------
        time : numpy.timedelta64, default: numpy.timedelta64(0, 'm')
            Plot time relative to ``self.event_time``.
        fig : matplotlib.figure.Figure, optional, default: None
            Existing figure from :meth:`create_CS_figure` to update. If
            ``None``, a new figure is created.
        ezie_path_buffer : tuple of numpy.timedelta64, default:
            (numpy.timedelta64(-20, 'm'), numpy.timedelta64(20, 's'))
            Time window around the plotted inversion time used for drawing EZIE
            beam tracks.
        mag_quiver_kwargs : dict, default: {'scale': 6e-6, 'zorder': 100}
            Keyword arguments for modeled magnetic-field quivers.
        mag_quiver_key_kwargs : dict, default:
            {'U': 800e-9, 'zorder': 500, 'fontproperties': {'size': 30}}
            Keyword arguments for magnetic-field quiver keys.
        mag_scatter_kwargs : dict, default:
            {'zorder': 120, 's': 600, 'marker': 'o', 'edgecolors': 'black'}
            Keyword arguments for measured SuperMAG station scatter markers.
        current_quiver_kwargs : dict, default: {'scale': 7e0, 'zorder': 100}
            Keyword arguments for equivalent-current quivers.
        current_quiver_key_kwargs : dict, default:
            {'U': 800e-3, 'zorder': 500, 'fontproperties': {'size': 30}}
            Keyword arguments for equivalent-current quiver keys.
        **create_fig_kwargs
            Passed to :meth:`create_CS_figure` when ``fig`` is ``None``.
            Common options include ``Br_lim`` and ``secs_lim``.

        Returns
        -------
        matplotlib.figure.Figure
            Updated Cartesian event figure.

        Notes
        -----
        File-backed ``SM_data``, ``SML_data``, ``estimated``, and ``amplitudes``
        are opened temporarily and restored to their original paths before
        returning.
        """

        if 'Br_lim' not in create_fig_kwargs:
            create_fig_kwargs['Br_lim']= 500
        if 'secs_lim' not in create_fig_kwargs:
            create_fig_kwargs['secs_lim']= 2

        file_paths= {'SML_data_path': None, 'estimated_path': None, 'amplitudes_path': None, 'SM_data_path': None}
        for var in ['SM_data', 'SML_data']:
            value= getattr(self, var)
            if isinstance(value, str):
                file_paths[f'{var}_path']= value
                setattr(self, var, pd.read_csv(value, parse_dates=['Date_UTC']))
        for var in ['estimated', 'amplitudes']:
            value= getattr(self, var)
            if isinstance(value, str):
                file_paths[f'{var}_path']= value
                setattr(self, var, xr.open_dataset(value))


        if fig is None:
            fig= self.create_CS_figure(**create_fig_kwargs)
        mag_ax, mag_cax, secs_ax, secs_cax, sml_ax, leg_ax= fig.axes

        # When reusing a figure for animation frames, remove only the dynamic
        # artists and keep the fixed axes, colorbars, coastlines, and layout.
        for ax, attrs in [
            (mag_ax, ['Magqk1', 'Magqk2', 'Q_B', 'Q_B2', 'Br', 'Mag_scatter', 'clon', 'clat']),
            (secs_ax, ['Curqk', 'Q_J', 'A', 'ezie_pass', 'clon', 'clat']),
             (sml_ax, ['time_vline']),
        ]:
            for var in attrs:
                artists = getattr(ax, var, None)
                if artists is None:
                    continue
                if not isinstance(artists, (list, tuple)):
                    artists = [artists]
                stack = list(artists)
                while stack:
                    artist = stack.pop()
                    if isinstance(artist, (list, tuple)):
                        stack.extend(artist)
                        continue
                    if hasattr(artist, 'remove'):
                        try:
                            artist.remove()
                        except ValueError:
                            pass
                        continue
                    if hasattr(artist, 'collections'):
                        for item in artist.collections:
                            try:
                                item.remove()
                            except ValueError:
                                pass
                setattr(ax, var, [] if var in ['ezie_pass', 'coastlines', 'clon', 'clat'] else None)




        mag_ax.Br= mag_ax.pcolormesh(self.eval_grid.xi_mesh, self.eval_grid.eta_mesh, self.estimated.sel(time=self.event_time+time).Bu_df.values.T*1e9,
                        cmap='seismic', vmin=-create_fig_kwargs['Br_lim'], vmax=create_fig_kwargs['Br_lim'], zorder=-1)
        mag_ax.Q_B= mag_ax.quiver(self.eval_grid.xi[::3, ::3], self.eval_grid.eta[::3, ::3],
                self.estimated.sel(time=self.event_time+time).Be_df.T[::3, ::3], self.estimated.sel(time=self.event_time+time).Bn_df.T[::3, ::3], **mag_quiver_kwargs)

        mag_quiver_kwargs_measured= mag_quiver_kwargs.copy()
        mag_quiver_kwargs_measured.update({'zorder': mag_quiver_kwargs['zorder']+100, 'color': 'darkorange'})
        mag_ax.Mag_scatter= mag_ax.scatter(*self.SM_data.loc[self.SM_data.Date_UTC==self.event_time+time, ['xi', 'eta']].values.T,
                       c=-self.SM_data.loc[self.SM_data.Date_UTC==self.event_time+time, 'Z_geo'],
                       vmin=-create_fig_kwargs['Br_lim'], vmax=create_fig_kwargs['Br_lim'], cmap='seismic', **mag_scatter_kwargs)
        mag_ax.Q_B2= mag_ax.quiver(*self.SM_data.loc[self.SM_data.Date_UTC==self.event_time+time, ['xi', 'eta', 'Be_cs', 'Bn_cs']].values.T, **mag_quiver_kwargs_measured)

        secs_ax.A= secs_ax.pcolormesh(self.node_grid.xi_mesh, self.node_grid.eta_mesh, (self.amplitudes.sel(time=self.event_time+time).Amplitude_df/self.amplitudes.Area.sel(time=self.event_time+time)).values.T, cmap='seismic',
                        vmin= -create_fig_kwargs['secs_lim'], vmax=create_fig_kwargs['secs_lim'], zorder=-1)
        secs_ax.Q_J= secs_ax.quiver(self.eval_grid.xi[::3, ::3], self.eval_grid.eta[::3, ::3],
                    self.estimated.sel(time=self.event_time+time).Je_df.T[::3, ::3], self.estimated.sel(time=self.event_time+time).Jn_df.T[::3, ::3],
                    **current_quiver_kwargs)

        mag_ax.Magqk1 = mag_ax.quiverkey(
            mag_ax.Q_B, 0.26, 0.23, label='', labelpos='E',
            coordinates='figure', transform=fig.transFigure, **mag_quiver_key_kwargs)
        if mag_ax.text1 is None:
            mag_ax.text1 = quiverkey_left_label(
                fig, mag_ax.Magqk1, 'Modelled',
                zorder=501, size=30
            )

        mag_ax.Magqk2 = mag_ax.quiverkey(
            mag_ax.Q_B2,
            mag_ax.Magqk1.X,
            mag_ax.Magqk1.Y - .03,
            label=f"{int(mag_quiver_key_kwargs['U'] * 1e9)} nT" + '\n',
            labelpos='E',
            coordinates='figure',
            transform=fig.transFigure,
            **mag_quiver_key_kwargs,
        )
        if mag_ax.text2 is None:
            mag_ax.text2 = quiverkey_left_label(
                fig, mag_ax.Magqk2, 'Measured',
                zorder=501, size=30
            )
        secs_ax.Curqk= secs_ax.quiverkey(secs_ax.Q_J, 0.5, 0.23,
            label=f"{int(current_quiver_key_kwargs['U']*1e3)} mA/m", labelpos='E',
            coordinates='figure', transform=fig.transFigure, **current_quiver_key_kwargs)
        sml_ax.time_vline= sml_ax.axvline(self.event_time+time, color='darkblue', linestyle='--', lw=8, label='Inversion Time', zorder=100)


        ezie = Dataset(self.ezie_l2_file)
        try:
            for beam, color in zip(range(1, 5), ['orange', 'green', 'red', 'blue']):
                ezie_time, lon, lat= ezie['Time']['time_utc'][:].astype('datetime64[ns]'), ezie['Geolocation'][f'obs_lon{beam}'][:], ezie['Geolocation'][f'obs_lat{beam}'][:]
                # ind= ((time-Event_date)>np.timedelta64(100, 'm'))&((time-Event_date)<np.timedelta64(105, 'm'))
                ind= (ezie_time > self.amplitudes.sel(time=self.event_time+time).time.values+ezie_path_buffer[0]-\
                      np.timedelta64(1, 'm'))&(ezie_time<=self.amplitudes.sel(time=self.event_time+time).time.values+ezie_path_buffer[1])
                alt= 80
                lon, lat= lon.filled(np.nan), lat.filled(np.nan)
                xi, eta= self.node_grid.projection.geo2cube(lon, lat)
                secs_ax.ezie_pass.extend(secs_ax.plot(xi[ind], eta[ind], label=f'beam {beam}', color=color, marker='d', lw=3, zorder=100_000))
        finally:
            ezie.close()
        mlat, mlt= self.A.convert(self.node_grid.lat, self.node_grid.lon, source='GEO', dest='mlt', datetime=pd.Timestamp(self.amplitudes.sel(time=self.event_time+time).time.values), height=80)
        mlt[mlt>12]-=24
        mlt[mlat>80]=np.nan
        for ax in [mag_ax, secs_ax]:
            clon=ax.contour(self.node_grid.xi, self.node_grid.eta, mlt, levels = np.sort(np.r_[-12:12:1]), colors = 'black', alpha=.5, linewidths=.5)
            clat=ax.contour(self.node_grid.xi, self.node_grid.eta, mlat, levels = np.r_[-90:90:10], colors = 'black', alpha=.5, linewidths=.5)
            ax.clon=[clon, contour_labels(clon, fmt=self._mlt_contour_fmt, sides=['bottom'], xpad=.06, ypad=.04, size=30)]
            ax.clat=[clat, contour_labels(clat, fmt=lambda x: f'{int(x)}'+r'$^{\circ}$'+'\nmlat', xpad=.04, ypad=.07, sides=['left', 'right'], size=30)]
            ax.axis('off')
        mag_ax.set_title('Magnetic Field', size=50)
        secs_ax.set_title('Current', size=50)
        fig.suptitle(self.amplitudes.sel(time=self.event_time+time, method='nearest').time.values.astype('datetime64[m]').tolist(), size=70, y=.86)
        for var in ['SM_data', 'SML_data', 'estimated', 'amplitudes']:
            if file_paths[f'{var}_path'] is not None:
                value= getattr(self, var)
                if isinstance(value, xr.Dataset):
                    value.close()
                setattr(self, var, file_paths[f'{var}_path'])
        if leg_ax.leg is None:
            handles, labels = [], []
            for ax in [mag_ax, secs_ax, sml_ax]:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            # Remove duplicates while preserving order
            seen = set()
            unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
            if unique:
                handles, labels = zip(*unique)
                leg_ax.legend(handles, labels, loc='upper center', fontsize=30, frameon=False, ncols=2)
        return fig
    def plot_event(self, plot_type='cartesian', start= np.timedelta64(-30, 'm'), end= np.timedelta64(30, 'm'),
                   time_step= np.timedelta64(1, 'm'), save_paths=None, create_video= True, video_kwargs=None, **kwargs):
        """Render event frames and optionally combine them into videos.

        Parameters
        ----------
        plot_type : str or list of str, default: 'cartesian'
            Plot renderer(s) to use. Currently ``'cartesian'`` is supported.
        start : numpy.timedelta64, default: numpy.timedelta64(-30, 'm')
            First frame time relative to ``self.event_time``.
        end : numpy.timedelta64, default: numpy.timedelta64(30, 'm')
            Last frame time relative to ``self.event_time``.
        time_step : numpy.timedelta64, default: numpy.timedelta64(1, 'm')
            Time spacing between frames.
        save_paths : str, list of str, or None, default: None
            Directory or directories where frame PNGs are written. If ``None``,
            event-plot folders are created under ``self.inversion_repo``.
        create_video : bool, default: True
            If ``True``, call ``SECSpy.utils.create_video_from_images`` for each
            rendered frame folder.
        video_kwargs : dict or None, default: None
            Additional options passed to ``create_video_from_images``.
        **kwargs
            Passed to the selected plot method, for example
            :meth:`cartesian_plot`.

        Returns
        -------
        str or list of str
            Resolved frame directory path, or one path per ``plot_type``.

        Raises
        ------
        ValueError
            If ``save_paths`` does not match the number of requested plot types.
        """
        times= np.arange(start, end+time_step, time_step)
        fig= None
        n_digits= len(str(max(len(times)-1, 0)))
        if not isinstance(plot_type, list):
            plot_type= [plot_type]
        if save_paths is None:
            save_paths= [None] * len(plot_type)
        elif not isinstance(save_paths, list):
            save_paths=[save_paths]
        if len(save_paths) != len(plot_type):
            raise ValueError('save_paths must be None, a single path, or one path per plot_type.')
        if video_kwargs is None:
            video_kwargs= {}
        resolved_save_paths = []
        for ptype, save_path in zip(plot_type, save_paths):
            if save_path is None:
                save_path= self._repo_path(f'{ptype}_Event_Plots', '')
            os.makedirs(save_path, exist_ok=True)
            resolved_save_paths.append(save_path)
        for ptype, save_path in zip(plot_type, resolved_save_paths):
            for i, time in enumerate(tqdm(times, desc='Plotting event', unit='frame')):
                if ptype.lower()=='cartesian':
                    fig= self.cartesian_plot(time=time, fig=fig, **kwargs)
                fig.savefig(os.path.join(save_path, f'{i:0{n_digits}d}.png'), bbox_inches='tight')
            if create_video:
                from SECSpy.utils import create_video_from_images
                video_options = video_kwargs.copy()
                video_options.setdefault('start_number', 0)
                video_options.setdefault('filename', os.path.join(self.inversion_repo, ptype))
                create_video_from_images(save_path, num_digits=n_digits, **video_options)

        return resolved_save_paths[0] if len(resolved_save_paths) == 1 else resolved_save_paths
    def run_all(self, username=None, grid_buffer=(.1, .1), grid_kwargs=None, start= -np.timedelta64(1, 'h'), end= np.timedelta64(1, 'h'), save=None,
            l1=None, l2=None, plot_kwargs={}, plot_type=['cartesian'], plot_start= np.timedelta64(-30, 'm'), plot_end= np.timedelta64(30, 'm'),
            plot_time_step= np.timedelta64(1, 'm'), plot_save_path=None, verbose=False, auto_remove=True):
        """Run the full EZIE inversion workflow from grid creation to plots.

        Parameters
        ----------
        username : str, optional, default: None
            SuperMAG username passed to :meth:`download_data`.
        grid_buffer : tuple of float, default: (0.1, 0.1)
            Fractional grid-bound expansion passed to :meth:`download_data`.
        grid_kwargs : dict or None, default: None
            Keyword arguments passed to :meth:`create_grid`.
        start : numpy.timedelta64, default: numpy.timedelta64(-1, 'h')
            Data download and inversion start time relative to ``event_time``.
        end : numpy.timedelta64, default: numpy.timedelta64(1, 'h')
            Data download and inversion end time relative to ``event_time``.
        save : object, default: None
            Reserved for workflow compatibility; individual steps currently save
            into ``self.inversion_repo`` using empty-prefix paths.
        l1, l2 : float, optional, default: None
            Regularization weights passed to :meth:`run_inversion`.
        plot_kwargs : dict, default: {}
            Keyword arguments passed to :meth:`plot_event`.
        plot_type : str or list of str, default: ['cartesian']
            Plot type(s) passed to :meth:`plot_event`.
        plot_start : numpy.timedelta64, default: numpy.timedelta64(-30, 'm')
            First plot frame time relative to ``event_time``.
        plot_end : numpy.timedelta64, default: numpy.timedelta64(30, 'm')
            Last plot frame time relative to ``event_time``.
        plot_time_step : numpy.timedelta64, default: numpy.timedelta64(1, 'm')
            Plot frame cadence.
        plot_save_path : str, list of str, or None, default: None
            Frame output path(s) passed to :meth:`plot_event`.
        verbose : bool, default: False
            If ``True``, print progress messages before each workflow step.
        auto_remove : bool, default: True
            Passed to :meth:`download_data` for stations with incomplete data.

        Returns
        -------
        tuple
            ``(self, self.inversion_repo)``.
        """
        if grid_kwargs is None:
            grid_kwargs = {}
        if verbose:
            print('Creating grids...')
        self.create_grid(**grid_kwargs)
        if verbose:
            print('Creating SECS poles...')
        self.create_SECS()        
        if verbose:
            print('Downloading data...')
        self.download_data(username=username, grid_buffer=grid_buffer, start=start, end=end, save='', auto_remove=auto_remove)
        if verbose:
            print('Plotting satellite pass...')
        self.plot_pass(save='')
        if verbose:
            print('Running inversion...')
        self.run_inversion(l1=l1, l2=l2, start=start, end=end, save='')
        if verbose:
            print('Plotting event...')
        self.plot_event(plot_type=plot_type, start=plot_start, end=plot_end, time_step=plot_time_step, save_paths=plot_save_path, **plot_kwargs)
        return self, self.inversion_repo

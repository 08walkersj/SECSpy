import numpy as np
from .SECS import theta, RE
def GridCheck(longitude, latitude, MagLon, MagLat, GridSpacing_km, limit=20, evaluation_altitude=RE, theta_prime=None):
    """
    Parameters
    ----------
    longitude : numpy.ndarray (Degrees)
        Longitude of the SEC poles to check
    latitude : numpy.ndarray (Degrees)
        Latitude of the SEC poles to check
    MagLon : numpy.ndarray (Degrees)
        Longitude of the magnetometers that are being used.
    MagLat : numpy.ndarray (Degrees)
        Latitude of the magnetometers that are being used.
    GridSpacing_km : int/float (kilometers)
        Average spacing between the poles in kilometers.
    limit : int/float, optional
        Percentage of the grid spacing that is the closest the SEC poles can be to magnetometers. The default is 20.
    evaluation_altitude: int/float (kilometers)
        Altitude of the evaluation points/ magnetometers. The default is RE.
    Returns
    -------
    Longitude : numpy.ndarray (Degrees)
        Longitude values of the poles that are problematic.
    Latitude : numpy.ndarray (Degrees)
        Latitude values of the poles that are problematic.
    Index: numpy.ndarray
        Index of the position of the problem poles. Index[0] is the problem measurements, Index[1] is the problem poles

    """
    print('Analysing grid')
    if theta_prime is None:
        theta_prime= theta(latitude, longitude, MagLat, MagLon)
    if len(np.where(GridSpacing_km *1e3 *limit/100>theta_prime*evaluation_altitude)[0])==0:
        print('Grid within requirements')
        return [None]*3
    else:
        print('Grid not optimal returning longitude and latitude values of the poles that are a problem and their index')
        I= np.where(GridSpacing_km *1e3 *limit/100>theta_prime*evaluation_altitude)
        #I[0] is the problem measurements, I[1] is the problem poles
        return longitude[I[1]], latitude[I[1]], I
def ImproveGrid(lon_centre, lat_centre, MagLon, MagLat, GridSpacing_km, length, height, evaluation_altitude=RE, date=None, east=None, north=None, limit=20, movement_limit=5):
    """
    Parameters
    ----------
    lon_centre : int/float (Degrees)
        longitude of the centre of the node grid.
    lat_centre : int/float ((Degrees))
        lattitude of the centre of the node grid.
    MagLon : numpy.ndarray (Degrees)
        Longitude of the magnetometers.
    MagLat : numpy.ndarray (Degrees)
        Latitude of the magnetometers.
    GridSpacing_km : float
        Spacing between nodes in km for the node grid.
    length : int/float
        length of the node grid.
    height : int/float
        height of the node grid.
    date : datetime.datetime
        Date being used.
    limit : int/float, optional
        Percentage of grid spacing that will provide a limit of closeness of magnetometers with the pole nodes. The default is 20.
    evaluation_altitude: int/float (kilometers)
        Altitude of the evaluation points/ magnetometers. The default is RE.
    Returns
    -------
        Change: numpy.ndarray (Degrees)
            Recommended change in the central position of the grid ([[lon_change, lat_change]]).
        Index: numpy.ndarray
            Index for problem poles for recommended grid (empty if no problem poles).
        Num_Bad_Poles: numpy.ndarray
            number of problem poles with each adjustment.
        Changes_Tried: numpy.ndarray
            all adjustments tried.
        Index: numpy.ndarray:
            Index where for each adjusted grid there is a problem pole.
    """
    try:
        import progressbar
        bar=True
    except ModuleNotFoundError:
        bar=False
    if bar:
        prog_bar = progressbar.ProgressBar(max_value=len(np.arange(-5,5,0.1))**2) 
        pbi=0
    if date is not None:
        from apexpy import Apex
    num_minimum=[]
    change=[]
    index=[]
    # from pysymmetry.utils.CSgrid import CSgrid
    from pysymmetry.utils import cubedsphere as CS
    if date is not None:
        A = Apex(date=date)
        f1, f2 = A.basevectors_qd(lat_centre, lon_centre, 0, coords = 'geo')
        qd_north = f2 / np.linalg.norm(f2)
        East, North= qd_north[1], -qd_north[0]
    else:
        East, North= east, north
    # node_grid= CSgrid((lon_centre, lat_centre), [East,North], height, length, GridSpacing_km, GridSpacing_km)
    Gridproj= CS.CSprojection((lon_centre, lat_centre), [East, North])
    node_grid=CS.CSgrid(Gridproj, height, length, GridSpacing_km, GridSpacing_km)
    node_lons, node_lats= node_grid.lon.flatten(), node_grid.lat.flatten() 
    thetap= theta(node_lats, node_lons, MagLat, MagLon)
    print('\n number of problem poles initially ',len(np.where(GridSpacing_km* 1e3 *limit/100 > thetap*evaluation_altitude)[0]))
    if len(np.where(GridSpacing_km* 1e3 *limit/100 > thetap*evaluation_altitude)[0])==0:
        print('Grid already optimal')
        if bar:
            prog_bar.finish()
        return [None]*5
    print('Looking for grid improvements')
    for dc in np.arange(-movement_limit, movement_limit, 0.1): 
         for d in np.arange(-movement_limit, movement_limit, 0.1):
             if date is not None:
                 A = Apex(date=date)
                 f1, f2 = A.basevectors_qd(lat_centre+dc, lon_centre+d, 0, coords = 'geo')
                 qd_north = f2 / np.linalg.norm(f2)
                 East, North= qd_north[1], -qd_north[0]
             else:
                 East, North= east, north
             Gridproj= CS.CSprojection((lon_centre+d, lat_centre+dc), [East, North])
             node_grid=CS.CSgrid(Gridproj, height, length, GridSpacing_km, GridSpacing_km)
             node_lons, node_lats= node_grid.lon.flatten(), node_grid.lat.flatten()
             thetap= theta(node_lats, node_lons, MagLat, MagLon)
             num_minimum.append(len(np.where(GridSpacing_km* 1e3 *limit/100 > thetap*evaluation_altitude)[0]))
             change.append([d,dc])
             index.append([np.where(GridSpacing_km* 1e3 *limit/100 > thetap*evaluation_altitude)])
             if bar:
                 pbi+=1
                 prog_bar.update(pbi)
    if bar:
        prog_bar.finish()
    num_minimum= np.array(num_minimum)
    change= np.array(change)
    index= np.array(index)
    I=np.where(num_minimum== np.min(num_minimum))
    I2= np.where(np.sum(abs(change[I]), axis=1)==np.min(np.sum(abs(change[I]), axis=1)))
    print('Completed! Information returned')
    print('Number of problem poles with this optimisation ',
          num_minimum[I][I2])
    return change[I][I2], index[I][I2], num_minimum, change, index

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

def create_grid(lon_center, lat_center, Apex, L=6000, W=6000, Lres=150., Wres=150., center_shift=(0, 0),
                auto_center=False, auto_center_step=.25, auto_center_max=3., **grid_kwargs):
    from secsy.cubedsphere import CSgrid, CSprojection
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
        center_lon = lon_center + shift[0]
        center_lat = lat_center + shift[1]
        f1, f2= Apex.basevectors_qd(center_lat, center_lon%360, 0, coords='geo')
        qd_north = f2/np.linalg.norm(f2)
        East, North= qd_north[1], -qd_north[0]
        grid_proj= CSprojection((center_lon, center_lat), (East, North))


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
                pole_direction = -np.sign(lat_center) * (shift[1] - center_shift[1])
                candidates.append((distance, -pole_direction, shift))
        candidates.sort(key=lambda item: (item[0], item[1], abs(item[2][0]), abs(item[2][1])))

        original_diagnostics = _local2geo_rotation_diagnostics_for_grid(node_grid, center_lon, center_lat)
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
            grid_lon_center = center_lon
            grid_lat_center = center_lat
            grid_center_shift = center_shift
            node_grid = node_grid
            eval_grid = eval_grid
            raise ValueError(
                original_diagnostics
                + f'\nNo valid center shift found within +/-{auto_center_max} degrees at {auto_center_step} degree resolution.'
            )

    grid_lon_center = center_lon
    grid_lat_center = center_lat
    grid_center_shift = center_shift
    return node_grid, eval_grid
def shift_grid(node_grid_dic, glon, glat, Apex, shift_lat=False):
    """Shift a saved cubed-sphere grid and rebuild its evaluation grid.

    Parameters
    ----------
    node_grid_dic : dict
        Dictionary representation of the node grid, such as one returned by
        ``CSgrid.to_dictionary()``.
    glon : int/float (Degrees)
        Geographic longitude used to shift the grid.
    glat : int/float (Degrees)
        Geographic latitude used to shift the grid.
    Apex : apexpy.Apex
        Apex object used to convert between geographic and apex coordinates.
    shift_lat : bool, optional
        If ``True``, shift both geographic longitude and latitude directly.
        If ``False``, keep the original apex latitude and shift only to the
        apex longitude associated with ``glon`` and ``glat``. The default is
        ``False``.

    Returns
    -------
    node_grid : secsy.cubedsphere.CSgrid
        Shifted node grid used for SECS pole amplitudes.
    eval_grid : secsy.cubedsphere.CSgrid
        One-cell-larger grid used to evaluate modeled fields and currents.
    """
    from secsy.cubedsphere import CSgrid, CSprojection, from_dictionary
    node_grid_dic= node_grid_dic.copy()
    if shift_lat:
        node_grid_dic['projection']['position']= glon, glat
    else:
        # Convert the requested geographic position to apex longitude, then
        # project it back at the original apex latitude.
        lon_cent, lat_cent= node_grid_dic['projection']['position']
        mlat, mlon= Apex.geo2apex(lat_cent, lon_cent, height=0)
        mlat_new, mlon_new= Apex.geo2apex(glat, glon, height=0)
        glat, glon, _= Apex.apex2geo(mlat, mlon_new, height=0)
        node_grid_dic['projection']['position']= glon, glat
    node_grid= from_dictionary(node_grid_dic)
    # The evaluation grid is one cell larger so fields can be evaluated on
    # cell edges while amplitudes remain on the SECS node grid.
    xi_e  = np.hstack((node_grid.xi_mesh[0]    , node_grid.xi_mesh [0 , - 1] + node_grid.dxi )) - node_grid.dxi /2
    eta_e = np.hstack((node_grid.eta_mesh[:, 0], node_grid.eta_mesh[-1,   0] + node_grid.deta)) - node_grid.deta/2
    eval_grid= CSgrid(node_grid.projection,
                        node_grid.L + node_grid.Lres, node_grid.W + node_grid.Wres,
                        node_grid.Lres, node_grid.Wres,
                        edges = (xi_e, eta_e), R = node_grid.R)
    return node_grid, eval_grid
def _recalculate_cs_vectors(ds, grid):
    """Recalculate cubed-sphere vector components after changing grid coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing geographic east/north vector components on
        ``xi``/``eta`` coordinates.
    grid : secsy.cubedsphere.CSgrid
        Grid whose projection should be used for the cubed-sphere components.
    """
    def project_pair(east_name, north_name, xi_name, eta_name):
        """Project one east/north vector pair and restore its original dimensions."""
        extra_dims = [dim for dim in ds[east_name].dims if dim not in ['xi', 'eta']]
        extra_shape = [ds[east_name].sizes[dim] for dim in extra_dims]

        # secsy expects flat vectors and matching flat lon/lat arrays. Keep
        # time or other non-grid dimensions last so each grid point is repeated
        # for every extra coordinate value.
        east = ds[east_name].transpose('eta', 'xi', *extra_dims).values.reshape(grid.size, -1)
        north = ds[north_name].transpose('eta', 'xi', *extra_dims).values.reshape(grid.size, -1)
        n_extra = east.shape[1]
        lon = np.repeat(np.asarray(ds['lon']).T.reshape(-1), n_extra)
        lat = np.repeat(np.asarray(ds['lat']).T.reshape(-1), n_extra)

        xi_values, eta_values = grid.projection.vector_cube_projection(
            east.reshape(-1),
            north.reshape(-1),
            lon,
            lat,
            return_xi_eta=False,
        )

        # Restore xarray's (xi, eta, ...) coordinate order after secsy's
        # flattened (eta, xi, ...) calculation order.
        projected_shape = (*grid.shape, *extra_shape)
        axes = (1, 0, *range(2, len(projected_shape)))
        dims = ('xi', 'eta', *extra_dims)
        ds[xi_name] = (dims, xi_values.reshape(projected_shape).transpose(axes))
        ds[eta_name] = (dims, eta_values.reshape(projected_shape).transpose(axes))

    if {'Be', 'Bn'}.issubset(ds):
        project_pair('Be', 'Bn', 'Be_cs', 'Bn_cs')
    if {'Je', 'Jn'}.issubset(ds):
        project_pair('Je', 'Jn', 'Je_cs', 'Jn_cs')

def change_xarray_grid(ds, new_grid):
    """Replace the cubed-sphere grid coordinates in an xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``xi``/``eta`` grid coordinates and cubed-sphere grid
        metadata in ``attrs['cs_grid']``.
    new_grid : secsy.cubedsphere.CSgrid
        Grid whose coordinates and dictionary representation will replace the
        existing grid information in ``ds``.

    Returns
    -------
    ds : xarray.Dataset
        Copy of ``ds`` with coordinates reassigned to ``new_grid`` and
        ``attrs['cs_grid']`` updated.
    """
    import json

    ds= ds.copy()
    # xarray stores the two-dimensional geographic coordinates as
    # (xi, eta), while secsy grid arrays are indexed as (eta, xi).
    new_coords= dict(xi=new_grid.xi[0,:],
                        eta=new_grid.eta[:,0],
                        lon=(['xi', 'eta'], new_grid.lon.T),
                        lat=(['xi', 'eta'], new_grid.lat.T),
                        xi_edge=np.arange(new_grid.xi_mesh.shape[1]),
                        eta_edge=np.arange(new_grid.eta_mesh.shape[0]),
                        xi_mesh=(['xi_edge', 'eta_edge'], new_grid.xi_mesh.T),
                        eta_mesh=(['xi_edge', 'eta_edge'], new_grid.eta_mesh.T))
    ds = ds.assign_coords(new_coords)
    _recalculate_cs_vectors(ds, new_grid)

    # Keep the serialized grid metadata consistent with the reassigned
    # coordinates so the dataset can be reconstructed later.
    ds.attrs['cs_grid']= json.dumps(new_grid.to_dictionary())
    return ds

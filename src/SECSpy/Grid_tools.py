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
        East, North= qd_north[0], qd_north[1]
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
                 East, North= qd_north[0], qd_north[1]
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
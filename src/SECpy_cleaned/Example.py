import numpy as np

def ExampleGrid():
    """
    Returns
    -------
    node_lons : numpy.ndarray (Degrees)
        SEC pole node longitudes.
    node_lats : numpy.ndarray (Degrees)
        SEC pole node lattitudes.
    lon : numpy.ndarray (Degrees)
        Evaluated node longitudes.
    lat : numpy.ndarray (Degrees)
        Evaluated node lattitudes.
    """
    dlat=0.6
    dlon=1.4
    
    a1= round(58.26/dlat)*dlat
    a2= round(78.92/dlat)*dlat
    b1=round(4.84/dlon)*dlon
    b2= round(36.08/dlon)*dlon
    
    SECSlat,SECSlon = np.meshgrid(np.arange(a1-4.5*dlat,a2+4.5*dlat,dlat), np.arange(b1-4.5*dlon,b2+4.5*dlon,dlon))
    node_lons, node_lats= SECSlon.flatten(),SECSlat.flatten()
    lat,lon = np.meshgrid(np.arange(a1,a2,dlat), np.arange(b1,b2,dlon))
    lat,lon= lat.flatten(),lon.flatten()
    return node_lons, node_lats, lon, lat
def ExampleData():
    """
    Returns
    -------
    South : numpy.ndarray (Tesla)
        Component of the magnetic field measurement(s) in the theta unit vector from the magnetometer(s).
    East : numpy.ndarray (Tesla)
        Component of the magnetic field measurement(s) in the phi unit vector from the magnetometer(s).
    Up : numpy.ndarray (Tesla)
        Component of the magnetic field measurement(s) in the radial unit vector from the magnetometer(s).
    MagLon : numpy.ndarray (Degrees)
        Longitudinal location of the magnetometers.
    MagLat : numpy.ndarray (Degrees)
        Latitudinal location of the magnetometers.
    """
    #Bnames=['TAR','KAR','UPS','NUR','SOL','DOB','HAN','MEK','OUJ','LYC','RVK','RAN','DON','JCK','PEL','SOD','KIR','MUO','ABK','IVA','KIL','AND','MAS','TRO','KEV','SOR','NOR','BJN','HOP','HOR','LYR','NAL','SUW','BRZ']
    #Date= 23:34 08/11/2015
    BX = -np.array([-14.1,   -6.2,  -11.7,  -32.9,  -21.2,  -43.6,  -61.1,  -85.7, -159.3, -199.9, -336.5, -267.9, -560.3, -465.7, -436.0, -483.1, -673.7, -667.4, -789.5, -688.4, -784.5, -599.0, -714.7, -534.3, -674.2, -321.1, -251.1,   -3.7,   13.0,   37.2,   37.4,   57.3,   -7.1,   -8.2])
    BY = np.array([ -6.2,   52.1,   11.2,   -9.3,   61.0,   37.1,  -25.6,  -31.1,  -40.6,   -3.7,  109.5,  -37.2,  162.0,  105.4,    3.1,  -14.4,   84.1,  109.1,  133.1,   83.7,  158.8,  179.4,  115.1,   78.0,  121.7,   52.2,   46.6,   20.4,   13.6,   17.5,   10.1,   10.2,    6.6,  -0.9])
    BZ = -np.array([-48.2,  -66.9,  -79.6, -101.7, -134.4, -176.1, -144.1, -155.4, -221.3, -255.5, -306.7, -284.0, -284.9, -385.3, -340.3, -270.8, -193.8, -217.3,  -87.3, -149.6,  228.2,  510.4,  376.7,  480.7,  384.6,  492.0,  413.9,  247.0,  180.7,  176.3,  154.6,  159.3,  -23.9,  -32.2])
    MagLat=  np.array([58.26, 59.21, 59.90, 60.50, 61.08, 62.07, 62.30, 62.77, 64.52, 64.61, 64.94, 65.90, 66.11, 66.40, 66.90, 67.37, 67.83, 68.02, 68.35, 68.56, 69.02, 69.30, 69.46, 69.66, 69.76, 70.54, 71.09, 74.50, 76.51, 77.00, 78.20, 78.92, 54.01, 56.17])
    MagLon=  np.array([26.46,  5.24, 17.35, 24.65,  4.84,  9.12, 26.65, 30.97, 27.23, 18.75, 10.98, 26.41, 12.50, 16.98, 24.08, 26.63, 20.42, 23.53, 18.82, 27.29, 20.79, 16.03, 23.70, 18.94, 27.01, 22.22, 25.79, 19.20, 25.01, 15.60, 15.82, 11.95, 23.18, 24.86])
    return BX*1e-9, BY*1e-9, BZ*1e-9, MagLon, MagLat
#%% Imports
import pandas as pd
import numpy as np
from apexpy import Apex
from secsy import cubedsphere as cs
import json
from File_tools.Filename_tools import date2filename
import os
Event_date = np.datetime64('2025-05-04T07:22')
A= Apex(pd.Timestamp(Event_date.astype('datetime64[Y]')))

#%% From SuperMag
SM_index = pd.read_csv('../../Data/SM_Indices_2025.csv', sep=',', parse_dates=['Date_UTC'],)
SM_index.replace(999999, np.nan, inplace=True)

mlt_, mlat_= SM_index.loc[SM_index['Date_UTC'] == Event_date, ['SML_MLT', 'SML_MLAT']].values[0]
glat, glon = A.convert(mlat_, mlt_, 'mlt', 'geo', datetime=pd.Timestamp(Event_date))


#%% From EZIE
from netCDF4 import Dataset
folder= '/home/simon/gits/SECpy_cleaned/Data/Sample data_ 2025-05-27/L1_TargetDays/'
file = [file for file in os.listdir(folder) if f'{Event_date.tolist().year}{Event_date.tolist().month:02d}{Event_date.tolist().day:02d}' in file][0]
ezie= Dataset(folder+file)
time= ezie['Time']['time_utc'][:].astype('datetime64[ns]')
ind= np.argmax(np.isfinite(ezie['Geolocation'][f'obs_lon{1}'][:].filled(np.nan))&\
    (np.arange(len(time))>=np.argmin(abs(time-(Event_date+np.timedelta64(3, 'm'))))))
closest= {'time':time[ind]}
closest.update({f'lon{i}':ezie['Geolocation'][f'obs_lon{i}'][:][ind] for i in range(1, 5)})
closest.update({f'lat{i}':ezie['Geolocation'][f'obs_lat{i}'][:][ind] for i in range(1, 5)})
closest.update({'sat_lat':ezie['Geolocation']['sat_lat'][:][ind],
                'sat_lon':ezie['Geolocation']['sat_lon'][:][ind],
                'sat_vel_ecef':ezie['Geolocation']['sat_vel_ecef'][:][ind]})
glon, glat= closest['lon1'], closest['lat1']

# Convert ECEF velocity to ENU (East-North-Up) at (lat, lon)
def ecef_to_enu(v_ecef, lat, lon):
    # Convert degrees to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    # Rotation matrix
    R = np.array([
        [-np.sin(lon),              np.cos(lon),             0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])
    return R @ v_ecef

vel_enu = ecef_to_enu(closest['sat_vel_ecef'], closest['sat_lat'], closest['sat_lon'])

# Compute the angle from North (y-axis in ENU) to the velocity vector
east, north, up = vel_enu
sat_heading = np.arctan2(north, east)  # radians, 0=north, +eastward



#%% North Projection
f1, f2= A.basevectors_qd(glat, glon%360, 0, coords='geo')

qd_north = f2/np.linalg.norm(f2)
East, North= qd_north[0], qd_north[1]
grid_proj= cs.CSprojection((glon, glat), np.rad2deg(np.arctan2(East, North)))


#%% Sat Projection
grid_proj= cs.CSprojection((glon%360, glat), np.rad2deg(sat_heading))

#%% Create Grid
node_grid= cs.CSgrid(grid_proj, 8000, 6000, 100., 100.)
xi_e  = np.hstack((node_grid.xi_mesh[0]    , node_grid.xi_mesh [0 , - 1] + node_grid.dxi )) - node_grid.dxi /2 
eta_e = np.hstack((node_grid.eta_mesh[:, 0], node_grid.eta_mesh[-1,   0] + node_grid.deta)) - node_grid.deta/2 
eval_grid= cs.CSgrid(grid_proj,
                               node_grid.L + node_grid.Lres, node_grid.W + node_grid.Wres, 
                               node_grid.Lres, node_grid.Wres, 
                               edges = (xi_e, eta_e), R = node_grid.R)


folder= f"../../Data/{date2filename(Event_date)}/"

os.makedirs(folder, exist_ok=True)
with open(folder+"node_grid.json", "w") as f:
    json.dump(node_grid.to_dictionary(), f)
with open(folder+"eval_grid.json", "w") as f:
    json.dump(eval_grid.to_dictionary(), f)

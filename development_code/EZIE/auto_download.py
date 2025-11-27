from secsy import cubedsphere as cs
import json
from File_tools.Filename_tools import date2filename
import numpy as np

#%% Load Grid
Event_date = np.datetime64("2025-05-04T07:22")
folder= f"../../Data/{date2filename(Event_date)}/"

with open(folder+"node_grid.json", "r") as f:
    node_grid = cs.from_dictionary(json.load(f))
with open(folder+"eval_grid.json", "r") as f:
    eval_grid = cs.from_dictionary(json.load(f))

#%% Get Data
from SuperMAG.supermag_api_simon import SuperMAGGetData
status, data= SuperMAGGetData('08walkersj', Event_date-np.timedelta64(60, 'm'), 2*60*60+60, ['GEO', 'decl'],
                      f'(GEOLAT>={node_grid.lat.min()}) & (GEOLAT<={node_grid.lat.max()}) &'
                       f'(GEOLON>={(node_grid.lon%360).min()}) & (GEOLON<={(node_grid.lon%360).max()})'
                       )
decl_rad = np.deg2rad(data['decl'])
data['N_geo'] = data['N_nez'] * np.cos(decl_rad) - data['E_nez'] * np.sin(decl_rad)
data['E_geo'] = data['N_nez'] * np.sin(decl_rad) + data['E_nez'] * np.cos(decl_rad)
#%% Correct Data
data.replace(999999, np.nan, inplace=True)
data.dropna(inplace=True)
data['xi'], data['eta']= node_grid.projection.geo2cube(data.glon, data.glat)

#%% Crop Data
grid_ind= (data.xi>node_grid.xi[0, 0])&(data.xi<node_grid.xi[-1, -1])
grid_ind&= (data.eta>node_grid.eta[0, 0])&(data.eta<node_grid.eta[-1, -1])
data= data[grid_ind]
data.to_csv(folder+'magnetometer_data.csv', index=False)

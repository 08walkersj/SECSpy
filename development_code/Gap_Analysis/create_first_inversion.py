#%%
from SECSpy import SECS
from SECSpy.Grid_tools import create_grid
from SECSpy.SuperMAG_tools import (cubedsphere_stations,
                                   download_in_grid, format_data_vector, read_data_vector)
from SECSpy.utils import date2filename
from SECSpy.constants import RE
from apexpy import Apex
import pandas as pd
import numpy as np
import os
from pathlib import Path
from xarray import load_dataset
# date= pd.Timestamp('2011-06-01T08:01')

date= pd.Timestamp('2024-05-10T00:00')


# lon_center, lat_center = 258, 61
lon_center, lat_center = 20, 64
A= Apex(date)-

# node_grid, eval_grid = create_grid(lon_center, lat_center, A)
node_grid, eval_grid= create_grid(lon_center, lat_center, A, L=1000, W=2000, Lres=50., Wres=50.)

stations = cubedsphere_stations(node_grid)

data= download_in_grid('08walkersj', date, stations, node_grid, auto_remove=True, start=-np.timedelta64(1, 'D'), end=np.timedelta64(5, 'D'))
poles= SECS(node_grid.lon.flatten(), node_grid.lat.flatten(), eval_grid.lon.flatten(), eval_grid.lat.flatten(),
            mode='image', image_current_radius=RE-500e3, current_radius=RE+110e3)

data= format_data_vector(data)
folder= Path('../Data/Gap_Analysis/'+date2filename(date, resolution='M'))
os.makedirs(folder, exist_ok=True)
data.to_csv(folder/Path('SM_data.csv'))

data= read_data_vector(folder/Path('SM_data.csv'))

#%%
import matplotlib.pyplot as plt
fig= plt.figure()
ax= fig.add_subplot()
ax.scatter(node_grid.xi, node_grid.eta, marker='x', alpha=.3)
node_grid.add_natural_earth_feature(ax, 'coastline', '10m')
ax.scatter(data.xi.iloc[0, :], data.eta.iloc[0, :], marker='p')
ax.set_aspect('equal')
#%% inversion

poles.Fitting_Matrix_Regularisation(l1= 3e3, l2= 6e3, grid=node_grid,
                                   glon=data.glon.iloc[0, :].values,
                                   glat=data.glat.iloc[0, :].values, Apex=Apex(date))

amps= poles.time_inversion(data[['South_ground', 'East_ground', 'Up_ground']].to_numpy())
amps_xr= poles.amps_to_xarray(amps, data.index.values, node_grid,
                              list(data.columns.levels[data.columns.names.index("iaga")]),
                              data.glon.iloc[0, :].values, data.glat.iloc[0, :].values,
                              meas_xi=data.xi.iloc[0, :].values, meas_eta=data.eta.iloc[0, :].values)
estimated= poles.estimated_to_xarray(data.index.values, eval_grid, node_grid, amps, mag_code=list(data.columns.levels[data.columns.names.index("iaga")]),
                                     meas_glon=data.glon.iloc[0, :].values, meas_glat=data.glat.iloc[0, :].values, 
                                     meas_xi=data.xi.iloc[0, :].values, meas_eta=data.eta.iloc[0, :].values)
#%% Save
amps_xr.to_netcdf(folder/Path('amplitudes.ncdf4'))
estimated.to_netcdf(folder/Path('estimated.ncdf4'))

#%% Load
amps_xr= load_dataset(folder/Path('amplitudes.ncdf4'))
estimated= load_dataset(folder/Path('estimated.ncdf4'))
# #%% Test Plot
from SECSpy.Plotting_Tools import cartesian_plot, time_loop
import numpy as np
date= np.datetime64('2024-05-10T18:32:00')
cartesian_plot(amps_xr, estimated, date, data, A, node_grid=node_grid)
1/0
#%% Plot All
time_loop(amps_xr, estimated, event_time=date, start=-np.timedelta64(1, 'D'), end=np.timedelta64(5, 'D'),
          SM_data=data, Apex=A, node_grid=node_grid, save_root='Gap_Analysis',
          savefig_kwargs={'bbox_inches': 'tight'})


#%% Shift

def shift_grid(node_grid_dic, glon, glat, Apex, shift_lat=False):
    from secsy.cubedsphere import CSgrid, CSprojection
    node_grid_dic= node_grid_dic.copy()
    if shift_lat:
        node_grid_dic['projection']['position']= glon, glat
    else:
        lon_cent, lat_cent= node_grid_dic['projection']['position']
        mlat, mlon= Apex.geo2apex(lat_cent, lon_cent, height=0)
        mlat_new, mlon_new= Apex.geo2apex(glat, glon, height=0)
        glat, glon, _= Apex.apex2geo(mlat, mlon_new, height=0)
        node_grid_dic['position']= glon, glat
    node_grid= CSgrid(projection=CSprojection(**node_grid_dic.pop('projection'), **node_grid_dic))
    xi_e  = np.hstack((node_grid.xi_mesh[0]    , node_grid.xi_mesh [0 , - 1] + node_grid.dxi )) - node_grid.dxi /2
    eta_e = np.hstack((node_grid.eta_mesh[:, 0], node_grid.eta_mesh[-1,   0] + node_grid.deta)) - node_grid.deta/2
    eval_grid= CSgrid(node_grid.projection,
                        node_grid.L + node_grid.Lres, node_grid.W + node_grid.Wres,
                        node_grid.Lres, node_grid.Wres,
                        edges = (xi_e, eta_e), R = node_grid.R)
    return node_grid, eval_grid
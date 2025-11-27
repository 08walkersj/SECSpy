#%% Imports
import matplotlib.pyplot as plt
import xarray as xr
from secsy import cubedsphere as cs
from File_tools.Filename_tools import date2filename
from apexpy import Apex
import pandas as pd
import numpy as np
import json
from Plotting_Tools.Subplot_alignment_tools import subplot_align
#%% Load Data
Event_date = np.datetime64("2025-05-04T07:22")
folder= f"../../Data/{date2filename(Event_date)}/"

with open(folder+"node_grid.json", "r") as f:
    node_grid = cs.from_dictionary(json.load(f))
with open(folder+"eval_grid.json", "r") as f:
    eval_grid = cs.from_dictionary(json.load(f))
A= Apex(pd.Timestamp(Event_date.astype('datetime64[Y]')))

amps = xr.load_dataset(folder+'Amplitudes.ncdf4').isel(time=0)
amps['Amplitude_df']= amps['Amplitude_df']/amps.Area
estimated= xr.load_dataset(folder+'estimated.ncdf4').isel(time=0)

data= pd.read_csv(folder+'magnetometer_data.csv',
                  parse_dates=['Date_UTC'])
vals, counts= np.unique(data.iaga.values, return_counts=True)
data.sort_values(['Date_UTC', 'glat'], inplace=True)
data.index= list(range(len(data)))
data['Be_cs'], data['Bn_cs']= node_grid.projection.vector_cube_projection(data.E_geo, data.N_geo, data.glon, data.glat, return_xi_eta=False)
data= data.loc[data.Date_UTC==Event_date]
#%% Create Figure
fig= plt.figure(figsize=(40, 20))
gs= fig.add_gridspec(2, 3, height_ratios=[1, .1], hspace=0)

mag_ax= fig.add_subplot(gs[0, 0])
mag_cax= fig.add_subplot(gs[1, 0])

secs_ax= fig.add_subplot(gs[0, 1])
secs_cax= fig.add_subplot(gs[1, 1])

cur_ax= fig.add_subplot(gs[0, 2])
cur_cax= fig.add_subplot(gs[1, 2])

##%% Plot Estimated
Q_B= mag_ax.quiver(eval_grid.xi[::3, ::3], eval_grid.eta[::3, ::3],
                   estimated.Be_df[::3, ::3], estimated.Bn_df[::3, ::3], zorder=100)
B_lim= 500
# Br= mag_ax.pcolormesh(eval_grid.xi_mesh, eval_grid.eta_mesh, estimated.Bu_df.values.T*1e9,
#                       cmap='seismic', vmin=-B_lim, vmax=B_lim, zorder=-1)
# Br_cbar= fig.colorbar(Br, cax=mag_cax, orientation='horizontal',
#                       label='Br [nT]', extend='both')
secs_lim= 2
A= secs_ax.pcolormesh(node_grid.xi_mesh, node_grid.eta_mesh, amps.Amplitude_df.values.T, cmap='seismic',
                      vmin= -secs_lim, vmax=secs_lim, zorder=-1)
A_cbar= fig.colorbar(A, cax= secs_cax, orientation='horizontal',
                     label='Equivalent FAC [$\mu/m^{2}$]', extend='both')
Q_J= secs_ax.quiver(eval_grid.xi[::3, ::3], eval_grid.eta[::3, ::3],
                    estimated.Je_df[::3, ::3], estimated.Jn_df[::3, ::3], zorder=100)


##%% Add Measured
mag_ax.scatter(data.xi, data.eta, marker='*', color='darkorange', zorder=30, s=3000)
mag_ax.scatter(data.xi, data.eta, c=-data.Z_geo, zorder=50, s=800, vmin=-B_lim, vmax=B_lim, cmap='seismic')
mag_ax.quiver(data.xi, data.eta, data.Be_cs, data.Bn_cs, color='red', zorder=200)


##%% Add Coastlines
plots= []
for ax in [mag_ax, secs_ax, cur_ax]:
    for line in node_grid.projection.get_projected_coastlines('10m'):
        plots.extend(ax.plot(line[0], line[1], color='black', zorder=10))
    ax.set_xlim(node_grid.xi_min, node_grid.xi_max)
    ax.set_ylim(node_grid.eta_min, node_grid.eta_max)
    ax.set_aspect('equal')
for ax, cax in zip([mag_ax, secs_ax, cur_ax], [mag_cax, secs_cax, cur_cax]):
    pos= ax.get_position()
    pos2= cax.get_position()
    cax.set_position([pos2.x0, pos.y0-.13, pos.width, pos2.height])
    subplot_align(cax, ax, dim='x')

fig
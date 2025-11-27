from polplot.polplot import Polarplot as polar

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from apexpy import Apex
from secsy import cubedsphere as cs
import json
from File_tools.Filename_tools import date2filename
import os

Event_date = np.datetime64('2025-05-04T07:22')

folder= f"../../Data/{date2filename(Event_date)}/"

with open(folder+"node_grid.json", "r") as f:
    node_grid = cs.from_dictionary(json.load(f))
with open(folder+"eval_grid.json", "r") as f:
    eval_grid = cs.from_dictionary(json.load(f))

A= Apex(pd.Timestamp(Event_date.astype('datetime64[Y]')))

SM_index = pd.read_csv('../../Data/SM_Indices_2025.csv', sep=',', parse_dates=['Date_UTC'],)
SM_index.replace(999999, np.nan, inplace=True)

mlt_, mlat_= SM_index.loc[SM_index['Date_UTC'] == Event_date, ['SML_MLT', 'SML_MLAT']].values[0]
glat, glon = A.convert(mlat_, mlt_, 'mlt', 'geo', datetime=pd.Timestamp(Event_date))

from netCDF4 import Dataset
folder= '/home/simon/gits/SECpy_cleaned/Data/Sample data_ 2025-05-27/L1_TargetDays/'
file = [file for file in os.listdir(folder) if f'{Event_date.tolist().year}{Event_date.tolist().month:02d}{Event_date.tolist().day:02d}' in file][0]
ezie= Dataset(folder+file)
fig= plt.figure(figsize=(20, 20))
ax= polar(fig.add_subplot())
ax.scatter(mlat_, mlt_, s=100)
for beam, color in zip(range(1, 5), ['orange', 'green', 'red', 'blue']):
    time, lon, lat= ezie['Time']['time_utc'][:].astype('datetime64[ns]'), ezie['Geolocation'][f'obs_lon{beam}'][:], ezie['Geolocation'][f'obs_lat{beam}'][:]
    # ind= ((time-Event_date)>np.timedelta64(100, 'm'))&((time-Event_date)<np.timedelta64(105, 'm'))
    ind= abs(time-Event_date)<np.timedelta64(10, 'm')
    alt= 80
    lon, lat= lon.filled(np.nan), lat.filled(np.nan)
    mlat, mlt= A.convert(lat, lon, source='GEO', dest='mlt', datetime=pd.Timestamp(Event_date), height=alt)
    p= ax.plot(mlat[ind], mlt[ind], label=f'beam {beam}', color=color)
    # ax.plot(mlat, mlt, label=f'beam {beam}', linestyle='--', color=p[0].get_color(), zorder=20)

# lon=[]
# lat=[]
# #left
# lon.append(node_grid.lon[:, 0])
# lat.append(node_grid.lat[:, 0])
# #top
# lon.append(node_grid.lon[-1, :])
# lat.append(node_grid.lat[-1, :])
# #right
# lon.append(node_grid.lon[::-1, -1])
# lat.append(node_grid.lat[::-1, -1])
# #bottom
# lon.append(node_grid.lon[0, ::-1])
# lat.append(node_grid.lat[0, ::-1])

# mlat, mlt= A.convert(np.concatenate(lat), np.concatenate(lon), source='GEO',
#                      dest='mlt', datetime=pd.Timestamp(Event_date))
# ax.plot(mlat, mlt, zorder=10)

mlat, mlt= A.convert(node_grid.lat, node_grid.lon, source='GEO', dest='mlt', datetime=pd.Timestamp(Event_date))
ax.scatter(mlat, mlt, color='red', marker='+')
ax.coastlines(mag=A, color='black', zorder=-1)
ax.ax.legend()
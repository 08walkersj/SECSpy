from SECSpy.utils import date2filename
from pathlib import Path
from xarray import load_dataset
import pandas as pd
import json
from SECSpy.Grid_tools import shift_grid, change_xarray_grid
from apexpy import Apex

date= pd.Timestamp('2024-05-10T00:00')
A= Apex(date)

folder= Path('../Data/Gap_Analysis/'+date2filename(date, resolution='M'))
amplitudes= load_dataset(folder/Path('amplitudes.ncdf4'))

estimates= load_dataset(folder/Path('estimated.ncdf4'))

node_grid, eval_grid= shift_grid(json.loads(amplitudes.cs_grid), 352, 53.4, A,
                                 shift_lat=True)


amplitudes= change_xarray_grid(amplitudes, node_grid)
estimates= change_xarray_grid(estimates, eval_grid)

amplitudes.to_netcdf(folder/Path('amplitudes_ireland.ncdf4'))
estimates.to_netcdf(folder/Path('estimated_ireland.ncdf4'))


import matplotlib.pyplot as plt
fig= plt.figure()
ax= fig.add_subplot()
ax.scatter(node_grid.xi, node_grid.eta, marker='x', alpha=.3)
node_grid.add_natural_earth_feature(ax, 'coastline', '10m')
ax.set_aspect('equal')
ax.scatter(estimates.meas_xi.values, estimates.meas_eta.values, marker='p')
# ax.scatter(data.xi.iloc[0, :], data.eta.iloc[0, :], marker='p')
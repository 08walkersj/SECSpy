from SECSpy.SECS import SECS, RE
from apexpy import Apex
import xarray as xr
import numpy as np
from File_tools.Filename_tools import date2filename
from secsy import cubedsphere as cs
import json
import pandas as pd
Event_date = np.datetime64("2025-05-04T07:22")


folder= f"../../Data/{date2filename(Event_date)}/"

with open(folder+"node_grid.json", "r") as f:
    node_grid = cs.from_dictionary(json.load(f))
with open(folder+"eval_grid.json", "r") as f:
    eval_grid = cs.from_dictionary(json.load(f))
A= Apex(pd.Timestamp(Event_date.astype('datetime64[Y]')))


poles= SECS(node_grid.lon.flatten(), node_grid.lat.flatten(), eval_grid.lon.flatten(), eval_grid.lat.flatten(), 
            mode='image', current_radius=RE+110e3, image_current_radius=RE-500e3)
singularity_limit= np.min([node_grid.Wres, node_grid.Lres])/2

Le, Ln=node_grid.get_Le_Ln()
node_f1, node_f2= A.basevectors_qd(node_grid.lat.flatten(), node_grid.lon.flatten(), 110, coords='geo')
e= node_f1/np.linalg.norm(node_f1, axis=0)
n= node_f2/np.linalg.norm(node_f2, axis=0)
L_e= np.diag(e[0]).dot(Le) + np.diag(e[1]).dot(Ln)

l2=1
l3=1
λ2= l2* np.dot(L_e.T, L_e)/np.max(np.abs(np.dot(L_e.T, L_e)))
λ3= l3

L1= -np.identity(node_grid.shape[0])
for i in range(node_grid.shape[0]-1):
    L1[i, i+1]= 1
L1[-1]=0

L= [L1]+ [np.zeros(L1.shape)]*(node_grid.shape[-1]-1)
L1=[]
for i in range(node_grid.shape[-1]):
    if i==0:
        L1.append(np.concatenate(L, axis=1))
    else:
        L1.append(np.concatenate(L[i::-1][:-1] +L[:-i], axis=1))
L1= np.vstack(L1)

L2= -np.identity(L1.shape[0])
for i in range(L2.shape[0]-node_grid.shape[1]):
    L2[i, i+node_grid.shape[1]]=1
L2[-node_grid.shape[1]:]=0
L_cube= L1.T@L1+L2.T@L2
# L_cube_cf=L_cube*λ3_cf
# L_cube_df=L_cube*λ3_df
L_cube*=λ3
data= pd.read_csv(folder+'magnetometer_data.csv',
                parse_dates=['Date_UTC'])
vals, counts= np.unique(data.iaga.values, return_counts=True)
data.sort_values(['Date_UTC', 'glat'], inplace=True)
data.index= list(range(len(data)))
if len(np.unique(counts))>1:
    raise ValueError('inconsistent availability of magnetometers')
MagLon, MagLat= data.loc[data.Date_UTC==data.iloc[0].Date_UTC, ['glon', 'glat']].values.T
G_df= poles.G_Matrix(MagLon, MagLat)
GTG= G_df.T@G_df
scale= np.median(GTG)
matrix = GTG+ λ2*scale + L_cube*scale
Inverse= poles.Inverse_Matrix(matrix, cond=0)
poles.fitting_matrix['divergence-free']= np.dot(Inverse, G_df.T)
GJE_df, GJN_df= poles.eval_G_Matrix_J()
(GBR_df, GBE_df, GBN_df), (GBR2, GBE2, GBN2)= poles.eval_G_Matrix_B()
GBR_df+=GBR2
GBE_df+=GBE2
GBN_df+=GBN2


date= Event_date
East_ground= data[data.Date_UTC==date].E_geo.values*1e-9
South_ground=-data[data.Date_UTC==date].N_geo.values*1e-9
Up_ground= -data[data.Date_UTC==date].Z_geo.values*1e-9

d_df= np.concatenate((South_ground, East_ground, Up_ground))
DF_amps = poles.fitting_matrix['divergence-free']@d_df
DF_amps= np.zeros(DF_amps.shape).reshape(node_grid.shape)+1

DF_amps[10:15, 10:15]=10000
# DF_amps[20:25, 20:25]= -10000
DF_amps= DF_amps.T.flatten()

BE, BN= node_grid.projection.vector_cube_projection(GBE_df@DF_amps, GBN_df@DF_amps, poles.eval_longitude, 
                                                    poles.eval_latitude, return_xi_eta=False)
BE_df, BN_df, BU_df= BE.reshape(eval_grid.shape), BN.reshape(eval_grid.shape), (GBR_df@DF_amps).reshape(eval_grid.shape)



JE, JN= node_grid.projection.vector_cube_projection(GJE_df@DF_amps, GJN_df@DF_amps, poles.eval_longitude, 
                                                    poles.eval_latitude, return_xi_eta=False)
JE_df, JN_df= JE.reshape(eval_grid.shape), JN.reshape(eval_grid.shape)

#%%
import matplotlib.pyplot as plt

fig= plt.figure()
ax= fig.add_subplot()
ax.pcolormesh(node_grid.xi_mesh, node_grid.eta_mesh,
              DF_amps.reshape(node_grid.shape), vmin=-100, vmax=100, cmap='seismic')
ax.quiver(eval_grid.xi, eval_grid.eta, JE_df, JN_df)
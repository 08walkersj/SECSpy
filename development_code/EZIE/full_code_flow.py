#%%
import numpy as np
from SECSpy.Filename_tools import date2filename
from secsy import cubedsphere as cs
import json
import pandas as pd
from SECSpy.user_input_tools import validinput
def create_grid(Event_date, glon=False, glat=False, use_sml=False,
                use_ezie=True, mag_north=False, sat_orb=True, t_offset=5):
    #'../../Data/SM_Indices_2025.csv'
    from apexpy import Apex
    from secsy import cubedsphere as cs
    import os
    A= Apex(pd.Timestamp(Event_date.astype('datetime64[Y]')))

    if use_sml:
        SM_index = pd.read_csv('../../Data/SM_Indices_2025.csv', sep=',', parse_dates=['Date_UTC'],)
        SM_index.replace(999999, np.nan, inplace=True)

        mlt_, mlat_= SM_index.loc[SM_index['Date_UTC'] == Event_date, ['SML_MLT', 'SML_MLAT']].values[0]
        glat, glon = A.convert(mlat_, mlt_, 'mlt', 'geo', datetime=pd.Timestamp(Event_date))
    elif use_ezie:
        from netCDF4 import Dataset
        folder= '/home/simon/gits/SECpy_cleaned/Data/Sample data_ 2025-05-27/L1_TargetDays/'
        file = [file for file in os.listdir(folder) if f'{Event_date.tolist().year}{Event_date.tolist().month:02d}{Event_date.tolist().day:02d}' in file][0]
        ezie= Dataset(folder+file)
        time= ezie['Time']['time_utc'][:].astype('datetime64[ns]')
        ind= np.argmax(np.isfinite(ezie['Geolocation'][f'obs_lon{1}'][:].filled(np.nan))&\
            (np.arange(len(time))>=np.argmin(abs(time-(Event_date+np.timedelta64(t_offset, 'm'))))))
        closest= {'time':time[ind]}
        closest.update({f'lon{i}':ezie['Geolocation'][f'obs_lon{i}'][:][ind] for i in range(1, 5)})
        closest.update({f'lat{i}':ezie['Geolocation'][f'obs_lat{i}'][:][ind] for i in range(1, 5)})
        closest.update({'sat_lat':ezie['Geolocation']['sat_lat'][:][ind],
                        'sat_lon':ezie['Geolocation']['sat_lon'][:][ind],
                        'sat_vel_ecef':ezie['Geolocation']['sat_vel_ecef'][:][ind]})
        glon, glat= closest['lon1'], closest['lat1']
        # glon+=10

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

        vel_enu = ecef_to_enu(closest['sat_vel_ecef'], closest['sat_lat'], closest['sat_lon'])[:-1]
        East, North= vel_enu/np.linalg.norm(vel_enu)
        if North<0:
            # East*=-1
            North*=-1
        # Compute the angle from North (y-axis in ENU) to the velocity vector
        # east, north, up = vel_enu
        sat_heading = np.rad2deg(np.arctan2(East, North))  # radians, 0=north, +eastward
        # if north<0:
        #     sat_heading+=np.pi
        # sat_heading+np.pi/2
    if mag_north:
        f1, f2= A.basevectors_qd(glat, glon%360, 0, coords='geo')

        qd_north = f2/np.linalg.norm(f2)
        East, North= qd_north[0], qd_north[1]
        grid_proj= cs.CSprojection((glon, glat), (East, North))
        print('sat_heading', East, North)
        print('glon', glon%360)
        print('glat', glat)
    elif sat_orb:
        print('sat_heading', East, North)
        print('glon', glon%360)
        print('glat', glat)
        grid_proj= cs.CSprojection((glon%360, glat), sat_heading)

    node_grid= cs.CSgrid(grid_proj, 6000, 6000, 100., 100.)
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
def plot_pass(Event_date, save=True):
    from polplot.polplot import Polarplot as polar

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from apexpy import Apex
    from secsy import cubedsphere as cs
    import json
    from SECSpy.Filename_tools import date2filename
    import os


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
    ax= polar(fig.add_subplot(), color='grey', linewidth=2)
    ax.scatter(mlat_, mlt_, s=500, color='darkblue', zorder=1000, label='SML Index Magnetometer')
    for beam, color in zip(range(1, 5), ['orange', 'green', 'red', 'blue']):
        time, lon, lat= ezie['Time']['time_utc'][:].astype('datetime64[ns]'), ezie['Geolocation'][f'obs_lon{beam}'][:], ezie['Geolocation'][f'obs_lat{beam}'][:]
        # ind= ((time-Event_date)>np.timedelta64(100, 'm'))&((time-Event_date)<np.timedelta64(105, 'm'))
        ind= abs(time-Event_date)<np.timedelta64(10, 'm')
        alt= 80
        lon, lat= lon.filled(np.nan), lat.filled(np.nan)
        mlat, mlt= A.convert(lat, lon, source='GEO', dest='mlt', datetime=pd.Timestamp(Event_date), height=alt)
        p= ax.plot(mlat[ind], mlt[ind], label=f'beam {beam}', color=color, zorder=100, marker='d', lw=2)

    mlat, mlt= A.convert(node_grid.lat, node_grid.lon, source='GEO', dest='mlt', datetime=pd.Timestamp(Event_date))
    ax.scatter(mlat, mlt, color='red', marker='+', label='SECS Grid', zorder=50)
    ax.coastlines(mag=A, color='black', zorder=-1)
    ax.writeLTlabels(size=40)
    ax.writeLATlabels(lt=9, size=40, color='black', bbox= {'facecolor':'white', 'edgecolor':'none', 'alpha':.7, 'zorder':2})
    ax.plot(*A.convert(node_grid.lat[:, -1], node_grid.lon[:, -1], source='GEO', dest='mlt', datetime=pd.Timestamp(Event_date)),
            color='grey')
    ax.ax.legend(fontsize=30, ncols=3)
    folder= f"../../Figures/{date2filename(Event_date)}/"
    os.makedirs(folder, exist_ok=True)
    fig.suptitle(f'EZIE Pass for {Event_date.tolist()}', size=50, y=.9)
    if save:
        fig.savefig(folder+"EZIE_Pass.png")

    return fig
def download_data(Event_date):
    folder= f"../../Data/{date2filename(Event_date)}/"

    with open(folder+"node_grid.json", "r") as f:
        node_grid = cs.from_dictionary(json.load(f))
    with open(folder+"eval_grid.json", "r") as f:
        eval_grid = cs.from_dictionary(json.load(f))

    from SuperMAG.supermag_api_simon import SuperMAGGetData
    status, data= SuperMAGGetData('08walkersj', Event_date-np.timedelta64(60, 'm'), 2*60*60+60, ['GEO', 'decl'],
                        f'(GLAT>={node_grid.lat.min()}) & (GLAT<={node_grid.lat.max()}) &'
                        f'(GLON>={(node_grid.lon%360).min()}) & (GLON<={(node_grid.lon%360).max()})'
                        )
    decl_rad = np.deg2rad(data['decl'])
    data['N_geo'] = data['N_nez'] * np.cos(decl_rad) - data['E_nez'] * np.sin(decl_rad)
    data['E_geo'] = data['N_nez'] * np.sin(decl_rad) + data['E_nez'] * np.cos(decl_rad)
    data.replace(999999, np.nan, inplace=True)
    data.dropna(inplace=True)
    data['xi'], data['eta']= node_grid.projection.geo2cube(data.glon, data.glat)

    grid_ind= (data.xi>node_grid.xi[0, 0])&(data.xi<node_grid.xi[-1, -1])
    grid_ind&= (data.eta>node_grid.eta[0, 0])&(data.eta<node_grid.eta[-1, -1])
    data= data[grid_ind]
    data.to_csv(folder+'magnetometer_data.csv', index=False)


def run_inversion(times, Event_date, l2=1e2, l3=1e1):

    from SECSpy.SECS import SECS, RE
    from apexpy import Apex
    import xarray as xr
    save_option=True

    folder= f"../../Data/{date2filename(Event_date)}/"

    with open(folder+"node_grid.json", "r") as f:
        node_grid = cs.from_dictionary(json.load(f))
    with open(folder+"eval_grid.json", "r") as f:
        eval_grid = cs.from_dictionary(json.load(f))
    A= Apex(pd.Timestamp(Event_date.astype('datetime64[Y]')))

    if not l2 and not l3:
        with open(folder+'regularisation.json', 'r') as f:
            reg= json.load(f)
        l2, l3= reg['l2'], reg['l3']
        print('l2', l2, 'l3', l3)
        save_option=False

    poles= SECS(node_grid.lon.flatten(), node_grid.lat.flatten(), eval_grid.lon.flatten(), eval_grid.lat.flatten(), 
                mode='image', current_radius=RE+110e3, image_current_radius=RE-500e3)
    singularity_limit= np.min([node_grid.Wres, node_grid.Lres])/2

    Le, Ln=node_grid.get_Le_Ln()
    node_f1, node_f2= A.basevectors_qd(node_grid.lat.flatten(), node_grid.lon.flatten(), 110, coords='geo')
    e= node_f1/np.linalg.norm(node_f1, axis=0)
    n= node_f2/np.linalg.norm(node_f2, axis=0)
    L_e= np.diag(e[0]).dot(Le) + np.diag(e[1]).dot(Ln)

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

    L_cube*=λ3
    data= pd.read_csv(folder+'magnetometer_data.csv',
                    parse_dates=['Date_UTC'])
    vals, counts= np.unique(data.iaga.values, return_counts=True)
    data.sort_values(['Date_UTC', 'glat'], inplace=True)
    data.index= list(range(len(data)))
    if len(np.unique(counts))>1:
        print('inconsistent availability of magnetometers\n'+'\n'.join([f'{v}:{c}' for v, c in zip(vals, counts)]))
        if validinput('Remove Problem Site(s)? (y/n)', 'y', 'n'):
            sites= eval(input('Site keys as list'))
            for site in sites:
                data= data.loc[data.iaga!=site]
            data.to_csv(folder+'magnetometer_data.csv', index=False)
        else:
            raise ValueError('inconsistent availability of magnetometers\n'+'\n'.join([f'{v}:{c}' for v, c in zip(vals, counts)]))
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
    amps= []
    estimated= []
    for date in times:
        # date= Event_date
        East_ground= data[data.Date_UTC==date].E_geo.values*1e-9
        South_ground=-data[data.Date_UTC==date].N_geo.values*1e-9
        Up_ground= -data[data.Date_UTC==date].Z_geo.values*1e-9

        d_df= np.concatenate((South_ground, East_ground, Up_ground))
        DF_amps = poles.fitting_matrix['divergence-free']@d_df

        # Amplitude Dataset
        amps_= xr.Dataset(data_vars=dict(Amplitude_df=(['time', 'xi', 'eta'], [DF_amps.reshape(node_grid.shape).T], {'units': 'A'}),
                                        Area=(['xi', 'eta'], node_grid.A.T, {'units': 'm^2'})), 
                coords= dict(xi=node_grid.xi[0,:],
                                eta=node_grid.eta[:,0],
                                lon=(['xi', 'eta'], node_grid.lon.T),
                                lat=(['xi', 'eta'], node_grid.lat.T),
                                time=[date.astype('datetime64[ns]')]),
                attrs={'cs_grid': json.dumps(node_grid.to_dictionary())}
                )

        BE, BN= node_grid.projection.vector_cube_projection(GBE_df@DF_amps, GBN_df@DF_amps, poles.eval_longitude, 
                                                            poles.eval_latitude, return_xi_eta=False)
        BE_df, BN_df, BU_df= BE.reshape(eval_grid.shape), BN.reshape(eval_grid.shape), (GBR_df@DF_amps).reshape(eval_grid.shape)



        JE, JN= node_grid.projection.vector_cube_projection(GJE_df@DF_amps, GJN_df@DF_amps, poles.eval_longitude, 
                                                            poles.eval_latitude, return_xi_eta=False)
        JE_df, JN_df= JE.reshape(eval_grid.shape), JN.reshape(eval_grid.shape)

        # Magnetic Field and Current Dataset
        estimated_= xr.Dataset(data_vars=dict(Je_df=(['time', 'xi', 'eta'], [JE_df.T], {'units':'A/m'}),
                                            Jn_df=(['time', 'xi', 'eta'], [JN_df.T], {'units':'A/m'}),
                                    Be_df=(['time', 'xi', 'eta'], [BE_df.T], {'units':'T'}),
                                    Bn_df=(['time', 'xi', 'eta'], [BN_df.T], {'units':'T'}),
                                    Bu_df=(['time', 'xi', 'eta'], [BU_df.T], {'units':'T'}),),
                        coords=dict(xi= eval_grid.xi[0, :],
                                    eta=eval_grid.eta[:,0],
                                lon=(['xi', 'eta'], eval_grid.lon.T),
                                    lat=(['xi', 'eta'], eval_grid.lat.T),
                                    time=[date.astype('datetime64[ns]')]),
                attrs={'cs_grid': json.dumps(eval_grid.to_dictionary())}
                        )
        amps.append(amps_)
        estimated.append(estimated_)
    xr.concat(estimated, 'time').to_netcdf(folder+'estimated.ncdf4')
    xr.concat(amps, 'time').to_netcdf(folder+'Amplitudes.ncdf4')
    if save_option:
        if validinput('Save Regularisation Parameters? (y/n)', 'y', 'n'):
            with open(folder+"regularisation.json", "w") as f:
                json.dump({'l2':l2, 'l3':l3}, f)
    

def create_figure(Event_date, t=0, fig=False, ezie_pass=True, save=False):
    import matplotlib.pyplot as plt
    import xarray as xr
    from secsy import cubedsphere as cs
    from SECSpy.Filename_tools import date2filename
    from apexpy import Apex
    import pandas as pd
    import json
    from SECSpy.Subplot_alignment_tools import subplot_align
    from SECSpy.Contour_labels import contour_labels
    import os
    def mlt_contour_fmt(mlt):
        if mlt<0:
            mlt+=24
        try:
            return f'{int(mlt):02d}\nMLT'
        except:
            return 'wtf'
    folder= f"../../Data/{date2filename(Event_date)}/"

    with open(folder+"node_grid.json", "r") as f:
        node_grid = cs.from_dictionary(json.load(f))
    with open(folder+"eval_grid.json", "r") as f:
        eval_grid = cs.from_dictionary(json.load(f))
    A= Apex(pd.Timestamp(Event_date.astype('datetime64[Y]')))

    amps = xr.load_dataset(folder+'Amplitudes.ncdf4').isel(time=t)
    amps['Amplitude_df']= amps['Amplitude_df']/amps.Area
    estimated= xr.load_dataset(folder+'estimated.ncdf4').isel(time=t)

    data= pd.read_csv(folder+'magnetometer_data.csv',
                    parse_dates=['Date_UTC'])
    vals, counts= np.unique(data.iaga.values, return_counts=True)
    data.sort_values(['Date_UTC', 'glat'], inplace=True)
    data.index= list(range(len(data)))
    data['Be_cs'], data['Bn_cs']= node_grid.projection.vector_cube_projection(data.E_geo*1e-9, data.N_geo*1e-9, data.glon, data.glat, return_xi_eta=False)
    data= data.loc[data.Date_UTC==amps.time.values]
    if not fig:
        fig= plt.figure(figsize=(40, 20))
        gs= fig.add_gridspec(2, 3, height_ratios=[1, .1], hspace=0, wspace=.3)

        mag_ax= fig.add_subplot(gs[0, 0])
        mag_cax= fig.add_subplot(gs[1, 0])

        secs_ax= fig.add_subplot(gs[0, 1])
        secs_cax= fig.add_subplot(gs[1, 1])

        sml_ax= fig.add_subplot(gs[0, 2])
        leg_ax= fig.add_subplot(gs[1, 2])
        cbars=True
        sml=True
    else:
        mag_ax, mag_cax, secs_ax, secs_cax, sml_ax, leg_ax= fig.axes
        for ax in [mag_ax, secs_ax]: ax.clear()
        cbars= False
        sml= False

    Q_B= mag_ax.quiver(eval_grid.xi[::3, ::3], eval_grid.eta[::3, ::3],
                    estimated.Be_df.T[::3, ::3], estimated.Bn_df.T[::3, ::3], zorder=100,
                    scale= 6e-6)
    B_lim= 500
    Br= mag_ax.pcolormesh(eval_grid.xi_mesh, eval_grid.eta_mesh, estimated.Bu_df.values.T*1e9,
                          cmap='seismic', vmin=-B_lim, vmax=B_lim, zorder=-1)
    if cbars:
        Br_cbar= fig.colorbar(Br, cax=mag_cax, orientation='horizontal',
                            extend='both')
        Br_cbar.set_label(label='Br [nT]', size=30)
    secs_lim= 2
    Apc= secs_ax.pcolormesh(node_grid.xi_mesh, node_grid.eta_mesh, amps.Amplitude_df.values.T, cmap='seismic',
                        vmin= -secs_lim, vmax=secs_lim, zorder=-1)
    if cbars:
        A_cbar= fig.colorbar(Apc, cax= secs_cax, orientation='horizontal',
                            extend='both')
        A_cbar.set_label(label='Equivalent FAC [$\mu/m^{2}$]', size=30)
    Q_J= secs_ax.quiver(eval_grid.xi[::3, ::3], eval_grid.eta[::3, ::3],
                        estimated.Je_df.T[::3, ::3], estimated.Jn_df.T[::3, ::3], zorder=100,
                        scale=7e0)
    Magqk1 = mag_ax.quiverkey(
        Q_B, 0.26, 0.23, 800e-9, '', labelpos='E',
        coordinates='figure', zorder=500, transform=fig.transFigure
    )
    text1 = fig.text(
        Magqk1.X-.04, Magqk1.Y, 'Modelled', transform=fig.transFigure, zorder=501, size=30, va='center', ha='right'
    )

    Curqk= secs_ax.quiverkey(Q_J, 0.5, 0.23, 800e-3, '800 mA/m', labelpos='E',
        coordinates='figure', zorder=500, transform=fig.transFigure, fontproperties={'size':30}
    )

    mag_ax.scatter(data.xi, data.eta, marker='*', color='darkorange', zorder=110, s=3000, label='Magnetometer Site')
    mag_ax.scatter(data.xi, data.eta, c=-data.Z_geo, zorder=120, s=600, vmin=-B_lim, vmax=B_lim, cmap='seismic')
    Q_B2= mag_ax.quiver(data.xi, data.eta, data.Be_cs, data.Bn_cs, color='darkorange', zorder=200,
                        scale= 6e-6)


    Magqk2 = mag_ax.quiverkey(
        Q_B2, Magqk1.X, Magqk1.Y-.03, 800e-9, '800 nT'+'\n'*1, labelpos='E',
        coordinates='figure', zorder=500, transform=fig.transFigure, fontproperties={'size':30}
    )
    text2 = fig.text(
        Magqk2.X-0.04, Magqk2.Y, 'Measured', transform=fig.transFigure, zorder=501, size=30, va='center', ha='right'
    )
    if sml:
        SM_index = pd.read_csv('../../Data/SM_Indices_2025.csv', sep=',', parse_dates=['Date_UTC'],)
        SM_index.replace(999999, np.nan, inplace=True)
        ind= (SM_index.Date_UTC>Event_date-np.timedelta64(1, 'h'))&\
            (SM_index.Date_UTC<Event_date+np.timedelta64(1, 'h'))

        sml_ax.plot(SM_index.Date_UTC[ind], SM_index.SML[ind])
        sml_ax.axhline(0, color='black', lw=3)
        sml_ax.axvline(Event_date, color='darkred', linestyle='-', lw=8, label='Event Time', zorder=10)
    else:
        fig.time_vline.remove()
    fig.time_vline= sml_ax.axvline(amps.time.values, color='darkblue', linestyle='--', lw=8, label='Inversion Time', zorder=100)
    if ezie_pass:
        from netCDF4 import Dataset
        folder= '/home/simon/gits/SECpy_cleaned/Data/Sample data_ 2025-05-27/L1_TargetDays/'
        file = [file for file in os.listdir(folder) if f'{Event_date.tolist().year}{Event_date.tolist().month:02d}{Event_date.tolist().day:02d}' in file][0]
        ezie= Dataset(folder+file)
        for beam, color in zip(range(1, 5), ['orange', 'green', 'red', 'blue']):
            time, lon, lat= ezie['Time']['time_utc'][:].astype('datetime64[ns]'), ezie['Geolocation'][f'obs_lon{beam}'][:], ezie['Geolocation'][f'obs_lat{beam}'][:]
            # ind= ((time-Event_date)>np.timedelta64(100, 'm'))&((time-Event_date)<np.timedelta64(105, 'm'))
            ind= (time > amps.time.values-np.timedelta64(31, 'm'))&(time<=amps.time.values+np.timedelta64(30, 's'))
            alt= 80
            lon, lat= lon.filled(np.nan), lat.filled(np.nan)
            xi, eta= node_grid.projection.geo2cube(lon, lat)
            p= secs_ax.plot(xi[ind], eta[ind], label=f'beam {beam}', color=color, marker='d', lw=3, zorder=100_000)

    if sml:
        sml_ax.tick_params(labelsize=30)
        for ax in [mag_cax, secs_cax]:
            ax.tick_params(labelsize=30)
        import matplotlib.dates as mdates
        sml_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        for label in sml_ax.get_xticklabels():
            label.set_rotation(45)
    mlat, mlt= A.convert(node_grid.lat, node_grid.lon, source='GEO', dest='mlt', datetime=pd.Timestamp(amps.time.values), height=80)
    mlt[mlt>12]-=24
    mlt[mlat>80]=np.nan
    plots= []
    for ax in [mag_ax, secs_ax]:
        for line in node_grid.projection.get_projected_coastlines('10m'):
            plots.extend(ax.plot(line[0], line[1], color='black', zorder=10))
        ax.set_xlim(node_grid.xi_min, node_grid.xi_max)
        ax.set_ylim(node_grid.eta_min, node_grid.eta_max)
        ax.set_aspect('equal')
        clon=ax.contour(node_grid.xi, node_grid.eta, mlt, levels = np.sort(np.r_[-12:12:1]), colors = 'black', alpha=.5, linewidths=.5)
        clat=ax.contour(node_grid.xi, node_grid.eta, mlat, levels = np.r_[-90:90:10], colors = 'black', alpha=.5, linewidths=.5)
        ax.clon=contour_labels(clon, fmt=mlt_contour_fmt, sides=['bottom'], xpad=.06, ypad=.04, size=30)
        ax.clat=contour_labels(clat, fmt=lambda x: f'{int(x)}'+r'$^{\circ}$'+'\nmlat', xpad=.04, ypad=.07, sides=['left', 'right'], size=30)
        ax.axis('off')
    leg_ax.axis('off')
    sml_ax.spines[['top', 'right']].set_visible(False)
    if sml:
        subplot_align(sml_ax, secs_ax, dim='y')
    if sml:
        for ax, cax in zip([mag_ax, secs_ax, sml_ax], [mag_cax, secs_cax, leg_ax]):
            pos= ax.get_position()
            pos2= cax.get_position()
            cax.set_position([pos2.x0, pos.y0-.2, pos.width, pos2.height])
            subplot_align(cax, ax, dim='x')
        handles, labels = [], []
        for ax in [mag_ax, secs_ax, sml_ax]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        # Remove duplicates while preserving order
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if unique:
            handles, labels = zip(*unique)
            leg_ax.legend(handles, labels, loc='center', fontsize=30, frameon=False, ncols=2)
        leg_ax.axis('off')
    else:
        for ax, cax in zip([mag_ax, secs_ax], [mag_cax, secs_cax]):
            pos= ax.get_position()
            pos2= cax.get_position()
            cax.set_position([pos2.x0, pos.y0-.2, pos.width, pos2.height])
            subplot_align(cax, ax, dim='x')

    mag_ax.set_title('Magnetic Field', size=50)
    secs_ax.set_title('Current', size=50)
    sml_ax.set_title('SML Index', size=50)
    sml_ax.set_ylabel('Index [nT]', size=40, labelpad=-15)
    sml_ax.set_xlabel('Universal Time', size=40, labelpad=40)
    fig.suptitle(amps.time.values.astype('datetime64[m]').tolist(), size=70, y=.86)
    if save:
        folder= f"../../Figures/{date2filename(Event_date)}/"
        os.makedirs(folder, exist_ok=True)
        fig.savefig(folder+f"{t:02d}.png", tight_layout=True)
    return fig



#%%
# Event_date = np.datetime64("2025-05-04T07:22")
# Event_date = np.datetime64("2025-05-07T09:22")
Event_date = np.datetime64("2025-05-05T04:22")
# 1/0
# Event_date = np.datetime64("2025-04-21T02:19")
#%%
# create_grid(Event_date, mag_north=False, t_offset=1)
# #%%
# fig= plot_pass(Event_date, save=False)
# #%%
# download_data(Event_date)
# #%%
# run_inversion(np.arange(Event_date-np.timedelta64(30, 'm'),
#                         Event_date+np.timedelta64(31, 'm'),
#                         np.timedelta64(1, 'm')),
#                         Event_date,
#                         l2=3e3, l3=6e3
#                         # l2= False, l3=False
#                         )

#%%
t=0
fig= create_figure(Event_date, t=t, ezie_pass=False, save=True)
for t in range(1, 61):
    fig= create_figure(Event_date, t=t, fig=fig, ezie_pass=False, save=True)
#%%
from SECSpy.animate_png import process_all_folders
process_all_folders(base_folder= f"../../Figures/", num_digits=2, fps=2)
from netCDF4 import Dataset
import numpy as np
from secsy.cubedsphere import CSgrid, CSprojection
from secsy import cubedsphere as cs
import pandas as pd
from SECSpy import SECS
from SECSpy.constants import RE
from apexpy import Apex
from SECSpy.utils import date2filename
import os
import time
import matplotlib.pyplot as plt
from polplot import Polarplot as polar
import json
import xarray as xr

class inversion(object):
    def __init__(self, ezie_l2_file, event_time, SML_file=None, inversion_repo= None) -> None:
        self.ezie_l2_file = ezie_l2_file
        self.event_time = event_time
        self.SML_file = SML_file
        if inversion_repo is None:
            self.inversion_repo= f'../../Inversions/{date2filename(event_time)}/'
        else:
            self.inversion_repo= inversion_repo
        os.makedirs(self.inversion_repo, exist_ok=True)
        self.sat_heading= self.get_sat_vector()
        self.stations= None
        self.username= None
        self.l1= 1e2
        self.l2= 1e1
        self.poles= None
        self.A= None
        self.SM_data= None
        self.amps= None
        self.estimated= None
        self.SECS_kwargs= None
        
    def get_sat_vector(self):
        ezie = Dataset(self.ezie_l2_file)
        time= ezie['Time']['time_utc'][:].astype('datetime64[ns]')
        ind= np.argmax(np.isfinite(ezie['Geolocation'][f'obs_lon{1}'][:].filled(np.nan))&\
            (np.arange(len(time))>=np.argmin(abs(time-(self.event_time+np.timedelta64(0, 'm'))))))
        closest= {'time':time[ind]}
        closest.update({f'lon{i}':ezie['Geolocation'][f'obs_lon{i}'][:][ind] for i in range(1, 5)})
        closest.update({f'lat{i}':ezie['Geolocation'][f'obs_lat{i}'][:][ind] for i in range(1, 5)})
        closest.update({'sat_lat':ezie['Geolocation']['sat_lat'][:][ind],
                        'sat_lon':ezie['Geolocation']['sat_lon'][:][ind],
                        'sat_vel_ecef':ezie['Geolocation']['sat_vel_ecef'][:][ind]})
        glon, glat= closest['lon1'], closest['lat1']

        vel_enu = self._ecef_to_enu(closest['sat_vel_ecef'], closest['sat_lat'], closest['sat_lon'])[:-1]
        East, North= vel_enu/np.linalg.norm(vel_enu)
        if North<0:
            North*=-1
        self.lon_center, self.lat_center= glon, glat
        return np.rad2deg(np.arctan2(East, North))

    def _ecef_to_enu(self, v_ecef, lat, lon):
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
    def create_grid(self, magnetic_north=False, L=6000, W=6000, Lres=100, Wres=100, **grid_kwargs):
        self.A= Apex(pd.Timestamp(self.event_time.astype('datetime64[Y]')))
        if magnetic_north:
            f1, f2= self.A.basevectors_qd(self.lat_center, self.lon_center%360, 0, coords='geo')
            qd_north = f2/np.linalg.norm(f2)
            East, North= qd_north[0], qd_north[1]
            grid_proj= CSprojection((self.lon_center, self.lat_center), (East, North))
        else:
            grid_proj= CSprojection((self.lon_center, self.lat_center), self.sat_heading)

        self.node_grid= CSgrid(grid_proj, L, W, Lres, Wres, **grid_kwargs)
        xi_e  = np.hstack((self.node_grid.xi_mesh[0]    , self.node_grid.xi_mesh [0 , - 1] + self.node_grid.dxi )) - self.node_grid.dxi /2 
        eta_e = np.hstack((self.node_grid.eta_mesh[:, 0], self.node_grid.eta_mesh[-1,   0] + self.node_grid.deta)) - self.node_grid.deta/2
        self.eval_grid= CSgrid(grid_proj,
                            self.node_grid.L + self.node_grid.Lres, self.node_grid.W + self.node_grid.Wres, 
                            self.node_grid.Lres, self.node_grid.Wres, 
                            edges = (xi_e, eta_e), R = self.node_grid.R)
    def create_SECS(self, **SECS_kwargs):
        if not 'mode' in SECS_kwargs:
            SECS_kwargs['mode']= 'image'
        if not 'current_radius' in SECS_kwargs:
            SECS_kwargs['current_radius']= RE+110e3
        if not 'image_current_radius' in SECS_kwargs:
            SECS_kwargs['image_current_radius']= RE-500e3
        self.SECS_kwargs= SECS_kwargs.copy()
        self.poles= SECS(self.node_grid.lon.flatten(), self.node_grid.lat.flatten(), self.eval_grid.lon.flatten(), self.eval_grid.lat.flatten(), **SECS_kwargs)

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(val) for val in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit='ns')
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.timedelta64):
            return int(value.astype('timedelta64[ns]').astype(int))
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _save_file_backed_state(self, value, filename, file_type):
        if value is None:
            return None
        if isinstance(value, str):
            return os.path.abspath(value)

        path= os.path.abspath(os.path.join(self.inversion_repo, filename))
        if file_type == 'csv' and isinstance(value, pd.DataFrame):
            value.to_csv(path, index=False)
            return path
        if file_type == 'netcdf' and isinstance(value, xr.Dataset):
            value.to_netcdf(path)
            return path

        raise TypeError(f'Cannot save {filename}: unsupported object type {type(value)}')

    def _repo_path(self, filename, prefix=None):
        if prefix is None or prefix is False:
            return None
        if prefix is True:
            prefix= ''
        return os.path.join(self.inversion_repo, f'{prefix}{filename}')

    def save_state(self, save_file=None):
        if save_file is None:
            save_file= os.path.join(self.inversion_repo, 'inversion_state.json')
        save_file= os.path.abspath(save_file)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        SM_data_path= self._save_file_backed_state(self.SM_data, 'magnetometer_data.csv', 'csv')
        amps_path= self._save_file_backed_state(self.amps, 'amps.ncdf4', 'netcdf')
        estimated_path= self._save_file_backed_state(self.estimated, 'estimated.ncdf4', 'netcdf')
        stations_path= self._save_file_backed_state(self.stations, 'stations.csv', 'csv')

        state= {
            'state_file': save_file,
            'ezie_l2_file': os.path.abspath(self.ezie_l2_file) if self.ezie_l2_file is not None else None,
            'event_time': self.event_time,
            'SML_file': os.path.abspath(self.SML_file) if self.SML_file is not None else None,
            'inversion_repo': os.path.abspath(self.inversion_repo),
            'sat_heading': self.sat_heading,
            'lon_center': getattr(self, 'lon_center', None),
            'lat_center': getattr(self, 'lat_center', None),
            'username': self.username,
            'l1': self.l1,
            'l2': self.l2,
            'SECS_kwargs': self.SECS_kwargs,
            'files': {
                'SM_data': SM_data_path,
                'amps': amps_path,
                'estimated': estimated_path,
                'stations': stations_path,
            },
            'grids': {
                'node_grid': self.node_grid.to_dictionary() if hasattr(self, 'node_grid') else None,
                'eval_grid': self.eval_grid.to_dictionary() if hasattr(self, 'eval_grid') else None,
            },
        }
        state= self._json_safe(state)

        with open(save_file, 'w') as f:
            json.dump(state, f, indent=2)

        return state

    @staticmethod
    def _load_csv_state(path):
        if path is None:
            return None
        data= pd.read_csv(path)
        if 'Date_UTC' in data.columns:
            data['Date_UTC']= pd.to_datetime(data['Date_UTC']).values
        return data

    @staticmethod
    def _load_netcdf_state(path):
        if path is None:
            return None
        data= xr.open_dataset(path)
        loaded= data.load()
        data.close()
        return loaded

    @classmethod
    def load_state(cls, save_file):
        save_file= os.path.abspath(save_file)
        with open(save_file, 'r') as f:
            state= json.load(f)

        inv= cls.__new__(cls)
        inv.state_file= save_file
        inv.ezie_l2_file= state.get('ezie_l2_file')
        inv.event_time= np.datetime64(state['event_time']) if state.get('event_time') is not None else None
        inv.SML_file= state.get('SML_file')
        inv.inversion_repo= state.get('inversion_repo', os.path.dirname(save_file))
        os.makedirs(inv.inversion_repo, exist_ok=True)

        inv.sat_heading= state.get('sat_heading')
        inv.lon_center= state.get('lon_center')
        inv.lat_center= state.get('lat_center')
        inv.username= state.get('username')
        inv.l1= state.get('l1', 1e2)
        inv.l2= state.get('l2', 1e1)
        inv.SECS_kwargs= state.get('SECS_kwargs')
        inv.poles= None

        files= state.get('files', {})
        inv.SM_data= cls._load_csv_state(files.get('SM_data'))
        inv.stations= cls._load_csv_state(files.get('stations'))
        inv.amps= cls._load_netcdf_state(files.get('amps'))
        inv.estimated= cls._load_netcdf_state(files.get('estimated'))

        grids= state.get('grids', {})
        inv.node_grid= cs.from_dictionary(grids['node_grid']) if grids.get('node_grid') is not None else None
        inv.eval_grid= cs.from_dictionary(grids['eval_grid']) if grids.get('eval_grid') is not None else None

        inv.A= Apex(pd.Timestamp(inv.event_time.astype('datetime64[Y]'))) if inv.event_time is not None else None
        if inv.node_grid is not None and inv.eval_grid is not None and inv.SECS_kwargs is not None:
            inv.poles= SECS(inv.node_grid.lon.flatten(), inv.node_grid.lat.flatten(),
                            inv.eval_grid.lon.flatten(), inv.eval_grid.lat.flatten(),
                            **inv.SECS_kwargs)

        return inv
    def download_data(self, username=None, grid_buffer=(.1, .1),
                      start= -np.timedelta64(1, 'h'), end= np.timedelta64(1, 'h'), save=None):
        from SECSpy.SuperMAG_tools import SuperMAGGetData, _default_stations_path, _STATION_COLUMNS
        from SECSpy.utils import validinput
        if self.stations is None or 'xi' not in self.stations.columns:
            stations_path = _default_stations_path()
            stations = pd.read_csv(stations_path, names=_STATION_COLUMNS, skiprows=1)
            stations['xi'], stations['eta']= self.node_grid.projection.geo2cube(stations.GEOLON, stations.GEOLAT)
            self.stations= stations
        if username is not None:
            self.username= username
        elif self.username is None:
            raise ValueError('Username is required to download SuperMAG data.')
        else:
            username= self.username
        extent= (end-start + np.timedelta64(1, 'm')).astype('timedelta64[s]').astype(int)
        start = self.event_time + start
        xi_min=self.node_grid.xi_min.copy()
        xi_max=self.node_grid.xi_max.copy()
        xi_min-=grid_buffer[0]*xi_min
        xi_max+=grid_buffer[0]*xi_max
        eta_min=self.node_grid.eta_min.copy()
        eta_max=self.node_grid.eta_max.copy()
        eta_min-=grid_buffer[1]*eta_min
        eta_max+=grid_buffer[1]*eta_max
        self.SM_data= SuperMAGGetData(username, start-np.timedelta64(1, 'm'), extent, ['GEO', 'decl'], f'xi>={xi_min} & xi<={xi_max} & eta>={eta_min} & eta<={eta_max}', stations=self.stations)[1].sort_values(['Date_UTC', 'glat'])
        self.SM_data.index= list(range(len(self.SM_data)))
        decl_rad = np.deg2rad(self.SM_data['decl'])
        self.SM_data['N_geo'] = self.SM_data['N_nez'] * np.cos(decl_rad) - self.SM_data['E_nez'] * np.sin(decl_rad)
        self.SM_data['E_geo'] = self.SM_data['N_nez'] * np.sin(decl_rad) + self.SM_data['E_nez'] * np.cos(decl_rad)
        self.SM_data.replace(999999, np.nan, inplace=True)
        self.SM_data.dropna(inplace=True)
        vals, counts= np.unique(self.SM_data.iaga.values, return_counts=True)
        if len(np.unique(counts))>1:
            print('inconsistent availability of magnetometers\n'+'\n'.join([f'{v}:{c}' for v, c in zip(vals, counts)]))
            time.sleep(1)
            if validinput('Remove Problem Site(s)? (y/n)', 'y', 'n'):
                sites= input('Enter site IAGA codes to remove, separated by commas. Leave blank to remove all sites with incomplete data: ').strip()
                if sites=='':
                    sites= vals[counts!=np.max(counts)]
                else:
                    sites= [site.strip() for site in sites.split(',') if site.strip()]
                self.SM_data= self.SM_data.loc[~self.SM_data.iaga.isin(sites)]
                # self.SM_data.to_csv(folder+'magnetometer_data.csv', index=False)
            else:
                raise ValueError('inconsistent availability of magnetometers\n'+'\n'.join([f'{v}:{c}' for v, c in zip(vals, counts)]))
        save_path= self._repo_path('magnetometer_data.csv', save)
        if save_path is not None:
            self.SM_data.to_csv(save_path, index=False)
            self.SM_data= save_path
    def run_inversion(self, l1=None, l2=None, start=-np.timedelta64(30, 'm'), end=np.timedelta64(30, 'm'), save=None):
        if l1 is not None:
            self.l1= l1
        if l2 is not None:
            self.l2= l2
        l1= self.l1
        l2= self.l2
        if self.A is None:
            self.A= Apex(pd.Timestamp(self.event_time.astype('datetime64[Y]')))
        

        Le, Ln=self.node_grid.get_Le_Ln()
        node_f1, node_f2= self.A.basevectors_qd(self.node_grid.lat.flatten(), self.node_grid.lon.flatten(), 110, coords='geo')
        e= node_f1/np.linalg.norm(node_f1, axis=0)
        n= node_f2/np.linalg.norm(node_f2, axis=0)
        L_e= np.diag(e[0]).dot(Le) + np.diag(e[1]).dot(Ln)

        λ2= l1* np.dot(L_e.T, L_e)/np.max(np.abs(np.dot(L_e.T, L_e)))
        λ3= l2

        L1= -np.identity(self.node_grid.shape[0])
        for i in range(self.node_grid.shape[0]-1):
            L1[i, i+1]= 1
        L1[-1]=0

        L= [L1]+ [np.zeros(L1.shape)]*(self.node_grid.shape[-1]-1)
        L1=[]
        for i in range(self.node_grid.shape[-1]):
            if i==0:
                L1.append(np.concatenate(L, axis=1))
            else:
                L1.append(np.concatenate(L[i::-1][:-1] +L[:-i], axis=1))
        L1= np.vstack(L1)

        L2= -np.identity(L1.shape[0])
        for i in range(L2.shape[0]-self.node_grid.shape[1]):
            L2[i, i+self.node_grid.shape[1]]=1
        L2[-self.node_grid.shape[1]:]=0
        L_cube= L1.T@L1+L2.T@L2
        L_cube*=λ3
        if isinstance(self.SM_data, str):
            SM_path= self.SM_data
            self.SM_data= pd.read_csv(self.SM_data)
        else:
            SM_path= None


        MagLon, MagLat= self.SM_data.loc[self.SM_data.Date_UTC==self.SM_data.iloc[0].Date_UTC, ['glon', 'glat']].values.T
        G_df= self.poles.G_Matrix(MagLon, MagLat)
        GTG= G_df.T@G_df
        scale= np.median(GTG)
        matrix = GTG+ λ2*scale + L_cube*scale
        Inverse= self.poles.Inverse_Matrix(matrix, cond=0)
        self.poles.fitting_matrix['divergence-free']= np.dot(Inverse, G_df.T)
        GJE_df, GJN_df= self.poles.eval_G_Matrix_J()
        (GBR_df, GBE_df, GBN_df), (GBR2, GBE2, GBN2)= self.poles.eval_G_Matrix_B()
        GBR_df+=GBR2
        GBE_df+=GBE2
        GBN_df+=GBN2
        amps= []
        estimated= []

        for date in np.arange(self.event_time+start, self.event_time+end+np.timedelta64(1, 'm'), np.timedelta64(1, 'm')):
            East_ground= self.SM_data[self.SM_data.Date_UTC==date].E_geo.values*1e-9
            South_ground=-self.SM_data[self.SM_data.Date_UTC==date].N_geo.values*1e-9
            Up_ground= -self.SM_data[self.SM_data.Date_UTC==date].Z_geo.values*1e-9

            d_df= np.concatenate((South_ground, East_ground, Up_ground))
            DF_amps = self.poles.fitting_matrix['divergence-free']@d_df

            # Amplitude Dataset
            amps_= xr.Dataset(data_vars=dict(Amplitude_df=(['time', 'xi', 'eta'], [DF_amps.reshape(self.node_grid.shape).T], {'units': 'A'}),
                                            Area=(['xi', 'eta'], self.node_grid.A.T, {'units': 'm^2'})), 
                    coords= dict(xi=self.node_grid.xi[0,:],
                                    eta=self.node_grid.eta[:,0],
                                    lon=(['xi', 'eta'], self.node_grid.lon.T),
                                    lat=(['xi', 'eta'], self.node_grid.lat.T),
                                    time=[date.astype('datetime64[ns]')]),
                    attrs={'cs_grid': json.dumps(self.node_grid.to_dictionary())})

            BE, BN= self.node_grid.projection.vector_cube_projection(GBE_df@DF_amps, GBN_df@DF_amps, self.poles.eval_longitude, 
                                                                self.poles.eval_latitude, return_xi_eta=False)
            BE_df, BN_df, BU_df= BE.reshape(self.eval_grid.shape), BN.reshape(self.eval_grid.shape), (GBR_df@DF_amps).reshape(self.eval_grid.shape)



            JE, JN= self.node_grid.projection.vector_cube_projection(GJE_df@DF_amps, GJN_df@DF_amps, self.poles.eval_longitude, 
                                                                self.poles.eval_latitude, return_xi_eta=False)
            JE_df, JN_df= JE.reshape(self.eval_grid.shape), JN.reshape(self.eval_grid.shape)

            # Magnetic Field and Current Dataset
            estimated_= xr.Dataset(data_vars=dict(Je_df=(['time', 'xi', 'eta'], [JE_df.T], {'units':'A/m'}),
                                                Jn_df=(['time', 'xi', 'eta'], [JN_df.T], {'units':'A/m'}),
                                        Be_df=(['time', 'xi', 'eta'], [BE_df.T], {'units':'T'}),
                                        Bn_df=(['time', 'xi', 'eta'], [BN_df.T], {'units':'T'}),
                                        Bu_df=(['time', 'xi', 'eta'], [BU_df.T], {'units':'T'}),),
                            coords=dict(xi= self.eval_grid.xi[0, :],
                                        eta=self.eval_grid.eta[:,0],
                                    lon=(['xi', 'eta'], self.eval_grid.lon.T),
                                        lat=(['xi', 'eta'], self.eval_grid.lat.T),
                                        time=[date.astype('datetime64[ns]')]),
                    attrs={'cs_grid': json.dumps(self.eval_grid.to_dictionary())})
            amps.append(amps_)
            estimated.append(estimated_)
        self.amps= xr.concat(amps, dim='time')
        self.estimated= xr.concat(estimated, dim='time')
        if SM_path is not None:
            self.SM_data= SM_path
        amps_path= self._repo_path('amps.ncdf4', save)
        estimated_path= self._repo_path('estimated.ncdf4', save)
        if amps_path is not None:
            self.amps.to_netcdf(amps_path)
            self.amps= amps_path
            self.estimated.to_netcdf(estimated_path)
            self.estimated= estimated_path
        return self.amps, self.estimated
    def plot_pass(self, save=None):
        fig= plt.figure(figsize=(20, 20))
        ax= polar(fig.add_subplot(), color='grey', linewidth=2)
        ezie = Dataset(self.ezie_l2_file)
        time= ezie['Time']['time_utc'][:].astype('datetime64[ns]')
        for beam, color in zip(range(1, 5), ['orange', 'green', 'red', 'blue']):
            time, lon, lat= ezie['Time']['time_utc'][:].astype('datetime64[ns]'), ezie['Geolocation'][f'obs_lon{beam}'][:], ezie['Geolocation'][f'obs_lat{beam}'][:]
            ind= abs(time-self.event_time)<np.timedelta64(10, 'm')
            alt= 80
            lon, lat= lon.filled(np.nan), lat.filled(np.nan)
            mlat, mlt= self.A.convert(lat, lon, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time), height=alt)
            p= ax.plot(mlat[ind], mlt[ind], label=f'beam {beam}', color=color, zorder=100, marker='d', lw=2)

        mlat, mlt= self.A.convert(self.node_grid.lat, self.node_grid.lon, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time))
        ax.scatter(mlat, mlt, color='red', marker='+', label='SECS Grid', zorder=50)
        ax.coastlines(mag=self.A, color='black', zorder=-1)
        ax.writeLTlabels(size=40)
        ax.writeLATlabels(lt=9, size=40, color='black', bbox= {'facecolor':'white', 'edgecolor':'none', 'alpha':.7, 'zorder':2})
        ax.plot(*self.A.convert(self.node_grid.lat[:, -1], self.node_grid.lon[:, -1], source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time)),
                color='grey')
        
        if self.SM_data is not None:
            if isinstance(self.SM_data, str):
                SM_data= pd.read_csv(self.SM_data)
                lon, lat= SM_data.glon[np.unique(SM_data.iaga, return_index=True)[1]], SM_data.glat[np.unique(SM_data.iaga, return_index=True)[1]]
            elif isinstance(self.SM_data, pd.DataFrame):
                lon, lat= self.SM_data.glon[np.unique(self.SM_data.iaga, return_index=True)[1]], self.SM_data.glat[np.unique(self.SM_data.iaga, return_index=True)[1]]
            mlat, mlt= self.A.convert(lat, lon, source='GEO', dest='mlt', datetime=pd.Timestamp(self.event_time))
            ax.scatter(mlat, mlt, color='darkorange', marker='h', label='SuperMAG Stations', zorder=100, s=100, edgecolor='black')
        else:
            print('No SuperMAG data')
        
        

        ax.ax.legend(fontsize=30, ncols=3)
        save_path= self._repo_path('EZIE_Pass.png', save)
        if save_path is not None:
            fig.savefig(save_path)
        return fig

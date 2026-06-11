
def download_ingrid(node_grid, start_time, end_time, supermag_id, save_file='./magnetometer_data.csv'):
    from secsy import cubedsphere as cs
    import json
    import numpy as np
    from SuperMAG.supermag_api_simon import SuperMAGGetData

    #%% Load Grid
    if isinstance(node_grid, str):
        with open(node_grid, "r") as f:
            node_grid = cs.from_dictionary(json.load(f))
    duration= (end_time - start_time + np.timedelta64(1, 'm')).astype('timedelta64[s]').astype(int)
    print((end_time-start_time).astype('timedelta64[s]'))
    #%% Get Data
    status, data= SuperMAGGetData(supermag_id, start_time-np.timedelta64(60, 's'), duration, ['GEO', 'decl'],
                        f'(GLAT>={node_grid.lat.min()}) & (GLAT<={node_grid.lat.max()}) &'
                        f'(GLON>={(node_grid.lon%360).min()}) & (GLON<={(node_grid.lon%360).max()})'
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
    data.to_csv(save_file, index=False)
    return save_file
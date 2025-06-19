"""Classes"""
from .SECpy import ArgumentError
import numpy as np
#Plotting tools that can be used alongisde the SECS analysis
#Cartopy is used to allow plotting of SECS models onto maps
def Geocentric_to_PlateCarree_vector_components(east, north, latitude):
    """ convert east north vector components to Plate Carree projection 

        parameters
        ----------
        east : array-like
            eastward components
        north : array-like
            westward components
        latitude : array-like
            latitude of each vector

        returns
        -------
        east, north in Plate Carree projection
        Preserveres shape and norm

    """
    magnitude = np.sqrt(east**2 + north**2)

    east_pc = east / np.cos(latitude * np.pi / 180)

    magnitude_pc = np.sqrt(east_pc**2 + north**2)

    east_pc  = east_pc * magnitude / magnitude_pc
    north_pc = north * magnitude / magnitude_pc

    return east_pc, north_pc
def features(axis, coastlines_only=False, features_list=False, features_kwargs=None):
    """
    Parameters
    ----------
    axis : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis object that map features are to be added to.

    Returns
    -------
    None.

    """
    import cartopy
    axis.coastlines(resolution='50m', zorder=200)
    if not coastlines_only and not features_list:
        axis.add_feature(cartopy.feature.OCEAN, facecolor='aqua')
        # axis.add_feature(cartopy.feature.COASTLINE, zorder=200)
        
        axis.add_feature(cartopy.feature.BORDERS, linestyle=':')
        axis.add_feature(cartopy.feature.LAKES, alpha=0.5, facecolor= 'mediumseagreen')
        axis.add_feature(cartopy.feature.RIVERS, edgecolor= 'mediumseagreen')
        axis.add_feature(cartopy.feature.LAND, facecolor='green')
    elif coastlines_only and features_list:
        raise ArgumentError('Either coast lines only or a list of features both cannot be inputs')
    elif not coastlines_only and features_list:
        for i, Feature in enumerate(features_list):
            if features_kwargs is None:
                if Feature.lower()=='borders':
                    axis.add_feature(cartopy.feature.BORDERS, linestyle=':')
                elif Feature.lower()=='lakes':
                    axis.add_feature(cartopy.feature.LAKES, alpha=0.5, facecolor= 'mediumseagreen')
                elif Feature.lower()=='rivers':
                    axis.add_feature(cartopy.feature.RIVERS, edgecolor= 'mediumseagreen')
                elif Feature.lower()=='land':
                    axis.add_feature(cartopy.feature.LAND, facecolor='green')
                elif Feature.lower()== 'ocean':
                    axis.add_feature(cartopy.feature.OCEAN, facecolor='aqua')
                elif Feature.lower()=='coastlines':
                    axis.coastlines(resolution='50m', zorder=200)
            else:
                exec('axis.add_feature(cartopy.feature.'+Feature+',**features_kwargs[i])')
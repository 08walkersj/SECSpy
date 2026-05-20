"""Classes"""
from .SECS import ArgumentError
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
def add_image(ax, url, location=(0.5, 0), zoom=0.3, alpha=1.0):
    """
    Download an image and attach it to a Matplotlib axis as an annotation artist.

    Parameters
    ----------
    ax : object
        Matplotlib axis that will receive the image artist.
    url : str
        URL of the image to download.

    Returns
    -------
    object
        Artist returned by ``ax.add_artist()``.
    """
    from io import BytesIO
    from urllib.request import urlopen
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    img_data = urlopen(url).read()
    img = Image.open(BytesIO(img_data)).convert("RGBA")
    arr = np.asarray(img)

    # create offset image (zoom controls displayed size)
    imagebox = OffsetImage(arr, zoom=zoom, alpha=alpha)
    ab = AnnotationBbox(imagebox, location, frameon=False)  # attach at data coords
    return ax.add_artist(ab)


def align_axis_below(reference_axis, lower_axis, pad=0.1):
    """
    Place one Matplotlib axis directly below another and keep their widths aligned.

    The alignment is applied immediately and is also reapplied whenever the
    parent figure is resized.

    Parameters
    ----------
    reference_axis : matplotlib.axes.Axes
        Axis that defines the horizontal position, width, and upper anchor.
    lower_axis : matplotlib.axes.Axes
        Axis to move below ``reference_axis``. This is typically a colour-bar
        axis.
    pad : float, optional
        Vertical separation between the two axes in figure-coordinate units.
        The default is 0.1.

    Returns
    -------
    None
        Matplotlib ``set_position`` result from the initial alignment.
    """
    import functools

    def onresize(reference_axis, lower_axis, event):
        reference_position = reference_axis.get_position()
        lower_position = lower_axis.get_position()
        return lower_axis.set_position([
            lower_position.x0,
            reference_position.y0 - pad - lower_position.height,
            reference_position.width,
            lower_position.height,
        ])

    onresize_wrapper = functools.partial(onresize, reference_axis, lower_axis)
    callback_id = reference_axis.figure.canvas.mpl_connect(
        'resize_event',
        onresize_wrapper,
    )
    lower_axis._secspy_align_axis_below_callback_id = callback_id
    return onresize(reference_axis, lower_axis, None)


def add_land(ax, grid, resolution='50m', facecolor='darkgreen', **kwargs):
    """
    Add land polygons to a map axis using projected coastlines from a grid object.

    Parameters
    ----------
    ax : object
        Matplotlib axis that will receive the polygon patches.
    grid : object
        Grid object providing ``projection.get_projected_coastlines()``.
    resolution : str, optional
        Coastline resolution passed to the grid projection.
    facecolor : str, optional
        Fill color used for land polygons.
    **kwargs : dict
        Additional keyword arguments passed to ``PathPatch``.
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    plot_kwargs = dict(facecolor=facecolor, edgecolor='none', zorder=1)
    plot_kwargs.update(kwargs)
    land_patches = []
    for cl in grid.projection.get_projected_coastlines(resolution=resolution):
        x, y = cl
        if len(x) < 3:
            continue  # can't form a polygon
        # Build a closed Path; assume each cl represents a coastline ring
        verts = np.column_stack([x, y])
        # ensure closed
        if not (verts[0] == verts[-1]).all():
            verts = np.vstack([verts, verts[0]])

        # Path codes: MOVETO, then LINETO..., then CLOSEPOLY
        codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

        path = Path(verts, codes)
        patch = PathPatch(path, **plot_kwargs)  # land under outlines/grid
        land_patches.append(patch)

    for p in land_patches:
        ax.add_patch(p)


def add_ocean(ax, grid, facecolor='#3498db', **kwargs):
    """
    Add a rectangular ocean background patch covering the full grid extent.

    Parameters
    ----------
    ax : object
        Matplotlib axis that will receive the rectangle patch.
    grid : object
        Grid object exposing ``xi_min``, ``xi_max``, ``eta_min``, and ``eta_max``.
    facecolor : str, optional
        Fill color used for the ocean patch.
    **kwargs : dict
        Additional keyword arguments passed to ``Rectangle``.
    """
    from matplotlib.patches import Rectangle

    plot_kwargs = dict(facecolor=facecolor, edgecolor='none', zorder=-5)
    plot_kwargs.update(kwargs)

    x0, x1 = grid.xi_min, grid.xi_max
    y0, y1 = grid.eta_min, grid.eta_max

    ocean_patch = Rectangle((x0, y0), x1 - x0, y1 - y0, **plot_kwargs)
    ax.add_patch(ocean_patch)

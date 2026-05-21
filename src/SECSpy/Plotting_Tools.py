from SECSpy.SECS import ArgumentError
import numpy as np
from .natural_earth import (
    grid_extent as _grid_extent,
    natural_earth_land_mask,
    natural_earth_polygon_mask,
    project_natural_earth,
    projected_parts_to_path,
)


def _natural_earth_axis_state(ax):
    state = getattr(ax, "_secspy_natural_earth_state", None)
    if state is None:
        state = {}
        ax._secspy_natural_earth_state = state
    return state


def _remove_artist(artist):
    if artist is None:
        return
    try:
        artist.remove()
    except ValueError:
        pass


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


def add_natural_earth_feature(
    ax,
    grid,
    layer,
    resolution='50m',
    facecolor='none',
    edgecolor='black',
    linewidth=0.5,
    clip=True,
    geographic_buffer=2.0,
    **kwargs,
):
    """
    Add any bundled Natural Earth feature to a cubed-sphere map axis.

    ``land`` and ``ocean`` use the robust SECSpy-specific renderers by default:
    land is drawn as a cubed-sphere raster mask, and ocean is drawn as the map
    water background. Other polygon and line layers are projected from the
    bundled Natural Earth vectors.

    ``land`` returns a ``QuadMesh`` by default, ``ocean`` returns a background
    patch, other polygon layers return a ``PathPatch``, and line layers
    (``coastline``, ``borders``, ``states_provinces``,
    ``rivers_lake_centerlines``) return a ``LineCollection``.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import PathPatch, Rectangle

    extent = _grid_extent(grid)
    state = _natural_earth_axis_state(ax)

    if layer == 'land':
        if facecolor == 'none':
            facecolor = 'darkgreen'
        _remove_artist(state.pop('ocean_land_background', None))
        method = kwargs.pop('method', 'mask')
        mask_size = kwargs.pop('mask_size', 350)
        require_closed = kwargs.pop('require_closed', True)
        if method == 'mask':
            mesh_kwargs = dict(zorder=kwargs.pop('zorder', 1))
            if 'alpha' in kwargs:
                mesh_kwargs['alpha'] = kwargs.pop('alpha')
            mesh_kwargs.update(kwargs)
            mesh_kwargs.pop('facecolor', None)
            mesh_kwargs.pop('edgecolor', None)
            xi, eta, land_mask = natural_earth_land_mask(
                grid,
                resolution=resolution,
                nx=mask_size,
                geographic_buffer=geographic_buffer,
                extent=extent,
            )
            values = np.where(land_mask, 1.0, np.nan)
            artist = ax.pcolormesh(
                xi,
                eta,
                values,
                cmap=ListedColormap([facecolor]),
                shading='nearest',
                **mesh_kwargs,
            )
            state['land'] = artist
            return artist
        if method != 'polygon':
            raise ValueError("land method must be 'mask' or 'polygon'")

        plot_kwargs = dict(facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
        plot_kwargs.update(kwargs)
        parts = project_natural_earth(
            grid,
            "land",
            resolution=resolution,
            extent=extent,
            part_types={"polygon_exterior", "polygon_hole"},
            geographic_buffer=geographic_buffer,
            require_closed=require_closed,
        )
        path = projected_parts_to_path(parts)
        if path is None:
            return None
        artist = PathPatch(path, **plot_kwargs)
        if clip:
            x0, x1, y0, y1 = extent
            artist.set_clip_path(Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                transform=ax.transData,
            ))
        ax.add_patch(artist)
        state['land'] = artist
        return artist

    if layer == 'ocean':
        if facecolor == 'none':
            facecolor = '#3498db'
        use_natural_earth = kwargs.pop('use_natural_earth', False)
        mask_land = kwargs.pop('mask_land', True)
        mask_size = kwargs.pop('mask_size', 350)
        plot_kwargs = dict(facecolor=facecolor, edgecolor='none', zorder=-5)
        plot_kwargs.update(kwargs)
        x0, x1, y0, y1 = extent
        if use_natural_earth:
            parts = project_natural_earth(
                grid,
                "ocean",
                resolution=resolution,
                extent=extent,
                part_types={"polygon_exterior", "polygon_hole"},
                geographic_buffer=geographic_buffer,
            )
            path = projected_parts_to_path(parts)
            if path is not None:
                artist = PathPatch(path, **plot_kwargs)
                if clip:
                    artist.set_clip_path(Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        transform=ax.transData,
                    ))
                ax.add_patch(artist)
                return artist
        artist = Rectangle((x0, y0), x1 - x0, y1 - y0, **plot_kwargs)
        artist.set_clip_on(clip)
        ax.add_patch(artist)
        state['ocean'] = artist
        if mask_land and 'land' not in state and 'ocean_land_background' not in state:
            xi, eta, land_mask = natural_earth_land_mask(
                grid,
                resolution=resolution,
                nx=mask_size,
                geographic_buffer=geographic_buffer,
                extent=extent,
            )
            values = np.where(land_mask, 1.0, np.nan)
            background_land = ax.pcolormesh(
                xi,
                eta,
                values,
                cmap=ListedColormap([ax.get_facecolor()]),
                shading='nearest',
                zorder=plot_kwargs.get('zorder', -5) + 0.1,
            )
            state['ocean_land_background'] = background_land
        return artist

    if layer == 'lakes':
        if facecolor == 'none':
            facecolor = '#3498db'
        method = kwargs.pop('method', 'mask')
        mask_size = kwargs.pop('mask_size', 350)
        if method == 'mask':
            mesh_kwargs = dict(zorder=kwargs.pop('zorder', 1))
            if 'alpha' in kwargs:
                mesh_kwargs['alpha'] = kwargs.pop('alpha')
            mesh_kwargs.update(kwargs)
            mesh_kwargs.pop('facecolor', None)
            mesh_kwargs.pop('edgecolor', None)
            xi, eta, lake_mask = natural_earth_polygon_mask(
                grid,
                "lakes",
                resolution=resolution,
                nx=mask_size,
                geographic_buffer=geographic_buffer,
                extent=extent,
            )
            values = np.where(lake_mask, 1.0, np.nan)
            artist = ax.pcolormesh(
                xi,
                eta,
                values,
                cmap=ListedColormap([facecolor]),
                shading='nearest',
                **mesh_kwargs,
            )
            state['lakes'] = artist
            return artist
        if method != 'polygon':
            raise ValueError("lakes method must be 'mask' or 'polygon'")

    polygon_layer = layer in {"land", "ocean", "lakes"}
    parts = project_natural_earth(
        grid,
        layer,
        resolution=resolution,
        extent=extent,
        geographic_buffer=geographic_buffer,
        require_closed=polygon_layer,
    )
    if not parts:
        return None

    if polygon_layer:
        plot_kwargs = dict(
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        plot_kwargs.update(kwargs)
        artist = PathPatch(projected_parts_to_path(parts), **plot_kwargs)
    else:
        plot_kwargs = dict(colors=edgecolor, linewidths=linewidth)
        plot_kwargs.update(kwargs)
        if 'linestyle' in plot_kwargs and 'linestyles' not in plot_kwargs:
            plot_kwargs['linestyles'] = plot_kwargs.pop('linestyle')
        segments = [
            np.column_stack([part["x"], part["y"]])
            for part in parts
            if len(part["x"]) >= 2
        ]
        if not segments:
            return None
        artist = LineCollection(segments, **plot_kwargs)

    if clip:
        x0, x1, y0, y1 = extent
        clip_patch = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            transform=ax.transData,
        )
        artist.set_clip_path(clip_patch)
    ax.add_artist(artist)
    return artist


def contour_labels(contour, xpad=None, ypad=None, sides=['left', 'right'], rtol=.1, fmt=lambda x: f'{x}', x_splits=None, y_splits=None, **text_kwargs):
    """
    function for creating labels for contour lines at the edge of the subplot

    Parameters
    ----------
    contour : matplotlib.contour
        matplotlib contour object.
    xpad : float/int, optional
        shift the left or right labels. The default is None and uses .15% of absolute max of limits.
    ypad : float/int, optional
        shift the bottom or top labels. The default is None and uses .15% of absolute max of limits.
    sides : list/string, optional
        sides the labels are for. The default is ['left', 'right'].
    rtol : int/float, optional
        Defines how close the line must be to the axes limit for a label to be made. The default is .1, which is 10% of the limit away. Passed to numpy.isclose and atol=0.
    fmt : definition, optional
        format for the label string The default is lambda x: f'{x}'.
    x_splits : list/str, optional
        Used to define the area where labels should be found. The default is None. This is useful of the contour line intersects the axes limits in more than one place.
        Can define a string as positive (only look at the postive side of the x axis) or negative. Alternatively string can be provided as e.g. 'x>10' or any string with boolean design where only x is used.
        If multiple sides will be done to all if x_splits is a list is not provided. Use None to not activate this.
    y_splits : TYPE, optional
        Same as x_splits but for the y axis. The default is None.
    **text_kwargs : dictionary
        kwargs to be passed to matplotlib.axes.text.

    Raises
    ------
    ValueError
        Raised if an argument is not usable.

    Returns
    -------
    list
        list of text objects. If more than one side provided then will be a list of lists where the first list is a list of text objects corresponding to the first side provided etc.

    """
    import numpy as np
    ax= contour.axes
    if xpad is None:
        xpad= np.max(np.abs(ax.get_xlim()))*.015
    if ypad is None:
        ypad= np.max(np.abs(ax.get_ylim()))*.015
    try:
        if isinstance(sides, (str, np.str)):
            sides= [sides]
    except AttributeError:
        if isinstance(sides, (str)):
            sides= [sides]
    if x_splits is None or isinstance(x_splits, (str, np.str)):
        x_splits= [x_splits]*len(sides)
    if y_splits is None or isinstance(y_splits, (str, np.str)):
        y_splits= [y_splits]*len(sides)
    labels= [[] for i in range(len(sides))]
    for collection, level in zip(contour.collections, contour.levels):
        if not len(collection.get_paths()):
            continue
        for i, (side, x_split, y_split) in enumerate(zip(sides, x_splits, y_splits)):
            x, y= np.concatenate([path.vertices for path in collection.get_paths()], axis=0).T
            if side=='left':
                if y_split=='negative':
                    x= x[y<0]
                    y= y[y<0]
                elif y_split=='positive':
                    x= x[y>0]
                    y= y[y>0]
                elif not y_split is None and ('<' in y_split or '>' in y_split):
                    x= x[eval(y_split)]
                    y= y[eval(y_split)]
                elif not y_split is None:
                    raise ArgumentError(f"y_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {y_split}")
                if not np.any(np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)):
                    continue
                y= y[np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)]
                x= x[np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)]
                y= y[np.argmin(abs(x-ax.get_xlim()[0]))]
                x= x[np.argmin(abs(x-ax.get_xlim()[0]))]
                x= ax.get_xlim()[0]
                Xpad=xpad*-1
                Ypad=0
            elif side=='right':
                if y_split=='negative':
                    x= x[y<0]
                    y= y[y<0]
                elif y_split=='positive':
                    x= x[y>0]
                    y= y[y>0]
                elif not y_split is None and ('<' in y_split or '>' in y_split):
                    x= x[eval(y_split)]
                    y= y[eval(y_split)]
                elif not y_split is None:
                    raise ArgumentError(f"y_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {y_split}")
                if not np.any(np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)):
                    continue
                y= y[np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)]
                x= x[np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)]
                y= y[np.argmin(abs(x-ax.get_xlim()[1]))]
                x= x[np.argmin(abs(x-ax.get_xlim()[1]))]
                x= ax.get_xlim()[1]
                Xpad=xpad
                Ypad=0
            elif side=='bottom':
                if x_split=='negative':
                    y= y[x<0]
                    x= x[x<0]
                elif x_split=='positive':
                    y= y[x>0]
                    x= x[x>0]
                elif not x_split is None and ('<' in x_split or '>' in x_split):
                    y= y[eval(x_split)]
                    x= x[eval(x_split)]
                elif not x_split is None:
                    raise ArgumentError(f"x_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {x_split}")
                if not np.any(np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)):
                    continue
                x= x[np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)]
                y= y[np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)]
                x= x[np.argmin(abs(y-ax.get_ylim()[0]))]
                y= y[np.argmin(abs(y-ax.get_ylim()[0]))]
                y= ax.get_ylim()[0]
                Xpad=0
                Ypad=ypad*-1
            elif side=='top':
                if x_split=='negative':
                    y= y[x<0]
                    x= x[x<0]
                elif x_split=='positive':
                    y= y[x>0]
                    x= x[x>0]
                elif not x_split is None and ('<' in x_split or '>' in x_split):
                    y= y[eval(x_split)]
                    x= x[eval(x_split)]
                elif not x_split is None:
                    raise ArgumentError(f"x_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {x_split}")
                if not np.any(np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)):
                    continue
                x= x[np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)]
                y= y[np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)]
                x= x[np.argmin(abs(y-ax.get_ylim()[1]))]
                y= y[np.argmin(abs(y-ax.get_ylim()[1]))]
                y= ax.get_ylim()[1]
                Xpad=0
                Ypad=ypad
            else:
                raise ArgumentError(f"Invalid choice for side. Please choose either or any combination of: 'left', 'right', 'top' or 'bottom'\n you chose: {side}")
            labels[i].append(ax.text(x+Xpad, y+Ypad, fmt(level), va='center', ha='center', **text_kwargs))
    if len(sides)==1:
        return labels[0]
    return labels
def subplot_matchx(axis1, axis2):
    """

    Parameters
    ----------
    axis1 : matplotlib subplot
        subplot to be adjusted.
    axis2 : matplotlib subplot
        subplots who's x dimensions is to be matched.

    Returns
    -------
    Set position output.

    """
    def onresize(axis1, axis2, event):
        pos= axis2.get_position()
        return axis1.set_position([pos.x0, axis1.get_position().y0, pos.width, axis1.get_position().height])
    import functools
    onresize_wrapper=functools.partial(onresize, axis1, axis2)
    cid = axis1.figure.canvas.mpl_connect('resize_event', onresize_wrapper)
    return onresize(axis1, axis2, None)
def subplot_matchy(axis1, axis2):
    """

    Parameters
    ----------
    axis1 : matplotlib subplot
        subplot to be adjusted.
    axis2 : matplotlib subplot
        subplots who's y dimensions is to be matched.

    Returns
    -------
    Set position output.

    """
    def onresize(axis1, axis2, event):
        pos= axis2.get_position()
        return axis1.set_position([axis1.get_position().x0, pos.y0, axis1.get_position().width, pos.height])
    import functools
    onresize_wrapper=functools.partial(onresize, axis1, axis2)
    cid = axis1.figure.canvas.mpl_connect('resize_event', onresize_wrapper)
    return onresize(axis1, axis2, None)

def subplot_align(axis1, *axes, dim='x'):
    """
    Aligns the position of the given primary axis (axis1) with other specified axes based on the specified dimension.

    This function adjusts the position of the primary axis `axis1` to align with the other given axes along the specified dimension ('x', 'y', or 'both'). It listens for resize events on the figure canvas and dynamically adjusts the position of `axis1` to maintain alignment.

    Parameters:
    axis1 (matplotlib.axes.Axes): The primary axis whose position will be adjusted.
    *axes (matplotlib.axes.Axes): Additional axes to which the primary axis should be aligned.
    dim (str): The dimension along which to align the axes. Must be one of 'x', 'y', or 'both'. Default is 'x'.

    Raises:
    ArgumentError: If the specified dimension is not 'x', 'y', or 'both'.

    Returns:
    None

    Example:
    --------
    >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    >>> subplot_align(ax1, ax2, ax3, dim='y')
    
    This will align `ax1` with `ax2` and `ax3` along the y-axis.
    """
    
    # Check if the specified dimension is valid
    if not dim.lower() in ['x', 'y', 'both']:
        raise ArgumentError(f'dimension to align is not understood. You chose: {dim}. Please specify either "x" or "y" or "both".')
    
    def onresize(axis1, axes, event):
        """
        Inner function to handle the resize event by adjusting the position of axis1.
        """
        if dim.lower() == 'x':
            # Concatenate all x-axis positions of the specified axes
            x = np.concatenate([[ax.get_position().x0, ax.get_position().x1] for ax in axes])
            # Calculate the new width and x-position for axis1
            width = max(x) - min(x)
            x = min(x)
            y = axis1.get_position().y0
            height = axis1.get_position().height
        elif dim.lower() == 'y':
            # Concatenate all y-axis positions of the specified axes
            y = np.concatenate([[ax.get_position().y0, ax.get_position().y1] for ax in axes])
            # Calculate the new height and y-position for axis1
            height = max(y) - min(y)
            y = min(y)
            x = axis1.get_position().x0
            width = axis1.get_position().width
        elif dim.lower() == 'both':
            # Concatenate all x and y positions of the specified axes
            x = np.concatenate([[ax.get_position().x0, ax.get_position().x1] for ax in axes])
            width = max(x) - min(x)
            x = min(x)
            y = np.concatenate([[ax.get_position().y0, ax.get_position().y1] for ax in axes])
            height = max(y) - min(y)
            y = min(y)
        # Set the new position of axis1
        return axis1.set_position([x, y, width, height])
    
    import functools
    # Create a partial function that binds the axes and the onresize function
    onresize_wrapper = functools.partial(onresize, axis1, axes)
    # Connect the resize event of the figure to the onresize_wrapper function
    cid = axis1.figure.canvas.mpl_connect('resize_event', onresize_wrapper)
    
    # Initially align the axis1 with the specified axes
    return onresize(axis1, axes, None)

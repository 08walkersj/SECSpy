from SECSpy.SECS import ArgumentError
import numpy as np
import os
import tempfile
import pandas as pd
import inspect
from .natural_earth import (
    grid_extent as _grid_extent,
    natural_earth_land_mask,
    natural_earth_polygon_mask,
    project_natural_earth,
    projected_parts_to_path,
)

SECSPY_PLOT_FONT_FAMILY = 'DejaVu Sans'


def _ensure_writable_matplotlib_cache():
    cache_root = os.path.join(tempfile.gettempdir(), 'secspy-matplotlib')
    os.environ.setdefault('MPLCONFIGDIR', cache_root)
    os.environ.setdefault('XDG_CACHE_HOME', cache_root)
    os.makedirs(cache_root, exist_ok=True)


_ensure_writable_matplotlib_cache()

import matplotlib.pyplot as plt


def use_consistent_plot_fonts(font_family=SECSPY_PLOT_FONT_FAMILY):
    """Make normal Matplotlib text and mathtext use the same font family."""
    _ensure_writable_matplotlib_cache()

    import matplotlib as mpl

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [font_family],
        'font.cursive': [font_family],
        'text.usetex': False,
        'mathtext.fontset': 'custom',
        'mathtext.default': 'regular',
        'mathtext.rm': font_family,
        'mathtext.it': f'{font_family}:italic',
        'mathtext.bf': f'{font_family}:bold',
        'mathtext.cal': font_family,
        'mathtext.sf': font_family,
        'mathtext.tt': font_family,
        'axes.unicode_minus': False,
    })
    return font_family


def quiverkey_left_label(fig, qk, text, pad=0.008, **text_kwargs):
    if not hasattr(qk, 'vector'):
        qk._init()

    vertices = qk.verts.reshape(-1, 2)
    vertices_display = qk.vector.get_transform().transform(vertices)
    offset_display = qk.vector.get_offset_transform().transform(qk.vector.get_offsets()[0])
    arrow_display = vertices_display + offset_display
    arrow_figure = fig.transFigure.inverted().transform(arrow_display)

    return fig.text(
        np.nanmin(arrow_figure[:, 0]) - pad,
        0.5 * (np.nanmin(arrow_figure[:, 1]) + np.nanmax(arrow_figure[:, 1])),
        text,
        transform=fig.transFigure,
        ha='right',
        va='center',
        **text_kwargs,
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
def _mlt_contour_fmt(mlt):
    """Format magnetic local time contour labels.

    Parameters
    ----------
    mlt : float
        Magnetic local time in hours.

    Returns
    -------
    str
        Two-line contour label, for example ``'03\\nMLT'``.
    """
    if mlt < 0:
        mlt+= 24
    if mlt >= 24:
        mlt-= 24
    return f'{int(mlt):02d}\nMLT'


def _continuous_mlt_for_contours(mlt):
    """Wrap MLT around the plotted domain so contouring avoids a 24-hour jump."""
    mlt = np.asarray(mlt, dtype=float).copy()
    finite = np.isfinite(mlt)
    if not np.any(finite):
        return mlt
    angles = mlt[finite] * 2 * np.pi / 24
    center = np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles))) * 24 / (2 * np.pi)
    center %= 24
    mlt[finite] = ((mlt[finite] - center + 12) % 24) - 12 + center
    return mlt


def _first_column_values(frame, column):
    values = frame.loc[:, column]
    if isinstance(values, pd.DataFrame):
        return values.to_numpy().reshape(-1)
    return values.to_numpy()


def _has_station_component(frame, component):
    if isinstance(frame.columns, pd.MultiIndex):
        return component in frame.columns.get_level_values(0)
    return component in frame


def _set_station_component(frame, component, values, like_component):
    if isinstance(frame.columns, pd.MultiIndex):
        like = frame.loc[:, like_component]
        values = np.asarray(values).reshape(like.shape)
        component_columns = pd.MultiIndex.from_product(
            [[component], like.columns],
            names=frame.columns.names,
        )
        component_frame = pd.DataFrame(values, index=frame.index, columns=component_columns)
        other_columns = frame.columns.get_level_values(0) != component
        return pd.concat([frame.loc[:, other_columns], component_frame], axis=1)
    frame = frame.copy()
    frame[component] = values
    return frame


def _as_float_array(values):
    if hasattr(values, 'values'):
        values = values.values
    return np.asarray(values, dtype=float)


def _dataarray_unit(values):
    return str(getattr(values, 'attrs', {}).get('units', '')).strip().lower()


def _magnetic_values_nt(values):
    data = _as_float_array(values)
    unit = _dataarray_unit(values)
    if unit in {'t', 'tesla'}:
        return data * 1e9
    return data


def _magnetic_values_t(values):
    data = _as_float_array(values)
    unit = _dataarray_unit(values)
    if unit == 'nt':
        return data * 1e-9
    return data


def _sheet_current_values_a_per_m(values):
    data = _as_float_array(values)
    unit = _dataarray_unit(values).replace(' ', '')
    if unit in {'ua/m', 'microa/m', 'μa/m', 'ua/m^2', 'microa/m^2', 'μa/m^2'}:
        return data * 1e-6
    if unit in {'', 'ma/m', 'ma/m^2'}:
        return data * 1e-3
    return data


def _fac_values_microamps_per_m2(amplitude, area):
    fac = amplitude / area
    area_unit = _dataarray_unit(area).replace(' ', '')
    if area_unit in {'m^2', 'm2'}:
        return fac * 1e6
    return fac


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
    def iter_level_vertices(contour):
        if hasattr(contour, 'collections'):
            for collection, level in zip(contour.collections, contour.levels):
                paths= collection.get_paths()
                if len(paths):
                    yield [path.vertices for path in paths], level
        elif hasattr(contour, 'allsegs'):
            for segments, level in zip(contour.allsegs, contour.levels):
                vertices= [segment for segment in segments if len(segment)]
                if len(vertices):
                    yield vertices, level
        else:
            raise AttributeError("contour object does not expose collections or allsegs")

    ax= contour.axes
    if xpad is None:
        xpad= np.max(np.abs(ax.get_xlim()))*.015
    if ypad is None:
        ypad= np.max(np.abs(ax.get_ylim()))*.015
    if isinstance(sides, str):
        sides= [sides]
    if x_splits is None or isinstance(x_splits, str):
        x_splits= [x_splits]*len(sides)
    if y_splits is None or isinstance(y_splits, str):
        y_splits= [y_splits]*len(sides)
    labels= [[] for i in range(len(sides))]
    for vertices, level in iter_level_vertices(contour):
        for i, (side, x_split, y_split) in enumerate(zip(sides, x_splits, y_splits)):
            x, y= np.concatenate(vertices, axis=0).T
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
def create_CS_figure(node_grid, figsize=(40, 20), Br_lim=500, secs_lim=2, **grid_spec_kwargs):
    """Create the reusable three-panel Cartesian event figure.

    Parameters
    ----------
    figsize : tuple of float, default: (40, 20)
        Matplotlib figure size in inches.
    Br_lim : float, default: 500
        Symmetric radial magnetic perturbation color limit in nT.
    secs_lim : float, default: 2
        Symmetric equivalent current color limit used by the current panel.
    **grid_spec_kwargs
        Extra keyword arguments passed to ``Figure.add_gridspec``. Defaults
        are ``hspace=0``, ``wspace=0.3``, and
        ``height_ratios=[1, 0.1]`` when not supplied.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with magnetic-field, current, SML, colorbar, and legend axes.
    """
    use_consistent_plot_fonts()
    if 'hspace' not in grid_spec_kwargs:
        grid_spec_kwargs['hspace']= 0
    if 'wspace' not in grid_spec_kwargs:
        grid_spec_kwargs['wspace']= .3
    if 'height_ratios' not in grid_spec_kwargs:
        grid_spec_kwargs['height_ratios']= [1, .1]
    fig= plt.figure(figsize=figsize)
    gs= fig.add_gridspec(2, 3,**grid_spec_kwargs)

    mag_ax= fig.add_subplot(gs[0, 0])
    mag_ax.set_title('Ground Magnetic Field', size=50)
    mag_ax.Q_B= None
    mag_ax.Q_B2= None
    mag_ax.Br= None
    mag_ax.Magqk1=None
    mag_ax.Mag_scatter= None
    mag_ax.text1= None
    mag_ax.text2= None
    mag_ax.coastlines= []
    mag_ax.clon= []
    mag_ax.clat= []
    mag_cax= fig.add_subplot(gs[1, 0])
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-Br_lim, vmax=Br_lim), cmap='seismic'), cax=mag_cax, orientation='horizontal',
                    extend='both')
    cbar.set_label(label='Br [nT]', size=30)

    secs_ax= fig.add_subplot(gs[0, 1])
    secs_ax.set_title('Divergence-Free Current', size=50)
    secs_ax.ezie_pass= []
    secs_ax.Q_J= None
    secs_ax.Curqk= None
    secs_ax.A= None
    secs_ax.coastlines= []
    secs_ax.clon= []
    secs_ax.clat= []
    secs_cax= fig.add_subplot(gs[1, 1])
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-secs_lim, vmax=secs_lim), cmap='seismic'), cax=secs_cax,
                    orientation='horizontal', 
                    extend='both')
    cbar.set_label(label=r'Equivalent FAC [$\mu\mathrm{A}/m^{2}$]', size=30)

    leg_ax= fig.add_subplot(gs[1, 2])
    leg_ax.axis('off')
    leg_ax.leg= None
    for ax in [mag_ax, secs_ax]:
        ax.set_xlim(node_grid.xi_min, node_grid.xi_max)
        ax.set_ylim(node_grid.eta_min, node_grid.eta_max)
        ax.set_aspect('equal')

    for ax, cax in zip([mag_ax, secs_ax], [mag_cax, secs_cax, leg_ax]):
        pos= ax.get_position()
        pos2= cax.get_position()
        cax.set_position([pos2.x0, pos.y0-.2, pos.width, pos2.height])
        subplot_align(cax, ax, dim='x')
    for ax in [mag_ax, secs_ax]:
        ax.coastlines= node_grid.add_natural_earth_feature(ax, 'coastline', resolution='10m')
        ax.axis('off')
    for ax in [mag_cax, secs_cax]:
        ax.tick_params(labelsize=30)

    
    return fig
def create_CS_figure_SML(node_grid, SML_data, event_time, figsize=(40, 20), Br_lim=500, secs_lim=2, **grid_spec_kwargs):
    """Create the reusable three-panel Cartesian event figure.

    Parameters
    ----------
    figsize : tuple of float, default: (40, 20)
        Matplotlib figure size in inches.
    Br_lim : float, default: 500
        Symmetric radial magnetic perturbation color limit in nT.
    secs_lim : float, default: 2
        Symmetric equivalent current color limit used by the current panel.
    **grid_spec_kwargs
        Extra keyword arguments passed to ``Figure.add_gridspec``. Defaults
        are ``hspace=0``, ``wspace=0.3``, and
        ``height_ratios=[1, 0.1]`` when not supplied.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with magnetic-field, current, SML, colorbar, and legend axes.
    """
    use_consistent_plot_fonts()
    if 'hspace' not in grid_spec_kwargs:
        grid_spec_kwargs['hspace']= 0
    if 'wspace' not in grid_spec_kwargs:
        grid_spec_kwargs['wspace']= .3
    if 'height_ratios' not in grid_spec_kwargs:
        grid_spec_kwargs['height_ratios']= [1, .1]
    fig= plt.figure(figsize=figsize)
    gs= fig.add_gridspec(2, 3,**grid_spec_kwargs)

    mag_ax= fig.add_subplot(gs[0, 0])
    mag_ax.set_title('Ground Magnetic Field', size=50)
    mag_ax.Q_B= None
    mag_ax.Q_B2= None
    mag_ax.Br= None
    mag_ax.Magqk1=None
    mag_ax.Mag_scatter= None
    mag_ax.text1= None
    mag_ax.text2= None
    mag_ax.coastlines= []
    mag_ax.clon= []
    mag_ax.clat= []
    mag_cax= fig.add_subplot(gs[1, 0])
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-Br_lim, vmax=Br_lim), cmap='seismic'), cax=mag_cax, orientation='horizontal',
                    extend='both')
    cbar.set_label(label='Br [nT]', size=30)

    secs_ax= fig.add_subplot(gs[0, 1])
    secs_ax.set_title('Divergence-Free Current', size=50)
    secs_ax.ezie_pass= []
    secs_ax.Q_J= None
    secs_ax.Curqk= None
    secs_ax.A= None
    secs_ax.coastlines= []
    secs_ax.clon= []
    secs_ax.clat= []
    secs_cax= fig.add_subplot(gs[1, 1])
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-secs_lim, vmax=secs_lim), cmap='seismic'), cax=secs_cax,
                    orientation='horizontal', 
                    extend='both')
    cbar.set_label(label=r'Equivalent FAC [$\mu\mathrm{A}/m^{2}$]', size=30)

    sml_ax= fig.add_subplot(gs[0, 2])
    sml_ax.set_title('SML Index', size=50)
    sml_ax.set_ylabel('Index [nT]', size=40, labelpad=-15)
    sml_ax.set_xlabel('UTC [hh:mm]', size=40, labelpad=40)
    ind= (SML_data.Date_UTC>event_time-np.timedelta64(1, 'h'))&\
        (SML_data.Date_UTC<event_time+np.timedelta64(1, 'h'))

    sml_ax.plot(SML_data.Date_UTC[ind], SML_data.SML[ind])
    sml_ax.axhline(0, color='black', lw=3)
    sml_ax.axvline(event_time, color='darkred', linestyle='-', lw=8, label='Event Time', zorder=10)



    leg_ax= fig.add_subplot(gs[1, 2])
    leg_ax.axis('off')
    leg_ax.leg= None
    for ax in [mag_ax, secs_ax]:
        ax.set_xlim(node_grid.xi_min, node_grid.xi_max)
        ax.set_ylim(node_grid.eta_min, node_grid.eta_max)
        ax.set_aspect('equal')

    subplot_align(sml_ax, secs_ax, dim='y')
    for ax, cax in zip([mag_ax, secs_ax, sml_ax], [mag_cax, secs_cax, leg_ax]):
        pos= ax.get_position()
        pos2= cax.get_position()
        cax.set_position([pos2.x0, pos.y0-.2, pos.width, pos2.height])
        subplot_align(cax, ax, dim='x')
    for ax in [mag_ax, secs_ax]:
        ax.coastlines= node_grid.add_natural_earth_feature(ax, 'coastline', resolution='10m')
        ax.axis('off')
    sml_ax.spines[['top', 'right']].set_visible(False)
    sml_ax.tick_params(labelsize=30)
    for ax in [mag_cax, secs_cax]:
        ax.tick_params(labelsize=30)
    import matplotlib.dates as mdates
    sml_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    for label in sml_ax.get_xticklabels():
        label.set_rotation(45)
    
    return fig
def cartesian_plot(amplitudes, estimated, event_time, SM_data, Apex, time= np.timedelta64(0, 'm'), fig=None,
                    mag_quiver_kwargs=None,
                    mag_quiver_key_kwargs=None,
                    mag_scatter_kwargs=None,
                    current_quiver_kwargs=None,
                    current_quiver_key_kwargs=None, SML=False, **create_fig_kwargs):
    """Draw or update a Cartesian inversion frame for one relative time.

    Parameters
    ----------
    time : numpy.timedelta64, default: numpy.timedelta64(0, 'm')
        Plot time relative to ``self.event_time``.
    fig : matplotlib.figure.Figure, optional, default: None
        Existing figure from :meth:`create_CS_figure` to update. If
        ``None``, a new figure is created.
    mag_quiver_kwargs : dict, default: {'scale': 6e-6, 'zorder': 100}
        Keyword arguments for modeled magnetic-field quivers.
    mag_quiver_key_kwargs : dict, default:
        {'U': 800e-9, 'zorder': 500, 'fontproperties': {'size': 30}}
        Keyword arguments for magnetic-field quiver keys.
    mag_scatter_kwargs : dict, default:
        {'zorder': 120, 's': 600, 'marker': 'o', 'edgecolors': 'black'}
        Keyword arguments for measured SuperMAG station scatter markers.
    current_quiver_kwargs : dict, default: {'scale': 7e0, 'zorder': 100}
        Keyword arguments for equivalent-current quivers.
    current_quiver_key_kwargs : dict, default:
        {'U': 800e-3, 'zorder': 500, 'fontproperties': {'size': 30}}
        Keyword arguments for equivalent-current quiver keys.
    **create_fig_kwargs
        Passed to :meth:`create_CS_figure` when ``fig`` is ``None``.
        Common options include ``Br_lim`` and ``secs_lim``.

    Returns
    -------
    matplotlib.figure.Figure
        Updated Cartesian event figure.

    Notes
    -----
    File-backed ``SM_data``, ``SML_data``, ``estimated``, and ``amplitudes``
    are opened temporarily and restored to their original paths before
    returning.
    """
    use_consistent_plot_fonts()
    if mag_quiver_kwargs is None:
        mag_quiver_kwargs = {'scale': 6e-6, 'zorder': 100}
    else:
        mag_quiver_kwargs = mag_quiver_kwargs.copy()
    if mag_quiver_key_kwargs is None:
        mag_quiver_key_kwargs = {'U':800e-9, 'zorder':500, 'fontproperties':{'size':30}}
    else:
        mag_quiver_key_kwargs = mag_quiver_key_kwargs.copy()
    if mag_scatter_kwargs is None:
        mag_scatter_kwargs = {'zorder': 120, 's': 600, 'marker':'o', 'edgecolors': 'black'}
    else:
        mag_scatter_kwargs = mag_scatter_kwargs.copy()
    if current_quiver_kwargs is None:
        current_quiver_kwargs = {'scale': 7e0, 'zorder': 100}
    else:
        current_quiver_kwargs = current_quiver_kwargs.copy()
    if current_quiver_key_kwargs is None:
        current_quiver_key_kwargs = {'U': 800e-3, 'zorder':500, 'fontproperties':{'size':30}}
    else:
        current_quiver_key_kwargs = current_quiver_key_kwargs.copy()
    mag_scatter_kwargs.pop('c', None)

    if 'Br_lim' not in create_fig_kwargs:
        create_fig_kwargs['Br_lim']= 500
    if 'secs_lim' not in create_fig_kwargs:
        create_fig_kwargs['secs_lim']= 2



    if fig is None:
        if not SML:
            fig= create_CS_figure(**create_fig_kwargs)
        else:
            fig= create_CS_figure_SML(**create_fig_kwargs)
    if not SML:
        mag_ax, mag_cax, secs_ax, secs_cax, leg_ax= fig.axes
        sml_ax = None
    else:
        mag_ax, mag_cax, secs_ax, secs_cax, sml_ax, leg_ax= fig.axes
    

    # When reusing a figure for animation frames, remove only the dynamic
    # artists and keep the fixed axes, colorbars, coastlines, and layout.
    if SML:
        coupled= [
        (mag_ax, ['Magqk1', 'Magqk2', 'Q_B', 'Q_B2', 'Br', 'Mag_scatter', 'clon', 'clat']),
            (secs_ax, ['Curqk', 'Q_J', 'A', 'clon', 'clat']),
                (sml_ax, ['time_vline']),
        ]
    else:
        coupled= [
        (mag_ax, ['Magqk1', 'Magqk2', 'Q_B', 'Q_B2', 'Br', 'Mag_scatter', 'clon', 'clat']),
        (secs_ax, ['Curqk', 'Q_J', 'A', 'clon', 'clat'])
        ]
    for ax, attrs in coupled:
        for var in attrs:
            artists = getattr(ax, var, None)
            if artists is None:
                continue
            if not isinstance(artists, (list, tuple)):
                artists = [artists]
            stack = list(artists)
            while stack:
                artist = stack.pop()
                if isinstance(artist, (list, tuple)):
                    stack.extend(artist)
                    continue
                if hasattr(artist, 'remove'):
                    try:
                        artist.remove()
                    except ValueError:
                        pass
                    continue
                if hasattr(artist, 'collections'):
                    for item in artist.collections:
                        try:
                            item.remove()
                        except ValueError:
                            pass
            setattr(ax, var, [] if var in ['coastlines', 'clon', 'clat'] else None)
    if not _has_station_component(SM_data, 'Be_cs') or not _has_station_component(SM_data, 'Bn_cs'):
        node_grid = create_fig_kwargs.get('node_grid')
        if node_grid is None:
            from secsy.cubedsphere import CSgrid, CSprojection
            import json
            dic= json.loads(amplitudes.attrs['cs_grid'])
            node_grid= CSgrid(projection=CSprojection(**dic.pop('projection')), **dic)
        Be_cs, Bn_cs = node_grid.projection.vector_cube_projection(
            SM_data['East_ground'],
            -SM_data['South_ground'],
            SM_data['glon'],
            SM_data['glat'],
            return_xi_eta=False,
        )
        SM_data = _set_station_component(SM_data, 'Be_cs', Be_cs, 'East_ground')
        SM_data = _set_station_component(SM_data, 'Bn_cs', Bn_cs, 'South_ground')
    frame = estimated.sel(time=event_time + time)

    skip = 3




    mag_ax.Br= mag_ax.pcolormesh(
                    _as_float_array(estimated.xi_mesh).T,
                    _as_float_array(estimated.eta_mesh).T,
                    _magnetic_values_nt(frame.Bu).T,
                    cmap='seismic', vmin=-create_fig_kwargs['Br_lim'], vmax=create_fig_kwargs['Br_lim'], zorder=-1)
    u = frame.Be_cs.transpose('eta', 'xi').isel(
        eta=slice(None, None, skip),
        xi=slice(None, None, skip),
    )
    v = frame.Bn_cs.transpose('eta', 'xi').isel(
        eta=slice(None, None, skip),
        xi=slice(None, None, skip),
    )
    mag_ax.Q_B= mag_ax.quiver(u.xi.values, u.eta.values, _magnetic_values_t(u), _magnetic_values_t(v), **mag_quiver_kwargs)

    mag_quiver_kwargs_measured= mag_quiver_kwargs.copy()
    mag_quiver_kwargs_measured.update({'zorder': mag_quiver_kwargs['zorder']+100, 'color': 'darkorange'})
    sm_frame = SM_data.loc[SM_data.index == event_time + time]
    mag_ax.Mag_scatter= mag_ax.scatter(_first_column_values(sm_frame, 'xi'), _first_column_values(sm_frame, 'eta'),
                    c=_first_column_values(sm_frame, 'Up_ground')*1e9,
                    vmin=-create_fig_kwargs['Br_lim'], vmax=create_fig_kwargs['Br_lim'], cmap='seismic', **mag_scatter_kwargs)
    mag_ax.Q_B2= mag_ax.quiver(_first_column_values(sm_frame, 'xi'), _first_column_values(sm_frame, 'eta'),
                                _first_column_values(sm_frame, 'Be_cs'), _first_column_values(sm_frame, 'Bn_cs'),
                                **mag_quiver_kwargs_measured)

    amplitude_frame = amplitudes.sel(time=event_time+time).Amplitude
    area = amplitudes.Area
    if 'time' in area.dims or 'time' in area.coords:
        area = area.sel(time=event_time+time)
    secs_ax.A= secs_ax.pcolormesh(
                    _as_float_array(amplitudes.xi_mesh).T,
                    _as_float_array(amplitudes.eta_mesh).T,
                    _fac_values_microamps_per_m2(amplitude_frame, area).values.T, cmap='seismic',
                    vmin= -create_fig_kwargs['secs_lim'], vmax=create_fig_kwargs['secs_lim'], zorder=-1)
    u = frame.Je_cs.transpose('eta', 'xi').isel(
        eta=slice(None, None, skip),
        xi=slice(None, None, skip),
    )
    v = frame.Jn_cs.transpose('eta', 'xi').isel(
        eta=slice(None, None, skip),
        xi=slice(None, None, skip),
    )
    secs_ax.Q_J= secs_ax.quiver(u.xi, u.eta, _sheet_current_values_a_per_m(u), _sheet_current_values_a_per_m(v),
                **current_quiver_kwargs)

    mag_ax.Magqk1 = mag_ax.quiverkey(
        mag_ax.Q_B, 0.26, 0.23, label='', labelpos='E',
        coordinates='figure', transform=fig.transFigure, **mag_quiver_key_kwargs)
    if mag_ax.text1 is None:
        mag_ax.text1 = quiverkey_left_label(
            fig, mag_ax.Magqk1, 'Modelled',
            zorder=501, size=30
        )

    mag_ax.Magqk2 = mag_ax.quiverkey(
        mag_ax.Q_B2,
        mag_ax.Magqk1.X,
        mag_ax.Magqk1.Y - .03,
        label=f"{int(mag_quiver_key_kwargs['U'] * 1e9)} nT" + '\n',
        labelpos='E',
        coordinates='figure',
        transform=fig.transFigure,
        **mag_quiver_key_kwargs,
    )
    if mag_ax.text2 is None:
        mag_ax.text2 = quiverkey_left_label(
            fig, mag_ax.Magqk2, 'Measured',
            zorder=501, size=30
        )
    secs_ax.Curqk= secs_ax.quiverkey(secs_ax.Q_J, 0.5, 0.23,
        label=f"{int(current_quiver_key_kwargs['U']*1e3)} mA/m", labelpos='E',
        coordinates='figure', transform=fig.transFigure, **current_quiver_key_kwargs)
    if SML:
        sml_ax.time_vline= sml_ax.axvline(event_time+time, color='darkblue', linestyle='--', lw=8, label='Inversion Time', zorder=100)


    lat = _as_float_array(amplitudes.lat)
    lon = _as_float_array(amplitudes.lon)
    mlat, mlt= Apex.convert(lat, lon, source='GEO', dest='mlt',
                         datetime=pd.Timestamp(amplitudes.sel(time=event_time+time).time.values), height=80)
    mlat = _as_float_array(mlat)
    mlt = _as_float_array(mlt)
    mlt[mlat>80]=np.nan
    mlt = _continuous_mlt_for_contours(mlt)
    xi = _as_float_array(amplitudes.xi)
    eta = _as_float_array(amplitudes.eta)
    mlt_yx = mlt.T
    mlat_yx = mlat.T
    mlt_levels = np.arange(np.floor(np.nanmin(mlt)), np.ceil(np.nanmax(mlt)) + 1)
    for ax in [mag_ax, secs_ax]:
        clon=ax.contour(xi, eta, mlt_yx, levels = mlt_levels, colors = 'black', alpha=.5, linewidths=.5)
        clat=ax.contour(xi, eta, mlat_yx, levels = np.r_[-90:90:10], colors = 'black', alpha=.5, linewidths=.5)
        ax.clon=[clon, contour_labels(clon, fmt=_mlt_contour_fmt, sides=['bottom'], xpad=.06, ypad=.04, size=30)]
        ax.clat=[clat, contour_labels(clat, fmt=lambda x: f'{int(x)}'+r'$^{\circ}$'+'\nmlat', xpad=.04, ypad=.07, sides=['left', 'right'], size=30)]
        ax.axis('off')
    mag_ax.set_title('Ground Magnetic Field', size=50)
    secs_ax.set_title('Divergence-Free Current', size=50)
    fig.suptitle(f"{amplitudes.sel(time=event_time+time, method='nearest').time.values.astype('datetime64[m]').tolist()} UTC", size=70, y=.86)

    if leg_ax.leg is None:
        handles, labels = [], []
        legend_axes = [mag_ax, secs_ax]
        if SML:
            legend_axes.append(sml_ax)
        for ax in legend_axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        # Remove duplicates while preserving order
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if unique:
            handles, labels = zip(*unique)
            leg_ax.legend(handles, labels, loc='upper center', fontsize=30, frameon=False, ncols=2)
    return fig


def _time_values_from_dataset(dataset):
    if 'time' not in dataset.coords:
        raise ValueError("Dataset must have a 'time' coordinate.")
    return np.asarray(dataset.time.values, dtype='datetime64[ns]')


_TIME_LOOP_RESERVED_KWARGS = {
    'save_path',
    'save_dir',
    'save_root',
    'save_template',
    'savefig_kwargs',
}


def _accepted_kwargs(function, state, passthrough_kwargs=None):
    if passthrough_kwargs is None:
        passthrough_kwargs = {}
    signature = inspect.signature(function)
    accepted = {key: value for key, value in state.items() if key in signature.parameters}
    accepted.update({
        key: value
        for key, value in passthrough_kwargs.items()
        if key in signature.parameters
    })
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        accepted.update({
            key: value
            for key, value in passthrough_kwargs.items()
            if key not in _TIME_LOOP_RESERVED_KWARGS
        })
        return accepted
    return accepted


def _call_time_loop_function(function, state, passthrough_kwargs=None):
    kwargs = _accepted_kwargs(function, state, passthrough_kwargs=passthrough_kwargs)
    result = function(**kwargs)
    if result is None:
        return state['fig']
    if isinstance(result, tuple):
        result = result[0]
    return result


def save_time_loop_figure(fig, frame_index, n_digits, absolute_time, event_time, save_path=None,
                          save_dir=None, save_root=None, save_template=None, savefig_kwargs=None):
    """Save a frame produced by :func:`time_loop`.

    ``save_root``, ``save_path``, or ``save_dir`` is treated as the root
    directory. Frames are written into an event-day subfolder named with
    :func:`SECSpy.utils.date2filename`.
    """
    if fig is None:
        return fig
    from SECSpy.utils import date2filename

    root = save_root if save_root is not None else save_path
    if root is None:
        root = save_dir
    if root is None:
        root = './'
    event_date = np.datetime64(pd.Timestamp(event_time).to_datetime64(), 'ns')
    target_dir = os.path.join(root, date2filename(event_date, resolution='D'))
    if save_template is None:
        save_template = '{frame_index:0{n_digits}d}.png'
    if savefig_kwargs is None:
        savefig_kwargs = {}

    os.makedirs(target_dir, exist_ok=True)
    filename = save_template.format(
        frame_index=frame_index,
        n_digits=n_digits,
        absolute_time=absolute_time,
        event_day=date2filename(event_date, resolution='D'),
        frame_time=date2filename(np.datetime64(pd.Timestamp(absolute_time).to_datetime64(), 'ns')),
    )
    fig.savefig(os.path.join(target_dir, filename), **savefig_kwargs)
    return fig


def time_loop(amplitudes, estimated, functions=None, event_time=None, times=None, start=None, end=None,
              time_step=None, fig=None, draw=False, progress=True, **kwargs):
    """Loop over xarray time frames and reuse one Matplotlib figure.

    Parameters
    ----------
    amplitudes, estimated : xarray.Dataset
        Inversion datasets with a ``time`` coordinate.
    functions : callable or list of callable, optional
        Functions to run for each time frame, in order. Each function is called
        with the subset of this state that it accepts by name:
        ``amplitudes``, ``estimated``, ``event_time``, ``time``, ``absolute_time``,
        ``frame_index``, ``fig``, plus any extra ``**kwargs`` supplied to
        :func:`time_loop`. A function may return an updated figure or ``None``.
        Defaults to ``[cartesian_plot, save_time_loop_figure]``.
    event_time : datetime-like, optional
        Reference time used to convert absolute frame times to relative
        ``time`` offsets. Defaults to the first selected frame time.
    times : array-like of datetime-like or timedelta-like, optional
        Explicit frames to run. Datetime-like values are converted to relative
        offsets from ``event_time``; timedelta-like values are used directly.
    start, end, time_step : numpy.timedelta64, optional
        Relative-time range used when ``times`` is omitted. If omitted, the
        shared time coordinate from ``amplitudes`` and ``estimated`` is used.
    fig : matplotlib.figure.Figure, optional
        Existing figure to update and reuse across frames.
    draw : bool, default: False
        If ``True``, draw the canvas after each frame.
    progress : bool, default: True
        If ``True``, show a tqdm progress bar while processing frames.
    **kwargs
        Extra arguments forwarded to matching function parameters. For
        ``cartesian_plot`` this is where ``SM_data``, ``Apex`` and ``node_grid``
        should be supplied. For saving, pass ``save_root``, ``save_path``, or
        ``save_dir`` to choose the root; otherwise ``./`` is used.

    Returns
    -------
    matplotlib.figure.Figure
        The final reused figure.
    """
    if functions is None:
        functions = [cartesian_plot, save_time_loop_figure]
    elif callable(functions):
        functions = [functions]
    else:
        functions = list(functions)

    amplitude_times = _time_values_from_dataset(amplitudes)
    estimated_times = _time_values_from_dataset(estimated)
    shared_times = np.intersect1d(amplitude_times, estimated_times)
    if shared_times.size == 0:
        raise ValueError("amplitudes and estimated have no shared time values.")

    if event_time is None:
        event_time = pd.Timestamp(shared_times[0])
    event_time64 = np.datetime64(pd.Timestamp(event_time).to_datetime64(), 'ns')

    if times is None:
        if time_step is not None:
            if start is None:
                start = shared_times[0] - event_time64
            if end is None:
                end = shared_times[-1] - event_time64
            relative_times = np.arange(start, end + time_step, time_step)
        else:
            if start is None and end is None:
                selected_times = shared_times
            else:
                start_time = shared_times[0] if start is None else event_time64 + start
                end_time = shared_times[-1] if end is None else event_time64 + end
                selected_times = shared_times[(shared_times >= start_time) & (shared_times <= end_time)]
            relative_times = selected_times - event_time64
    else:
        times = np.asarray(times)
        if np.issubdtype(times.dtype, np.datetime64):
            relative_times = times.astype('datetime64[ns]') - event_time64
        else:
            relative_times = times.astype('timedelta64[ns]')

    n_digits = len(str(max(len(relative_times) - 1, 0)))
    frame_times = relative_times
    if progress:
        from tqdm import tqdm
        frame_times = tqdm(relative_times, desc='Plotting event', unit='frame')
    for frame_index, relative_time in enumerate(frame_times):
        absolute_time = event_time64 + relative_time
        state = dict(
            amplitudes=amplitudes,
            estimated=estimated,
            event_time=event_time,
            time=relative_time,
            absolute_time=pd.Timestamp(absolute_time),
            frame_index=frame_index,
            n_digits=n_digits,
            fig=fig,
        )
        for function in functions:
            fig = _call_time_loop_function(function, state, passthrough_kwargs=kwargs)
            state['fig'] = fig
        if draw and fig is not None:
            fig.canvas.draw()

    return fig

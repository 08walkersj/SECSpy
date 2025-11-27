import inspect
import numpy as np
class ArgumentError(Exception):
     pass
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


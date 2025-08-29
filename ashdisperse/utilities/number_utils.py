from math import ceil, floor
import numpy as np


def nice_round_up(val, mag=None):
    """Rounds up a number to a nice value of the form a*10^b for integer
        a and b.

    Args:
      val: the number to round.
      **mag: Optional; the value of the exponent b.

    Returns:
      A number rounded.
    """
    if mag is None:
        mag = 10**int(np.log10(val))
    return ceil(val/mag)*mag


def nice_round_down(val, mag=None):
    """Rounds down a number to a nice value of the form a*10^b for integer
        a and b.

    Args:
      val: the number to round.
      **mag: Optional; the value of the exponent b.

    Returns:
      A number rounded.
    """
    if mag is None:
        mag = 10**int(np.log10(val))
    return floor(val/mag)*mag


def log_levels(vmin, vmax):
    mag_low = floor(np.log10(vmin))
    mag_high = floor(np.log10(vmax))

    vals = np.arange(1, 10)
    levels = []
    for mag in range(mag_low, mag_high+1):
        levels.append(vals*10**mag)

    levels = np.asarray(levels).flatten()

    return levels[(levels >= vmin) & (levels <= vmax)]

def log_steps(vmin, vmax, step=10, include_max=True):
    if vmin<=0:
        raise ValueError('vmin must be positive, received {}'.format(vmin))
    levels = []
    this_level = vmin
    while this_level<vmax:
        levels.append(this_level)
        this_level *= step
    
    if include_max:
        if vmax not in levels:
            levels.append(vmax)

    return levels

def lin_levels(vmin, vmax, num=10):
    return np.linspace(vmin, vmax, num)

from numpy import (pi, arange, hstack)

def grid_freq(num_points):
    """Constructs the coordinates and wavenumbers for a fourier grid on [-pi,pi]

    Args:
        num_points: the number of points (should be a power of 2)

    Returns:
        x: the coordinates as a 1d numpy array
        k: the wavenumbers as a 1d numpy array
    """
    x = -pi + arange(0, num_points)*2*pi/num_points
    delta_k = 1
    k = delta_k*hstack((arange(0, num_points/2), arange(-num_points/2, 0)))
    return x, k

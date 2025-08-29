import numpy as np
from numba import complex128, float64, int64, jit

# from numba.pycc import CC
from numba.types import Tuple

# cc = CC('cheb')
# cc.verbose = True


# @cc.export('cheb', Tuple((float64[::1], float64[:, ::1], float64[:, ::1], float64[:, ::1]))(int64))
@jit(
    Tuple((float64[::1], float64[:, ::1], float64[:, ::1], float64[:, ::1]))(int64),
    nopython=True,
    cache=True,
)
def cheb(N):
    t = np.zeros((N), dtype=np.float64)
    x = np.zeros((N), dtype=np.float64)
    T = np.zeros((N, N), dtype=np.float64)
    dT = np.zeros((N, N), dtype=np.float64)
    d2T = np.zeros((N, N), dtype=np.float64)

    t[:] = np.pi * np.arange(0, N) / (N - 1)
    t = t[::-1]
    sint = np.sin(t)
    cost = np.cos(t)
    sin2t = sint * sint
    sin3t = sin2t * sint
    x = cost
    a = np.ones((N), dtype=np.float64)
    a[1::2] = -1.0
    T[:, 0] = 1.0
    dT[:, 0] = 0.0
    d2T[:, 0] = 0.0
    T[:, 1] = x
    dT[:, 1] = 1.0
    d2T[:, 1] = 0.0
    T[:, 2] = 2.0 * x * x - 1.0
    dT[:, 2] = 4.0 * x
    d2T[:, 2] = 4.0
    for n in range(3, N):
        T[1:-1, n] = np.cos(n * t[1:-1])
        T[0, n] = a[n]
        T[-1, n] = 1.0
        dT[1:-1, n] = n * np.sin(n * t[1:-1]) / sint[1:-1]
        dT[0, n] = -a[n] * n * n
        dT[-1, n] = n * n
        d2T[1:-1, n] = (
            n * cost[1:-1] * np.sin(n * t[1:-1]) / sin3t[1:-1]
            - n * n * np.cos(n * t[1:-1]) / sin2t[1:-1]
        )
        d2T[0, n] = a[n] * (n**4 - n**2) / 3
        d2T[-1, n] = (n**4 - n**2) / 3
    return x, T, dT, d2T


# @cc.export('ChebMat', complex128[:, ::1](int64, float64[::1]))
@jit(complex128[:, ::1](int64, float64[::1]), nopython=True, cache=True)
def ChebMat(N, x):
    """Evaluation at +1 of a function defined by Chebyshev coefficients.

    Args:
        coeffs: the Chebyshev coefficients as an array of complex numbers.

    Returns:
        The function evaluated at +1.
    """
    T = np.zeros((x.size, N), dtype=complex128)
    acos = np.arccos(x)
    for k in range(0, N):
        T[:, k] = np.cos(k * acos)
    return T


# @cc.export('cheb_val_p1', complex128(complex128[::1]))
@jit(complex128(complex128[::1]), nopython=True, cache=True)
def cheb_val_p1(coeffs):
    """Evaluates at +1 of a function defined by Chebyshev coefficients.

    Args:
        coeffs: the Chebyshev coefficients as an array of complex numbers.

    Returns:
        The function evaluated at +1.
    """
    return np.sum(coeffs)


# @cc.export('cheb_dif_p1', complex128(complex128[::1]))
@jit(complex128(complex128[::1]), nopython=True, cache=True)
def cheb_dif_p1(coeffs):
    """Evaluates at +1 of the first derivative of a function
        defined by Chebyshev coefficients.

    Args:
        coeffs: the Chebyshev coefficients as an array of complex numbers.

    Returns:
        The derivative of the function evaluated at +1.
    """
    n2 = np.arange(len(coeffs), dtype=np.complex128) ** 2
    df = np.dot(n2, np.ascontiguousarray(coeffs))
    return df


# @cc.export('cheb_val_m1', complex128(complex128[::1]))
@jit(complex128(complex128[::1]), nopython=True, cache=True)
def cheb_val_m1(coeffs):
    """Evaluates at -1 of a function defined by Chebyshev coefficients.

    Args:
        coeffs: the Chebyshev coefficients as an array of complex numbers.

    Returns:
        The function evaluated at -1.
    """
    a = np.ones((len(coeffs)))
    a[1::2] = -1.0
    f = np.dot(a.astype(np.complex128), np.ascontiguousarray(coeffs))
    return f


# @cc.export('cheb_dif_m1', complex128(complex128[::1]))
@jit(complex128(complex128[::1]), nopython=True, cache=True)
def cheb_dif_m1(coeffs):
    """Evaluation at -1 of the first derivative of a function
        defined by Chebyshev coefficients.

    Args:
        coeffs: the Chebyshev coefficients as an array of complex numbers.

    Returns:
        The derivative of the function evaluated at -1.
    """
    N = len(coeffs)
    n2 = np.arange(N) ** 2
    a = np.ones((N), dtype=np.int64)
    a[0::2] = -1.0
    b = a * n2
    func_deriv = np.dot(b.astype(np.complex128), np.ascontiguousarray(coeffs))
    return func_deriv


# @cc.export('convergedCoeffs', complex128[::1](complex128[::1], float64))
@jit(complex128[::1](complex128[::1], float64), nopython=True, cache=True)
def convergedCoeffs(coeffs, eps):

    N = coeffs.size

    a = np.abs(coeffs)
    max_a = np.amax(a)

    if max_a > 0:
        neg = (a < eps) * 1

        if neg[-1] and neg[-2]:
            n = np.arange(0, N)
            nGood = n[neg == 0]
            Ntrunc = nGood[-1] + 1
        else:
            Ntrunc = N

        new_coeffs = coeffs[:Ntrunc].flatten()

    else:
        new_coeffs = np.zeros((1), dtype=np.complex128)

    return new_coeffs


# if __name__ == "__main__":
#     cc.compile()

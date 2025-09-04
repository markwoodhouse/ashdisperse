import numpy as np
from numba import boolean, complex128, float64, int64, jit
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


@jit(Tuple((float64, float64))(int64[::1], float64[::1]), nopython=True, cache=True)
def linear_fit_coeffs(idx, coeff):
    """
    Compute slope and intercept for coeff ~ slope*idx + intercept
    using least squares. idx is int64 array, coeff is float64 array.
    """
    n = len(idx)
    # sums
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0

    for i in range(n):
        xi = float(idx[i])
        yi = coeff[i]
        sx += xi
        sy += yi
        sxx += xi * xi
        sxy += xi * yi

    denom = n * sxx - sx * sx
    if denom == 0.0:
        return 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept

# @jit(boolean(float64[::1], float64, int64), nopython=True, cache=True)
@jit(nopython=True, cache=True)
def cheb_tail_test(a_abs, tol_tail=1e-8, M=8):
    """
    Numba-friendly tail test.
    Input:
      a_abs : float64[:]  -- ABSOLUTE values of Chebyshev coeffs (non-negative)
      tol_tail : float64   -- normalized tail tolerance
      M : int              -- number of last coefficients to use in simple tail check
    Returns:
      converged : boolean
      tail_ratio : float64
      est_error : float64   (normalized estimate of remainder)
    """
    N = a_abs.size
    if N == 0:
        return True, 0.0, 0.0

    max_a = 0.0
    for i in range(N):
        if a_abs[i] > max_a:
            max_a = a_abs[i]

    if max_a == 0.0:
        # all coefficients zero
        return True, 0.0, 0.0

    # Simple tail max ratio (use available M but clamp)
    M_use = M
    if M_use > N:
        M_use = N
    tail_max = 0.0
    for i in range(N - M_use, N):
        v = a_abs[i]
        if v > tail_max:
            tail_max = v
    tail_ratio = tail_max / max_a

    # Exponential fit on last L coefficients
    L = 20
    # choose L = min(20, max(6, int(0.2*N)))
    tmp = int(0.2 * N)
    if tmp > 6:
        if tmp < L:
            L = tmp
    if L < 6:
        L = 6
    if L > N:
        L = N

    # build index array idx = [N-L, ..., N-1] as int64
    idx = np.empty(L, dtype=np.int64)
    for i in range(L):
        idx[i] = N - L + i

    # build coeffs vector (normalized)
    coeffs = np.empty(L, dtype=np.float64)
    for i in range(L):
        v = a_abs[N - L + i]
        if v <= 0.0:
            v = 1e-300
        coeffs[i] = np.log(v / max_a)  # use normalized log to get alpha' directly

    beta, alpha_prime = linear_fit_coeffs(idx, coeffs)  # coeffs is log(b_n)

    # beta is slope of log(b_n) vs n, so b_n ~ exp(alpha' + beta n)
    est_error = 1.0
    # Only compute geometric remainder if beta < 0 and not too close to 0
    if beta < -1e-12:
        # b_{N-1} = exp(alpha' + beta*(N-1))
        b_last = np.exp(alpha_prime + beta * (N - 1))
        exp_beta = np.exp(beta)
        denom_geom = 1.0 - exp_beta
        if denom_geom <= 0.0:
            est_error = 1.0
        else:
            geom_factor = exp_beta / denom_geom
            est_error = b_last * geom_factor
    else:
        # no credible exponential decay
        est_error = 1.0

    converged = (tail_ratio < tol_tail) and (est_error < tol_tail)
    return converged, tail_ratio, est_error


# @cc.export('convergedCoeffs', complex128[::1](complex128[::1], float64))
@jit(complex128[::1](complex128[::1], float64, float64), nopython=True, cache=True)
def truncateCoeffs(coeffs, eps, plateau_factor):
    """
    Truncate coefficients to those above plateau. Numba-friendly.

    Inputs:
      coeffs : complex128[:]  -- full coefficient array
      eps : float64           -- relative tolerance used by cheb_tail_test
      plateau_factor : float64 -- factor above noise floor to treat as meaningful (e.g. 10.0)

    Returns:
      truncated coeffs (1-d complex128 array)
    """
    N = coeffs.size
    if N == 0:
        return np.zeros(0, dtype=np.complex128)

    # abs values
    a = np.empty(N, dtype=np.float64)
    max_a = 0.0
    min_a = 1e300
    for i in range(N):
        av = np.abs(coeffs[i])
        a[i] = av
        if av > max_a:
            max_a = av
        if av < min_a:
            min_a = av

    if max_a <= 0.0:
        # all zero -> return a single zero coeff (or empty, choose single zero)
        out = np.zeros(1, dtype=np.complex128)
        out[0] = 0.0 + 0.0j
        return out

    converged, tail_ratio, est_error = cheb_tail_test(a, tol_tail=eps, M=8)

    if converged:
        # noise_floor: avoid tiny min_a by comparing with eps*max_a
        noise_floor = min_a
        # enforce lower bound
        thresh = eps * max_a
        if noise_floor < thresh:
            noise_floor = thresh

        # find last coefficient > plateau_factor * noise_floor
        Ntrunc = 1  # default if nothing found
        thr = plateau_factor * noise_floor
        for i in range(N - 1, -1, -1):
            if a[i] > thr:
                Ntrunc = i + 1
                break

        # slice and return
        out = np.empty(Ntrunc, dtype=np.complex128)
        for i in range(Ntrunc):
            out[i] = coeffs[i]
        return out
    else:
        # not converged -> return original array (no truncation)
        out = np.empty(N, dtype=np.complex128)
        for i in range(N):
            out[i] = coeffs[i]
        return out

    # if max_a > 0: # Should always be true, unless all coefficients are zero
    #     below_tol = (a < eps * max_a) * 1 # Find where coefficients are below tolerance

    #     # Only truncate if the final two coefficients are both below tolerance.
    #     # This guards against situations where some intermediate coefficients are below tolerance, 
    #     # for example if there is a parity (e.g. odd or even functions).
    #     if below_tol[-1] and below_tol[-2]:
    #         # Find last index where coefficient is above tolerance
    #         n = np.arange(0, N)
    #         nGood = n[below_tol == 0]
    #         Ntrunc = nGood[-1] + 1
    #     else: # otherwise, don't truncate
    #         Ntrunc = N

    #     new_coeffs = coeffs[:Ntrunc].flatten()

    # else: # Safety catch
    #     new_coeffs = np.zeros((1), dtype=np.complex128)

    # return new_coeffs

# if __name__ == "__main__":
#     cc.compile()

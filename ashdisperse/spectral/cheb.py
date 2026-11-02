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


@jit(float64[::1](int64), nopython=True, cache=True)
def cheb_lobatto(M):
    """Chebyshev-Lobatto points on [-1, 1] in ascending order (as in cheb)."""
    t = np.pi * np.arange(M - 1, -1, -1) / (M - 1)
    return np.cos(t)


@jit(float64[::1](float64[::1]), nopython=True, cache=True)
def cheb_fit(f):
    """Chebyshev coefficients of the polynomial interpolating f at the
    Chebyshev-Lobatto points returned by cheb_lobatto(f.size) (ascending x).

    Uses the discrete cosine transform (type I) written out explicitly; M is
    small (tens of points) so the O(M^2) cost is negligible.
    """
    M = f.size
    n = M - 1
    # reorder to descending x, i.e. theta_j = pi j / n
    g = f[::-1].copy()
    c = np.zeros(M, dtype=np.float64)
    for k in range(M):
        s = 0.0
        for j in range(M):
            w = 0.5 if (j == 0 or j == n) else 1.0
            s += w * g[j] * np.cos(np.pi * k * j / n)
        c[k] = 2.0 * s / n
    c[0] *= 0.5
    c[n] *= 0.5
    return c


@jit(float64[::1](float64[::1], float64[::1]), nopython=True, cache=True)
def cheb_eval(c, x):
    """Evaluate the Chebyshev series with coefficients c at points x (Clenshaw)."""
    out = np.empty(x.size, dtype=np.float64)
    M = c.size
    for i in range(x.size):
        xi = x[i]
        b1 = 0.0
        b2 = 0.0
        for k in range(M - 1, 0, -1):
            b0 = 2.0 * xi * b1 - b2 + c[k]
            b2 = b1
            b1 = b0
        out[i] = xi * b1 - b2 + c[0]
    return out


@jit(float64[::1](float64[::1]), nopython=True, cache=True)
def cheb_deriv(c):
    """Chebyshev coefficients of the derivative of the series c."""
    M = c.size
    d = np.zeros(M, dtype=np.float64)
    if M < 2:
        return d
    d[M - 2] = 2.0 * (M - 1) * c[M - 1]
    for k in range(M - 3, -1, -1):
        d[k] = d[k + 2] + 2.0 * (k + 1) * c[k + 1]
    d[0] *= 0.5
    return d


@jit(int64(float64[::1], float64), nopython=True, cache=True)
def standard_chop(a_abs, tol):
    """Chop point for a Chebyshev series (Aurentz & Trefethen, 2017).

    Port of Chebfun's standardChop.  Given absolute values of Chebyshev
    coefficients, returns the number of coefficients to keep.  If the series
    shows no plateau of size ~tol (i.e. it is not resolved), returns
    a_abs.size, which callers treat as "not converged".
    """
    n = a_abs.size
    cutoff = n
    if tol >= 1.0:
        return 1
    if n < 17:
        return n

    # Step 1: monotone envelope, normalised
    m = np.empty(n, dtype=np.float64)
    m[n - 1] = a_abs[n - 1]
    for j in range(n - 2, -1, -1):
        m[j] = max(a_abs[j], m[j + 1])
    if m[0] == 0.0:
        return 1
    env = m / m[0]

    # Step 2: find a plateau
    logtol = np.log(tol)
    plateau_point = -1
    j2 = 0
    for j in range(1, n):  # j is 0-based; Chebfun's j is j+1
        j2 = int(np.floor(1.25 * (j + 1) + 5.5)) - 1  # MATLAB round (half away from 0)
        if j2 > n - 1:
            return cutoff  # no plateau: not resolved
        e1 = env[j]
        e2 = env[j2]
        r = 3.0 * (1.0 - np.log(e1) / logtol) if e1 > 0.0 else 0.0
        if e1 == 0.0 or e2 / e1 > r:
            plateau_point = j - 1
            break
    if plateau_point < 0:
        return cutoff

    # Step 3: fix the cutoff at a point where envelope + linear function is minimal
    if env[plateau_point] == 0.0:
        return plateau_point + 1
    tol76 = tol ** (7.0 / 6.0)
    j3 = 0
    for j in range(n):
        if env[j] >= tol76:
            j3 += 1
    jj2 = j2 + 1  # 1-based length
    if j3 < jj2:
        jj2 = j3 + 1
        env[jj2 - 1] = tol76
    best = 0
    best_val = 1e300
    for j in range(jj2):
        v = np.log10(env[j]) + (-1.0 / 3.0) * np.log10(tol) * j / max(jj2 - 1, 1)
        if v < best_val:
            best_val = v
            best = j
    return max(best, 1)


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
      a_abs : float64[:]   -- absolute values of Chebyshev coeffs (non-negative)
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
    Truncate a Chebyshev series where it is resolved to relative tolerance eps.

    Uses standard_chop (Aurentz & Trefethen 2017).  If the series is not
    resolved, the full array is returned and callers move to a finer level.

    Inputs:
      coeffs : complex128[:]  -- full coefficient array
      eps : float64           -- relative tolerance
      plateau_factor : float64 -- unused; kept for backwards compatibility

    Returns:
      truncated coeffs (1-d complex128 array)
    """
    N = coeffs.size
    if N == 0:
        return np.zeros(0, dtype=np.complex128)
    a = np.empty(N, dtype=np.float64)
    for i in range(N):
        a[i] = np.abs(coeffs[i])
    cutoff = standard_chop(a, eps)
    out = np.empty(cutoff, dtype=np.complex128)
    for i in range(cutoff):
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

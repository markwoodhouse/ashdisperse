import numpy as np
from numba import (boolean, complex128, float64, int64, njit,
                   parallel_chunksize, prange, set_parallel_chunksize)
from numba.types import Tuple

import ashdisperse.core.core as core
import ashdisperse.spectral.cheb as cheb
from ashdisperse.containers import ChebContainer_type, VelocityContainer_type
from ashdisperse.met.met import MetData_type
from ashdisperse.params import Parameters_type

# @njit(
#     Tuple((complex128[::1], complex128[:, ::1]))(
#         float64, float64, complex128[::1], Parameters_type, MetData_type
#     ),
#     cache=True,
#     parallel=False,
#     fastmath=True,
# )
# def ade_mode_rk_solve(kx, ky, fxy_ij, parameters, Met):

#     z = np.zeros((parameters.output.Nz), dtype=np.float64)
#     z = parameters.output.altitudes
#     z = z / parameters.source.PlumeHeight

#     conc_0_ft = np.zeros((parameters.grains.bins), dtype=np.complex128)
#     conc_z_ft = np.zeros((z.size, parameters.grains.bins), dtype=np.complex128)

#     for grain_i in range(parameters.grains.bins):
#         C = core.rkBVP(kx, ky, fxy_ij[grain_i], grain_i, parameters, Met)

#         conc_0_ft[grain_i] = C[-1]
#         conc_z_ft[:, grain_i] = C[::-1]

#     return conc_0_ft, conc_z_ft


@njit(
    Tuple((complex128, complex128[::1], int64, int64))(
        int64,
        float64,
        float64,
        complex128,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
        int64,
        int64,
    ),
    cache=True,
    parallel=False,
    fastmath=False,
)
def ade_mode_solve_grain_from(grain_i, kx, ky, fxy_ij, ChebContainer, parameters, VelocityContainer, k_lower, k_upper):
    """Solve one Fourier mode, starting the Chebyshev ladders at the given
    levels.  Returns surface value, profile, and the levels used."""

    z = np.zeros((parameters.output.Nz), dtype=np.float64)
    z = parameters.output.altitudes
    z = z / parameters.source.PlumeHeight
    Nlow = z[z <= 1].size

    conc_0_ft = np.complex128(0.0)
    conc_z_ft = np.zeros((z.size), dtype=np.complex128)

    #Lower domain.
    # Solve for:
    #   inhomogeneous part (called p1) with bcs p1'(-1) = 0, p1(1) = 1
    #   homogeneous part (called r1) with bcs r1'(-1) = 1, r1(1) = 1
    #   note, other homogeneous part (called l1) with bcs l1'(-1) = 0, l1(1) = 0 has trivial solution l1==0
    #   with _c for coefficients, _v for values)
    
    p1_c, r1_c, k_lo = core.LowerODE_from(
        kx,
        ky,
        fxy_ij,
        grain_i,
        ChebContainer,
        parameters,
        VelocityContainer,
        k_lower,
    )

    # Lower domain solution:
    # Evaluate derivative of particular solution and homogeneous solutions at x=1
    # for matching to upper domain
    dp1_p1 = cheb.cheb_dif_p1(p1_c)
    dr1_p1 = cheb.cheb_dif_p1(r1_c)

    # Lower domain solution:
    # Evaluate particular solution and homogeneous solutions at x=-1
    # for surface value
    p1_m1 = cheb.cheb_val_m1(p1_c)
    r1_m1 = cheb.cheb_val_m1(r1_c)
    
    # Upper domain.
    #     Solve for:
    #       inhomogeneous part (called p2) with null boundary conditions
    #       homogeneous part (called r2) with bcs r2(1) = 1, r2(-1) = 0
    #       with _c for coefficients, _v for values)
    l2_c, k_up = core.UpperODE_from(
        kx, ky, grain_i, ChebContainer, parameters, VelocityContainer, k_upper
    )

    # Upper domain solution:
    # Evaluate derivative of homogeneous solutions at x=-1
    # for matching to lower domain
    dl2_m1 = cheb.cheb_dif_m1(l2_c)

    # Flux continuity at z = H.  d/dz = 2 d/dx below and (2/ell) d/dx above.
    ell = VelocityContainer.ell[grain_i]
    a = dp1_p1 / (dl2_m1 / ell - dr1_p1)

    # Evaluate the series at the output altitudes directly from the stored
    # Chebyshev matrices (no per-mode zero-padded copies).
    T_low = parameters.output.Cheb_lower
    for iz in range(Nlow):
        p1_v = 0.0 + 0.0j
        for n in range(p1_c.size):
            p1_v += T_low[iz, n] * p1_c[n]
        r1_v = 0.0 + 0.0j
        for n in range(r1_c.size):
            r1_v += T_low[iz, n] * r1_c[n]
        conc_z_ft[iz] = p1_v + a * r1_v

    for iz in range(z.size - Nlow):
        l2_v = 0.0 + 0.0j
        for n in range(l2_c.size):
            l2_v += VelocityContainer.Cheb_upper[grain_i, iz, n] * l2_c[n]
        conc_z_ft[Nlow + iz] = a * l2_v

    conc_0_ft = p1_m1 + a*r1_m1 # Evaluate lower solution on Z=0 <=> x=-1

    return conc_0_ft, conc_z_ft, k_lo, k_up


@njit(
    Tuple((complex128, complex128[::1]))(
        int64, float64, float64, complex128,
        ChebContainer_type, Parameters_type, VelocityContainer_type,
    ),
    cache=True,
)
def ade_mode_solve_grain(grain_i, kx, ky, fxy_ij, ChebContainer, parameters, VelocityContainer):
    """Solve one Fourier mode with the full Chebyshev ladder."""
    c0, cz, _, _ = ade_mode_solve_grain_from(
        grain_i, kx, ky, fxy_ij, ChebContainer, parameters, VelocityContainer, 0, 0
    )
    return c0, cz


@njit(int64(int64[:, :, ::1], int64, int64, int64, int64), cache=True)
def _start_level(lev, j, i, k, Ny):
    """Starting Chebyshev level for interior mode (j, i): the lowest level
    that the neighbouring axis modes (same ky on kx = 0, same kx on the
    ky = 0 / ky = -1 rows) needed.  The axis modes are solved first, so the
    choice is deterministic."""
    s = lev[j, 0, k]
    if lev[0, i, k] < s:
        s = lev[0, i, k]
    if lev[Ny - 1, i, k] < s:
        s = lev[Ny - 1, i, k]
    return s


@njit(
    Tuple((int64[::1], int64[::1], int64[::1]))(int64, int64),
    cache=True,
)
def _axis_modes(Nx, Ny):
    """Fourier modes on the kx = 0 and ky = 0 axes of the rfft layout.

    Returns (i, j, line) with line = 0 for the i = 0 column (all j) and
    line = 1 for the j = 0 and j = Ny - 1 rows (i >= 1).  These rows are the
    two members of the j = 0 pair (j, Ny - 1 - j) used in the interior loop.
    """
    half_Nx = Nx // 2
    n = Ny + 2 * half_Nx
    ii = np.empty(n, dtype=np.int64)
    jj = np.empty(n, dtype=np.int64)
    line = np.empty(n, dtype=np.int64)
    m = 0
    for j in range(Ny):
        ii[m] = 0
        jj[m] = j
        line[m] = 0
        m += 1
    for i in range(1, half_Nx + 1):
        ii[m] = i
        jj[m] = 0
        line[m] = 1
        m += 1
        ii[m] = i
        jj[m] = Ny - 1
        line[m] = 1
        m += 1
    return ii, jj, line


@njit(
    Tuple((int64[::1], int64[::1]))(complex128[:, :, ::1], float64, int64, int64),
    cache=True,
)
def _adaptive_stops(conc_0_fft, fft_tol, Nx, Ny):
    """Truncation indices from the axis modes (computed beforehand).

    Ystop[k]: first j along kx = 0 where the pair (j, Ny - 1 - j) is below
    fft_tol.  Xstop[k]: first i along ky = 0 with 0 < |C| < fft_tol.  These
    are the rules of the original single-pass loop, applied once all axis
    modes are known, so the result does not depend on thread scheduling.
    """
    Ng = conc_0_fft.shape[2]
    half_Ny = (Ny + 1) // 2
    Xstop = (Nx // 2 + 1) * np.ones(Ng, dtype=np.int64)
    Ystop = Ny // 2 * np.ones(Ng, dtype=np.int64)
    for k in range(Ng):
        for j in range(half_Ny):
            j_upper = Ny - j - 1
            if np.abs(conc_0_fft[j, 0, k]) < fft_tol and np.abs(conc_0_fft[j_upper, 0, k]) < fft_tol:
                Ystop[k] = j
                break
        for i in range(Nx // 2 + 1):
            a = np.abs(conc_0_fft[0, i, k])
            if 0.0 < a < fft_tol:
                Xstop[k] = i
                break
    return Xstop, Ystop


@njit(
    Tuple((complex128[:, :, ::1], complex128[:, :, :, ::1]))(
        float64[::1],
        float64[::1],
        complex128[:, :, ::1],
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
    ),
    parallel=True,
    cache=True,
    fastmath=False,
)
def ade_ft_system(kx, ky, fxy_f, cheby, params, velocities):
    """Solve every retained Fourier mode for every grain class.

    Two parallel passes make the result independent of thread scheduling:
      1. modes on the kx = 0 and ky = 0 axes, which set the adaptive
         truncation indices Xstop/Ystop (per grain);
      2. interior modes, skipped beyond the truncation indices.
    """
    Nx = kx.size
    Ny = ky.size

    Ng = params.grains.bins
    Nz = params.output.Nz

    fft_tol = params.solver.fft_tol

    conc_0_fft = np.zeros((Ny, Nx//2+1, Ng), dtype=np.complex128)
    conc_z_fft = np.zeros(
        (Ny, Nx//2+1, Nz, Ng), dtype=np.complex128
    )

    # Pass 1: axis modes
    lev_lo = np.zeros((Ny, Nx // 2 + 1, Ng), dtype=np.int64)
    lev_up = np.zeros((Ny, Nx // 2 + 1, Ng), dtype=np.int64)
    ai, aj, _ = _axis_modes(Nx, Ny)
    NA = ai.size * Ng
    for idx in prange(NA):
        m = idx // Ng
        k = Ng - 1 - (idx % Ng)
        i = ai[m]
        j = aj[m]
        c0, cz, klo, kup = ade_mode_solve_grain_from(
            k, kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities, 0, 0
        )
        conc_0_fft[j, i, k] = c0
        conc_z_fft[j, i, :, k] = cz
        lev_lo[j, i, k] = klo
        lev_up[j, i, k] = kup

    Xstop, Ystop = _adaptive_stops(conc_0_fft, fft_tol, Nx, Ny)

    # Pass 2: interior modes, in pairs (j, Ny - 1 - j) for j >= 1
    half_Ny = (Ny + 1) // 2
    nI = Nx // 2
    nJ = half_Ny - 1
    N = nI * nJ * Ng
    for idx in prange(N):
        i = 1 + idx // (nJ * Ng)
        j = 1 + (idx // Ng) % nJ
        k = Ng - 1 - (idx % Ng)  # coarse-to-fine order spreads the work
        j_upper = Ny - j - 1

        if (i > Xstop[k]) or (j > Ystop[k]):
            continue  # arrays are zero-initialised

        c0, cz, _, _ = ade_mode_solve_grain_from(
            k, kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities,
            _start_level(lev_lo, j, i, k, Ny), _start_level(lev_up, j, i, k, Ny),
        )
        conc_0_fft[j, i, k] = c0
        conc_z_fft[j, i, :, k] = cz

        if j_upper != j:
            c0, cz, _, _ = ade_mode_solve_grain_from(
                k, kx[i], ky[j_upper], fxy_f[j_upper, i, k], cheby, params, velocities,
                _start_level(lev_lo, j_upper, i, k, Ny), _start_level(lev_up, j_upper, i, k, Ny),
            )
            conc_0_fft[j_upper, i, k] = c0
            conc_z_fft[j_upper, i, :, k] = cz

    return conc_0_fft, conc_z_fft


@njit(
    Tuple((complex128[:, :, ::1], complex128[:, :, :, ::1]))(
        complex128[:, :, ::1],
        complex128[:, :, :, ::1],
        float64[::1],
        float64[::1],
        complex128[:, :, ::1],
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
        boolean,
    ),
    parallel=True,
    cache=True,
    fastmath=False,
)
def ade_ft_refine(conc_0_fft_old, conc_z_fft_old, kx, ky, fxy_f, cheby, params, velocities, full=False):
    """Solve on a finer Fourier grid, reusing modes from a previous solution.

    Same two-pass structure as ade_ft_system.  Axis modes are always
    recomputed; an interior mode is copied from the old solution when it
    exists there and |C0| >= fft_tol (unless full=True), otherwise solved.
    (The former moving-window predictor shared state between threads and
    has been removed.)
    """
    Ny_old = conc_0_fft_old.shape[0]
    half_Nx_old = conc_0_fft_old.shape[1] - 1
    half_Ny_old = Ny_old // 2

    Nx = kx.size
    Ny = ky.size
    half_Nx = Nx // 2
    half_Ny = (Ny + 1) // 2
    Ng = params.grains.bins
    Nz = params.output.Nz
    fft_tol = params.solver.fft_tol

    conc_0_fft = np.zeros((Ny, half_Nx+1, Ng), dtype=np.complex128)
    conc_z_fft = np.zeros((Ny, half_Nx+1, Nz, Ng), dtype=np.complex128)

    lev_lo = np.zeros((Ny, Nx // 2 + 1, Ng), dtype=np.int64)
    lev_up = np.zeros((Ny, Nx // 2 + 1, Ng), dtype=np.int64)
    ai, aj, _ = _axis_modes(Nx, Ny)
    NA = ai.size * Ng
    for idx in prange(NA):
        m = idx // Ng
        k = Ng - 1 - (idx % Ng)
        i = ai[m]
        j = aj[m]
        c0, cz, klo, kup = ade_mode_solve_grain_from(
            k, kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities, 0, 0
        )
        conc_0_fft[j, i, k] = c0
        conc_z_fft[j, i, :, k] = cz
        lev_lo[j, i, k] = klo
        lev_up[j, i, k] = kup

    Xstop, Ystop = _adaptive_stops(conc_0_fft, fft_tol, Nx, Ny)

    nI = half_Nx
    nJ = half_Ny - 1
    N_total = nI * nJ * Ng
    for idx in prange(N_total):
        i = 1 + idx // (nJ * Ng)
        j = 1 + (idx // Ng) % nJ
        k = Ng - 1 - (idx % Ng)
        j_upper = Ny - j - 1
        j_upper_old = Ny_old - j - 1

        if (i > Xstop[k]) or (j > Ystop[k]):
            continue

        use_old = False
        if (not full) and (i < half_Nx_old) and (j < half_Ny_old):
            if np.abs(conc_0_fft_old[j, i, k]) >= fft_tol:
                use_old = True

        if use_old:
            conc_0_fft[j, i, k] = conc_0_fft_old[j, i, k]
            conc_z_fft[j, i, :, k] = conc_z_fft_old[j, i, :, k]
            if j_upper != j:
                conc_0_fft[j_upper, i, k] = conc_0_fft_old[j_upper_old, i, k]
                conc_z_fft[j_upper, i, :, k] = conc_z_fft_old[j_upper_old, i, :, k]
        else:
            c0, cz, _, _ = ade_mode_solve_grain_from(
                k, kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities,
                _start_level(lev_lo, j, i, k, Ny), _start_level(lev_up, j, i, k, Ny),
            )
            conc_0_fft[j, i, k] = c0
            conc_z_fft[j, i, :, k] = cz
            if j_upper != j:
                c0, cz, _, _ = ade_mode_solve_grain_from(
                    k, kx[i], ky[j_upper], fxy_f[j_upper, i, k], cheby, params, velocities,
                    _start_level(lev_lo, j_upper, i, k, Ny), _start_level(lev_up, j_upper, i, k, Ny),
                )
                conc_0_fft[j_upper, i, k] = c0
                conc_z_fft[j_upper, i, :, k] = cz

    return conc_0_fft, conc_z_fft

# @njit(
#     Tuple((complex128[:, :, ::1], complex128[:, :, :, ::1]))(
#         float64[::1],
#         float64[::1],
#         complex128[:, :, ::1],
#         Parameters_type,
#         MetData_type,
#     ),
#     parallel=True,
#     cache=True,
#     fastmath=True,
# )
# def ade_ft_system_rk(kx, ky, fxy_f, params, Met):

#     Nx = kx.size
#     Ny = ky.size

#     conc_0_fft = np.zeros((Ny, Nx, params.grains.bins), dtype=np.complex128)
#     conc_z_fft = np.zeros(
#         (Ny, Nx, params.output.Nz, params.grains.bins), dtype=np.complex128
#     )

#     # do kx = ky = 0:
#     conc_0_mode_fft, conc_z_mode_fft = ade_mode_rk_solve(
#         0.0, 0.0, fxy_f[0, 0, :], params, Met
#     )
#     conc_0_fft[0, 0, :] = conc_0_mode_fft
#     conc_z_fft[0, 0, :, :] = conc_z_mode_fft

#     # do kx = 0, ky = 1 ... Ny/2-1, -Ny/2
#     # and we get kx = 0, ky = -Ny/2+1, ... , -1 for free by conjugation
#     for j in prange(1, Ny // 2 + 1):

#         conc_0_mode_fft, conc_z_mode_fft = ade_mode_rk_solve(
#             0.0, ky[j], fxy_f[j, 0, :], params, Met
#         )

#         conc_0_fft[j, 0, :] = conc_0_mode_fft
#         conc_0_fft[Ny - j, 0, :] = np.conj(conc_0_mode_fft)

#         conc_z_fft[j, 0, :, :] = conc_z_mode_fft
#         conc_z_fft[Ny - j, 0, :, :] = np.conj(conc_z_mode_fft)

#     # do ky = 0, kx = 1 ... Nx/2-1, -Nx/2
#     # and we get ky = 0, kx = -Nx/2+1, ... , -1 for free by conjugation
#     for i in prange(1, Nx // 2 + 1):

#         conc_0_mode_fft, conc_z_mode_fft = ade_mode_rk_solve(
#             kx[i], 0.0, fxy_f[0, i, :], params, Met
#         )

#         conc_0_fft[0, i, :] = conc_0_mode_fft
#         conc_0_fft[0, Nx - i, :] = np.conj(conc_0_mode_fft)

#         conc_z_fft[0, i, :, :] = conc_z_mode_fft
#         conc_z_fft[0, Nx - i, :, :] = np.conj(conc_z_mode_fft)

#     # Do first quadrant, kx = 1 ... Nx/2-1, ky = 1 .. Ny/2-1
#     # and get the 4th quadrant for free by conjugation.
#     # Also do second quadrant, kx = -Nx/2+1 ... -1, ky = 1 .. Ny/2-1
#     # and get the third quadrant for free by conjugation.
#     for i in prange(1, Nx // 2):
#         for j in range(1, Ny // 2):

#             conc_0_mode_fft, conc_z_mode_fft = ade_mode_rk_solve(
#                 kx[i], ky[j], fxy_f[j, i, :], params, Met
#             )
#             conc_0_fft[j, i, :] = conc_0_mode_fft
#             conc_z_fft[j, i, :, :] = conc_z_mode_fft

#             conc_0_mode_fft, conc_z_mode_fft = ade_mode_rk_solve(
#                 kx[Nx - i], ky[j], fxy_f[j, Nx - i, :], params, Met
#             )
#             conc_0_fft[j, Nx - i, :] = conc_0_mode_fft
#             conc_z_fft[j, Nx - i, :, :] = conc_z_mode_fft

#             conc_0_fft[Ny - j, Nx - i, :] = np.conj(conc_0_fft[j, i, :])
#             conc_0_fft[Ny - j, i, :] = np.conj(conc_0_fft[j, Nx - i, :])

#             conc_z_fft[Ny - j, Nx - i, :, :] = np.conj(conc_z_fft[j, i, :, :])
#             conc_z_fft[Ny - j, i, :, :] = np.conj(conc_z_fft[j, Nx - i, :, :])

#     return conc_0_fft, conc_z_fft
#     return conc_0_fft, conc_z_fft

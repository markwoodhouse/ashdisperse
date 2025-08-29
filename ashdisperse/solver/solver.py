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
    Tuple((complex128, complex128[::1]))(
        int64,
        float64,
        float64,
        complex128,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
    ),
    cache=True,
    parallel=False,
    fastmath=True,
)
def ade_mode_solve_grain(grain_i, kx, ky, fxy_ij, ChebContainer, parameters, VelocityContainer):

    z = np.zeros((parameters.output.Nz), dtype=np.float64)
    z = parameters.output.altitudes
    z = z / parameters.source.PlumeHeight
    Nlow = z[z <= 1].size

    conc_0_ft = np.complex128(0.0)
    conc_z_ft = np.zeros((z.size), dtype=np.complex128)

    """Lower domain.
    Solve for:
      inhomogeneous part (called p1) with null boundary conditions
      homogeneous part (called r1) with bcs r1(1) = 1, r1(-1) = 0
      homogeneous part (called l1) with bcs l1(1) = 0, l1(-1) = 1
      with _c for coefficients, _v for values)
    """
    p1_c, l1_c, r1_c = core.LowerODE(
        kx,
        ky,
        fxy_ij,
        grain_i,
        ChebContainer,
        parameters,
        VelocityContainer,
    )

    dp1_p1 = cheb.cheb_dif_p1(p1_c)
    dr1_p1 = cheb.cheb_dif_p1(r1_c)
    dl1_p1 = cheb.cheb_dif_p1(l1_c)

    dp1_m1 = cheb.cheb_dif_m1(p1_c)
    dr1_m1 = cheb.cheb_dif_m1(r1_c)
    dl1_m1 = cheb.cheb_dif_m1(l1_c)

    """Upper domain.
         Solve for:
           inhomogeneous part (called p2) with null boundary conditions
           homogeneous part (called r2) with bcs r2(1) = 1, r2(-1) = 0
           with _c for coefficients, _v for values)
    """
    l2_c = core.UpperODE(
        kx, ky, grain_i, ChebContainer, parameters, VelocityContainer
    )

    dl2_m1 = cheb.cheb_dif_m1(l2_c)

    a = dl1_m1
    b = dr1_m1
    c = dl1_p1
    d = dr1_p1 - dl2_m1

    b0 = -dp1_m1
    b1 = -dp1_p1

    delta = a * d - b * c

    match0 = (b0 * d - b1 * b) / delta
    match1 = (b1 * a - b0 * c) / delta

    Cheb_r1 = np.zeros(
        (parameters.output.Cheb_lower.shape[0], r1_c.size), dtype=np.complex128
    )
    Cheb_l1 = np.zeros(
        (parameters.output.Cheb_lower.shape[0], l1_c.size), dtype=np.complex128
    )
    Cheb_p1 = np.zeros(
        (parameters.output.Cheb_lower.shape[0], p1_c.size), dtype=np.complex128
    )

    Cheb_r1[:, : r1_c.size] = parameters.output.Cheb_lower[:, : r1_c.size]
    Cheb_l1[:, : l1_c.size] = parameters.output.Cheb_lower[:, : l1_c.size]
    Cheb_p1[:, : p1_c.size] = parameters.output.Cheb_lower[:, : p1_c.size]

    r1_v = Cheb_r1 @ r1_c
    l1_v = Cheb_l1 @ l1_c
    p1_v = Cheb_p1 @ p1_c

    conc_z_ft[:Nlow] = p1_v + match0 * l1_v + match1 * r1_v

    Cheb_l2 = np.zeros(
        (parameters.output.Cheb_upper.shape[0], l2_c.size), dtype=np.complex128
    )
    Cheb_l2[:, : l2_c.size] = parameters.output.Cheb_upper[:, : l2_c.size]
    l2_v = Cheb_l2 @ l2_c

    conc_z_ft[Nlow:] = match1 * l2_v

    conc_0_ft = match0  # (b0*d - b1*b)/delta

    return conc_0_ft, conc_z_ft


@njit(
    Tuple((complex128[::1], complex128[:, ::1]))(
        float64,
        float64,
        complex128[::1],
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
    ),
    cache=True,
    parallel=False,
    fastmath=True,
)
def ade_mode_solve(kx, ky, fxy_ij, ChebContainer, parameters, VelocityContainer):

    z = np.zeros((parameters.output.Nz), dtype=np.float64)
    z = parameters.output.altitudes
    z = z / parameters.source.PlumeHeight
    Nlow = z[z <= 1].size

    conc_0_ft = np.zeros((parameters.grains.bins), dtype=np.complex128)
    conc_z_ft = np.zeros((z.size, parameters.grains.bins), dtype=np.complex128)

    for grain_i in range(parameters.grains.bins):
        """Lower domain.
        Solve for:
          inhomogeneous part (called p1) with null boundary conditions
          homogeneous part (called r1) with bcs r1(1) = 1, r1(-1) = 0
          homogeneous part (called l1) with bcs l1(1) = 0, l1(-1) = 1
          with _c for coefficients, _v for values)
        """
        p1_c, l1_c, r1_c = core.LowerODE(
            kx,
            ky,
            fxy_ij[grain_i],
            np.int64(grain_i),
            ChebContainer,
            parameters,
            VelocityContainer,
        )

        dp1_p1 = cheb.cheb_dif_p1(p1_c)
        dr1_p1 = cheb.cheb_dif_p1(r1_c)
        dl1_p1 = cheb.cheb_dif_p1(l1_c)

        dp1_m1 = cheb.cheb_dif_m1(p1_c)
        dr1_m1 = cheb.cheb_dif_m1(r1_c)
        dl1_m1 = cheb.cheb_dif_m1(l1_c)

        """Upper domain.
             Solve for:
               inhomogeneous part (called p2) with null boundary conditions
               homogeneous part (called r2) with bcs r2(1) = 1, r2(-1) = 0
               with _c for coefficients, _v for values)
        """
        l2_c = core.UpperODE(
            kx, ky, np.int64(grain_i), ChebContainer, parameters, VelocityContainer
        )

        dl2_m1 = cheb.cheb_dif_m1(l2_c)

        a = dl1_m1
        b = dr1_m1
        c = dl1_p1
        d = dr1_p1 - dl2_m1

        b0 = -dp1_m1
        b1 = -dp1_p1

        delta = a * d - b * c

        match0 = (b0 * d - b1 * b) / delta
        match1 = (b1 * a - b0 * c) / delta

        Cheb_r1 = np.zeros(
            (parameters.output.Cheb_lower.shape[0], r1_c.size), dtype=np.complex128
        )
        Cheb_l1 = np.zeros(
            (parameters.output.Cheb_lower.shape[0], l1_c.size), dtype=np.complex128
        )
        Cheb_p1 = np.zeros(
            (parameters.output.Cheb_lower.shape[0], p1_c.size), dtype=np.complex128
        )

        Cheb_r1[:, : r1_c.size] = parameters.output.Cheb_lower[:, : r1_c.size]
        Cheb_l1[:, : l1_c.size] = parameters.output.Cheb_lower[:, : l1_c.size]
        Cheb_p1[:, : p1_c.size] = parameters.output.Cheb_lower[:, : p1_c.size]

        r1_v = Cheb_r1 @ r1_c
        l1_v = Cheb_l1 @ l1_c
        p1_v = Cheb_p1 @ p1_c

        conc_z_ft[:Nlow, grain_i] = p1_v + match0 * l1_v + match1 * r1_v

        Cheb_l2 = np.zeros(
            (parameters.output.Cheb_upper.shape[0], l2_c.size), dtype=np.complex128
        )
        Cheb_l2[:, : l2_c.size] = parameters.output.Cheb_upper[:, : l2_c.size]
        l2_v = Cheb_l2 @ l2_c

        conc_z_ft[Nlow:, grain_i] = match1 * l2_v

        conc_0_ft[grain_i] = match0  # (b0*d - b1*b)/delta

    return conc_0_ft, conc_z_ft


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

    Nx = kx.size
    Ny = ky.size

    Ng = params.grains.bins
    Nz = params.output.Nz

    fft_tol = params.solver.fft_tol

    conc_0_fft = np.zeros((Ny, Nx//2+1, Ng), dtype=np.complex128)
    conc_z_fft = np.zeros(
        (Ny, Nx//2+1, Nz, Ng), dtype=np.complex128
    )

    Xstop = (Nx//2+1) * np.ones((Ng), dtype=np.int64)
    Ystop = Ny//2 * np.ones((Ng), dtype=np.int64)

    half_Ny = (Ny+1)//2

    N = (Nx//2 + 1) * (half_Ny) * Ng
    for idx in prange(N):
        i = idx // (half_Ny * Ng)  # Compute i
        j = (idx // Ng) % (half_Ny)  # Compute j
        j_upper = Ny - j - 1  # Mirror j
        #k = idx % Ng          # Compute k
        k = Ng - 1 - (idx % Ng) # Compute k in reverse

        if (i<=Xstop[k] and j<=Ystop[k]):

            conc_0_mode_fft, conc_z_mode_fft = ade_mode_solve_grain(k,
                    kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities
                )
            
            conc_0_fft[j, i, k] = conc_0_mode_fft
            conc_z_fft[j, i, :, k] = conc_z_mode_fft
            
            conc_0_mode_fft_upper, conc_z_mode_fft_upper = ade_mode_solve_grain(k,
                    kx[i], ky[j_upper], fxy_f[j_upper, i, k], cheby, params, velocities
                )
        
            conc_0_fft[j_upper, i, k] = conc_0_mode_fft_upper
            conc_z_fft[j_upper, i, :, k] = conc_z_mode_fft_upper

            if i==0:
                if np.absolute(conc_0_mode_fft)< fft_tol and np.absolute(conc_0_mode_fft_upper) < fft_tol:
                    Ystop[k] = np.minimum(j, Ystop[k])
            if j==0:
                if 0 < np.absolute(conc_0_mode_fft) < fft_tol:
                    Xstop[k] = np.minimum(i, Xstop[k])
        else:
            conc_0_fft[j, i, k] = np.complex128(0.0)
            conc_z_fft[j, i, :, k] = np.zeros((Nz), dtype=np.complex128)
            conc_0_fft[j_upper, i, k] = np.complex128(0.0)
            conc_z_fft[j_upper, i, :, k] = np.zeros((Nz), dtype=np.complex128)
    
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
    fastmath=True,
)
def ade_ft_refine(conc_0_fft_old, conc_z_fft_old, kx, ky, fxy_f, cheby, params, velocities, full=False):

    Ny_old = conc_0_fft_old.shape[0]
    half_Nx_old = conc_0_fft_old.shape[1] - 1

    Nx_old = half_Nx_old * 2
    half_Ny_old = Ny_old // 2

    Nx = kx.size
    Ny = ky.size

    half_Nx = Nx // 2 # Only need the non-negative modes but symmetry of FFT of real-valued function
    half_Ny = (Ny + 1) // 2

    Ng = params.grains.bins
    Nz = params.output.Nz

    fft_tol = params.solver.fft_tol

    Xstop = (Nx//2+1) * np.ones((Ng), dtype=np.int64)
    Ystop = Ny//2 * np.ones((Ng), dtype=np.int64)

    conc_0_fft = np.zeros((Ny, half_Nx+1, Ng), dtype=np.complex128)
    conc_z_fft = np.zeros((Ny, Nx+1, Nz, Ng), dtype=np.complex128)

    N = (Nx//2 + 1) * (half_Ny) * Ng
    for idx in prange(N):
        i = idx // (half_Ny * Ng)  # Compute i
        j = (idx // Ng) % (half_Ny)  # Compute j
        j_upper = Ny - j - 1  # Mirror j
        k = idx % Ng          # Compute k

        j_upper = Ny - j - 1 # Mirror j
        j_upper_old = Ny_old - j - 1 # Mirror j

        is_special = (i == 0) or (j == 0)

        if full:
            if (not is_special and i<half_Nx_old and j<half_Ny_old):
            
                if np.absolute(conc_0_fft_old[j, i, k])<fft_tol:
                    conc_0_mode_fft, conc_z_mode_fft = ade_mode_solve_grain(k,
                        kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities
                    )
                else:
                    conc_0_mode_fft = conc_0_fft_old[j, i, k]
                    conc_z_mode_fft = conc_z_fft_old[j, i, :, k]

                if np.absolute(conc_0_fft_old[j, i, k])<fft_tol:
                    conc_0_mode_fft_upper, conc_z_mode_fft_upper = ade_mode_solve_grain(k,
                        kx[i], ky[j_upper], fxy_f[j_upper, i, k], cheby, params, velocities
                    )
                else:
                    conc_0_mode_fft_upper = conc_0_fft_old[j_upper_old, i, k]
                    conc_z_mode_fft_upper = conc_z_fft_old[j_upper_old, i, :, k]
        
            else:
                conc_0_mode_fft, conc_z_mode_fft = ade_mode_solve_grain(k,
                    kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities
                )
                
                conc_0_mode_fft_upper, conc_z_mode_fft_upper = ade_mode_solve_grain(k,
                    kx[i], ky[j_upper], fxy_f[j_upper, i, k], cheby, params, velocities
                )
            
            conc_0_fft[j, i, k] = conc_0_mode_fft
            conc_z_fft[j, i, :, k] = conc_z_mode_fft

            conc_0_fft[j_upper, i, k] = conc_0_mode_fft_upper
            conc_z_fft[j_upper, i, :, k] = conc_z_mode_fft_upper
        
        else:
            if (not is_special and i<half_Nx_old and j<half_Ny_old):
                
                conc_0_mode_fft = conc_0_fft_old[j, i, k]
                conc_z_mode_fft = conc_z_fft_old[j, i, :, k]

                conc_0_mode_fft_upper = conc_0_fft_old[j_upper_old, i, k]
                conc_z_mode_fft_upper = conc_z_fft_old[j_upper_old, i, :, k]

            else:
                if (i<=Xstop[k] and j<=Ystop[k]):

                    conc_0_mode_fft, conc_z_mode_fft = ade_mode_solve_grain(k,
                        kx[i], ky[j], fxy_f[j, i, k], cheby, params, velocities
                    )

                    conc_0_mode_fft_upper, conc_z_mode_fft_upper = ade_mode_solve_grain(k,
                        kx[i], ky[j_upper], fxy_f[j_upper, i, k], cheby, params, velocities
                    )

                    if i==0:
                        if np.absolute(conc_0_mode_fft)< fft_tol and np.absolute(conc_0_mode_fft_upper) < fft_tol:
                            Ystop[k] = np.minimum(j, Ystop[k])
                    if j==0:
                        if 0 < np.absolute(conc_0_mode_fft) < fft_tol:
                            Xstop[k] = np.minimum(i, Xstop[k])
                
                else:
                    conc_0_mode_fft = np.complex128(0.0)
                    conc_z_mode_fft = np.zeros((Nz), dtype=np.complex128)
                    conc_0_mode_fft_upper = np.complex128(0.0)
                    conc_z_mode_fft_upper = np.zeros((Nz), dtype=np.complex128)
            
            conc_0_fft[j, i, k] = conc_0_mode_fft
            conc_z_fft[j, i, :, k] = conc_z_mode_fft

            conc_0_fft[j_upper, i, k] = conc_0_mode_fft_upper
            conc_z_fft[j_upper, i, :, k] = conc_z_mode_fft_upper
            
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

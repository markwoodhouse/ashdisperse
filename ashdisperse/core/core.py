import numpy as np
from numba import complex128, float64, int64, jit, njit
# from numba.pycc import CC
from numba.types import Tuple

from ..containers.cheb import ChebContainer_type
from ..containers.velocities import VelocityContainer_type
from ..met.met import MetData_type
from ..params.params import Parameters_type
from ..spectral.cheb import truncateCoeffs
from .getters import (Source_z_dimless, lower_dWsdz, lower_U, lower_V,
                      lower_Ws, upper_dWsdz, upper_U, upper_V, upper_Ws)

# from .rk import integrateTo

# pylint: disable=C0103, E501

# cc = CC('core')
# cc.verbose = True


@jit(
    complex128[:, ::1](
        float64[::1], float64[::1], complex128[::1],
        float64[:, :], float64[:, :], float64[:, :],
    ),
    nopython=True,
    cache=True,
    fastmath=False,
)
def assemble_operator(c2, c1, c0, Tn, dTn, d2Tn):
    """Collocation matrix L[r, s] = c2[r] d2Tn[r, s] + c1[r] dTn[r, s] + c0[r] Tn[r, s].

    One pass over the N x N matrices; replaces np.repeat/reshape/multiply,
    which allocated four temporary N x N arrays per call.
    """
    N = Tn.shape[0]
    L = np.empty((N, N), dtype=np.complex128)
    for r in range(N):
        a2 = c2[r]
        a1 = c1[r]
        a0 = c0[r]
        for s in range(N):
            L[r, s] = (a2 * d2Tn[r, s] + a1 * dTn[r, s]) + a0 * Tn[r, s]
    return L


# LowerODE
# @cc.export('LowerODE', Tuple((complex128[::1], complex128[::1]))(float64, float64, complex128, int64, ChebContainer_type, Parameters_type, VelocityContainer_type))
@jit(
    Tuple((complex128[::1], complex128[::1]))(
        float64,
        float64,
        complex128,
        int64,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
    ),
    nopython=True,
    cache=True,
    fastmath=False,
)
def LowerODE(kx, ky, fxy_ij, grain_i, cheby, parameters, velocities):
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    rPe = 1.0 / parameters.model.Peclet_number # reciprocal Peclet number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio

    for k, N in enumerate(cheby.N):
        x, Tn, dTn, d2Tn = cheby.get_cheb(k)
        U = lower_U(velocities, k) # dimensionless U
        V = lower_V(velocities, k) # dimensionless V
        Ws = lower_Ws(velocities, k)[:, grain_i] # dimensionless
        dWsdz = lower_dWsdz(velocities, k)[:, grain_i] # dimensionless

        c2 = 4.0 * rPe * Vratio / R * np.ones(N, dtype=np.float64)
        c1 = 2 * Ws

        c0 = (
            -1j * pi * (kx / Lx * U + ky / Ly * V)
            + dWsdz
            - pi * pi * rPe / Vratio * ((kx / Lx) ** 2 + (ky / Ly) ** 2)
        )

        L = assemble_operator(c2, c1, c0, Tn, dTn, d2Tn)

        b = np.zeros((N, 2), dtype=np.complex128)
        fz = Source_z_dimless(0.5 * (x + 1), 
                              parameters.emission.profile[grain_i],
                              parameters.emission.Suzuki_k[grain_i],
                              parameters.emission.lower[grain_i],
                              parameters.emission.upper[grain_i],
                              parameters.source.PlumeHeight)

        L[0, :] = dTn[0, :]
        L[-1, :] = Tn[-1, :]

        b[:, 0] = -fxy_ij * fz
        b[0, 0] = 0.0
        b[-1, 0] = 0.0

        b[0, 1] = 0.0
        b[-1, 1] = 1.0

        coeffs = np.linalg.solve(L, b)  # .astype(np.complex128)

        co0 = truncateCoeffs(coeffs[:, 0].flatten(), parameters.solver.epsilon, parameters.solver.plateau_factor)
        co1 = truncateCoeffs(coeffs[:, 1].flatten(), parameters.solver.epsilon, parameters.solver.plateau_factor)

        if (co0.size < N) and (co1.size < N):
            break

    return co0, co1


# UpperODE
# @cc.export('UpperODE', complex128[::1](float64, float64, int64, ChebContainer_type, Parameters_type, VelocityContainer_type))
@jit(
    complex128[::1](
        float64,
        float64,
        int64,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
    ),
    nopython=True,
    cache=True,
    fastmath=False,
)
def UpperODE(kx, ky, grain_i, cheby, parameters, velocities):
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    rPe = 1.0 / parameters.model.Peclet_number # reciprocal Peclet number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio
    # map z/H = 1 + ell (1 + x)/(1 - x): d/dz = (1-x)^2/(2 ell) d/dx,
    # d2/dz2 = (1-x)^4/(4 ell^2) d2/dx2 - (1-x)^3/(2 ell^2) d/dx
    ell = velocities.ell[grain_i]

    for k, N in enumerate(cheby.N):
        x, Tn, dTn, d2Tn = cheby.get_cheb(k)
        U = upper_U(velocities, k, grain_i)
        V = upper_V(velocities, k, grain_i)
        Ws = upper_Ws(velocities, k)[:, grain_i]
        dWsdz = upper_dWsdz(velocities, k)[:, grain_i]

        c2 = np.zeros(N, dtype=np.float64)
        c2[:-1] = ((1 - x[:-1]) ** 4) / (4 * ell * ell) * rPe * Vratio / R

        c1 = np.zeros(N, dtype=np.float64)
        c1[:-1] = ((1 - x[:-1]) ** 2) / (2 * ell) * (
            Ws - (1 - x[:-1]) * rPe * Vratio / R / ell
        )

        c0 = np.zeros(N, dtype=np.complex128)
        c0[:-1] = (
            -1j * pi * (kx / Lx * U + ky / Ly * V)
            + dWsdz
            - pi * pi * rPe / Vratio * ((kx / Lx) ** 2 + (ky / Ly) ** 2)
        )

        L = assemble_operator(c2, c1, c0, Tn, dTn, d2Tn)
        b = np.zeros((N, 1), dtype=np.complex128)

        L[0, :] = Tn[0, :]
        L[-1, :] = Tn[-1, :]

        b[0, 0] = 1.0

        coeffs = np.linalg.solve(L, b)  # .astype(np.complex128)

        co = truncateCoeffs(coeffs[:, 0].flatten(), parameters.solver.epsilon, parameters.solver.plateau_factor)

        if co.size < N:
            break

    # co = np.ascontiguousarray(new_coeffs[0].flatten())

    return co


# @jit(
#     complex128[::1](
#         float64,
#         float64,
#         complex128,
#         int64,
#         Parameters_type,
#         MetData_type,
#     ),
#     nopython=True,
#     cache=True,
#     fastmath=True,
# )
# # @njit(cache=True, fastmath=True)
# def rkBVP(kx, ky, fxy_ij, grain_i, parameters, Met):

#     Nz = parameters.output.Nz

#     x0 = 0.0

#     y = np.zeros((4), dtype=np.complex128)
#     y[0] = 0.0
#     y[1] = 0.0
#     y[2] = 0.0
#     y[3] = 1.0

#     dy = np.zeros((4), dtype=np.complex128)
#     dy[0] = 0.0
#     dy[1] = 0.0
#     dy[2] = 1.0
#     dy[3] = 0.0

#     h = 1e-8

#     u = np.zeros((Nz), dtype=np.complex128)
#     v = np.zeros((Nz), dtype=np.complex128)
#     C = np.zeros((Nz), dtype=np.complex128)

#     for j in range(Nz):

#         z = parameters.output.altitudes[-j]

#         x = np.exp(-z)

#         y, dy, h = integrateTo(
#             x0, x, y, dy, h, grain_i, kx, ky, fxy_ij, parameters, Met
#         )

#         u[j] = y[0]
#         v[j] = y[1]

#         x0 = x

#     x = 1.0
#     y, dy, h = integrateTo(x0, x, y, dy, h, grain_i, kx, ky, fxy_ij, parameters, Met)

#     du1 = y[1]
#     dv1 = y[3]

#     A = -du1 / dv1
#     C = u + A * v

#     return C


# if __name__ == "__main__":
#     cc.compile()

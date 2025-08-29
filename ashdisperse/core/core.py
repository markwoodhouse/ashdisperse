import numpy as np
from numba import complex128, float64, int64, jit, njit
# from numba.pycc import CC
from numba.types import Tuple

from ..containers.cheb import ChebContainer_type
from ..containers.velocities import VelocityContainer_type
from ..met.met import MetData_type
from ..params.params import Parameters_type
from ..spectral.cheb import convergedCoeffs
from .getters import (Source_z_dimless, lower_dWsdz, lower_U, lower_V,
                      lower_Ws, upper_dWsdz, upper_U, upper_V, upper_Ws)

# from .rk import integrateTo

# pylint: disable=C0103, E501

# cc = CC('core')
# cc.verbose = True


# LowerODE
# @cc.export('LowerODE', Tuple((complex128[::1], complex128[::1], complex128[::1]))(float64, float64, complex128, int64, ChebContainer_type, Parameters_type, VelocityContainer_type))
@jit(
    Tuple((complex128[::1], complex128[::1], complex128[::1]))(
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
    fastmath=True,
)
def LowerODE(kx, ky, fxy_ij, grain_i, cheby, parameters, velocities):
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    Pe = parameters.model.Peclet_number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio

    for k, N in enumerate(cheby.N):
        x, Tn, dTn, d2Tn = cheby.get_cheb(k)
        U = lower_U(velocities, k) # dimensionless U
        V = lower_V(velocities, k) # dimensionless V
        Ws = lower_Ws(velocities, k)[:, grain_i] # dimensionless
        dWsdz = lower_dWsdz(velocities, k)[:, grain_i] # dimensionless

        c2 = 4.0 * Pe * Vratio / R * np.ones_like(x, dtype=np.complex128)
        c2a = np.reshape(np.repeat(c2, N), (N, N))
        L2 = np.multiply(c2a, d2Tn)

        c1 = 2 * Ws
        c1a = np.reshape(np.repeat(c1, N), (N, N))
        L1 = np.multiply(c1a, dTn)

        c0 = (
            -1j * pi * (kx / Lx * U + ky / Ly * V)
            + dWsdz
            - pi * pi * Pe / Vratio * ((kx / Lx) ** 2 + (ky / Ly) ** 2)
        )
        c0a = np.reshape(np.repeat(c0, N), (N, N))
        L0 = np.multiply(c0a, Tn)

        L = L2 + L1 + L0

        b = np.zeros((N, 3), dtype=np.complex128)
        fz = Source_z_dimless(0.5 * (x + 1), 
                              parameters.emission.profile[grain_i],
                              parameters.emission.Suzuki_k[grain_i],
                              parameters.emission.lower[grain_i],
                              parameters.emission.upper[grain_i],
                              parameters.source.PlumeHeight)

        L[0, :] = Tn[0, :]
        L[-1, :] = Tn[-1, :]

        b[:, 0] = -fxy_ij * fz
        b[0, 0] = 0.0
        b[-1, 0] = 0.0

        b[0, 1] = 1.0
        b[-1, 1] = 0.0

        b[0, 2] = 0.0
        b[-1, 2] = 1.0

        coeffs = np.linalg.solve(L, b)  # .astype(np.complex128)

        co0 = convergedCoeffs(coeffs[:, 0].flatten(), parameters.solver.epsilon)
        co1 = convergedCoeffs(coeffs[:, 1].flatten(), parameters.solver.epsilon)
        co2 = convergedCoeffs(coeffs[:, 2].flatten(), parameters.solver.epsilon)

        if (co0.size < N) and (co1.size < N) and (co2.size < N):
            break

    return co0, co1, co2


# LowerODE
# @cc.export('LowerODE', Tuple((complex128[::1], complex128[::1], complex128[::1]))(float64, float64, complex128, int64, ChebContainer_type, Parameters_type, VelocityContainer_type))
@jit(
    complex128[::1](
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
    fastmath=True,
)
def LowerODE_f(kx, ky, fxy_ij, grain_i, cheby, parameters, velocities):
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    Pe = parameters.model.Peclet_number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio

    for k, N in enumerate(cheby.N):
        x, Tn, dTn, d2Tn = cheby.get_cheb(k)
        U = lower_U(velocities, k)
        V = lower_V(velocities, k)
        Ws = lower_Ws(velocities, k)[:, grain_i]
        dWsdz = lower_dWsdz(velocities, k)[:, grain_i]

        c2 = 4.0 * Pe * Vratio / R * np.ones_like(x, dtype=np.complex128)
        c2a = np.reshape(np.repeat(c2, N), (N, N))
        L2 = np.multiply(c2a, d2Tn)

        c1 = 2 * Ws
        c1a = np.reshape(np.repeat(c1, N), (N, N))
        L1 = np.multiply(c1a, dTn)

        c0 = (
            -1j * pi * (kx / Lx * U + ky / Ly * V)
            + dWsdz
            - pi * pi * Pe / Vratio * ((kx / Lx) ** 2 + (ky / Ly) ** 2)
        )
        c0a = np.reshape(np.repeat(c0, N), (N, N))
        L0 = np.multiply(c0a, Tn)

        L = L2 + L1 + L0

        b = np.zeros((N, 1), dtype=np.complex128)
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

        coeffs = np.linalg.solve(L, b)  # .astype(np.complex128)

        co0 = convergedCoeffs(coeffs[:, 0].flatten(), parameters.solver.epsilon)

        if co0.size < N:
            break

    return co0


@jit(
    complex128[::1](
        float64,
        float64,
        int64,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
        int64,
        complex128,
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def LowerODE_0(kx, ky, grain_i, cheby, parameters, velocities, b_indx, b_m):
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    Pe = parameters.model.Peclet_number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio

    for k, N in enumerate(cheby.N):
        x, Tn, dTn, d2Tn = cheby.get_cheb(k)
        U = lower_U(velocities, k)
        V = lower_V(velocities, k)
        Ws = lower_Ws(velocities, k)[:, grain_i]
        dWsdz = lower_dWsdz(velocities, k)[:, grain_i]

        c2 = 4.0 * Pe * Vratio / R * np.ones_like(x, dtype=np.complex128)
        c2a = np.reshape(np.repeat(c2, N), (N, N))
        L2 = np.multiply(c2a, d2Tn)

        c1 = 2 * Ws
        c1a = np.reshape(np.repeat(c1, N), (N, N))
        L1 = np.multiply(c1a, dTn)

        c0 = (
            -1j * pi * (kx / Lx * U + ky / Ly * V)
            + dWsdz
            - pi * pi * Pe / Vratio * ((kx / Lx) ** 2 + (ky / Ly) ** 2)
        )
        c0a = np.reshape(np.repeat(c0, N), (N, N))
        L0 = np.multiply(c0a, Tn)

        L = L2 + L1 + L0

        b = np.zeros((N, 1), dtype=np.complex128)

        L[0, :] = dTn[0, :]
        L[-1, :] = Tn[b_indx, :]

        b[0, 0] = 0.0
        b[-1, 0] = b_m

        coeffs = np.linalg.solve(L, b)  # .astype(np.complex128)

        co = convergedCoeffs(coeffs[:, 0].flatten(), parameters.solver.epsilon)

        if co.size < N:
            break

    return co


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
    fastmath=True,
)
def UpperODE(kx, ky, grain_i, cheby, parameters, velocities):
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    Pe = parameters.model.Peclet_number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio

    for k, N in enumerate(cheby.N):
        x, Tn, dTn, d2Tn = cheby.get_cheb(k)
        U = upper_U(velocities, k)
        V = upper_V(velocities, k)
        Ws = upper_Ws(velocities, k)[:, grain_i]
        dWsdz = upper_dWsdz(velocities, k)[:, grain_i]

        c2 = np.zeros_like(x, dtype=np.complex128)
        c2[:-1] = ((1 - x[:-1]) ** 4) / 4 * Pe * Vratio / R
        c2a = np.reshape(np.repeat(c2, N), (N, N))
        L2 = np.multiply(c2a, d2Tn)

        c1 = np.zeros_like(x, dtype=np.complex128)
        c1[:-1] = ((1 - x[:-1]) ** 2) / 2 * (Ws - (1 - x[:-1]) * Pe * Vratio / R)
        c1[-1] = 0
        c1a = np.reshape(np.repeat(c1, N), (N, N))
        L1 = np.multiply(c1a, dTn)

        c0 = np.zeros_like(x, dtype=np.complex128)
        c0[:-1] = (
            -1j * pi * (kx / Lx * U + ky / Ly * V)
            + dWsdz
            - pi * pi * Pe / Vratio * ((kx / Lx) ** 2 + (ky / Ly) ** 2)
        )
        c0[-1] = 0
        c0a = np.reshape(np.repeat(c0, N), (N, N))
        L0 = np.multiply(c0a, Tn)

        L = L2 + L1 + L0
        b = np.zeros((N, 1), dtype=np.complex128)

        L[0, :] = Tn[0, :]
        L[-1, :] = Tn[-1, :]

        b[0, 0] = 1.0

        coeffs = np.linalg.solve(L, b)  # .astype(np.complex128)

        co = convergedCoeffs(coeffs[:, 0].flatten(), parameters.solver.epsilon)

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

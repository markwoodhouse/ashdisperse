import numpy as np
from numba import boolean, complex128, float64, int64, jit, njit
# from numba.pycc import CC
from numba.types import Tuple

from ..containers.cheb import ChebContainer_type
from ..containers.velocities import VelocityContainer_type
from ..met.met import MetData_type
from ..params.params import Parameters_type
from ..spectral.cheb import (cheb_dif_m1, cheb_dif_p1, standard_chop,
                             truncateCoeffs)
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
    Tuple((complex128[::1], complex128[::1], int64))(
        float64,
        float64,
        complex128,
        int64,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
        int64,
    ),
    nopython=True,
    cache=True,
    fastmath=False,
)
def LowerODE_from(kx, ky, fxy_ij, grain_i, cheby, parameters, velocities, k_start):
    """Lower-domain solve starting the Chebyshev ladder at level k_start.

    Returns the particular and homogeneous coefficients and the level used.
    """
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    rPe = 1.0 / parameters.model.Peclet_number # reciprocal Peclet number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio

    k_used = cheby.N.size - 1
    for k in range(min(k_start, cheby.N.size - 1), cheby.N.size):
        N = cheby.N[k]
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
            k_used = k
            break

    return co0, co1, k_used


@jit(
    Tuple((complex128[::1], complex128[::1]))(
        float64, float64, complex128, int64,
        ChebContainer_type, Parameters_type, VelocityContainer_type,
    ),
    nopython=True,
    cache=True,
)
def LowerODE(kx, ky, fxy_ij, grain_i, cheby, parameters, velocities):
    co0, co1, _ = LowerODE_from(kx, ky, fxy_ij, grain_i, cheby, parameters, velocities, 0)
    return co0, co1


# UpperODE
# @cc.export('UpperODE', complex128[::1](float64, float64, int64, ChebContainer_type, Parameters_type, VelocityContainer_type))
@jit(
    Tuple((complex128[::1], int64))(
        float64,
        float64,
        int64,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
        int64,
    ),
    nopython=True,
    cache=True,
    fastmath=False,
)
def UpperODE_from(kx, ky, grain_i, cheby, parameters, velocities, k_start):
    """Upper-domain solve starting the Chebyshev ladder at level k_start.

    Returns the coefficients and the level used.
    """
    pi = np.pi
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    rPe = 1.0 / parameters.model.Peclet_number # reciprocal Peclet number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio
    # map z/H = 1 + ell (1 + x)/(1 - x): d/dz = (1-x)^2/(2 ell) d/dx,
    # d2/dz2 = (1-x)^4/(4 ell^2) d2/dx2 - (1-x)^3/(2 ell^2) d/dx
    ell = velocities.ell[grain_i]

    k_used = cheby.N.size - 1
    for k in range(min(k_start, cheby.N.size - 1), cheby.N.size):
        N = cheby.N[k]
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
            k_used = k
            break

    return co, k_used


@jit(
    complex128[::1](
        float64, float64, int64,
        ChebContainer_type, Parameters_type, VelocityContainer_type,
    ),
    nopython=True,
    cache=True,
)
def UpperODE(kx, ky, grain_i, cheby, parameters, velocities):
    co, _ = UpperODE_from(kx, ky, grain_i, cheby, parameters, velocities, 0)
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


@jit(
    Tuple((complex128[:, ::1], int64[::1], complex128[::1], int64))(
        float64,
        float64,
        complex128,
        int64,
        ChebContainer_type,
        Parameters_type,
        VelocityContainer_type,
        int64,
        complex128,
    ),
    nopython=True,
    cache=True,
    fastmath=False,
)
def LowerElements(kx, ky, fxy_ij, grain_i, cheby, parameters, velocities, k_start, upper_flux):
    """Lower-domain solve on spectral elements by static condensation.

    The lower domain [0, 1] (z/H) is split at velocities.bps[grain_i].  Each
    element e is solved on its own Chebyshev grid for a particular solution
    p_e (source, zero boundary data) and homogeneous solutions with unit
    value at one end:
        bottom element: p + V_1 h,          h'(0) = 0, h(top) = 1
        element e >= 1: p + V_e g + V_{e+1} h,  g(bot) = 1, h(top) = 1
    The interface values V (V_E = C(H)) follow from flux continuity at every
    interface and, at z = H, with the upper solution C = V_E l2, whose flux
    per unit C(H) is upper_flux.  This is algebraically the coupled
    spectral-element system, at the cost of one small LU per element.

    Convergence is tested on each element's combined coefficients (the
    individual homogeneous solutions contain settling layers that the sum
    does not), and only unresolved elements are refined.

    Returns combined coefficients (3, maxN), the number of coefficients of
    each element, the interface values (3) and the lowest level used.
    """
    pi = np.pi
    E = velocities.n_el[grain_i]
    nlev = cheby.N.size
    maxN = cheby.N[nlev - 1]
    eps = parameters.solver.epsilon
    Lx = parameters.model.Lx[grain_i]
    Ly = parameters.model.Ly[grain_i]
    rPe = 1.0 / parameters.model.Peclet_number
    Vratio = parameters.model.Velocity_ratio[grain_i]
    R = parameters.model.Diffusion_ratio
    a_diff = rPe * Vratio / R
    kk = -pi * pi * rPe / Vratio * ((kx / Lx) ** 2 + (ky / Ly) ** 2)

    raw = np.zeros((3, 3, maxN), dtype=np.complex128)  # (element, column, coeff)
    comb = np.zeros((3, maxN), dtype=np.complex128)
    sizes = np.zeros(3, dtype=np.int64)
    Js = np.zeros(3, dtype=np.float64)
    ks = np.zeros(3, dtype=np.int64)
    need = np.zeros(3, dtype=boolean)
    for e in range(E):
        ks[e] = min(k_start, nlev - 1)
        need[e] = True
        Js[e] = 0.5 * (velocities.bps[grain_i, e + 1] - velocities.bps[grain_i, e])
    Vv = np.zeros(3, dtype=np.complex128)

    while True:
        for e in range(E):
            if not need[e]:
                continue
            k = ks[e]
            N = cheby.N[k]
            _, Tn, dTn, d2Tn = cheby.get_cheb(k)
            J = Js[e]
            zt = velocities.el_z[:N, e, k, grain_i]
            U = velocities.el_U[:N, e, k, grain_i]
            V = velocities.el_V[:N, e, k, grain_i]
            Ws = velocities.el_Ws[:N, e, k, grain_i]
            dWs = velocities.el_dWs[:N, e, k, grain_i]
            c2 = a_diff / (J * J) * np.ones(N, dtype=np.float64)
            c1 = Ws / J
            c0 = -1j * pi * (kx / Lx * U + ky / Ly * V) + dWs + kk
            L = assemble_operator(c2, np.ascontiguousarray(c1), c0, Tn, dTn, d2Tn)
            b = np.zeros((N, 3), dtype=np.complex128)
            fz = Source_z_dimless(np.ascontiguousarray(zt),
                                  parameters.emission.profile[grain_i],
                                  parameters.emission.Suzuki_k[grain_i],
                                  parameters.emission.lower[grain_i],
                                  parameters.emission.upper[grain_i],
                                  parameters.source.PlumeHeight)
            b[:, 0] = -fxy_ij * fz
            b[0, 0] = 0.0
            b[N - 1, 0] = 0.0
            if e == 0:
                L[0, :] = dTn[0, :]   # C'(ground) = 0
                b[N - 1, 1] = 1.0     # h(top) = 1
            else:
                L[0, :] = Tn[0, :]
                b[0, 1] = 1.0         # g(bottom) = 1
                b[N - 1, 2] = 1.0     # h(top) = 1
            L[N - 1, :] = Tn[N - 1, :]
            co = np.linalg.solve(L, b)
            for col in range(3):
                for n in range(N):
                    raw[e, col, n] = co[n, col]
                for n in range(N, maxN):
                    raw[e, col, n] = 0.0
            sizes[e] = N
            need[e] = False

        # interface system: flux continuity at bps[m + 1], m = 0..E-1
        A = np.zeros((E, E), dtype=np.complex128)
        r = np.zeros(E, dtype=np.complex128)
        for m in range(E):
            # element m, from below (x = +1)
            N = sizes[m]
            J = Js[m]
            r[m] -= cheb_dif_p1(np.ascontiguousarray(raw[m, 0, :N])) / J
            if m == 0:
                A[m, 0] += cheb_dif_p1(np.ascontiguousarray(raw[m, 1, :N])) / J
            else:
                A[m, m - 1] += cheb_dif_p1(np.ascontiguousarray(raw[m, 1, :N])) / J
                A[m, m] += cheb_dif_p1(np.ascontiguousarray(raw[m, 2, :N])) / J
            if m + 1 < E:
                # element m + 1, from above (x = -1)
                N = sizes[m + 1]
                J = Js[m + 1]
                r[m] += cheb_dif_m1(np.ascontiguousarray(raw[m + 1, 0, :N])) / J
                A[m, m] -= cheb_dif_m1(np.ascontiguousarray(raw[m + 1, 1, :N])) / J
                A[m, m + 1] -= cheb_dif_m1(np.ascontiguousarray(raw[m + 1, 2, :N])) / J
            else:
                A[m, m] -= upper_flux
        sol = np.linalg.solve(A, r)
        for m in range(E):
            Vv[m] = sol[m]

        # combine and test convergence element by element
        any_need = False
        for e in range(E):
            N = sizes[e]
            for n in range(N):
                if e == 0:
                    comb[e, n] = raw[e, 0, n] + Vv[0] * raw[e, 1, n]
                else:
                    comb[e, n] = raw[e, 0, n] + Vv[e - 1] * raw[e, 1, n] + Vv[e] * raw[e, 2, n]
            for n in range(N, maxN):
                comb[e, n] = 0.0
            cut = standard_chop(np.abs(comb[e, :N]), eps)
            if cut >= N and ks[e] < nlev - 1:
                ks[e] += 1
                need[e] = True
                any_need = True
        if not any_need:
            break

    # The full series are returned: truncating at the chop cutoff saves
    # little here and, for element series, cost up to ~1e-5 in accuracy.
    k_min = ks[0]
    for e in range(E):
        if ks[e] < k_min:
            k_min = ks[e]
    return comb, sizes, Vv, k_min

# -*- coding: utf-8 -*-
"""velocities.py

This module defines a container for velocities used in AshDisperse,
stored as a numba jitclass.
"""

from collections import OrderedDict

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

from ..spectral.cheb import cheb_deriv, cheb_eval, cheb_fit, cheb_lobatto

# pylint: disable=C0103


Vel_spec = OrderedDict()
Vel_spec["N"] = int64[::1]
Vel_spec["z"] = float64[:, :, ::1]
Vel_spec["U"] = float64[:, :, ::1]
Vel_spec["V"] = float64[:, :, ::1]
Vel_spec["Ws"] = float64[:, :, :, ::1]
Vel_spec["dWs"] = float64[:, :, :, ::1]


@jitclass(Vel_spec)
class VelocityContainer:
    """A container for velocities at chebyshev collocation points.

    VelocityContainer contains wind and settling velocities evaluated at
    chebyshev collocation points for different degrees of approximation.

    Attributes:
        N (int[:]): Array of the degrees of the chebyshev approximations.
        z (float[:,:,:]): Altitudes of the wind data (in physical space).
                          The altitudes are split into those from ground to the
                          top of the plume (contained in z[:N[k],0,k] for
                          approximation index k) and those above (contained in
                          z[:N[k],1,k] for approximation index k).
        U (float[:,:,:]): The easting component of the wind velocity.
                          The wind velocity is split into those from ground to
                          the top of the plume (contained in U[:N[k],0,k] for
                          approximation index k) and those above (contained in
                          U[:N[k],1,k] for approximation index k).
        V (float[:,:,:]): The northin component of the wind velocity.
                          Structured as U.
        Ws (float[:,:,:,:]): The settling speed.
                             Each grain class has it's own settling speed,
                             which also varies with altitude.
                             The settling is split into those from ground to
                             the top of the plume (contained in Ws[:N[k],0,k,j]
                             for approximation index k and grain class index j)
                             and those above (contained in Ws[:N[k],1,k,j]
                             for approximation index k and grain class index
                             j).
        dWs (float[:,:,:,:]): The first derivative of the settling speed with
                              z. Structured as Ws.
    """

    def __init__(self, parameters, Met, x):
        chebIts = parameters.solver.chebIts
        maxN = parameters.solver.maxN
        bins = parameters.grains.bins
        self.N = np.zeros((chebIts), dtype=np.int64)
        self.z = np.zeros((maxN, 2, chebIts), dtype=np.float64)
        self.U = -999 * np.ones((maxN, 2, chebIts), dtype=np.float64)
        self.V = -999 * np.ones((maxN, 2, chebIts), dtype=np.float64)
        self.Ws = -999 * np.ones((maxN, 2, chebIts, bins), dtype=np.float64)
        self.dWs = -999 * np.ones((maxN, 2, chebIts, bins), dtype=np.float64)

        minN_log2 = parameters.solver.minN_log2
        U_scale = parameters.met.U_scale
        Ws_scale = parameters.met.Ws_scale
        H = parameters.source.PlumeHeight

        # Smooth representation of the met profiles in each sub-domain.
        # The met data are piecewise linear in z; using them directly as ODE
        # coefficients limits Chebyshev convergence to an algebraic rate, so
        # the adaptive solver never resolves the solution.  Instead the
        # profiles are interpolated by a Chebyshev polynomial on met_cheb_N
        # Lobatto points of the computational coordinate x, and dWs/dz is the
        # exact derivative of that interpolant.
        M = parameters.solver.met_cheb_N
        z_top = max(parameters.solver.z_top, 1.5 * H)
        xm = cheb_lobatto(M)

        # lower domain: z = H (x + 1) / 2
        zl = 0.5 * (xm + 1) * H
        # upper domain: z = 2 H / (1 - x), held constant above z_top
        zu = np.empty(M, dtype=np.float64)
        zu[: M - 1] = 2 * H / (1 - xm[: M - 1])
        zu[M - 1] = z_top
        zu = np.minimum(zu, z_top)

        cU_l = cheb_fit(np.ascontiguousarray(Met.wind_U_array(zl, U_scale)))
        cV_l = cheb_fit(np.ascontiguousarray(Met.wind_V_array(zl, U_scale)))
        cU_u = cheb_fit(np.ascontiguousarray(Met.wind_U_array(zu, U_scale)))
        cV_u = cheb_fit(np.ascontiguousarray(Met.wind_V_array(zu, U_scale)))
        Ws_l = Met.settling_speed_array(parameters, zl, Ws_scale)
        Ws_u = Met.settling_speed_array(parameters, zu, Ws_scale)
        cW_l = np.zeros((M, bins), dtype=np.float64)
        cW_u = np.zeros((M, bins), dtype=np.float64)
        cdW_l = np.zeros((M, bins), dtype=np.float64)
        cdW_u = np.zeros((M, bins), dtype=np.float64)
        for ig in range(bins):
            cW_l[:, ig] = cheb_fit(np.ascontiguousarray(Ws_l[:, ig]))
            cW_u[:, ig] = cheb_fit(np.ascontiguousarray(Ws_u[:, ig]))
            cdW_l[:, ig] = cheb_deriv(np.ascontiguousarray(cW_l[:, ig]))
            cdW_u[:, ig] = cheb_deriv(np.ascontiguousarray(cW_u[:, ig]))

        for k in range(0, chebIts):
            N = 2 ** (k + minN_log2)
            self.N[k] = np.int64(N)
            xl = np.ascontiguousarray(x[:N, k])
            xu = np.ascontiguousarray(x[: N - 1, k])

            self.z[:N, 0, k] = 0.5 * (xl + 1) * H
            self.U[:N, 0, k] = cheb_eval(cU_l, xl)
            self.V[:N, 0, k] = cheb_eval(cV_l, xl)

            self.z[: N - 1, 1, k] = 2 / (1 - xu) * H
            self.U[: N - 1, 1, k] = cheb_eval(cU_u, xu)
            self.V[: N - 1, 1, k] = cheb_eval(cV_u, xu)

            # d/dz (dimensionless z/H): lower dx/dz = 2, upper dx/dz = (1-x)^2/2
            dxdz_u = 0.5 * (1 - xu) ** 2
            for ig in range(bins):
                self.Ws[:N, 0, k, ig] = cheb_eval(np.ascontiguousarray(cW_l[:, ig]), xl)
                self.dWs[:N, 0, k, ig] = 2.0 * cheb_eval(
                    np.ascontiguousarray(cdW_l[:, ig]), xl
                )
                self.Ws[: N - 1, 1, k, ig] = cheb_eval(
                    np.ascontiguousarray(cW_u[:, ig]), xu
                )
                self.dWs[: N - 1, 1, k, ig] = dxdz_u * cheb_eval(
                    np.ascontiguousarray(cdW_u[:, ig]), xu
                )


# pylint: disable=E1101
VelocityContainer_type = VelocityContainer.class_type.instance_type

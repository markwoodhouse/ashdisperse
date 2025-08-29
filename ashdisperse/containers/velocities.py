# -*- coding: utf-8 -*-
"""velocities.py

This module defines a container for velocities used in AshDisperse,
stored as a numba jitclass.
"""

from collections import OrderedDict

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

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
        for k in range(0, chebIts):
            N = 2 ** (k + minN_log2)
            self.N[k] = np.int64(N)
            lower_z = np.zeros((N), dtype=np.float64)
            lower_z[:N] = 0.5 * (x[:N, k] + 1) * H
            self.z[:N, 0, k] = lower_z[:N]
            self.U[:N, 0, k] = Met.wind_U_array(lower_z[:N], scale=U_scale)
            self.V[:N, 0, k] = Met.wind_V_array(lower_z[:N], scale=U_scale)
            self.Ws[:N, 0, k, :] = Met.settling_speed_array(
                parameters, lower_z[:N], scale=Ws_scale
            )
            upper_z = np.zeros((N - 1), dtype=np.float64)
            upper_z[: N - 1] = 2 / (1 - x[: N - 1, k]) * H
            self.z[: N - 1, 1, k] = upper_z[: N - 1]
            self.U[: N - 1, 1, k] = Met.wind_U_array(upper_z[: N - 1], scale=U_scale)
            self.V[: N - 1, 1, k] = Met.wind_V_array(upper_z[: N - 1], scale=U_scale)
            self.Ws[: N - 1, 1, k, :] = Met.settling_speed_array(
                parameters, upper_z[: N - 1], scale=Ws_scale
            )
            self.fd_Ws(k, H)

    def fd_Ws(self, k, H):
        """Compute derivate of settling speed.

        The first derivative dWs/dz is computed using centred difference, and
        assigned to the appropriate slice of the class attribute dWs.

        Args:
            k (int): the approximation index.
            H (float): the dimensional plume height (in metres).
        """
        N = self.N[k]
        nG = self.Ws.shape[-1]
        z_l = self.z[:N, 0, k] / H
        W_l = self.Ws[:N, 0, k, :]

        z_u = self.z[: N - 1, 1, k] / H
        W_u = self.Ws[: N - 1, 1, k, :]

        dW_l = np.empty((N, nG), dtype=np.float64)
        dW_u = np.empty((N - 1, nG), dtype=np.float64)

        for ig in range(nG):
            dW_l[0, ig] = (W_l[1, ig] - W_l[0, ig]) / (z_l[1] - z_l[0])
            dW_l[1:-1, ig] = (W_l[2:, ig] - W_l[:-2, ig]) / (z_l[2:] - z_l[:-2])
            dW_l[-1, ig] = (W_l[-1, ig] - W_l[-2, ig]) / (z_l[-1] - z_l[-2])

            dW_u[0, ig] = (W_u[1, ig] - W_u[0, ig]) / (z_u[1] - z_u[0])
            dW_u[1:-1, ig] = (W_u[2:, ig] - W_u[:-2, ig]) / (z_u[2:] - z_u[:-2])
            dW_u[-1, ig] = (W_u[-1, ig] - W_u[-2, ig]) / (z_u[-1] - z_u[-2])

        self.dWs[:N, 0, k, :] = dW_l
        self.dWs[: N - 1, 1, k, :] = dW_u


# pylint: disable=E1101
VelocityContainer_type = VelocityContainer.class_type.instance_type

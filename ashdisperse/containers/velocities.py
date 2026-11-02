# -*- coding: utf-8 -*-
"""velocities.py

This module defines a container for velocities used in AshDisperse,
stored as a numba jitclass.
"""

from collections import OrderedDict

import numpy as np
from numba import complex128, float64, int64, njit
from numba.experimental import jitclass

from ..spectral.cheb import (ChebMat, cheb_deriv, cheb_eval, cheb_fit,
                             cheb_lobatto)

# pylint: disable=C0103


Vel_spec = OrderedDict()
Vel_spec["N"] = int64[::1]
Vel_spec["z"] = float64[:, :, :, ::1]
Vel_spec["U"] = float64[:, :, :, ::1]
Vel_spec["V"] = float64[:, :, :, ::1]
Vel_spec["Ws"] = float64[:, :, :, ::1]
Vel_spec["dWs"] = float64[:, :, :, ::1]
Vel_spec["ell"] = float64[::1]  # upper-domain map scale for each grain class
Vel_spec["Cheb_upper"] = complex128[:, :, ::1]  # (grain, output z > H, maxN)
# Lower-domain spectral elements (see element_breakpoints)
Vel_spec["n_el"] = int64[::1]  # number of lower-domain elements per grain (1 if element_factor == 0, else 3)
Vel_spec["bps"] = float64[:, ::1]  # (grain, 4) element breakpoints in z/H
# coefficients at element nodes: (maxN, element, level, grain)
Vel_spec["el_z"] = float64[:, :, :, ::1]  # z/H
Vel_spec["el_U"] = float64[:, :, :, ::1]
Vel_spec["el_V"] = float64[:, :, :, ::1]
Vel_spec["el_Ws"] = float64[:, :, :, ::1]
Vel_spec["el_dWs"] = float64[:, :, :, ::1]  # d Ws / d(z/H)

# Largest layer-element thickness (z/H); fine grains are split into thirds.
ELEMENT_MAX_FRACTION = 1.0 / 3.0


@njit(cache=True)
def settling_lengths(parameters, Ws_H):
    """Settling-diffusion length a / Ws(H) (in z/H) for each grain class,
    with a = kappa_v / (Ws_scale H) the dimensionless vertical diffusivity.
    Concentration changes over this length above the plume top, just below
    it, and next to the ground."""
    bins = parameters.grains.bins
    delta = np.ones(bins, dtype=np.float64)
    rPe = 1.0 / parameters.model.Peclet_number
    R = parameters.model.Diffusion_ratio
    for ig in range(bins):
        a = rPe * parameters.model.Velocity_ratio[ig] / R
        if Ws_H[ig] > 0:
            delta[ig] = a / Ws_H[ig]
    return delta


@njit(cache=True)
def element_breakpoints(parameters, delta):
    """Lower-domain element layout per grain: (n_el, breakpoints).

    With element_factor > 0 the lower domain is split into three elements
    at z/H = eta and 1 - eta, eta = min(element_factor * delta,
    ELEMENT_MAX_FRACTION).  For coarse grains eta is small and the thin
    layers at the ground and below the plume top lie in their own elements;
    for fine grains the equal split still pays, because the dense solve
    cost grows as N^3.  element_factor = 0 gives a single element."""
    bins = parameters.grains.bins
    n_el = np.ones(bins, dtype=np.int64)
    bps = np.zeros((bins, 4), dtype=np.float64)
    c = parameters.solver.element_factor
    for ig in range(bins):
        if c > 0:
            eta = min(c * delta[ig], ELEMENT_MAX_FRACTION)
            n_el[ig] = 3
            bps[ig, 1] = eta
            bps[ig, 2] = 1.0 - eta
            bps[ig, 3] = 1.0
        else:
            bps[ig, 1] = 1.0
    return n_el, bps


@njit(cache=True)
def upper_map_scales(parameters, Ws_H):
    """Scale ell of the upper-domain map z/H = 1 + ell (1 + x) / (1 - x).

    The concentration above the plume decays over the dimensionless length
    a / Ws(H), with a = kappa_v / (Ws_scale H) the vertical diffusivity in
    model units.  ell is that length times solver.upper_map_factor, limited
    to [1e-4, 1].  upper_map_factor <= 0 gives ell = 1 (the original map).
    """
    bins = parameters.grains.bins
    ell = np.ones(bins, dtype=np.float64)
    factor = parameters.solver.upper_map_factor
    if factor <= 0:
        return ell
    rPe = 1.0 / parameters.model.Peclet_number
    R = parameters.model.Diffusion_ratio
    for ig in range(bins):
        a = rPe * parameters.model.Velocity_ratio[ig] / R
        if Ws_H[ig] > 0:
            ell[ig] = min(1.0, max(1e-4, factor * a / Ws_H[ig]))
    return ell


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
        self.z = np.zeros((maxN, 2, chebIts, bins), dtype=np.float64)
        self.U = -999 * np.ones((maxN, 2, chebIts, bins), dtype=np.float64)
        self.V = -999 * np.ones((maxN, 2, chebIts, bins), dtype=np.float64)
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

        # Lower domain: z = H (x + 1) / 2
        zl = 0.5 * (xm + 1) * H
        cU_l = cheb_fit(np.ascontiguousarray(Met.wind_U_array(zl, U_scale)))
        cV_l = cheb_fit(np.ascontiguousarray(Met.wind_V_array(zl, U_scale)))
        Ws_l = Met.settling_speed_array(parameters, zl, Ws_scale)

        # Upper-domain profiles are smoothed in the fixed reference coordinate
        # xr = 1 - 2H/z (capped at z_top), independent of the grain-dependent
        # map below, so that ell changes only the discretisation, not the model.
        zu_ref = np.empty(M, dtype=np.float64)
        zu_ref[: M - 1] = 2 * H / (1 - xm[: M - 1])
        zu_ref[M - 1] = z_top
        zu_ref = np.minimum(zu_ref, z_top)
        cU_u = cheb_fit(np.ascontiguousarray(Met.wind_U_array(zu_ref, U_scale)))
        cV_u = cheb_fit(np.ascontiguousarray(Met.wind_V_array(zu_ref, U_scale)))
        Ws_u = Met.settling_speed_array(parameters, zu_ref, Ws_scale)

        # Upper domain: z/H = 1 + ell (1 + x) / (1 - x), one ell per grain.
        # Above the plume, concentration decays over the settling-diffusion
        # length kappa_v / Ws, which for coarse grains is metres.  Scaling the
        # map to that length clusters collocation points where the solution
        # varies.  ell = 1 recovers the original map z = 2H / (1 - x).
        self.ell = upper_map_scales(parameters, Ws_l[M - 1, :])

        # Lower-domain elements: met coefficients at the element nodes,
        # evaluated from the same smoothed profiles as the single domain.
        delta = settling_lengths(parameters, Ws_l[M - 1, :])
        self.n_el, self.bps = element_breakpoints(parameters, delta)
        self.el_z = np.zeros((maxN, 3, chebIts, bins), dtype=np.float64)
        self.el_U = np.zeros((maxN, 3, chebIts, bins), dtype=np.float64)
        self.el_V = np.zeros((maxN, 3, chebIts, bins), dtype=np.float64)
        self.el_Ws = np.zeros((maxN, 3, chebIts, bins), dtype=np.float64)
        self.el_dWs = np.zeros((maxN, 3, chebIts, bins), dtype=np.float64)
        for ig in range(bins):
            if self.n_el[ig] == 1:
                continue
            cW = cheb_fit(np.ascontiguousarray(Ws_l[:, ig]))
            cdW = cheb_deriv(cW)
            for e in range(self.n_el[ig]):
                za = self.bps[ig, e]
                zb = self.bps[ig, e + 1]
                for k in range(chebIts):
                    N = 2 ** (k + minN_log2)
                    xe = np.ascontiguousarray(x[:N, k])
                    zt = za + 0.5 * (zb - za) * (xe + 1)
                    xr = 2 * zt - 1  # reference coordinate of the smoothed profiles
                    self.el_z[:N, e, k, ig] = zt
                    self.el_U[:N, e, k, ig] = cheb_eval(cU_l, xr)
                    self.el_V[:N, e, k, ig] = cheb_eval(cV_l, xr)
                    self.el_Ws[:N, e, k, ig] = cheb_eval(cW, xr)
                    self.el_dWs[:N, e, k, ig] = 2.0 * cheb_eval(cdW, xr)

        for k in range(0, chebIts):
            N = 2 ** (k + minN_log2)
            self.N[k] = np.int64(N)
            xl = np.ascontiguousarray(x[:N, k])
            Ul = cheb_eval(cU_l, xl)
            Vl = cheb_eval(cV_l, xl)
            for ig in range(bins):
                self.z[:N, 0, k, ig] = 0.5 * (xl + 1) * H
                self.U[:N, 0, k, ig] = Ul
                self.V[:N, 0, k, ig] = Vl

        Nz = parameters.output.altitudes.size
        z_out = parameters.output.altitudes
        n_up = 0
        for iz in range(Nz):
            if z_out[iz] > H:
                n_up += 1
        z_out_up = np.empty(n_up, dtype=np.float64)
        j = 0
        for iz in range(Nz):
            if z_out[iz] > H:
                z_out_up[j] = z_out[iz] / H
                j += 1
        self.Cheb_upper = np.zeros((bins, n_up, maxN), dtype=np.complex128)

        for ig in range(bins):
            # derivative coefficients for Ws in the lower domain
            cW_l = cheb_fit(np.ascontiguousarray(Ws_l[:, ig]))
            cdW_l = cheb_deriv(cW_l)

            ell = self.ell[ig]
            cW_u = cheb_fit(np.ascontiguousarray(Ws_u[:, ig]))
            cdW_u = cheb_deriv(cW_u)

            for k in range(0, chebIts):
                N = self.N[k]
                xl = np.ascontiguousarray(x[:N, k])
                xu = np.ascontiguousarray(x[: N - 1, k])

                self.Ws[:N, 0, k, ig] = cheb_eval(cW_l, xl)
                # d/dz (dimensionless z/H) = 2 d/dx in the lower domain
                self.dWs[:N, 0, k, ig] = 2.0 * cheb_eval(cdW_l, xl)

                # computational x -> z/H -> reference coordinate xr
                ztu = 1 + ell * (1 + xu) / (1 - xu)
                xr = 1 - 2 / ztu
                self.z[: N - 1, 1, k, ig] = H * ztu
                self.U[: N - 1, 1, k, ig] = cheb_eval(cU_u, xr)
                self.V[: N - 1, 1, k, ig] = cheb_eval(cV_u, xr)
                self.Ws[: N - 1, 1, k, ig] = cheb_eval(cW_u, xr)
                # d/d(z/H) = (1 - xr)^2 / 2 d/dxr
                self.dWs[: N - 1, 1, k, ig] = 0.5 * (1 - xr) ** 2 * cheb_eval(
                    cdW_u, xr
                )

            # Chebyshev polynomials at the output altitudes above the plume
            x_out = (z_out_up - 1 - ell) / (z_out_up - 1 + ell)
            self.Cheb_upper[ig, :, :] = ChebMat(maxN, x_out)


# pylint: disable=E1101
VelocityContainer_type = VelocityContainer.class_type.instance_type

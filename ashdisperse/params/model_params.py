from collections import OrderedDict
from math import ceil

import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba.typed import List
from numba.types import ListType

model_spec = OrderedDict()
# Scales and dimensionless parameters as arrays, one for each grain size
model_spec["SettlingScale"] = ListType(float64)  # Settling speed
model_spec["Velocity_ratio"] = ListType(float64)  # wind speed/settling speed
# model_spec["xyScale"] =ListType(t64[::1)]  # Horizontal scale
model_spec["xScale"] = ListType(float64)  # Easting scale
model_spec["yScale"] = ListType(float64)  # Northing scale
model_spec["Lx"] = ListType(float64)  # Dimensionless extent in x
model_spec["Ly"] = ListType(float64)  # Dimensionless extent in y
model_spec["cScale"] = ListType(float64)  # Concentration
model_spec["QScale"] = ListType(float64)  # Source term
model_spec["Peclet_number"] = float64 # Peclet number
model_spec["Diffusion_ratio"] = float64 # Kappa_h/Kappa_v
model_spec["sigma_hat"] = ListType(float64)  # source radius/xyScale
model_spec["sigma_hat_scale"] = ListType(float64)  # sigma_hat scaled to -pi->pi


@jitclass(model_spec)
class ModelParameters:

    # pylint: disable=too-many-instance-attributes
    # These attributes are all needed in ModelParameters

    def __init__(self):
        self.SettlingScale = List.empty_list(np.float64)
        self.Velocity_ratio = List.empty_list(np.float64)
        self.xScale = List.empty_list(np.float64)
        self.yScale = List.empty_list(np.float64)
        self.Lx = List.empty_list(np.float64)
        self.Ly = List.empty_list(np.float64)
        self.cScale = List.empty_list(np.float64)
        self.QScale = List.empty_list(np.float64)
        self.Peclet_number = np.nan
        self.Diffusion_ratio = np.nan
        self.sigma_hat = List.empty_list(np.float64)
        self.sigma_hat_scale = List.empty_list(np.float64)

        
    def from_params(
        self, params, xScale, yScale,
    ):
        solver_params = params.solver
        met_params = params.met
        source_params = params.source
        grain_params = params.grains
        physical_params = params.physical
        
        N = met_params.Ws_scale.shape[0]
        
        Pe = (met_params.U_scale * source_params.PlumeHeight / physical_params.Kappa_h) # Peclet number
        Diffusion_ratio = physical_params.Kappa_h / physical_params.Kappa_v
        self.Peclet_number = Pe
        self.Diffusion_ratio = Diffusion_ratio

        self._empty_lists(N)

        for j in range(N):
            SettlingScale = met_params.Ws_scale[j]
            sigma_hat = source_params.radius / np.maximum(xScale[j], yScale[j])
            self.SettlingScale[j] = SettlingScale
            self.Velocity_ratio[j] = met_params.U_scale / SettlingScale
            self.xScale[j] = xScale[j]
            self.yScale[j] = yScale[j]
            self.Lx[j] = ceil(3 * source_params.radius / xScale[j]) * solver_params.domX
            self.Ly[j] = ceil(3 * source_params.radius / yScale[j]) * solver_params.domY
            self.cScale[j] = grain_params.proportion[j] * source_params.MER / source_params.radius**2 / SettlingScale
            self.QScale[j] = grain_params.proportion[j] * source_params.MER
            self.sigma_hat[j] = sigma_hat
            self.sigma_hat_scale[j] = np.pi * sigma_hat / np.sqrt(solver_params.domX * solver_params.domY)

            # self._add_to_list(
            #     SettlingScale,
            #     met_params.U_scale / SettlingScale,
            #     xScale[j],
            #     yScale[j],
            #     ceil(3 * source_params.radius / xScale[j]) * solver_params.domX,
            #     ceil(3 * source_params.radius / yScale[j]) * solver_params.domY,
            #     grain_params.proportion[j] * source_params.MER / source_params.radius**2 / SettlingScale,
            #     grain_params.proportion[j] * source_params.MER,
            #     sigma_hat,
            #     np.pi * sigma_hat / np.sqrt(solver_params.domX * solver_params.domY)
            # )

    def from_values(
        self,
        SettlingScale,
        Velocity_ratio,
        xScale,
        yScale,
        Lx,
        Ly,
        cScale,
        QScale,
        Peclet_number,
        Diffusion_ratio,
        sigma_hat,
        sigma_hat_scale,
    ):
        self.Diffusion_ratio = np.float64(Diffusion_ratio)
        self.Peclet_number = np.float64(Peclet_number)
        self._add_to_list(
            SettlingScale,
            Velocity_ratio,
            xScale,
            yScale,
            Lx,
            Ly,
            cScale,
            QScale,
            sigma_hat,
            sigma_hat_scale,
        )

    def from_lists(self,
        SettlingScale,
        Velocity_ratio,
        xScale,
        yScale,
        Lx,
        Ly,
        cScale,
        QScale,
        Peclet_number,
        Diffusion_ratio,
        sigma_hat,
        sigma_hat_scale,
    ):
        """Initialize model parameters from lists"""

        self.Diffusion_ratio = np.float64(Diffusion_ratio)
        self.Peclet_number = np.float64(Peclet_number)

        for (W, V, xS, yS, lx, ly, cS, QS, sh, shS) in zip(SettlingScale,
            Velocity_ratio,
            xScale,
            yScale,
            Lx,
            Ly,
            cScale,
            QScale,
            sigma_hat,
            sigma_hat_scale
        ):
            self._add_to_list(W, V, xS, yS, lx, ly, cS, QS, sh, shS)

        return
        

    def _add_to_list(self,
        SettlingScale,
        Velocity_ratio,
        xScale,
        yScale,
        Lx,
        Ly,
        cScale,
        QScale,
        sigma_hat,
        sigma_hat_scale,
    ):
        self.SettlingScale.append(np.float64(SettlingScale))
        self.Velocity_ratio.append(np.float64(Velocity_ratio))
        self.xScale.append(np.float64(xScale))
        self.yScale.append(np.float64(yScale))
        self.Lx.append(np.float64(Lx))
        self.Ly.append(np.float64(Ly))
        self.cScale.append(np.float64(cScale))
        self.QScale.append(np.float64(QScale))
        self.sigma_hat.append(np.float64(sigma_hat))
        self.sigma_hat_scale.append(np.float64(sigma_hat_scale))

    def _empty_lists(self,
        N
    ):
        self.SettlingScale = List.empty_list(np.float64)
        self.Velocity_ratio = List.empty_list(np.float64)
        self.xScale = List.empty_list(np.float64)
        self.yScale = List.empty_list(np.float64)
        self.Lx = List.empty_list(np.float64)
        self.Ly = List.empty_list(np.float64)
        self.cScale = List.empty_list(np.float64)
        self.QScale = List.empty_list(np.float64)
        self.sigma_hat = List.empty_list(np.float64)
        self.sigma_hat_scale = List.empty_list(np.float64)
        for j in range(N):
            self.SettlingScale.append(np.float64(-1.0))
            self.Velocity_ratio.append(np.float64(-1.0))
            self.xScale.append(np.float64(-1.0))
            self.yScale.append(np.float64(-1.0))
            self.Lx.append(np.float64(-1.0))
            self.Ly.append(np.float64(-1.0))
            self.cScale.append(np.float64(-1.0))
            self.QScale.append(np.float64(-1.0))
            self.sigma_hat.append(np.float64(-1.0))
            self.sigma_hat_scale.append(np.float64(-1.0))
    

    def describe(self):
        print("Model parameters for AshDisperse")
        print("  Settling speed scale = ", self.SettlingScale)
        print("  Velocity ratio = ", self.Velocity_ratio)
        print("  concentration scale = ", self.cScale)
        # print("  x and y scale = ", self.xyScale)
        print("  x scale = ", self.xScale)
        print("  y scale = ", self.yScale)
        print("  Lx = ", self.Lx)
        print("  Ly = ", self.Ly)
        print("  source flux scale = ", self.QScale)
        print("  Peclet number = ", self.Peclet_number)
        print("  Diffusion ratio = ", self.Diffusion_ratio)
        print("********************")


# pylint: disable=E1101
ModelParameters_type = ModelParameters.class_type.instance_type

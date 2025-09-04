import datetime
import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import toml
from numba import optional
from numba.experimental import jitclass
from numba.typed import List

from ..queryreport import is_valid_lat, is_valid_lon
from ..utilities import latlon_point_to_utm_code
from .emission_params import (EmissionParameters, EmissionParameters_type,
                              Suzuki_k_from_peak, _emission_dict)
from .grain_params import GrainParameters, GrainParameters_type, _grains_dict
from .met_params import MetParameters, MetParameters_type
from .model_params import ModelParameters, ModelParameters_type
from .output_params import (OutputParameters, OutputParameters_type,
                            _output_dict)
from .physical_params import (PhysicalParameters, PhysicalParameters_type,
                              _physical_dict)
from .solver_params import (SolverParameters, SolverParameters_type,
                            _solver_dict)
from .source_params import (SourceParameters, SourceParameters_type,
                            _source_dict)

param_spec = OrderedDict()
param_spec["solver"] = optional(SolverParameters_type)
param_spec["grains"] = optional(GrainParameters_type)
param_spec["emission"] = optional(EmissionParameters_type)
param_spec["source"] = optional(SourceParameters_type)
param_spec["physical"] = optional(PhysicalParameters_type)
param_spec["met"] = optional(MetParameters_type)
param_spec["model"] = optional(ModelParameters_type)
param_spec["output"] = optional(OutputParameters_type)


@jitclass(param_spec)
class Parameters:
    def __init__(self):
        self.source = None
        self.grains = None
        self.emission = None
        self.solver = None
        self.physical = None
        self.met = None
        self.model = None
        self.output = None

    def update(self, with_utm: bool=True):
        if with_utm:
            self.source.utm_from_latlon()
        _ = self.source.validate()
        _ = self.grains.validate()
        _ = self.emission.validate()
        _ = self.solver.validate()
        _ = self.physical.validate()
        _ = self.met.validate()
        _ = self.output.validate()

        self.output.ChebMats(self.solver.maxN, self.source.PlumeHeight)

    def describe(self):
        print("AshDisperse parameters")
        self.source.describe()
        self.grains.describe()
        self.emission.describe()
        self.solver.describe()
        self.physical.describe()
        if self.model is not None:
            self.model.describe()
        # self.met.describe()
        self.output.describe()


# pylint: disable=E1101
Parameters_type = Parameters.class_type.instance_type


def copy_parameters(A):
    new_A = Parameters()
    new_A.source = SourceParameters(
        A.source.latitude,
        A.source.longitude,
        A.source.utmcode,
        A.source.radius,
        A.source.PlumeHeight,
        A.source.MER,
        A.source.duration,
        name=A.source.name,
    )
    new_A.grains = GrainParameters()
    for j in range(A.grains.bins):
        new_A.grains.add_grain(
            A.grains.diameter[j], A.grains.density[j], A.grains.proportion[j]
        )
    new_A.emission = EmissionParameters()
    for j in range(A.emission.len()):
        new_A.emission.add_profile(
            A.emission.lower[j], A.emission.upper[j], A.emission.profile[j], A.emission.Suzuki_k[j]
        )
    new_A.solver = SolverParameters(
        domX=A.solver.domX,
        domY=A.solver.domY,
        minN_log2=A.solver.minN_log2,
        maxN_log2=A.solver.maxN_log2,
        epsilon=A.solver.epsilon,
        plateau_factor=A.solver.plateau_factor,
        fft_tol=A.solver.fft_tol,
        Nx_log2=A.solver.Nx_log2,
        Ny_log2=A.solver.Ny_log2,
    )
    new_A.physical = PhysicalParameters(
        A.physical.Kappa_h, A.physical.Kappa_v, A.physical.g, A.physical.mu
    )
    if A.met is not None:
        new_A.met = MetParameters(A.met.U_scale, A.met.Ws_scale)
    if A.model is not None:
        new_A.model = ModelParameters()
        new_A.model.from_lists(
            A.model.SettlingScale,
            A.model.Velocity_ratio,
            A.model.xScale,
            A.model.yScale,
            A.model.Lx,
            A.model.Ly,
            A.model.cScale,
            A.model.QScale,
            A.model.Peclet_number,
            A.model.Diffusion_ratio,
            A.model.sigma_hat,
            A.model.sigma_hat_scale,
        )
    new_A.output = OutputParameters(A.output.start, A.output.stop, A.output.step)
    new_A.output.set_altitudes()
    new_A.output.ChebMats(A.solver.maxN, A.source.PlumeHeight)
    return new_A


def update_parameters(
    A,
    name=None,
    domX=None,
    domY=None,
    minN_log2=None,
    maxN_log2=None,
    epsilon=None,
    plateau_factor=None,
    fft_tol=None,
    Nx_log2=None,
    Ny_log2=None,
    grains=None,
    emissions: Optional[ list[dict[str,int | float]] | str ] = None,
    latitude=None,
    longitude=None,
    radius=None,
    PlumeHeight=None,
    MER=None,
    duration=None,
    Kappa_h=None,
    Kappa_v=None,
    g=None,
    mu=None,
    start=None,
    stop=None,
    step=None,
):

    # solver
    if domX is not None:
        A.solver.domX = domX
    if domY is not None:
        A.solver.domY = domY
    if minN_log2 is not None:
        A.solver.minN_log2 = minN_log2
    if maxN_log2 is not None:
        A.solver.maxN_log2 = maxN_log2
    if plateau_factor is not None:
        A.solver.plateau_factor = plateau_factor
    if epsilon is not None:
        A.solver.epsilon = epsilon
    if fft_tol is not None:
        A.solver.fft_tol = fft_tol
    if Nx_log2 is not None:
        A.solver.Nx_log2 = Nx_log2
    if Ny_log2 is not None:
        A.solver.Ny_log2 = Ny_log2

    # grains
    if grains is not None:
        if not isinstance(grains, list):
            raise ValueError(
                "in update_parameters, grains must be a list of dicts; "
                + "received {}".format(grains)
            )
        if not all(
            "class" in g and "diameter" in g and "density" in g and "proportion" in g
            for g in grains
        ):
            raise ValueError(
                "in update parameters, grains must be a list of dicts with "
                + "dict containing the keys 'class', 'diameter', 'density',"
                + " 'proportion'; received {}".format(grains)
            )
        grains = sorted(grains, key=lambda p: p["class"], reverse=False)
        diameter = [g['diameter'] for g in grains]
        density = [g['density'] for g in grains]
        proportion = [g['proportion'] for g in grains]
        A.grains.from_lists(diameter, density, proportion)
        A.grains.validate()

    if emissions is not None:
        if isinstance(emissions, list):
            if not all(
                "class" in g and "lower" in g and "upper" in g and ("Suzuki_k" in g or "Suzuki_peak" in g)
                for g in emissions
            ):
                raise ValueError(
                    "in update parameters, emissions must be a list of dicts with "
                    + "dict containing the keys 'class', 'lower', 'upper',"
                    + f" and either 'Suzuki_k' or 'Suzuki_peak'; received {emissions}"
                )
            emissions = sorted(emissions, key=lambda p: p["class"], reverse=False)
            profile_list = []
            lower_list = []
            upper_list = []
            Suzuki_k_list = []
            for this_emission in emissions:
                profile = 0
                lower = this_emission["lower"]
                upper = this_emission["upper"]
                if "Suzuki_peak" in this_emission:
                    Suzuki_k = Suzuki_k_from_peak(this_emission["Suzuki_peak"],lower,upper)
                else:
                    Suzuki_k = this_emission["Suzuki_k"]
                profile_list.append(profile)
                lower_list.append(lower)
                upper_list.append(upper)
                Suzuki_k_list.append(Suzuki_k)
            
            A.emission.from_lists(lower_list, upper_list, profile_list, Suzuki_k_list)
        elif isinstance(emissions, str):
            if emissions == 'single':
                profile_list = [A.emission.profile[0] for _ in range(A.grains.bins)]
                lower_list = [A.emission.lower[0] for _ in range(A.grains.bins)]
                upper_list = [A.emission.upper[0] for _ in range(A.grains.bins)]
                Suzuki_k_list = [A.emission.Suzuki_k[0] for _ in range(A.grains.bins)]

                A.emission.from_lists(lower_list, upper_list, profile_list, Suzuki_k_list)
            else:
                raise ValueError(f"in update_parameters, emissions must be a list of dicts or the string 'single'; "
                    + f"received {emissions}")

        else:
            raise ValueError(
                f"in update_parameters, emissions must be a list of dicts or the string 'single'; "
                + f"received {emissions}"
            )
    
    if not (grains is not None) == (emissions is not None):
        raise ValueError("Need to update both grains and emissions")

    # source
    if name is not None:
        A.source.name = name
    if latitude is not None:
        if is_valid_lat(latitude):
            A.source.latitude = np.float64(latitude)
        else:
            raise ValueError(
                "In update_parameters, latitude must be a valid latitude in "
                + " decimal degrees"
            )
    if longitude is not None:
        if is_valid_lon(longitude):
            A.source.longitude = np.float64(longitude)
        else:
            raise ValueError(
                "In update_parameters, latitude must be a valid longitude in "
                + " decimal degrees"
            )
    if (latitude is not None) or (longitude is not None):
        A.source.utmcode = latlon_point_to_utm_code(
            A.source.latitude, A.source.longitude
        )

    if radius is not None:
        if radius < 0:
            raise ValueError("In update_parameters, radius must be positive")
        A.source.radius = np.float64(radius)

    if PlumeHeight is not None:
        if PlumeHeight < 0:
            raise ValueError("In update_parameters, PlumeHeight must be positive")
        A.source.PlumeHeight = np.float64(PlumeHeight)

    if MER is not None:
        if MER < 0:
            raise ValueError("In update_parameters, MER must be positive")
        A.source.MER = np.float64(MER)

    if duration is not None:
        if duration < 0:
            raise ValueError("In update_parameters, duration must be positive")
        A.source.duration = np.float64(duration)

    # physical
    if Kappa_h is not None:
        if Kappa_h < 0:
            raise ValueError("In PhysicalParameters, Kappa_h must be positive")
        A.physical.Kappa_h = np.float64(Kappa_h)

    if Kappa_v is not None:
        if Kappa_v < 0:
            raise ValueError("In PhysicalParameters, Kappa_v must be positive")
        A.physical.Kappa_v = np.float64(Kappa_v)

    if g is not None:
        if g <= 0:
            raise ValueError("In PhysicalParameters, g must be positive")
        A.physical.g = np.float64(g)

    if mu is not None:
        if mu <= 0:
            raise ValueError("In PhysicalParameters, mu must be positive")
        A.physical.mu = np.float64(mu)

    # wind_speed = met.max_wind_speed(A.source.PlumeHeight)
    # ws = met.settling_speed_value(A, 0.0)
    # A.met = MetParameters(wind_speed, ws)

    # A.model.from_params(A.solver, A.met, A.source, A.grains, A.physical)

    if start is not None:
        A.output.start = np.float64(start)
    if stop is not None:
        A.output.stop = np.float64(stop)
    if step is not None:
        A.output.step = np.float64(step)

    A.output.set_altitudes()
    A.output.ChebMats(A.solver.maxN, A.source.PlumeHeight)

    return A


def save_parameters(params, file="parameters.toml"):
    if os.path.exists(file):
        print(
            "WARNING: {outname} already exists and will be replaced".format(
                outname=file
            )
        )

    pdict = {
        "source": _source_dict(params.source),
        "grains": _grains_dict(params.grains),
        "emission": _emission_dict(params.emission),
        "solver": _solver_dict(params.solver),
        "physical": _physical_dict(params.physical),
        "output": _output_dict(params.output),
    }
    
    with open(file, "w") as f:
        toml.dump(pdict,f)


def load_parameters(file):
    if not os.path.exists(file):
        raise IOError("AshDisperse parameters file {} not found".format(file))

    try:
        with open(file, 'r') as f:
            paramset = toml.load(f)
    except:
        raise RuntimeError(f'Unable to read file {file}')

    params = parameters_from_dict(paramset)

    # windSpeed = met.max_wind_speed(params.source.PlumeHeight)
    # ws = met.settling_speed(params, np.atleast_1d(0.0))[0]
    # params.met = MetParameters(windSpeed, ws)

    # params.model = ModelParameters()
    # params.model.from_params(params.solver,
    #                          params.met,
    #                          params.source,
    #                          params.grains,
    #                          params.physical)

    return params

def parameters_from_dict(paramset: dict) -> Parameters_type:

    params = Parameters()

    params.source = SourceParameters(
        paramset["source"]["latitude"],
        paramset["source"]["longitude"],
        paramset["source"]["utmcode"],
        radius=paramset["source"]["radius"],
        PlumeHeight=paramset["source"]["PlumeHeight"],
        MER=paramset["source"]["MER"],
        duration=paramset["source"]["duration"],
        name=paramset["source"]["name"],
    )

    params.grains = GrainParameters()
    diameters = List()
    [diameters.append(d) for d in paramset["grains"]["diameter"]]
    densities = List()
    [densities.append(p) for p in paramset["grains"]["density"]]
    props = List()
    [props.append(p) for p in paramset["grains"]["proportion"]]
    params.grains.from_lists(diameters, densities, props)
    
    params.emission = EmissionParameters()
    lower = List()
    [lower.append(d) for d in paramset["emission"]["lower"]]
    upper = List()
    [upper.append(p) for p in paramset["emission"]["upper"]]
    profile = List()
    # Backward compatibility -- depreciate in future...
    if 'profile' in paramset["emission"].keys():
        [profile.append(p) for p in paramset["emission"]["profile"]]
    else:
        [profile.append(0) for p in paramset["emission"]["upper"]]
    Suzuki_k = List()
    [Suzuki_k.append(p) for p in paramset["emission"]["Suzuki_k"]]
    params.emission.from_lists(lower, upper, profile, Suzuki_k)

    params.solver = SolverParameters(
        domX=paramset["solver"]["domX"],
        domY=paramset["solver"]["domY"],
        minN_log2=paramset["solver"]["minN_log2"],
        maxN_log2=paramset["solver"]["maxN_log2"],
        epsilon=paramset["solver"]["epsilon"],
        plateau_factor=paramset["solver"]["plateau_factor"],
        fft_tol=paramset["solver"]["fft_tol"],
        Nx_log2=paramset["solver"]["Nx_log2"],
        Ny_log2=paramset["solver"]["Ny_log2"],
    )

    
    params.physical = PhysicalParameters(
        Kappa_h=paramset["physical"]["Kappa_h"],
        Kappa_v=paramset["physical"]["Kappa_v"],
        g=paramset["physical"]["g"],
        mu=paramset["physical"]["mu"]
    )

    params.output = OutputParameters(
        start=paramset["output"]["start"],
        stop=paramset["output"]["stop"],
        step=paramset["output"]["step"],
    )
    params.output.set_altitudes()
    params.output.ChebMats(params.solver.maxN, params.source.PlumeHeight)

    return params


# -*- coding: utf-8 -*-
import datetime
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename

import numpy as np
from scipy.integrate import solve_ivp
from siphon.catalog import TDSCatalog

from ..config import config
from ..met import NetcdfMet, load_met, met, save_met
from ..params import (EmissionParameters, GrainParameters, MetParameters,
                      ModelParameters, OutputParameters, Parameters,
                      PhysicalParameters, SolverParameters, SourceParameters,
                      Suzuki_k_from_peak, Suzuki_peak_from_k, load_parameters,
                      save_parameters)
from ..queryreport import (print_text, print_title, print_warning,
                           query_change_value, query_choices, query_datetime,
                           query_latlon, query_met_file, query_set_value,
                           query_yes_no)
from ..utilities import latlon_point_to_utm_code, string_to_datetime


def print_grain_parameters(grain_params):
    """Prints the grain classes in GrainParameters object.

    This function prints a nicely formatted description of a GrainParameters
    object.

    Args:
        grain_params: a GrainParameters instance.

    Raises:
        ValueError: if grain_params in not a GrainParameters instance.
    """
    if not isinstance(grain_params, GrainParameters):
        raise ValueError(
            "in print_grain_parameters," + " argument must be a GrainParameters object"
        )

    print_title("Grain parameters for AshDisperse")
    print_text("  Number of grain classes, N_grains = {}".format(grain_params.bins))
    for j, (diam, density, prop) in enumerate(
        zip(grain_params.diameter, grain_params.density, grain_params.proportion)
    ):
        print_text("  Grain class {}".format(j))
        print_text("    Grain diameter = {} m".format(diam))
        print_text("    Grain density = {} kg/m^3".format(density))
        print_text("    Proportion = {}".format(prop))
    print_text("********************")


def print_emission_parameters(emission_params):
    """Prints the emission profiles in EmissionParameters object.

    This function prints a nicely formatted description of a EmissionParameters
    object.

    Args:
        emission_params: a EmissionParameters instance.

    Raises:
        ValueError: if emission_params in not a EmissionParameters instance.
    """
    if not isinstance(emission_params, EmissionParameters):
        raise ValueError(
            "in print_emission_parameters," + " argument must be a EmissionParameters object"
        )

    print_title("Emission parameters for AshDisperse")
    print_text("  Number of emission profiles = {}".format(emission_params.len()))
    for j, (lower, upper, profile, Suzuki_k) in enumerate(
        zip(emission_params.lower, emission_params.upper, emission_params.profile, emission_params.Suzuki_k)
    ):
        print_text(f"  Emission profile for grain class {j}")
        print_text(f"    Lower altitude = {lower} m")
        print_text(f"    Upper altitude = {upper} m")
        if profile==0:
            print_text(f"    Profile type = 0 (Suzuki)")
            print_text(f"    Suzuki k = {Suzuki_k}")
            print_text(f"    Suzuki peak = {Suzuki_peak_from_k(Suzuki_k, lower, upper)}")
        else:
            print_text(f"    Profile type = 1 (Uniform)")
    print_text("********************")


def print_model_parameters(model_params):
    """Prints the model parameters in the ModelParameters object.

    This function prints a nicely formatted description of a ModelsParameters
    object.

    Args:
        model_params: a ModelParameters instance.

    Raises:
        ValueError: if grain_params in not a GrainParameters instance.
    """

    print_title("Model parameters for AshDisperse")
    print_text("  Settling speed scale = {} ".format(model_params.SettlingScale))
    print_text("  Velocity ratio = {} ".format(model_params.Velocity_ratio))
    # print_text("  x and y scale = {} ".format(model_params.xyScale))
    print_text("  x scale = {} ".format(model_params.xScale))
    print_text("  y scale = {} ".format(model_params.yScale))
    print_text("  Lx = {} ".format(model_params.Lx))
    print_text("  Ly = {} ".format(model_params.Ly))
    print_text("  concentration scale = {} ".format(model_params.cScale))
    print_text("  source flux scale = {} ".format(model_params.QScale))
    print_text("  Peclet number = {} ".format(model_params.Peclet_number))
    print_text("  Diffusion ratio = {} ".format(model_params.Diffusion_ratio))
    print_text("********************")


def print_output_parameters(output_params):
    print_title("Output parameters for AshDisperse")
    print_text("  Altitudes = {} m ".format(output_params.altitudes))
    print_text("********************")


def print_physical_parameters(physical_params):
    print_title("Physical parameters for AshDisperse")
    print_text(
        "  Horizontal diffusion coefficient Kappa_h = {} m^2/s ".format(
            physical_params.Kappa_h
        )
    )
    print_text(
        "  Vertical diffusion coefficient Kappa_v = {} m^2/s ".format(
            physical_params.Kappa_v
        )
    )
    print_text("  Gravitational acceleration g = {} m/s^2 ".format(physical_params.g))
    print_text("  Viscosity of air mu = {} kg/m/s ".format(physical_params.mu))
    print_text("********************")


def print_solver_parameters(solver_params):
    print_title("Solver parameters for AshDisperse")
    print_text("  Dimensionless domain size in x, domX = {}".format(solver_params.domX))
    print_text("  Dimensionless domain size in y, domY = {}".format(solver_params.domY))
    print_text(
        "  Minimum resolution in z, minN = {}, (minN_log2 = {})".format(
            solver_params.minN, solver_params.minN_log2
        )
    )
    print_text(
        "  Maximum resolution in z, maxN = {}, (maxN_log2 = {})".format(
            solver_params.maxN, solver_params.maxN_log2
        )
    )
    print_text("  Number of Chebyshev iterates = {}".format(solver_params.chebIts))
    print_text(
        "  Tolerance for Chebyshev series, epsilon = {}".format(solver_params.epsilon)
    )
    print_text(
        "  Resolution in x, Nx = {}, (Nx_log2 = {})".format(
            solver_params.Nx, solver_params.Nx_log2
        )
    )
    print_text(
        "  Resolution in y, Ny = {}, (Ny_log2 = {})".format(
            solver_params.Ny, solver_params.Ny_log2
        )
    )
    print_text("********************")


def print_source_parameters(source_params):
    """Print source parameters.

    Args:
        source_params (SourceParameters): A SourceParameters object
    """
    if not isinstance(source_params, SourceParameters):
        raise ValueError(
            "input to print_source_parameters must be a 'SourceParameters'"
            + " instance"
        )
    print_title("Source parameters for AshDisperse")
    print_text("  Mass eruption rate MER = {} kg/s".format(source_params.MER))
    print_text(
        "  Eruption duration = {} ".format(
            str(datetime.timedelta(seconds=source_params.MER))
        )
    )
    print_text("  Plume height H = {} m".format(source_params.PlumeHeight))
    print_text("  Gaussian source radius = {} m".format(source_params.radius))
    print_text("********************")


def set_grain_parameters(params_set, printout=True):
    """Set grain parameters for AshDisperse.

    The Parameters class object stores the parameters required for the model.
    set_grain_parameters runs a series of command line inputs to initialize
    the GrainParameters element of the Parameters object.

    Args:
        params_set (Parameters): The Parameters object to initialize.
        printout (bool, optional): Print GrainParameters after setup.

    Returns:
        params_set (Parameters): An updated Parameters object.

    Raises:
        ValueError: If 'params_set' is not a Parameters object.
                    If user inputs a negative diameter.
                    If user inputs a negative density.
                    If user inputs a proportion outside of [0,1).
                    If user inputs a proportion that is too large.
                    If grain parameters are not valid according to validate()
                        method of GrainParameters.
    """
    if not isinstance(params_set, Parameters):
        raise ValueError(
            "input to set_grain_parameters must be a 'Parameters' instance"
        )

    params_set.grains = GrainParameters()

    grain_bins = query_set_value("  Number of grain classes: ", answer_type=int)
    
    combined_prop = 0
    grains_added = 0
    while grains_added < grain_bins:
        print_title("Add grain class")
        grain_diam = query_set_value("  Grain diameter (m): ", answer_type=float)
        if grain_diam <= 0:
            raise ValueError("Grain diameter must be positive")
        grain_density = query_set_value("  Grain density (kg/m^3): ", answer_type=float)
        if grain_density <= 0:
            raise ValueError("Grain density must be positive")
        grain_proportion = query_set_value(
            "  Grain class proportion (0--{}): ".format(1 - combined_prop),
            answer_type=float,
        )
        if grain_proportion <= 0 or grain_proportion > 1:
            raise ValueError("Grain proportion must be between 0 and 1")
        combined_prop += grain_proportion
        params_set.grains.add_grain(grain_diam, grain_density, grain_proportion)
        grains_added += 1
    
    try:
        params_set.grains.validate()
    except ValueError as error:
        raise ValueError("Grain classes not valid")
    if printout:
        print_grain_parameters(params_set.grains)
    return params_set


def set_emission_parameters(params_set, printout=True):
    """Set emission parameters for AshDisperse.

    The Parameters class object stores the parameters required for the model.
    set_emission_parameters runs a series of command line inputs to initialize
    the EmissionParameters element of the Parameters object.

    set_emission_parameters should be called after set_grain_parameters

    Args:
        params_set (Parameters): The Parameters object to initialize.
        printout (bool, optional): Print EmissionParameters after setup.

    Returns:
        params_set (Parameters): An updated Parameters object.

    Raises:
        ValueError: If 'params_set' is not a Parameters object.
                    If 'params_set' does not have a GrainParameters element.
                    If user inputs a negative lower altitude.
                    If user inputs a negative upper altitude.
                    If user inputs a upper altitude < lower altitude.
                    If user inputs a negative Suzuki_k.
                    If user inputs a negative Suzuki_peak.
                    If user inputs a Suzuki_peak < lower altitude.
                    If user inputs a Suzuki_peak > upper altitude.
    """
    if not isinstance(params_set, Parameters):
        raise ValueError(
            "input to set_emission_parameters must be a 'Parameters' instance"
        )
    
    if not isinstance(params_set.grains, GrainParameters):
        raise ValueError(
            "input to set_emission_parameters must be a 'Parameters' instance "\
            "with GrainParameters set"
        )

    if params_set.grains.bins<1:
        raise ValueError(
            "input to set_emission_parameters must be a called after grains have been set\n" \
            f"Input currently has {params_set.grains.bins} grain classes."
        )

    params_set.emission = EmissionParameters()

    Ngrains = params_set.grains.bins

    if Ngrains == 1:
        single_profile = True
    else:
        single_profile = query_yes_no("Use a single emission profile for all grain classes?", default="yes")

    if single_profile:
        N = 1
    else:
        N = Ngrains

    profiles_added = 0
    while profiles_added < N:
        # profile = query_choices("  Select emission profile (0 = Suzuki, 1 = Uniform): ", choices=["0", "1"], default="0")
        profile = 0 # Currently only Suzuki
        profile = int(profile)

        lower = query_change_value("  Lower altitude for emission (m): ", answer_type=float, default=0., lower=0., upper=params_set.source.PlumeHeight)

        upper = query_change_value("  Upper altitude for emission (m): ", answer_type=float, default=params_set.source.PlumeHeight, lower=lower, upper=params_set.source.PlumeHeight)

        if profile==0:
            suzuki_choice = query_choices(
                "  Select Suzuki emission profile parameter: ",
                choices=["k", "peak"],
                default="k",
            )
            if suzuki_choice == "k":
                Suzuki_k = query_change_value(
                    "  Suzuki emission profile k-parameter",
                    default=10.0,
                    lower=0.,
                    answer_type=float,
                )
            else:
                Suzuki_peak = query_change_value(
                    "  Suzuki emission profile peak altitude (m)",
                    default=Suzuki_peak_from_k(10., lower, upper),
                    lower = lower,
                    upper = upper,
                    answer_type=float,
                )
                Suzuki_k = Suzuki_k_from_peak(Suzuki_peak, lower, upper)
        else:
            Suzuki_k = 0

        params_set.emission.add_profile(lower, upper, profile, Suzuki_k)
        profiles_added += 1

    if single_profile and Ngrains>1:
        for j in range(1,Ngrains):
            params_set.emission.add_profile(lower, upper, profile, Suzuki_k)
    
    if printout:
        print_emission_parameters(params_set.emission)
    return params_set


def set_output_parameters(params_set, printout=False):
    """Set output parameters for AshDisperse.

    The Parameters class object stores the parameters required for the model.
    set_output_parameters runs a series of command line inputs to initialize
    the OutputParameters element of the Parameters object.

    Args:
        params_set (Parameters): The Parameters object to initialize.
        printout (bool, optional): Print GrainParameters after setup.

    Returns:
        params_set (Parameters): An updated Parameters object.

    Raises:
        ValueError: If 'params_set' is not a Parameters object

    TODO (Mark): Add validation of parameters.
    """
    if not isinstance(params_set, Parameters):
        raise ValueError(
            "input to set_output_parameters must be a 'Parameters' instance"
        )
    params_set.output.start = query_change_value(
        "  Lower altitude", default=params_set.output.start, answer_type=float
    )
    params_set.output.stop = query_change_value(
        "  Upper altitude", default=params_set.output.stop, answer_type=float
    )
    params_set.output.step = query_change_value(
        "  Altitude step", default=params_set.output.step, answer_type=float
    )

    params_set.output.set_altitudes()
    params_set.output.ChebMats(params_set.solver.maxN, params_set.source.PlumeHeight)

    if printout:
        print_output_parameters(params_set.output)

    return params_set


def set_parameters():
    """Interface to set Parameters for AshDisperse.

    The Parameters class object stores the parameters required for the model.
    set_parameters runs a series of command line inputs to initialize
    the Parameters object.

    Args:
        None

    Returns:
        params_set (Parameters): A Parameters object.

    """
    print_title("Set parameters for AshDisperse")
    params_set = Parameters()

    print_title("Set source parameters")
    params_set = set_source_parameters(params_set)

    print_title("Set grain parameters")
    params_set = set_grain_parameters(params_set)

    print_title("Set emission parameters")
    params_set = set_emission_parameters(params_set)

    print_title("Set solver parameters")
    params_set.solver = SolverParameters()
    print_solver_parameters(params_set.solver)
    solver_change = query_yes_no("Change solver parameters?", default="no")
    if solver_change:
        params_set = set_solver_parameters(params_set)

    print_title("Set physical parameters")
    params_set.physical = PhysicalParameters()
    print_physical_parameters(params_set.physical)
    physical_change = query_yes_no("Change physical parameters?", default="no")
    if physical_change:
        params_set = set_physical_parameters(params_set)

    print_title("Set output parameters")
    params_set.output = OutputParameters(
        start=0,
        stop=1.1 * params_set.source.PlumeHeight,
        step=1.1 * params_set.source.PlumeHeight / 10,
    )
    print_output_parameters(params_set.output)
    output_change = query_yes_no("Change output parameters?", default="no")
    if output_change:
        params_set = set_output_parameters(params_set)
    else:
        params_set.output.set_altitudes()
        params_set.output.ChebMats(
            params_set.solver.maxN, params_set.source.PlumeHeight
        )

    return params_set


def set_physical_parameters(params_set, printout=False):
    """Set physical parameters for AshDisperse.

    The Parameters class object stores the parameters required for the model.
    set_physical_parameters runs a series of command line inputs to initialize
    the PhysicalParameters element of the Parameters object.

    Args:
        params_set (Parameters): The Parameters object to initialize.
        printout (bool, optional): Print GrainParameters after setup.

    Returns:
        params_set (Parameters): An updated Parameters object.

    Raises:
        ValueError: If 'params_set' is not a Parameters object

    TODO (Mark): Add validation of parameters.
    """
    if not isinstance(params_set, Parameters):
        raise ValueError(
            "input to set_physical_parameters must be a 'Parameters' instance"
        )

    params_set.physical.Kappa_h = query_change_value(
        "  Horizontal diffusion coefficient, kappa_h",
        default=params_set.physical.Kappa_h,
        answer_type=float,
    )
    params_set.physical.Kappa_v = query_change_value(
        "  Vertical diffusion coefficient, kappa_v",
        default=params_set.physical.Kappa_v,
        answer_type=float,
    )
    params_set.physical.g = query_change_value(
        "  Gravitational acceleration, g",
        default=params_set.physical.g,
        answer_type=float,
    )
    params_set.physical.mu = query_change_value(
        "  Viscosity of air, mu", default=params_set.physical.mu, answer_type=float
    )

    if printout:
        print_solver_parameters(params_set.solver)

    return params_set


def set_source_parameters(params_set):
    """Set source parameters for AshDisperse.

    The Parameters class object stores the parameters required for the model.
    set_source_parameters runs a series of command line inputs to initialize
    the SourceParameters element of the Parameters object.

    Args:
        params_set (Parameters): The Parameters object to initialize.

    Returns:
        params_set (Parameters): An updated Parameters object.

    Raises:
        ValueError: If 'params_set' is not a Parameters object
    """
    if not isinstance(params_set, Parameters):
        raise ValueError(
            "input to set_source_parameters must be a 'Parameters' instance"
        )

    name, lat, lon = query_latlon()

    params_set.source = SourceParameters(
        lat, lon, latlon_point_to_utm_code(lat, lon), name=name
    )

    params_set.source.MER = query_change_value(
        "  Mass eruption rate", default=1e6, answer_type=float, lower=0., upper=None
    )
    params_set.source.duration = query_change_value(
        "  Eruption duration", default=5*3600., answer_type=float
    )
    params_set.source.PlumeHeight = query_change_value(
        "  Plume height", default=10e3, answer_type=float
    )
    params_set.source.radius = query_change_value(
        "  Gaussian source radius", default=10e3, answer_type=float
    )
    return params_set


def set_solver_parameters(params_set, printout=False):
    """Set solver parameters for AshDisperse.

    The Parameters class object stores the parameters required for the model.
    set_solver_parameters runs a series of command line inputs to initialize
    the SolverParameters element of the Parameters object.

    Args:
        params_set (Parameters): The Parameters object to initialize.
        printout (bool, optional): Print GrainParameters after setup.

    Returns:
        params_set (Parameters): An updated Parameters object.

    Raises:
        ValueError: If 'params_set' is not a Parameters object

    TODO (Mark): Add validation of parameters.
    """
    if not isinstance(params_set, Parameters):
        raise ValueError(
            "input to set_solver_parameters must be a 'Parameters' instance"
        )

    params_set.solver.domX = query_change_value(
        "  Dimensionless domain size in x, domX",
        default=params_set.solver.domX,
        answer_type=float,
    )
    params_set.solver.domY = query_change_value(
        "  Dimensionless domain size in y, domY",
        default=params_set.solver.domY,
        answer_type=float,
    )
    params_set.solver.minN_log2 = query_change_value(
        "  Minimum resolution in z, log2(minN)",
        default=params_set.solver.minN_log2,
        answer_type=int,
    )
    params_set.solver.maxN_log2 = query_change_value(
        "  Maximum resolution in z, log2(maxN)",
        default=params_set.solver.maxN_log2,
        answer_type=int,
    )
    params_set.solver.epsilon = query_change_value(
        "  Tolerance for Chebyshev series, epsilon",
        default=params_set.solver.epsilon,
        answer_type=float,
    )
    params_set.solver.fft_tol = query_change_value(
        "  Tolerance for FFT terms, fft_tol",
        default=params_set.solver.fft_tol,
        answer_type=float,
    )
    params_set.solver.Nx_log2 = query_change_value(
        "  Resolution in x, log2(Nx)",
        default=params_set.solver.Nx_log2,
        answer_type=int,
    )
    params_set.solver.Ny_log2 = query_change_value(
        "  Resolution in y, log2(Ny)",
        default=params_set.solver.Ny_log2,
        answer_type=int,
    )

    # params_set.solver.rk = query_yes_no("  Use Runge-Kutta solver?", default="no")
    # if params_set.solver.rk:
    #     params_set.solver.rtol = query_change_value(
    #         "  Runge-Kutta relative error tolerance, rtol",
    #         default=params_set.solver.rtol,
    #         answer_type=float,
    #     )
    #     params_set.solver.maxStep = query_change_value(
    #         "  Runge-Kutta maximum step size, maxStep",
    #         default=params_set.solver.maxStep,
    #         answer_type=float,
    #     )

    if printout:
        print_solver_parameters(params_set.solver)

    return params_set


def set_met(params_set, source="interface", **kwargs):

    if source == "interface":
        if "gui" in kwargs.keys():
            gui = kwargs["gui"]
        else:
            gui = False
        met_set = _set_met_interface(params_set, gui=gui)
    elif source == "local":
        if "met_filename" not in kwargs.keys():
            raise RuntimeError(
                "In met_set, 'met_filename' must be given if source='local'"
            )
        met_filename = kwargs["met_filename"]
        met_set = _get_local_netcdf_met(
            met_filename, params_set.source.latitude, params_set.source.longitude
        )
    elif source == "gfs":
        if "datetime" not in kwargs.keys():
            raise RuntimeError("In met_set, 'date' must be given if source='gfs'")
        date = kwargs["datetime"]
        met_datetime = string_to_datetime(date)
        met_set = _get_gfs_met(
            params_set.source.latitude, params_set.source.longitude, met_datetime
        )
    elif source == "ecmwf":
        if "datetime" not in kwargs.keys():
            raise RuntimeError("In met_set, 'date' must be given if source='ecmwf'")
        date = kwargs["datetime"]
        met_datetime = string_to_datetime(date)
        if "met_filename" in kwargs.items():
            met_filename = kwargs["met_filename"]
        else:
            met_filename = "./MetData/era5_download.nc"
        met_set = _get_ecmwf_met(
            params_set.source.latitude,
            params_set.source.longitude,
            met_datetime,
            met_filename=met_filename,
        )

    # params_set = _set_met_parameters(params_set, met_set)

    return met_set


def set_met_parameters(params_set, met_set):

    wind_speed = met_set.max_wind_speed(params_set.source.PlumeHeight)

    settling_speed = met_set.calculate_settling_speed_value(params_set, 0.0)

    params_set.met = MetParameters(wind_speed, settling_speed)

    return params_set


def advected_settling_trajectories(params, met) -> list[dict[str, np.typing.NDArray]]:

    trajectories = []

    for j in range(params.grains.bins):
        def dxdz(z,x):
            ws = met.settling_speed_for_grain_class_value(params, j, z)
            dx = [-met.wind_U_value(z)/ws, -met.wind_V_value(z)/ws]
            return dx
        
        sol = solve_ivp(dxdz, [params.source.PlumeHeight,0], [0,0], max_step=10)
        trajectories.append({'x':sol.y[0,:], 'y':sol.y[1,:], 'z': sol.t})

    return trajectories


def estimate_dispersal_distance(params, met) -> np.typing.NDArray:
    dispersal_distance = np.zeros((params.grains.bins,2))

    trajectories = advected_settling_trajectories(params, met)

    for j in range(params.grains.bins):
        dispersal_distance[j,0] = np.max(np.abs(trajectories[j]['x']))
        dispersal_distance[j,1] = np.max(np.abs(trajectories[j]['y']))
    return dispersal_distance


def dispersal_distance_vs_grain_sizes(params, met) -> np.typing.NDArray:
    
    for j, dd in enumerate(params.grains.diameter):
        params.grains.diameter[0] = dd
        dispersal_distance = ad.estimate_dispersal_distance(params, met)
        dist[j] = np.amax(dispersal_distance)


def set_model_parameters(params, met, square=False):

    xScale = np.zeros((params.grains.bins))
    yScale = np.zeros((params.grains.bins))
    dispersal_distance = estimate_dispersal_distance(params, met)

    for j in range(params.grains.bins):

        if np.amax(dispersal_distance[j,:])>1000e3:
            print_warning(f"Grain {j} with diameter {params.grains.diameter[j]} m is expected to disperse for {np.amax(dispersal_distance[j,:])/1000:.0f} km")
        
        xScale[j] = dispersal_distance[j,0] + params.source.radius*3
        yScale[j] = dispersal_distance[j,1] + params.source.radius*3

        if square:
            xScale[j] = np.maximum(xScale[j],yScale[j])
            yScale[j] = xScale[j]

    params.model = ModelParameters()
    params.model.from_params(
        params,
        xScale, yScale
    )

    return params


def _set_met_interface(params_set, gui=False):
    """Interface to set MetData for AshDisperse.

    The MetData class object stores the meteorological data required for the
    model.
    set_met runs a series of user inputs to initialize the MetData object.
    Additionally, setting meteorological data finalizes the Parameters object
    that stores parameters for the model.

    Args:
        params_set (Parameters): A Parameters object containing parameters for
                                 the model.
        gui (bool, optional): Use a tkinter gui; defaults to False.

    Returns:
        met_set (MetData): A MetData object containing meteorological data.
        params_set (Parameters): An updated Parameters object.

    Raises:
        OSError: If user inputted file not found.
    """
    print_title("Set up meteorological inputs")
    met_source = int(
        query_choices(
            "Source of meteorological data: \n"
            + "   0 -> local netcdf file \n"
            + "   1 -> download ERA5 data \n"
            + "   2 -> standard atmosphere \n"
            + "   3 -> GFS forecast \n"
            + "   4 -> GFS archive \n",
            choices=["0", "1", "2", "3", "4"],
            default="0",
        )
    )

    if met_source == 0:

        if gui:
            tk.Tk().withdraw()
            met_file = askopenfilename(
                initialdir="./",
                title="Select met file (netCDF)",
                filetypes=(("netCDF", "*.nc"), ("all files", "*.*")),
            )
        else:
            met_file = query_met_file()

        met_nc = _get_local_netcdf_met(met_file)

        met_nc_time = met_nc.time

        if len(met_nc_time)>1:
            met_datetime = query_datetime(
               "Date and time for meteorological data (as yyyy-mm-dd hh:mm)"
            )
        elif len(met_nc_time)==1:
            met_datetime = met_nc.time[0]
        else:
            met_datetime = datetime.datetime.today()

        met_set = _extract_local_netcdf_met(met_nc, params_set.source.latitude, params_set.source.longitude, met_datetime)

    elif met_source == 1:
        met_file = query_change_value(
            "File name for download",
            default="./MetData/era5_download.nc",
            answer_type=str,
        )
        met_datetime = query_datetime(
            "Date and time for meteorological data (as yyyy-mm-dd hh:mm)"
        )

        met_set = _get_ecmwf_met(
            params_set.source.latitude,
            params_set.source.longitude,
            met_datetime,
            met_filename=met_file,
        )

    elif met_source == 2:
        print("Using Standard Atmosphere")
        met_set = met.MetData()

    elif met_source == 3:
        met_datetime = query_datetime(
            "Date and time for meteorological data (as yyyy-mm-dd hh:mm)"
        )
        met_set = _get_gfs_met(
            params_set.source.latitude, params_set.source.longitude, met_datetime
        )

    elif met_source == 4:
        met_datetime = query_datetime(
            "Date and time for GFS forecast cycle (as yyyy-mm-dd hh:00)"
        )
        forecast_hr = query_set_value(
            "Forecast hour (hours ahead of forecast cycle)", answer_type=int
        )
        met_set = _get_gfs_archive(
            params_set.source.latitude,
            params_set.source.longitude,
            met_datetime,
            forecast_hr,
        )

    return met_set


def _get_local_netcdf_met(met_file):
    if not os.path.isfile(met_file):
        raise OSError("Met file {} does not exist".format(met_file))
    nc_data = NetcdfMet(met_file)
    return nc_data

def _extract_local_netcdf_met(nc_data, latitude, longitude, met_datetime):
    nc_data.extract(lat=latitude, lon=longitude, datetime=met_datetime)
    met_set = met.MetData()
    met_set = met.netCDF_to_Met(met_set, nc_data)
    return met_set


def _get_ecmwf_met(
    latitude, longitude, met_datetime, met_filename="./MetData/era5_download.nc"
):
    cdf_data = NetcdfMet(met_filename)
    cdf_data.download_era5(lat=latitude, lon=longitude, datetime=met_datetime)
    cdf_data.extract(lat=latitude, lon=longitude, datetime=met_datetime)
    met_set = met.MetData()
    met_set = met.netCDF_to_Met(met_set, cdf_data)
    return met_set


def _get_gfs_met(latitude, longitude, met_datetime):
    gfs = TDSCatalog(
        "http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml"
        "?dataset=grib/NCEP/GFS/Global_0p25deg/Best"
    )
    gfs_ds = gfs.datasets[0]
    ncss = gfs_ds.subset()
    query = ncss.query()
    query.lonlat_point(longitude, latitude)
    query.time(met_datetime)
    # query.accept("netcdf4")
    query.variables(
        "Temperature_isobaric",
        "Geopotential_height_isobaric",
        "Relative_humidity_isobaric",
        "u-component_of_wind_isobaric",
        "v-component_of_wind_isobaric",
    )
    gfs_data = ncss.get_data(query)
    met_set = met.MetData()
    met_set = met.gfs_to_Met(met_set, gfs_data)
    return met_set


def _get_gfs_archive(latitude, longitude, cycle_datetime, forecast_hr):
    gfs_data = met.GFSarchive(
        cycle_datetime, forecast_hr, lat=latitude, lon=longitude
    ).profiles()

    met_set = met.MetData()
    met_set = met.gfs_archive_to_Met(met_set, gfs_data)
    return met_set


def save_inputs(params, met, filename="AshDisperse"):
    save_parameters(params, file=f"{filename}.params.txt")
    save_met(met, file=f"{filename}.met.npz")


def load_inputs(filename):
    params = load_parameters(f"{filename}.params.txt")
    met = load_met(f"{filename}.met.npz")

    return params, met
    return params, met

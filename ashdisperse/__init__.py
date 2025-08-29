from pkg_resources import resource_filename

from ashdisperse.ashdisperse import load_result, refine, setup, solve
from ashdisperse.core import AshDisperseResult
from ashdisperse.interface import (advected_settling_trajectories,
                                   estimate_dispersal_distance, load_inputs,
                                   save_inputs, set_met, set_parameters)
from ashdisperse.met import load_met, save_met, wind_plot
from ashdisperse.params import EmissionParameters as _EmissionParameters
from ashdisperse.params import GrainParameters as _GrainParameters
from ashdisperse.params import OutputParameters as _OutputParameters
from ashdisperse.params import Parameters as _Parameters
from ashdisperse.params import PhysicalParameters as _PhysicalParameters
from ashdisperse.params import SolverParameters as _SolverParameters
from ashdisperse.params import SourceParameters as _SourceParameters
from ashdisperse.params import (copy_parameters, load_parameters,
                                save_parameters, update_parameters)
from ashdisperse.version import __version__

__author__ = "Mark J. Woodhouse"

def __initialize__():

    params = _Parameters()
    params.source = _SourceParameters(
        51.456255,
        -2.604762,
        32632,
        10000,
        10000,
        1e6,
        18000,
        name='initialize',
    )
    params.grains = _GrainParameters()
    params.grains.add_grain(1e-3, 1200, 1)
    params.emission = _EmissionParameters()
    params.emission.add_profile(0, 10000, 0, 10)
    params.solver = _SolverParameters(
        domX=1.5,
        domY=1.5,
        minN_log2=4,
        maxN_log2=8,
        epsilon=1e-8,
        Nx_log2=5,
        Ny_log2=5,
    )
    params.physical = _PhysicalParameters()
    params.output = _OutputParameters(0, 10000, 1000)
    params.output.set_altitudes()
    params.output.ChebMats(params.solver.maxN, params.source.PlumeHeight)

    metdata = resource_filename(__name__, "/data/metdata.npz")

    met = load_met(metdata)

    return params, met
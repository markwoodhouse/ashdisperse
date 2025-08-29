# -*- coding: utf-8 -*-
"""AshDisperse -- a steady-state volcanic ash dispersion model.

AshDisperse implements an efficient and accurate numerical method for solving
the advection-diffusion-sedimentation equation for tephra classes of different
characteristics in a wind field.

Example:
    params, met = setup(gui=False)

    result = solve(params, met, timer=True)

    for grain_i in range(0, params.grains.bins):
        result.plot_settling_flux_for_grain_class(grain_i)
        result.plot_conc_for_grain_class(grain_i)
        result.plot_iso_conc_for_grain_class(grain_i, 1e-4)

    result.plot_ashload(resolution=500., vmin=1e-4, nodata=-1,
                        export_gtiff=True, export_name='AshLoad.tif')
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_SCHEDULE"] = "dynamic"

import time

import numpy as np
from numba import parallel_chunksize
from scipy.fft import ifft2, irfft2

from .config import (get_chunk_size, get_max_threads, get_num_threads,
                     set_default_threads, set_num_threads)
from .containers import ChebContainer, VelocityContainer
from .core import AshDisperseResult
from .interface import (set_met, set_met_parameters, set_model_parameters,
                        set_parameters)
from .solver import ade_ft_refine, ade_ft_system, source_xy_dimless
from .spectral import grid_freq

set_default_threads(1)


def setup(gui=False):
    """Set parameters and meteorological data for AshDisperse.

    The Parameters class object stores the parameters required for the model
    and the MetData class object stores meteorological data.
    setup() instantiates and initializes these objects.

    Args:
        gui (bool, optional): Use a tkinter gui; defaults to False.

    Returns:
        params_set (Parameters): A Parameters object containing parameters.
        met_set (MetData): A MetData object containing meteorological data.
    """
    params_set = set_parameters()
    met_set = set_met(params_set, gui=gui)
    return params_set, met_set


def solve(parameters, met_data, timer=False, square_grid=False):
    """Run the numerical solver using parameters and meteorological data.

    Args:
        parameters (Parameters): A set of parameters contained in a Parameters
                                 object.
        met_data (MetData): Meteorological data contained in a MetData object.
        timer (bool, optional): Run a timer of stages of the solver;
                                defaults to False.

    Returns:
        AshDisperseResult: An AshDisperseResult object containing the model
                           results.
    """

    if timer:
        timer_start = time.time()

    parameters = set_met_parameters(parameters, met_data)

    parameters = set_model_parameters(parameters, met_data, square=square_grid)

    parameters.update()

    cheby = ChebContainer(parameters)

    velocities = VelocityContainer(parameters, met_data, cheby.x)

    Nx = parameters.solver.Nx
    Ny = parameters.solver.Ny
    Ng = parameters.grains.bins
    Nz = parameters.output.Nz

    x, kx = grid_freq(Nx)
    y, ky = grid_freq(Ny)

    _, fxy_f = source_xy_dimless(x, y, parameters)

    if timer:
        timer_mid = time.time()
    
    conc_0_FT = np.zeros((Ny, Nx//2+1, Ng), dtype=np.complex128)
    conc_z_FT = np.zeros((Ny, Nx//2+1, Nz, Ng), dtype=np.complex128)
    # conc_0_FT = np.zeros((Ny, Nx, Ng), dtype=np.complex128)
    # conc_z_FT = np.zeros((Ny, Nx, Nz, Ng), dtype=np.complex128)
    
    with parallel_chunksize(get_chunk_size()):
        conc_0_FT, conc_z_FT = ade_ft_system(kx, ky, fxy_f, cheby, parameters, velocities)

    if timer:
        timer_end = time.time()
        print("Equation solve time : ", timer_end - timer_mid)
        print("Total solve time : ", timer_end - timer_start)

    return AshDisperseResult(parameters, conc_0_FT, conc_z_FT)


def refine(results, parameters, met_data, timer=False, full=False):

    if timer:
        timer_start = time.time()

    parameters = set_met_parameters(parameters, met_data)

    parameters = set_model_parameters(parameters, met_data)

    parameters.update()

    cheby = ChebContainer(parameters)

    velocities = VelocityContainer(parameters, met_data, cheby.x)

    Nx = parameters.solver.Nx
    Ny = parameters.solver.Ny
    Ng = parameters.grains.bins
    Nz = parameters.output.Nz

    x, kx = grid_freq(Nx)
    y, ky = grid_freq(Ny)

    _, fxy_f = source_xy_dimless(x, y, parameters)

    if timer:
        timer_mid = time.time()
    
    conc_0_FT = np.zeros((Ny, Nx//2+1, Ng), dtype=np.complex128)
    conc_z_FT = np.zeros((Ny, Nx//2+1, Nz, Ng), dtype=np.complex128)
    
    with parallel_chunksize(get_chunk_size()):
        conc_0_FT, conc_z_FT = ade_ft_refine(results.C0_FT, results.Cz_FT, kx, ky, fxy_f, cheby, parameters, velocities, full=full)

    if timer:
        timer_end = time.time()

        print("Total solve time : ", timer_end - timer_start)
        print("Equation solve time : ", timer_mid - timer_start)

    return AshDisperseResult(parameters, conc_0_FT, conc_z_FT)


def load_result(fname: str):
    result = AshDisperseResult.from_netcdf(fname)
    return result

if __name__ == "__main__":

    print("Running AshDisperse")

    params = set_parameters()

    Met = set_met(params)

    result = solve(params, Met, timer=True)

    for grain_i in range(0, params.grains.bins):
        result.plot_settling_flux_for_grain_class(grain_i)
        result.plot_conc_for_grain_class(grain_i)
        result.plot_iso_conc_for_grain_class(grain_i, 1e-4)

    _ = result.get_ashload(
        resolution=500.0,
        vmin=1e-4,
        nodata=-1,
        export_gtiff=True,
        export_name="AshLoad.tif",
    )

    result.plot_ashload(resolution=500, vmin=0.1)

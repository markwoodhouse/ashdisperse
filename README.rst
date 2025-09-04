AshDisperse
===========

**AshDisperse** is a numerical tool that solves the *steady-state* Advection-Diffusion-Sedimentation (ADS) Equation for a model volcanic eruption emission profile in a real wind field.

The solver is designed to be computationally efficient while solving the ADS equation accurately.

Quickstart
==========

Installation
------------

It is recommended to install AshDisperse in a virtual environment.

If using conda, it is recommended to first install the numpy, numba and tbb packages from conda-forge::

    conda install -c conda-forge numpy numba tbb

AshDisperse and its dependencies can be installed from PyPI with pip::

    pip install ashdisperse

Jupyter notebook example
------------------------

Installing AshDisperse will also download an example of use in a Jupyter notebook and create a command-line function `ashdisperse_nb` to launch  the notebook.


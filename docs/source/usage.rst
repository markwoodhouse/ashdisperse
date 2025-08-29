Usage
=====

.. _installation:

Installation
------------

.. note::
    To use *AshDisperse*, it is recommended to use a virtual environment (here called ``ash``).

First, install AshDisperse using pip:

.. code-block:: console

    (ash) $ pip install ashdisperse

This will install the package along with dependencies.


Command-line invocation
-----------------------

Once installed, you can run *AshDisperse* from the shell command-line using the command:

.. code-block:: console

    (ash) $ python -m ashdisperse

.. note::
    *AshDisperse* uses the `Numba <https://numba.pydata.org/>`_ package to accelerate computations.  Numba compiles the Python code into fast machine code.  The initial compilation step can take several seconds to complete, but results in substantial reduction in run times.

Following compilation, the *AshDisperse* command-line interface will appear, which allows the model to be set up by specifying parameters and meteorological data.

Interactive python session
--------------------------

*AshDisperse* can be used in an interactive Python session (e.g. in iPython) by importing the package.

First, launch an interactive Python session, e.g.

.. code-block:: console

    (ash) $ ipython

and then import *AshDisperse* (and some other packages that will be useful too):

.. nbinput:: ipython3
    :execution-count: 1
    
    import ashdisperse as ad
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime  # loaded to report compilation is complete
    print('AshDisperse compilation completed {}'.format(datetime.datetime.now()))

with the last line indicating that the Numba compilation is complete.  Here we have import matplotlib for plotting, numpy for numerical data, and datetime simply for reporting compilation.

The compiled *AshDisperse* package is now available with the alias ``ad``.

The command-line interface for setting the parameters and meteorological conditions can be accessed using the ``setup`` function, returning ``params`` and ``Met`` objects:

.. nbinput:: ipython3
    :execution-count: 2

    params, Met = ad.setup(gui=False)

Further details of the :ref:`parameters <parameters>` and :ref:`meteorology <meteorology>` are available.

Once the parameters and meteorology is set, the solver can be started using the ``solve`` function which returns a ``results`` object:

.. nbinput:: ipython3
    :execution-count: 3

    result = ad.solve(params, Met, timer=True)

Here we have included the optional ``timer`` keyword argument that indicates timing of elements of the calculation will be printed.

Further details of the :ref:`results <results>` are available.
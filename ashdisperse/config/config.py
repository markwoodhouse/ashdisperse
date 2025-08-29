import os
import warnings

import numba as nb
import numpy as np

_chunk_size = 8

def get_threading_layer():

    return nb.config.THREADING_LAYER


def set_threading_layer(thread_layer="tbb"):

    thread_layers = ["tbb", "omp", "workqueue"]

    if thread_layer not in thread_layers:
        raise ValueError("Invalid thread layer. Expected one of: %s" % thread_layers)

    nb.config.THREADING_LAYER = thread_layer
    return


def get_max_threads() -> int:
    cpu_count = os.cpu_count()
    if cpu_count:
        return cpu_count
    else:
        return 1


def get_num_threads() -> int:
    return nb.get_num_threads()


def set_num_threads(n):
    max_threads = get_max_threads()
    if n > max_threads:
        warnings.warn(
            "Request more threads than available. Setting to maximum recommended.",
            UserWarning,
        )
        nb.set_num_threads(max_threads - 1)
    elif n == max_threads:
        warnings.warn(
            "Setting number of threads equal to the maximum number of threads incurs a performance penalty.",
            UserWarning,
        )
        nb.set_num_threads(n)
    else:
        nb.set_num_threads(n)

    try:
        np.mkl.set_num_threads_local(1)
    except:
        pass
    return


def set_default_threads(n=None):
    if n is None:
        max_threads = get_max_threads()
        set_num_threads(max_threads - 1)
    else:
        set_num_threads(n)
    return

def get_chunk_size() -> int:
    return _chunk_size

def set_chunk_size(n: int=8) -> None:
    global _chunk_size
    _chunk_size = n
    return
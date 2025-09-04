import os
import warnings

import numba as nb
import numpy as np

_chunk_size = 8

def get_threading_layer() -> str:
    """
    Get the numba threading layer

    :return: threading layer used by numba
    :rtype: str

    :example:
        >>> ashdisperse.config.get_theading_layer()
        >>> "omp"

    """

    return nb.config.THREADING_LAYER


def set_threading_layer(thread_layer: str="tbb") -> None:
    """
    Set the numba threading layer

    :param thread_layer: threading layer selected from "tbb", "omp" and "workqueue", defaults to "tbb"
    :type thread_layer: str, optional
    :raises ValueError: if thread_layer not one of "tbb", "omp" or "workqueue"

    :example:
        >>> ashdisperse.config.set_threading_layer("omp")
        >>> ashdisperse.config.get_threading_layer()
        >>> "omp"
    """

    thread_layers = ["tbb", "omp", "workqueue"]

    if thread_layer not in thread_layers:
        raise ValueError("Invalid thread layer. Expected one of: %s" % thread_layers)

    nb.config.THREADING_LAYER = thread_layer
    return


def get_max_threads() -> int:
    """
    Get the maximum number of threads available on the system

    :return: maximum number of threads
    :rtype: int

    :example:
        >>> ashdisperse.config.get_max_threads()
        >>> 8
    """

    cpu_count = os.cpu_count()
    if cpu_count:
        return cpu_count
    else:
        return 1


def get_num_threads() -> int:
    """
    Get the number of threads available for computation.

    This function retrieves the number of threads that can be used, typically for parallel processing,
    by calling the underlying `numba.get_num_threads()` function.

    :return: The number of available threads for parallel processing.
    :rtype: int

    :example:
        >>> ashdisperse.config.get_num_threads()
        >>> 4
    """
    return nb.get_num_threads()


def set_num_threads(n):
    """
    Set the number of threads to be used for computation, ensuring it does not exceed the maximum recommended threads.
    Issues warnings if the requested number of threads is greater than or equal to the maximum, and adjusts accordingly.
    
    Also attempts to set MKL threads locally to 1 for compatibility.

    :param n: The desired number of threads to use for computation.
    :type n: int

    :example:
        >>> ashdisperse.config.set_num_threads(6)
        >>> ashdisperse.config.get_num_threads()
        >>> 6

    """
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
    """
    Set the default number of threads for processing.

    :param n: Number of threads to set. If None, sets to one less than the maximum available threads.
    :type n: int or None, optional

    :example:
        >>> ashdisperse.config.set_default_threads()      # Sets threads to max_threads - 1
        >>> ashdisperse.config.get_default_threads()
        >>> 7

        >>> ashdisperse.config.set_default_threads(4)     # Sets threads to 4
        >>> ashdisperse.config.get_default_threads()
        >>> 4
    """

    if n is None:
        max_threads = get_max_threads()
        set_num_threads(max_threads - 1)
    else:
        set_num_threads(n)
    return

def get_chunk_size() -> int:
    """
    Get the current chunk size used for processing.

    This function retrieves the value of the internal variable `_chunk_size`,
    which determines the number of items processed in each chunk.

    :return: The chunk size.
    :rtype: int

    :example:
        >>> ashdisperse.config.get_chunk_size()
        >>> 8
    """
    return _chunk_size

def set_chunk_size(n: int=8) -> None:
    """
    Set the global chunk size used for processing.

    :param n: The desired chunk size. Must be a positive integer. Defaults to 8.
    :type n: int, optional

    Example:
        >>> ashdisperse.config.set_chunk_size(16)  # Sets the chunk size to 16
        >>> ashdisperse.config.get_chunk_size()
        >>> 16
    """
    global _chunk_size
    _chunk_size = n
    return
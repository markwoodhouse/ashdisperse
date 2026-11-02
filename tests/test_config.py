import os

import numba as nb

from ashdisperse.config import (get_available_cpus, get_num_threads,
                                get_physical_cores,
                                set_default_threads, set_num_threads)


def test_physical_cores_bounds():
    n = get_physical_cores()
    assert 1 <= n <= (os.cpu_count() or 1)


def test_available_cpus_bounds():
    assert get_physical_cores() <= get_available_cpus() <= (os.cpu_count() or 1)


def test_default_threads_uses_all_logical_cpus():
    before = get_num_threads()
    try:
        set_default_threads()
        assert get_num_threads() == min(get_available_cpus(), nb.config.NUMBA_NUM_THREADS)
        set_default_threads(1)
        assert get_num_threads() == 1
    finally:
        set_num_threads(before)

import os

import numba as nb

from ashdisperse.config import (get_num_threads, get_physical_cores,
                                set_default_threads, set_num_threads)


def test_physical_cores_bounds():
    n = get_physical_cores()
    assert 1 <= n <= (os.cpu_count() or 1)


def test_default_threads_is_physical_cores():
    before = get_num_threads()
    try:
        set_default_threads()
        assert get_num_threads() == min(get_physical_cores(), nb.config.NUMBA_NUM_THREADS)
        set_default_threads(1)
        assert get_num_threads() == 1
    finally:
        set_num_threads(before)

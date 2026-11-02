"""Direct LAPACK calls for the per-mode collocation solves.

``np.linalg.solve`` in Numba copies the matrix into Fortran order, copies
the right-hand side, checks both for non-finite values and allocates its
pivot and info arrays on every call.  The solver performs millions of
small dense solves, so these overheads cost 8-20% of the solve time.

Here the C-ordered collocation matrix is passed straight to LAPACK, which
reads it as its transpose, and ``zgetrs`` is asked to solve with the
transpose of that, i.e. with the original matrix.  Right-hand sides are
stored one per row, (nrhs, N) in C order, which is LAPACK's (N, nrhs)
Fortran layout, and are overwritten by the solutions.

The LAPACK routines come from SciPy's Cython LAPACK exports and are
registered with LLVM under fixed names, so functions that call them can
still be cached by Numba.
"""
import llvmlite.binding as _llvm
import numpy as np
from numba import njit, types
from numba.extending import get_cython_function_address

for _name in ("zgetrf", "zgetrs"):
    _llvm.add_symbol(
        "ashdisperse_" + _name,
        get_cython_function_address("scipy.linalg.cython_lapack", _name),
    )

_vp = types.voidptr
_zgetrf = types.ExternalFunction(
    "ashdisperse_zgetrf", types.void(_vp, _vp, _vp, _vp, _vp, _vp)
)
_zgetrs = types.ExternalFunction(
    "ashdisperse_zgetrs", types.void(_vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp)
)


@njit(cache=True)
def solve_inplace(L, B):
    """Solve L X^T = B^T in place.

    :param L: (N, N) C-contiguous complex128 matrix; overwritten by its LU factors.
    :param B: (nrhs, N) C-contiguous complex128 right-hand sides, one per row;
              overwritten by the solutions.
    :raises ValueError: if L is singular.
    """
    N = L.shape[0]
    ints = np.empty(3, dtype=np.int32)  # n, nrhs, info
    ints[0] = N
    ints[1] = B.shape[0]
    ints[2] = 0
    trans = np.empty(1, dtype=np.uint8)
    trans[0] = 84  # 'T'
    ipiv = np.empty(N, dtype=np.int32)
    n_p = ints[0:].ctypes
    _zgetrf(n_p, n_p, L.ctypes, n_p, ipiv.ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        raise ValueError("Singular collocation matrix")
    _zgetrs(trans.ctypes, n_p, ints[1:].ctypes, L.ctypes, n_p,
            ipiv.ctypes, B.ctypes, n_p, ints[2:].ctypes)

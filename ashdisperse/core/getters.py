# from numba.pycc import CC
import numpy as np
from numba import float64, int64, jit

from ..containers.velocities import VelocityContainer_type

# cc = CC('getters')
# cc.verbose = True


# @cc.export("lower_z", float64[::1](VelocityContainer_type, int64))
@jit(
    float64[::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def lower_z(velocities, k):
    """ """
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In lower_z, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k]
    z = np.empty((N), dtype=np.float64)
    z[:N] = velocities.z[:N, 0, k]
    return z


# @cc.export('upper_z', float64[::1](VelocityContainer_type, int64))
@jit(
    float64[::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def upper_z(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In upper_z, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k] - 1
    z = np.empty((N), dtype=np.float64)
    z[:N] = velocities.z[:N, 1, k]
    return z


# @cc.export('lower_U', float64[::1](VelocityContainer_type, int64))
@jit(
    float64[::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def lower_U(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In lower_U, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k]
    u = np.empty((N), dtype=np.float64)
    u[:N] = velocities.U[:N, 0, k]
    return u


# @cc.export('upper_U', float64[::1](VelocityContainer_type, int64))
@jit(
    float64[::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def upper_U(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In upper_U, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k] - 1
    u = np.empty((N), dtype=np.float64)
    u[:N] = velocities.U[:N, 1, k]
    return u


# @cc.export('lower_V', float64[::1](VelocityContainer_type, int64))
@jit(
    float64[::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def lower_V(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In lower_V, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k]
    v = np.empty((N), dtype=np.float64)
    v[:N] = velocities.V[:N, 0, k]
    return v


# @cc.export('upper_V', float64[::1](VelocityContainer_type, int64))
@jit(
    float64[::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def upper_V(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In upper_V, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k] - 1
    v = np.empty((N), dtype=np.float64)
    v[:N] = velocities.V[:N, 1, k]
    return v


# @cc.export('lower_Ws', float64[:, ::1](VelocityContainer_type, int64))
@jit(
    float64[:, ::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def lower_Ws(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In lower_Ws, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k]
    ws = np.empty((N, velocities.Ws.shape[3]), dtype=np.float64)
    ws[:N, :] = velocities.Ws[:N, 0, k, :]
    return ws


# @cc.export('upper_Ws', float64[:, ::1](VelocityContainer_type, int64))
@jit(
    float64[:, ::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def upper_Ws(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In upper_Ws, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k] - 1
    ws = np.empty((N, velocities.Ws.shape[3]), dtype=np.float64)
    ws[:N, :] = velocities.Ws[:N, 1, k, :]
    return ws


# @cc.export('lower_dWsdz', float64[:, ::1](VelocityContainer_type, int64))
@jit(
    float64[:, ::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def lower_dWsdz(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In lower_dWsdz, k must be in the range "
            + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k]
    dws = np.empty((N, velocities.dWs.shape[3]), dtype=np.float64)
    dws[:N, :] = velocities.dWs[:N, 0, k, :]
    return dws


# @cc.export('upper_dWsdz', float64[:, ::1](VelocityContainer_type, int64))
@jit(
    float64[:, ::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def upper_dWsdz(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In upper_dWsdz, k must be in the range "
            + "[0, parameters.solver.chebIts]."
        )
    N = velocities.N[k] - 1
    dws = np.empty((N, velocities.dWs.shape[3]), dtype=np.float64)
    dws[:N, :] = velocities.dWs[:N, 1, k, :]
    return dws


# @cc.export('Ws_surface', float64[::1](VelocityContainer_type, int64))
@jit(
    float64[::1](VelocityContainer_type, int64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def Ws_surface(velocities, k):
    if k < 0 or k >= velocities.N.size:
        raise ValueError(
            "In lower_Ws, k must be in the range " + "[0, parameters.solver.chebIts]."
        )
    Nbins = velocities.Ws.shape[3]
    ws = np.empty((Nbins), dtype=np.float64)
    ws[:Nbins] = velocities.Ws[0, 0, k, :]
    return ws


# @cc.export('Source_z_dimless', float64[::1](float64[::1], float64))
@jit(float64[::1](float64[::1], int64, float64, float64, float64, float64), nopython=True, cache=True, fastmath=True)
def Source_z_dimless(z, profile, Suzuki_k, lower, upper, H):

    xi = (H*z-lower)/(upper-lower)

    if profile==0: # Suzuki profile
        k = Suzuki_k

        fz = k * k * (1 - xi) * np.exp(-k * (1 - xi)) / (1 - (1 + k) * np.exp(-k))
    else: # Uniform profile
        fz = np.ones_like(z)/(upper - lower)
    fz[xi>1] = 0
    fz[xi<0] = 0

    return fz


@jit(float64(float64, int64, float64, float64, float64, float64), nopython=True, cache=True, fastmath=True)
def Source_z_dimless_value(z, profile, Suzuki_k, lower, upper, H):

    xi = (H*z-lower)/(upper-lower)

    if xi > 1:
        return 0.0
    elif xi < 0:
        return 0.0

    if profile==0: # Suzuki profile
        k = Suzuki_k
        fz = k * k * (1 - xi) * np.exp(-k * (1 - xi)) / (1 - (1 + k) * np.exp(-k))
    else: # Uniform profile
        fz = 1.0/(upper - lower)

    return fz


# @cc.export('Source_z', float64[::1](float64[::1], float64, float64))
@jit(
    float64[::1](float64[::1], int64, float64, float64, float64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def Source_z(z, profile, Suzuki_k, lower, upper):

    xi = (z-lower) / (upper-lower)

    if profile==0: # Suzuki profile
        k = Suzuki_k

        fz = k * k * (1 - xi) * np.exp(-k * (1 - xi)) / (1 - (1 + k) * np.exp(-k)) / (upper-lower)
    else: # Uniform profile
        fz = 1.0/(upper - lower) * np.ones_like(z)
    
    fz[z > upper] = 0
    fz[z < lower] = 0

    return fz


@jit(
    float64(float64, int64, float64, float64, float64),
    nopython=True,
    cache=True,
    fastmath=True,
)
def Source_z_value(z, profile, Suzuki_k, lower, upper):

    if z > upper:
        return 0.0
    elif z < lower:
        return 0.0
    
    xi = (z - lower) / (upper - lower)
    if profile==0: # Suzuki profile
        k = Suzuki_k
        fz = k * k * (1 - xi) * np.exp(-k * (1 - xi)) / (1 - (1 + k) * np.exp(-k)) / (upper - lower)
    else: # Uniform profile
        fz = 1.0/(upper - lower)

    return fz


# if __name__ == "__main__":
#     cc.compile()

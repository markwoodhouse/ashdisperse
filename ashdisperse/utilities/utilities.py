# from numba.pycc import CC
import numpy as np
from numba import float64, njit

# utilities_cc = CC('utilities')
# utilities_cc.verbose = True


# @utilities_cc.export('interp_ex', 'float64[:](float64[:], float64[:], float64[:])')
@njit(float64[:](float64[:], float64[:], float64[:]))
def interp_ex_array(x, xp, fp):
    i = np.zeros(np.shape(x), dtype=np.int64)
    i[x < xp[0]] = -1
    i[x > xp[-1]] = 1
    y = np.zeros_like(x)
    y[i == -1] = fp[0] + (x[i == -1] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[i == 1] = fp[-1]
    y[i == 0] = np.interp(x[i == 0], xp, fp)
    return y


@njit(float64(float64, float64[:], float64[:]))
def interp_ex_value(x, xp, fp):
    if x < xp[0]:
        y = fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    elif x > xp[-1]:
        y = fp[-1]
    else:
        y = np.interp(x, xp, fp)
    return y


# Standard atmosphere functions
# @utilities_cc.export('SA_Temperature', 'float64[::1](float64[::1], float64, float64, float64, float64, float64)')
@njit(float64[::1](float64[::1], float64, float64, float64, float64, float64))
def SA_Temperature_array(z, Ta0, mu, omega, Ht, Hs):
    Ta = np.empty_like(z, dtype=np.float64)
    Ta[z <= Ht] = Ta0 - mu * z[z <= Ht]
    Ta[(Ht < z) * (z < Hs)] = Ta0 - mu * Ht
    Ta[z >= Hs] = Ta0 - mu * Ht + omega * (z[z >= Hs] - Hs)
    return Ta


@njit(float64(float64, float64, float64, float64, float64, float64))
def SA_Temperature_value(z, Ta0, mu, omega, Ht, Hs):
    if z <= Ht:
        Ta = Ta0 - mu * z
    elif z >= Hs:
        Ta = Ta0 - mu * Ht + omega * (z - Hs)
    else:
        Ta = Ta0 - mu * Ht
    return Ta


# @utilities_cc.export('SA_Pressure', 'float64[::1](float64[::1], float64, float64, float64, float64, float64, float64, float64, float64)')
@njit(
    float64[::1](
        float64[::1],
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    )
)
def SA_Pressure_array(z, Ta0, Pa0, mu, omega, Ht, Hs, g, Ra):
    P0 = Pa0 / np.power(Ta0, g / Ra / mu)

    a = g / Ra / mu
    b = g / Ra / omega

    Pa = np.empty_like(z, dtype=np.float64)

    Pa[z <= Ht] = P0 * np.power(Ta0 - mu * z[z <= Ht], a)
    Pa[(Ht < z) * (z < Hs)] = (
        P0
        * np.power(Ta0 - mu * Ht, a)
        * np.exp(-g * (z[(Ht < z) * (z < Hs)] - Ht) / Ra / (Ta0 - mu * Ht))
    )
    Pa[z >= Hs] = (
        P0
        * np.power(Ta0 - mu * Ht, (a + b))
        * np.exp(-g * (Hs - Ht) / Ra / (Ta0 - mu * Ht))
        * np.power(Ta0 - mu * Ht + omega * (z[z >= Hs] - Hs), -b)
    )
    return Pa


@njit(
    float64(
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    )
)
def SA_Pressure_value(z, Ta0, Pa0, mu, omega, Ht, Hs, g, Ra):
    P0 = Pa0 / np.power(Ta0, g / Ra / mu)

    a = g / Ra / mu
    b = g / Ra / omega

    if z <= Ht:
        Pa = P0 * np.power(Ta0 - mu * z, a)
    elif z >= Hs:
        Pa = (
            P0
            * np.power(Ta0 - mu * Ht, (a + b))
            * np.exp(-g * (Hs - Ht) / Ra / (Ta0 - mu * Ht))
            * np.power(Ta0 - mu * Ht + omega * (z - Hs), -b)
        )
    else:
        Pa = (
            P0
            * np.power(Ta0 - mu * Ht, a)
            * np.exp(-g * (z - Ht) / Ra / (Ta0 - mu * Ht))
        )

    return Pa


# @utilities_cc.export('SA_Density', 'float64[::1](float64[::1], float64, float64, float64, float64, float64, float64, float64, float64)')
@njit(
    float64[::1](
        float64[::1],
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    )
)
def SA_Density_array(z, Ta0, Pa0, mu, omega, Ht, Hs, g, Ra):

    P0 = Pa0 / np.power(Ta0, g / Ra / mu)

    a = g / Ra / mu
    b = g / Ra / omega

    Pa = np.empty_like(z, dtype=np.float64)

    Pa[z <= Ht] = P0 * np.power(Ta0 - mu * z[z <= Ht], a)
    Pa[(Ht < z) * (z < Hs)] = (
        P0
        * np.power(Ta0 - mu * Ht, a)
        * np.exp(-g * (z[(Ht < z) * (z < Hs)] - Ht) / Ra / (Ta0 - mu * Ht))
    )
    Pa[z >= Hs] = (
        P0
        * np.power(Ta0 - mu * Ht, (a + b))
        * np.exp(-g * (Hs - Ht) / Ra / (Ta0 - mu * Ht))
        * np.power(Ta0 - mu * Ht + omega * (z[z >= Hs] - Hs), -b)
    )

    Ta = np.empty_like(z, dtype=np.float64)
    Ta[z <= Ht] = Ta0 - mu * z[z <= Ht]
    Ta[(Ht < z) * (z < Hs)] = Ta0 - mu * Ht
    Ta[z >= Hs] = Ta0 - mu * Ht + omega * (z[z >= Hs] - Hs)

    rhoa = Pa / (Ra * Ta)
    return rhoa


@njit(
    float64(
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    )
)
def SA_Density_value(z, Ta0, Pa0, mu, omega, Ht, Hs, g, Ra):

    P0 = Pa0 / np.power(Ta0, g / Ra / mu)

    a = g / Ra / mu
    b = g / Ra / omega

    Pa = SA_Pressure_value(z, Ta0, Pa0, mu, omega, Ht, Hs, g, Ra)

    Ta = SA_Temperature_value(z, Ta0, mu, omega, Ht, Hs)

    rhoa = Pa / (Ra * Ta)
    return rhoa


# if __name__ == "__main__":
# utilities_cc.compile()

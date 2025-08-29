import numpy as np
from scipy.fft import fft2


def source_xy_dimless_flat(x, y, p):
    xx, yy = np.meshgrid(x, y)
    pi = np.pi

    f = np.zeros((len(y), len(x)), dtype=np.float64)
    fxy = np.zeros((len(y), len(x), p.grains.bins), dtype=np.float64)
    fxy_f = np.zeros((len(y), len(x), p.grains.bins), dtype=np.complex128)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Lr = p.solver.domX / p.solver.domY

    rr2 = Lr * np.power(xx, 2) + np.power(yy, 2) / Lr

    for j in range(p.grains.bins):
        f[:, :] = 1 / np.pi
        sigma2 = (
            p.source.radius**2
            # / p.model.xyScale[j] ** 2
            / p.model.xScale[j] * p.model.yScale[j]
            * pi**2
            / (p.solver.Lx * p.solver.Ly)
        )
        f[rr2 > sigma2] = 0.0

        fxy[:, :, j] = f
        fxy_f[:, :, j] = fft2(fxy[:, :, j]) * dx * dy

    return fxy, fxy_f


# def source_xy_dimless(x, y, p):
#     xx, yy = np.meshgrid(x, y)
#     pi = np.pi

#     fxy = np.zeros((len(y), len(x), p.grains.bins), dtype=np.float64)
#     fxy_f = np.zeros((len(y), len(x), p.grains.bins), dtype=np.complex128)

#     dx = (x[1] - x[0])
#     dy = (y[1] - y[0])

#     Lr = p.solver.domX/p.solver.domY

#     rr2 = Lr*np.power(xx, 2) + np.power(yy, 2)/Lr

#     for j in range(p.grains.bins):

#         sigma = (p.source.radius/3./p.model.xyScale[j]
#                  * pi/np.sqrt(p.solver.domX*p.solver.domY))

#         var = np.power(sigma, 2)

#         fxy[:, :, j] = np.exp(-0.5*rr2/var)/(2.0*pi)
#         fxy_f[:, :, j] = fft2(fxy[:, :, j])*dx*dy
#     return fxy, fxy_f


def source_xy_dimless(x, y, p):
    xx, yy = np.meshgrid(x, y)
    pi = np.pi
    pi2 = pi * pi

    fxy = np.zeros((len(y), len(x), p.grains.bins), dtype=np.float64)
    fxy_f = np.zeros((len(y), len(x), p.grains.bins), dtype=np.complex128)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Lx = p.solver.domX
    Ly = p.solver.domY

    # rr2 = Lx**2 * np.power(xx, 2) + Ly**2 * np.power(yy, 2)

    R2 = p.source.radius**2

    for j in range(p.grains.bins):

        # X2 = p.model.xScale[j] * p.model.yScale[j]

        # T = 9.0 * X2 * rr2 / (2.0 * pi2 * R2)

        x_dim = Lx*p.model.xScale[j]*xx/pi
        y_dim = Ly*p.model.yScale[j]*yy/pi

        r2_dim = np.power(x_dim, 2) + np.power(y_dim, 2)
        T = 0.5*9*r2_dim/R2

        fxy[:, :, j] = np.exp(-T) * 9.0 / (2.0 * pi) # note, no R^2 in denominator due to non-dimensionalization

        # fxy[:, :, j] = 9.0 * np.exp(-T) / (2.0 * pi)
        fxy_f[:, :, j] = fft2(fxy[:, :, j]) * dx * dy

    return fxy, fxy_f

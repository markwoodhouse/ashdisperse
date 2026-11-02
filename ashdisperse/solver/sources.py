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
    """Horizontal source distribution and its Fourier coefficients.

    The source is a Gaussian of radius R (f = 9/(2 pi) exp(-9 r^2 / (2 R^2))
    in model units).  Its Fourier coefficients are evaluated exactly,

        F(kx, ky) = (-1)^(kx+ky) A exp(-kx^2 / (4 ax) - ky^2 / (4 ay)),

    with ax = 4.5 (Lx xScale / pi)^2 / R^2 (similarly ay), A = 9/(2 sqrt(ax ay)),
    and the factor (-1)^(kx+ky) from the grid starting at -pi.  This replaces
    the FFT of the sampled Gaussian, which was wrong whenever the source is
    narrower than a grid cell (the total mass then scaled with the cell area;
    fine ash domains span thousands of km, the source a few km), and which
    had x along axis 0 although the solver reads axis 0 as y.  Periodic
    images of the source are neglected (the source is far smaller than the
    domain).

    Returns:
        fxy: the source sampled on the grid, shape (Ny, Nx, bins), x along
             axis 1 (for inspection; the solver uses only fxy_f)
        fxy_f: Fourier coefficients, shape (Ny, Nx, bins), [j, i] <-> (ky_j, kx_i)
    """
    Nx = len(x)
    Ny = len(y)
    kx = np.fft.fftfreq(Nx, 1.0 / Nx)
    ky = np.fft.fftfreq(Ny, 1.0 / Ny)
    phase = np.outer((-1.0) ** np.abs(ky), (-1.0) ** np.abs(kx))
    xx, yy = np.meshgrid(x, y)  # xx varies along axis 1

    fxy = np.zeros((Ny, Nx, p.grains.bins), dtype=np.float64)
    fxy_f = np.zeros((Ny, Nx, p.grains.bins), dtype=np.complex128)

    R2 = p.source.radius**2
    for j in range(p.grains.bins):
        # model coordinate x in [-pi, pi) <-> physical Lx * xScale * x / pi
        sx = p.model.Lx[j] * p.model.xScale[j] / np.pi
        sy = p.model.Ly[j] * p.model.yScale[j] / np.pi
        ax = 4.5 * sx**2 / R2
        ay = 4.5 * sy**2 / R2
        fxy[:, :, j] = 9.0 / (2.0 * np.pi) * np.exp(-ax * xx**2 - ay * yy**2)
        amp = 9.0 / (2.0 * np.sqrt(ax * ay))
        fxy_f[:, :, j] = phase * amp * np.exp(
            -(kx[None, :] ** 2) / (4.0 * ax) - (ky[:, None] ** 2) / (4.0 * ay)
        )

    return fxy, fxy_f

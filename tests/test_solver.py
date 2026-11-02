"""Integration tests for the spectral solver on a small, reproducible case."""
import numpy as np
import pytest

import ashdisperse as ad
from ashdisperse.config import get_num_threads, set_num_threads
from ashdisperse.met.met import MetData
from ashdisperse.params.params import parameters_from_dict


def _params(Nx_log2=5, Ny_log2=5, fft_tol=1e-10, grains=None, **solver):
    d = {
        "source": dict(name="test", latitude=63.983, longitude=-19.666, utmcode=32627,
                       radius=10000.0, PlumeHeight=10000.0, MER=1e6, duration=18000.0),
        "grains": dict(bins=2, diameter=[1e-3, 1e-4], density=[1400.0, 2000.0],
                       proportion=[0.5, 0.5]),
        "emission": dict(lower=[0.0, 0.0], upper=[10000.0, 10000.0], profile=[0, 0],
                         Suzuki_k=[10.0, 10.0]),
        "solver": dict(domX=1.5, domY=1.5, minN_log2=4, maxN_log2=8, Nx_log2=Nx_log2,
                       Ny_log2=Ny_log2, epsilon=1e-8, plateau_factor=10.0, fft_tol=fft_tol),
        "physical": dict(Kappa_h=100.0, Kappa_v=10.0, g=9.81, mu=1.85e-5),
        "output": dict(start=0.0, stop=11000.0, step=1100.0),
    }
    d["solver"].update(solver)
    if grains:
        d["grains"].update(grains)
    return parameters_from_dict(d)


@pytest.fixture(scope="module")
def met():
    m = MetData()
    z = np.linspace(0.0, 40e3, 81)
    spd = 5.0 + 25.0 * np.exp(-((z - 11e3) / 5e3) ** 2)
    ang = np.deg2rad(20.0 + 40.0 * np.tanh(z / 8e3))
    m.set_zuv_data(z, spd * np.cos(ang), spd * np.sin(ang))
    return m


@pytest.fixture(autouse=True)
def restore_threads():
    n = get_num_threads()
    yield
    set_num_threads(n)


def _solve(met, threads, **kw):
    set_num_threads(threads)
    return ad.solve(_params(**kw), met)


def test_solution_is_finite_and_conserves_mass(met):
    r = _solve(met, 2)
    assert np.all(np.isfinite(r.C0_FT)) and np.all(np.isfinite(r.Cz_FT))
    # (kx, ky) = (0, 0) mode: ground concentration is real and positive
    assert np.all(r.C0_FT[0, 0, :].real > 0)
    assert np.allclose(r.C0_FT[0, 0, :].imag, 0, atol=1e-12 * np.abs(r.C0_FT[0, 0, :]).max())


def test_reproducible_across_threads_with_truncation(met):
    # a large fft_tol makes the adaptive truncation skip many modes
    kw = dict(Nx_log2=6, Ny_log2=6, fft_tol=1e-6)
    r1 = _solve(met, 1, **kw)
    assert np.any(r1.C0_FT == 0), "test needs truncation to be active"
    for threads in (2, 4):
        for _ in range(2):
            r = _solve(met, threads, **kw)
            assert np.array_equal(r.C0_FT, r1.C0_FT)
            assert np.array_equal(r.Cz_FT, r1.Cz_FT)


def test_truncation_error_bounded(met):
    kw = dict(Nx_log2=6, Ny_log2=6)
    ref = _solve(met, 2, fft_tol=0.0, **kw)
    r = _solve(met, 2, fft_tol=1e-6, **kw)
    scale = np.abs(ref.C0_FT).max()
    assert np.max(np.abs(r.C0_FT - ref.C0_FT)) < 1e-4 * scale


def test_refine(met):
    set_num_threads(2)
    coarse = ad.solve(_params(Nx_log2=5, Ny_log2=5), met)
    fresh = ad.solve(_params(Nx_log2=6, Ny_log2=6), met)

    # full=True recomputes every mode: identical to a fresh solve
    full = ad.refine(coarse, _params(Nx_log2=6, Ny_log2=6), met, full=True)
    assert np.array_equal(full.C0_FT, fresh.C0_FT)

    # default: interior modes present on the coarse grid with |C0| >= fft_tol
    # are copied, everything else is solved afresh
    r = ad.refine(coarse, _params(Nx_log2=6, Ny_log2=6), met)
    tol = coarse.params.solver.fft_tol
    old = coarse.C0_FT
    for k in range(old.shape[2]):
        for i in range(1, old.shape[1] - 1):
            for j in range(1, old.shape[0] // 2):
                if abs(old[j, i, k]) >= tol:
                    assert r.C0_FT[j, i, k] == old[j, i, k]
                    assert r.C0_FT[64 - j - 1, i, k] == old[32 - j - 1, i, k]
    # axis modes are always recomputed
    assert np.array_equal(r.C0_FT[:, 0, :], fresh.C0_FT[:, 0, :])
    assert np.array_equal(r.C0_FT[0, :, :], fresh.C0_FT[0, :, :])


def test_lower_elements_match_single_domain_reference(met):
    # 1 mm grains use three lower-domain elements by default; compare with a
    # tightly converged single-domain solve (elements disabled, N <= 512).
    set_num_threads(2)
    el = ad.solve(_params(), met)
    from ashdisperse.containers import ChebContainer, VelocityContainer
    p = el.params
    vel = VelocityContainer(p, met, ChebContainer(p).x)
    assert vel.n_el[0] == 3
    ref = ad.solve(_params(maxN_log2=9, epsilon=1e-12, element_factor=0.0), met)
    for k in range(2):
        scale = np.abs(ref.C0_FT[..., k]).max()
        assert np.max(np.abs(el.C0_FT[..., k] - ref.C0_FT[..., k])) < 1e-8 * scale
    assert np.max(np.abs(el.Cz_FT - ref.Cz_FT)) < 1e-7 * np.abs(ref.Cz_FT).max()


def test_fine_grain_elements_split_into_thirds(met):
    # 30 um grains have no thin layers; their lower domain is split into
    # thirds, and must agree with a tightly converged single-domain solve.
    set_num_threads(2)
    g = dict(diameter=[3e-5, 3e-5], density=[2400.0, 2400.0])
    el = ad.solve(_params(grains=g), met)
    from ashdisperse.containers import ChebContainer, VelocityContainer
    vel = VelocityContainer(el.params, met, ChebContainer(el.params).x)
    assert np.all(vel.n_el == 3)
    assert np.allclose(vel.bps[:, 1], 1.0 / 3.0) and np.allclose(vel.bps[:, 2], 2.0 / 3.0)
    ref = ad.solve(_params(grains=g, maxN_log2=9, epsilon=1e-12, element_factor=0.0), met)
    scale = np.abs(ref.C0_FT[..., 0]).max()
    assert np.max(np.abs(el.C0_FT[..., 0] - ref.C0_FT[..., 0])) < 1e-8 * scale
    assert np.max(np.abs(el.Cz_FT - ref.Cz_FT)) < 1e-7 * np.abs(ref.Cz_FT).max()


def _model_params(met, **kw):
    from ashdisperse.interface import set_met_parameters, set_model_parameters
    p = _params(**kw)
    p = set_met_parameters(p, met)
    p = set_model_parameters(p, met)
    return p


def test_source_transform_matches_resolved_fft_and_orientation(met):
    from scipy.fft import fft2
    from ashdisperse.solver import source_xy_dimless
    from ashdisperse.spectral import grid_freq
    p = _model_params(met, Nx_log2=9, Ny_log2=9, domY=3.0)  # domY = 2 domX
    x, _ = grid_freq(512)
    y, _ = grid_freq(512)
    fxy, fxy_f = source_xy_dimless(x, y, p)
    dx = x[1] - x[0]
    # 1 mm grains: source well resolved on 512^2, so the FFT of the samples
    # equals the exact transform
    ref = fft2(fxy[:, :, 0]) * dx * dx
    assert np.max(np.abs(ref - fxy_f[:, :, 0])) < 1e-10 * np.abs(ref).max()
    # x along axis 1: with Ly*yScale ~ 2 Lx*xScale the source spans about
    # half as many model units (cells) along y (axis 0) as along x (axis 1)
    ratio = (p.model.Ly[0] * p.model.yScale[0]) / (p.model.Lx[0] * p.model.xScale[0])
    assert ratio > 1.5
    half = fxy[:, :, 0] > 0.5 * fxy[:, :, 0].max()
    nx = half.sum(axis=1).max()  # cells along x
    ny = half.sum(axis=0).max()  # cells along y
    assert abs(nx / ny - ratio) < 0.2 * ratio


def test_deposited_mass_independent_of_grid(met):
    # 100 um grains: the source is narrower than a grid cell on 32^2.  The
    # (kx, ky) = (0, 0) mode (total mass) must not depend on the grid.
    set_num_threads(2)
    a = ad.solve(_params(Nx_log2=5, Ny_log2=5), met)
    b = ad.solve(_params(Nx_log2=7, Ny_log2=7), met)
    assert np.allclose(a.C0_FT[0, 0, :], b.C0_FT[0, 0, :], rtol=1e-12, atol=0)


def _pad_to(c, Ny, Nx):
    """Place the modes of a smaller rfft-layout array on an Ny x (Nx/2+1) grid."""
    ny, nxh = c.shape[0], c.shape[1]
    out = np.zeros((Ny, Nx // 2 + 1) + c.shape[2:], dtype=c.dtype)
    out[: ny // 2, :nxh] = c[: ny // 2]
    out[Ny - ny // 2:, :nxh] = c[ny // 2:]
    return out


def test_per_grain_resolution_equals_own_grid(met):
    set_num_threads(2)
    mixed = ad.solve(_params(Nx_log2=6, Ny_log2=6, grain_N_log2=[5, 6]), met)
    coarse = ad.solve(_params(Nx_log2=5, Ny_log2=5), met)
    fine = ad.solve(_params(Nx_log2=6, Ny_log2=6), met)
    # grain 0 at 32^2, zero-padded to 64^2.  Column 16 (Nyquist) is solved at
    # kx = -16 on the 32 grid (grid_freq) and +16 here, so it is excluded.
    keep = np.ones(33, dtype=bool)
    keep[16] = False
    assert np.array_equal(mixed.C0_FT[..., 0][:, keep], _pad_to(coarse.C0_FT[..., 0], 64, 64)[:, keep])
    assert np.array_equal(mixed.Cz_FT[..., 0][:, keep], _pad_to(coarse.Cz_FT[..., 0], 64, 64)[:, keep])
    assert np.all(mixed.C0_FT[:, 17:, 0] == 0)
    # grain 1 at the global 64^2
    assert np.array_equal(mixed.C0_FT[..., 1], fine.C0_FT[..., 1])
    assert np.array_equal(mixed.Cz_FT[..., 1], fine.Cz_FT[..., 1])
    # grains combine as before
    _, _, tot_mixed = mixed.get_total_ashload(resolution=5000.0)
    _, _, tot_fine = fine.get_total_ashload(resolution=5000.0)
    assert tot_mixed.shape == tot_fine.shape
    assert np.ma.max(np.abs(tot_mixed - tot_fine)) < 0.05 * np.ma.max(tot_fine)


def test_grain_N_log2_validation(met):
    with pytest.raises(ValueError):
        ad.solve(_params(Nx_log2=5, Ny_log2=5, grain_N_log2=[5, 6]), met)   # exceeds global
    with pytest.raises(ValueError):
        ad.solve(_params(Nx_log2=6, Ny_log2=6, grain_N_log2=[5]), met)      # one per grain


def test_grain_N_log2_parameter_file_roundtrip(tmp_path):
    from ashdisperse.params import load_parameters, save_parameters
    p = _params(Nx_log2=6, Ny_log2=6, grain_N_log2=[5, 6])
    f = tmp_path / "p.toml"
    save_parameters(p, file=str(f))
    q = load_parameters(str(f))
    assert list(q.solver.grain_N_log2) == [5, 6]

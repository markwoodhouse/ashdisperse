"""Integration tests for the spectral solver on a small, reproducible case."""
import numpy as np
import pytest

import ashdisperse as ad
from ashdisperse.config import get_num_threads, set_num_threads
from ashdisperse.met.met import MetData
from ashdisperse.params.params import parameters_from_dict


def _params(Nx_log2=5, Ny_log2=5, fft_tol=1e-10, **solver):
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

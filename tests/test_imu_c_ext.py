"""Tests for the C+OpenMP IMU integration extension.

Verifies that each C batch integrator (Euler, Midpoint, RK4, Strapdown)
produces outputs numerically identical to the corresponding pure-Python
class on deterministic input — same algorithm, same floating-point
arithmetic — and that the extension loads cleanly even when the compiled
library is absent (all public functions raise RuntimeError gracefully).

Tests are skipped automatically when the C extension cannot be compiled
(CI without a C toolchain, Windows MSVC not installed, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.lib.algorithm.imu_c_integrators import (
    ensure_built,
    integrate_euler,
    integrate_midpoint,
    integrate_rk4,
    integrate_strapdown,
    is_available,
    monte_carlo_midpoint,
)

# Attempt one build at import time so skips are decided early.
ensure_built()

_c_available = pytest.mark.skipif(
    not is_available(), reason="C extension not built (no compiler or OpenMP)"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _straight_inputs(n: int, dt: float, v: float = 1.0):
    """Inputs for a robot moving straight at constant velocity v m/s.

    The ground-truth IMU readings are:
        omega = 0 (no rotation)
        ax    = 0 (constant velocity → zero acceleration after first step)
        ay    = 0

    For a perfectly steady state (skip the acceleration ramp) we just
    feed zeros: the integrator starts at rest and drifts by the number
    of input steps it receives.  We test that C and Python agree, not
    that the answer is v*t.
    """
    return np.zeros(n), np.zeros(n), np.zeros(n)


def _circle_inputs(n: int, dt: float, omega_val: float = 0.3, ax_val: float = 0.0):
    """Constant angular rate + zero body acceleration → circular arc."""
    omega = np.full(n, omega_val)
    ax = np.full(n, ax_val)
    ay = np.zeros(n)
    return omega, ax, ay


def _py_integrate(algo_cls, dt, omega, ax, ay):
    """Run a Python integrator step-by-step; return (px, py, theta) arrays."""
    from irsim.lib.algorithm.imu_pose_estimator import (
        EulerIntegrator,
        MidpointIntegrator,
        RK4Integrator,
        StrapdownIntegrator,
    )

    cls_map = {
        "euler": EulerIntegrator,
        "midpoint": MidpointIntegrator,
        "rk4": RK4Integrator,
        "strapdown": StrapdownIntegrator,
    }
    cls = cls_map[algo_cls]
    est = cls([0.0, 0.0, 0.0], dt=dt)
    pxs, pys, ths = [], [], []
    for i in range(len(omega)):
        accel = np.array([ax[i], ay[i]])
        est.update(float(omega[i]), accel)
        pxs.append(est.pos[0])
        pys.append(est.pos[1])
        ths.append(est.theta)
    return np.array(pxs), np.array(pys), np.array(ths)


# ── 1. Import is always safe ───────────────────────────────────────────────────


def test_import_does_not_raise():
    """imu_c_integrators imports without error regardless of build status."""
    import irsim.lib.algorithm.imu_c_integrators as m

    assert hasattr(m, "is_available")
    assert hasattr(m, "integrate_midpoint")


def test_unavailable_raises_runtime_error(monkeypatch):
    """When the library is not loaded, each integrate_* raises RuntimeError."""
    import irsim.lib.algorithm.imu_c_integrators as m

    monkeypatch.setattr(m, "_AVAILABLE", False)
    monkeypatch.setattr(m, "_lib", None)

    with pytest.raises(RuntimeError, match="not available"):
        integrate_euler(0.001, np.zeros(10), np.zeros(10), np.zeros(10))


# ── 2. Correctness vs Python reference ───────────────────────────────────────


@_c_available
@pytest.mark.parametrize("n", [1, 100, 5000])
def test_euler_matches_python(n):
    """C bench_euler output matches EulerIntegrator within 1e-12."""
    dt = 0.001
    omega, ax, ay = _circle_inputs(n, dt)
    c_px, c_py, c_th = integrate_euler(dt, omega, ax, ay)
    py_px, py_py, py_th = _py_integrate("euler", dt, omega, ax, ay)
    np.testing.assert_allclose(c_px, py_px, atol=1e-12, err_msg="Euler px mismatch")
    np.testing.assert_allclose(c_py, py_py, atol=1e-12, err_msg="Euler py mismatch")
    np.testing.assert_allclose(c_th, py_th, atol=1e-12, err_msg="Euler theta mismatch")


@_c_available
@pytest.mark.parametrize("n", [1, 100, 5000])
def test_midpoint_matches_python(n):
    """C bench_midpoint output matches MidpointIntegrator within 1e-12."""
    dt = 0.001
    omega, ax, ay = _circle_inputs(n, dt)
    c_px, c_py, c_th = integrate_midpoint(dt, omega, ax, ay)
    py_px, py_py, py_th = _py_integrate("midpoint", dt, omega, ax, ay)
    np.testing.assert_allclose(c_px, py_px, atol=1e-12)
    np.testing.assert_allclose(c_py, py_py, atol=1e-12)
    np.testing.assert_allclose(c_th, py_th, atol=1e-12)


@_c_available
@pytest.mark.parametrize("n", [1, 100, 5000])
def test_rk4_matches_python(n):
    """C bench_rk4 output matches RK4Integrator within 1e-12."""
    dt = 0.001
    omega, ax, ay = _circle_inputs(n, dt)
    c_px, c_py, c_th = integrate_rk4(dt, omega, ax, ay)
    py_px, py_py, py_th = _py_integrate("rk4", dt, omega, ax, ay)
    np.testing.assert_allclose(c_px, py_px, atol=1e-12)
    np.testing.assert_allclose(c_py, py_py, atol=1e-12)
    np.testing.assert_allclose(c_th, py_th, atol=1e-12)


@_c_available
@pytest.mark.parametrize("n", [1, 100, 5000])
def test_strapdown_matches_python(n):
    """C bench_strap output matches StrapdownIntegrator within 1e-12."""
    dt = 0.001
    omega, ax, ay = _circle_inputs(n, dt)
    c_px, c_py, c_th = integrate_strapdown(dt, omega, ax, ay)
    py_px, py_py, py_th = _py_integrate("strapdown", dt, omega, ax, ay)
    np.testing.assert_allclose(c_px, py_px, atol=1e-12)
    np.testing.assert_allclose(c_py, py_py, atol=1e-12)
    np.testing.assert_allclose(c_th, py_th, atol=1e-12)


# ── 3. Zero-input → zero-output ───────────────────────────────────────────────


@_c_available
@pytest.mark.parametrize(
    "fn", [integrate_euler, integrate_midpoint, integrate_rk4, integrate_strapdown]
)
def test_zero_input_stays_at_origin(fn):
    """Zero omega, zero acceleration: robot stays at the origin."""
    n = 200
    dt = 0.01
    omega = np.zeros(n)
    ax = np.zeros(n)
    ay = np.zeros(n)
    px, py, th = fn(dt, omega, ax, ay)
    np.testing.assert_allclose(px, 0.0, atol=1e-15)
    np.testing.assert_allclose(py, 0.0, atol=1e-15)
    np.testing.assert_allclose(th, 0.0, atol=1e-15)


# ── 4. Output shapes ─────────────────────────────────────────────────────────


@_c_available
def test_output_shapes():
    """Each integrate_* returns three arrays of the correct length."""
    n = 137
    dt = 0.005
    omega = np.random.default_rng(42).standard_normal(n) * 0.1
    ax = np.random.default_rng(1).standard_normal(n) * 0.5
    ay = np.zeros(n)

    for fn in (integrate_euler, integrate_midpoint, integrate_rk4, integrate_strapdown):
        px, py, th = fn(dt, omega, ax, ay)
        assert px.shape == (n,), f"{fn.__name__} px shape"
        assert py.shape == (n,), f"{fn.__name__} py shape"
        assert th.shape == (n,), f"{fn.__name__} th shape"
        assert px.dtype == np.float64
        assert py.dtype == np.float64
        assert th.dtype == np.float64


# ── 5. Input validation ───────────────────────────────────────────────────────


@_c_available
def test_mismatched_input_lengths_raises():
    """Mismatched array lengths raise ValueError."""
    dt = 0.001
    with pytest.raises(ValueError, match="equal length"):
        integrate_midpoint(dt, np.zeros(10), np.zeros(9), np.zeros(10))


@_c_available
def test_single_step():
    """Single step (n=1) is valid and returns an array of length 1."""
    dt = 0.001
    omega = np.array([0.5])
    ax = np.array([1.0])
    ay = np.array([0.0])
    for fn in (integrate_euler, integrate_midpoint, integrate_rk4, integrate_strapdown):
        px, _py, _th = fn(dt, omega, ax, ay)
        assert len(px) == 1


# ── 6. Monte Carlo midpoint ───────────────────────────────────────────────────


@_c_available
def test_mc_output_shape():
    """monte_carlo_midpoint returns an array of the requested trial count."""
    n_trials = 20
    n_steps = 500
    dt = 0.001
    omega = np.zeros(n_steps)
    ax = np.ones(n_steps) * 0.1
    ay = np.zeros(n_steps)
    rmse = monte_carlo_midpoint(n_trials, dt, omega, ax, ay)
    assert rmse.shape == (n_trials,)
    assert rmse.dtype == np.float64


@_c_available
def test_mc_rmse_positive():
    """RMSE values from MC are non-negative (trivially verified by math)."""
    n_trials = 10
    n_steps = 200
    dt = 0.001
    omega = np.full(n_steps, 0.3)
    ax = np.full(n_steps, 0.5)
    ay = np.zeros(n_steps)
    rmse = monte_carlo_midpoint(n_trials, dt, omega, ax, ay)
    assert np.all(rmse >= 0.0)


@_c_available
def test_mc_consistent_with_single_trial():
    """MC with 1 trial and deterministic input should equal single integrate_midpoint RMSE."""
    n_steps = 300
    dt = 0.001
    omega = np.full(n_steps, 0.2)
    ax = np.full(n_steps, 0.3)
    ay = np.zeros(n_steps)

    px, py, _ = integrate_midpoint(dt, omega, ax, ay)
    expected_rmse = float(np.sqrt(np.mean(px**2 + py**2)))

    rmse = monte_carlo_midpoint(1, dt, omega, ax, ay)
    np.testing.assert_allclose(rmse[0], expected_rmse, rtol=1e-10)


# ── 7. Non-trivial trajectory — straight-line heading stays zero ──────────────


@_c_available
def test_straight_line_zero_heading():
    """Straight-line motion with zero omega: heading stays zero for all integrators."""
    n = 1000
    dt = 0.001
    omega = np.zeros(n)
    ax = np.full(n, 1.0)  # constant body-x acceleration
    ay = np.zeros(n)

    for fn in (integrate_euler, integrate_midpoint, integrate_rk4, integrate_strapdown):
        _, _, th = fn(dt, omega, ax, ay)
        np.testing.assert_allclose(
            th, 0.0, atol=1e-14, err_msg=f"{fn.__name__} heading drift"
        )

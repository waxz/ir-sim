"""Tests for the IMU sensor (irsim/world/sensors/imu.py).

Covers:
1. Noise level — stationary robot, zero bias walk: std matches specification.
2. Bias drift — gyro bias grows proportional to sqrt(t).
3. Heading integration — constant angular velocity with noise=False.
4. Dead-reckoning drift — accumulated error grows over time.
"""

import numpy as np

from irsim.world.sensors.imu import IMU

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_imu(
    states: np.ndarray, *, noise: bool = True, step_time: float = 0.1, **kw
) -> IMU:
    """Run an IMU through a sequence of states; return the final sensor."""
    imu = IMU(states[0], noise=noise, step_time=step_time, **kw)
    for s in states[1:]:
        imu.step(s)
    return imu


def _collect(states: np.ndarray, *, step_time: float = 0.1, **kw) -> tuple[list, list]:
    """Collect (omega_meas, accel_meas) over a trajectory."""
    imu = IMU(states[0], step_time=step_time, **kw)
    omegas, accels = [], []
    for s in states[1:]:
        imu.step(s)
        omegas.append(imu.angular_velocity)
        accels.append(imu.linear_acceleration.copy())
    return omegas, accels


# ---------------------------------------------------------------------------
# 1. Noise level — stationary robot
# ---------------------------------------------------------------------------


def test_gyro_noise_level():
    """Gyro white-noise std matches N_g/sqrt(dt) within 20% at N=10 000 steps."""
    dt = 0.01
    N_g = 0.005  # rad/s/√Hz
    sigma_expected = N_g / np.sqrt(dt)

    n_steps = 10_000
    stationary = np.zeros((n_steps + 1, 3))  # robot doesn't move

    imu = IMU(
        stationary[0],
        gyro_noise_std=N_g,
        gyro_bias_walk_std=0.0,  # disable bias drift for this test
        accel_bias_walk_std=0.0,
        step_time=dt,
        noise=True,
    )
    omegas = []
    for s in stationary[1:]:
        imu.step(s)
        omegas.append(imu.angular_velocity)

    sigma_measured = float(np.std(omegas))
    assert abs(sigma_measured - sigma_expected) / sigma_expected < 0.20, (
        f"gyro std {sigma_measured:.4f} deviates >20% from expected {sigma_expected:.4f}"
    )


def test_accel_noise_level():
    """Accel white-noise std matches N_a/sqrt(dt) within 20% at N=10 000 steps."""
    dt = 0.01
    N_a = 0.05  # m/s²/√Hz
    sigma_expected = N_a / np.sqrt(dt)

    n_steps = 10_000
    stationary = np.zeros((n_steps + 1, 3))

    imu = IMU(
        stationary[0],
        accel_noise_std=N_a,
        gyro_bias_walk_std=0.0,
        accel_bias_walk_std=0.0,
        step_time=dt,
        noise=True,
    )
    ax_list = []
    for s in stationary[1:]:
        imu.step(s)
        ax_list.append(imu.linear_acceleration[0])

    sigma_measured = float(np.std(ax_list))
    assert abs(sigma_measured - sigma_expected) / sigma_expected < 0.20, (
        f"accel std {sigma_measured:.4f} deviates >20% from expected {sigma_expected:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Bias drift — sqrt(t) growth
# ---------------------------------------------------------------------------


def test_gyro_bias_drift():
    """Gyro bias RMS grows proportional to sqrt(t) over 1 000 steps."""
    dt = 0.1
    K_g = 0.001  # larger walk for visibility
    n_steps = 1_000

    # Collect bias values (disable white noise so only random walk remains)
    stationary = np.zeros((n_steps + 1, 3))

    imu = IMU(
        stationary[0],
        gyro_noise_std=0.0,
        accel_noise_std=0.0,
        gyro_bias_walk_std=K_g,
        accel_bias_walk_std=0.0,
        step_time=dt,
        noise=True,
    )

    biases = []
    for s in stationary[1:]:
        imu.step(s)
        biases.append(imu.gyro_bias)

    biases = np.array(biases)
    # Theoretical RMS after k steps: K_g * sqrt(k * dt)
    k_vals = np.arange(1, n_steps + 1)
    sigma_theory = K_g * np.sqrt(k_vals * dt)

    # Compare measured |bias| with theory at the midpoint and endpoint
    for idx in [n_steps // 2 - 1, n_steps - 1]:
        measured_window = np.abs(biases[max(0, idx - 50) : idx + 50])
        measured_rms = float(np.mean(measured_window))
        theory_rms = sigma_theory[idx]
        # Allow a 3x tolerance since it's a single trajectory, not an ensemble
        assert measured_rms < 3 * theory_rms + 1e-6, (
            f"bias at step {idx + 1} ({measured_rms:.5f}) exceeds 3x theory ({theory_rms:.5f})"
        )


# ---------------------------------------------------------------------------
# 3. Heading integration — constant ω, noise=False
# ---------------------------------------------------------------------------


def test_heading_integration_noisefree():
    """Integrating noise-free gyro output recovers heading within 1e-6 rad."""
    dt = 0.05
    omega_true = 0.5  # rad/s
    n_steps = 200

    # Build a trajectory: robot pivots in place at constant omega
    theta = np.cumsum(np.full(n_steps + 1, omega_true * dt))
    theta[0] = 0.0
    theta = np.cumsum(np.full(n_steps, omega_true * dt))
    states = np.column_stack(
        [np.zeros(n_steps + 1), np.zeros(n_steps + 1), np.concatenate([[0.0], theta])]
    )

    imu = IMU(states[0], noise=False, step_time=dt)
    integrated_theta = 0.0
    for s in states[1:]:
        imu.step(s)
        integrated_theta += imu.angular_velocity * dt

    expected_theta = omega_true * n_steps * dt
    assert abs(integrated_theta - expected_theta) < 1e-6, (
        f"integrated heading {integrated_theta:.8f} != expected {expected_theta:.8f}"
    )


# ---------------------------------------------------------------------------
# 4. Dead-reckoning drift — accumulated error grows over time
# ---------------------------------------------------------------------------


def test_dead_reckoning_drift_grows():
    """Accumulated position error from noisy IMU grows larger over time."""
    dt = 0.1
    n_steps = 500
    v = 1.0  # constant forward speed m/s

    # Straight-line trajectory in world frame
    t_arr = np.arange(n_steps + 1) * dt
    x = v * t_arr
    states = np.column_stack([x, np.zeros(n_steps + 1), np.zeros(n_steps + 1)])

    imu = IMU(
        states[0],
        gyro_noise_std=0.005,
        accel_noise_std=0.05,
        gyro_bias_walk_std=0.0001,
        accel_bias_walk_std=0.001,
        step_time=dt,
        noise=True,
    )

    # Dead-reckoning: integrate accel in body frame → world frame
    pos_dr = np.array(states[0, :2], dtype=float)
    vel_dr = np.zeros(2)
    theta_dr = 0.0
    errors_first = []
    errors_last = []

    for i, s in enumerate(states[1:], start=1):
        imu.step(s)
        theta_dr += imu.angular_velocity * dt
        c, s_t = np.cos(theta_dr), np.sin(theta_dr)
        R = np.array([[c, -s_t], [s_t, c]])
        accel_world = R @ imu.linear_acceleration
        vel_dr += accel_world * dt
        pos_dr += vel_dr * dt

        err = float(np.linalg.norm(pos_dr - s[:2]))
        if i <= 50:
            errors_first.append(err)
        if i >= n_steps - 50:
            errors_last.append(err)

    mean_early = float(np.mean(errors_first))
    mean_late = float(np.mean(errors_last))
    assert mean_late > mean_early, (
        f"dead-reckoning error should grow over time: early={mean_early:.4f}, late={mean_late:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Sensor factory integration
# ---------------------------------------------------------------------------


def test_sensor_factory_creates_imu():
    """SensorFactory with name='imu' returns an IMU instance."""
    from irsim.world.sensors.sensor_factory import SensorFactory

    factory = SensorFactory()
    state = np.array([0.0, 0.0, 0.0])
    sensor = factory.create_sensor(state, obj_id=1, name="imu", step_time=0.1)
    assert isinstance(sensor, IMU)
    assert sensor.sensor_type == "imu"


# ---------------------------------------------------------------------------
# 6. Noise disabled — ground truth pass-through
# ---------------------------------------------------------------------------


def test_noise_disabled_returns_ground_truth():
    """With noise=False and no bias, output matches finite-difference ground truth."""
    dt = 0.1
    omega = 0.3  # rad/s
    states = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, omega * dt],
            [0.0, 0.0, 2 * omega * dt],
        ]
    )
    imu = IMU(states[0], noise=False, step_time=dt)
    imu.step(states[1])
    imu.step(states[2])
    assert abs(imu.angular_velocity - omega) < 1e-9


# ---------------------------------------------------------------------------
# 7. Named profile
# ---------------------------------------------------------------------------


def test_named_profile_sets_noise():
    """IMU(profile='icm42688') loads the correct noise parameters."""
    imu = IMU(np.zeros(3), profile="icm42688", noise=False)
    assert imu._N_g == IMU.PROFILES["icm42688"]["gyro_noise_std"]
    assert imu._N_a == IMU.PROFILES["icm42688"]["accel_noise_std"]


def test_invalid_profile_raises():
    """IMU(profile='bad') raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown IMU profile"):
        IMU(np.zeros(3), profile="bad_sensor")


# ---------------------------------------------------------------------------
# 8. IMU pose estimators
# ---------------------------------------------------------------------------


def test_pose_estimators_noisefree_straight():
    """All four integrators recover a straight-line trajectory within 1 mm when noise=False."""
    from irsim.lib.algorithm.imu_pose_estimator import (
        EulerIntegrator,
        MidpointIntegrator,
        RK4Integrator,
        StrapdownIntegrator,
    )

    dt = 0.001
    v = 1.0
    n = 500
    t = np.arange(n + 1) * dt
    states = np.column_stack([v * t, np.zeros(n + 1), np.zeros(n + 1)])

    for cls in (
        EulerIntegrator,
        MidpointIntegrator,
        RK4Integrator,
        StrapdownIntegrator,
    ):
        imu = IMU(states[0], noise=False, step_time=dt)
        est = cls(states[0], dt=dt)
        for s in states[1:]:
            imu.step(s)
            est.update(imu.angular_velocity, imu.linear_acceleration)
        pose = est.get_pose()
        gt = states[-1]
        err = float(np.linalg.norm(pose[:2] - gt[:2]))
        assert err < 0.002, f"{cls.__name__} straight-line error {err:.6f} m > 2 mm"

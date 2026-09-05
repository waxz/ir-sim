"""Simulated IMU sensor with IEEE 517-style noise model.

Implements white noise (angle/velocity random walk) and bias random walk
on both gyroscope and accelerometer axes, consistent with the Moussaid-Helbing
2009 parameterisation used elsewhere in IR-SIM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from irsim.util.random import rng

if TYPE_CHECKING:
    from irsim.world.object_base import ObjectBase


class IMU:
    """Simulated 2-D IMU sensor (gyroscope + accelerometer).

    Derives ground-truth angular velocity and linear acceleration from the
    parent object's state history using finite differences, then corrupts
    the signal with white noise and a slowly-drifting bias (random walk).

    Default parameters match a consumer-grade MEMS IMU (e.g. MPU-6050).

    Args:
        state (np.ndarray): Initial [x, y, theta] state of the parent object.
        obj_id (int): ID of the associated object.
        gyro_noise_std (float): Gyroscope white-noise density (rad/s/√Hz).
        accel_noise_std (float): Accelerometer white-noise density (m/s²/√Hz).
        gyro_bias_walk_std (float): Gyroscope bias random-walk rate (rad/s/√s).
        accel_bias_walk_std (float): Accelerometer bias random-walk rate (m/s²/√s).
        step_time (float): Simulation step time in seconds.
        noise (bool): Enable noise and bias. Set False for ground-truth output.
        **kwargs: Ignored extra keyword arguments passed by SensorFactory.

    Attr:
        sensor_type (str): ``"imu"``.
        angular_velocity (float): Latest gyroscope measurement (rad/s).
        linear_acceleration (np.ndarray): Latest accelerometer measurement,
            shape (2,), in body frame (m/s²).
        gyro_bias (float): Current gyroscope bias estimate (rad/s).
        accel_bias (np.ndarray): Current accelerometer bias, shape (2,) (m/s²).
        parent (ObjectBase | None): Owning simulation object; set externally.
    """

    # ── Named IMU profiles (noise spectral densities & bias walk rates) ──────
    #
    # All values in SI:  N_g  rad/s/√Hz,  N_a  m/s²/√Hz,
    #                    K_g  rad/s/√s,   K_a  m/s²/√s
    #
    # Sources: product datasheets; µg converted via g=9.80665 m/s²
    #
    #  mpu6050     InvenSense MPU-6050  (budget mobile-robot/drone)
    #              N_g = 0.005 °/s/√Hz  = 8.73e-5 rad/s/√Hz
    #              N_a = 400 µg/√Hz     = 3.92e-3 m/s²/√Hz
    #  bmi088      Bosch BMI-088  (Pixhawk, mid-range robots)
    #              N_g = 0.014 °/s/√Hz  = 2.44e-4 rad/s/√Hz
    #              N_a = 230 µg/√Hz     = 2.26e-3 m/s²/√Hz
    #  icm42688    TDK ICM-42688-P  (high-performance mobile robots)
    #              N_g = 0.0028 °/s/√Hz = 4.89e-5 rad/s/√Hz
    #              N_a = 70 µg/√Hz      = 6.87e-4 m/s²/√Hz
    #  adis16448   Analog Devices ADIS-16448  (navigation grade)
    #              N_g = 0.066 °/s/√Hz  = 1.15e-3 rad/s/√Hz
    #              N_a = 0.158 mg/√Hz   = 1.55e-3 m/s²/√Hz
    PROFILES: ClassVar[dict[str, dict[str, float]]] = {
        "mpu6050": {
            "gyro_noise_std": 8.73e-5,
            "accel_noise_std": 3.92e-3,
            "gyro_bias_walk_std": 1.75e-4,
            "accel_bias_walk_std": 1.96e-4,
        },
        "bmi088": {
            "gyro_noise_std": 2.44e-4,
            "accel_noise_std": 2.26e-3,
            "gyro_bias_walk_std": 3.49e-5,
            "accel_bias_walk_std": 1.96e-4,
        },
        "icm42688": {
            "gyro_noise_std": 4.89e-5,
            "accel_noise_std": 6.87e-4,
            "gyro_bias_walk_std": 8.73e-6,
            "accel_bias_walk_std": 9.81e-5,
        },
        "adis16448": {
            "gyro_noise_std": 1.15e-3,
            "accel_noise_std": 1.55e-3,
            "gyro_bias_walk_std": 2.91e-5,
            "accel_bias_walk_std": 9.81e-5,
        },
    }

    def __init__(
        self,
        state: np.ndarray | None = None,
        obj_id: int = 0,
        gyro_noise_std: float = 8.73e-5,
        accel_noise_std: float = 3.92e-3,
        gyro_bias_walk_std: float = 1.75e-4,
        accel_bias_walk_std: float = 1.96e-4,
        step_time: float = 0.001,
        noise: bool = True,
        profile: str | None = None,
        **kwargs,
    ) -> None:
        self.sensor_type = "imu"
        self.obj_id = obj_id
        self.noise = noise
        self.step_time = step_time

        # Apply named profile if given (overrides explicit params)
        if profile is not None:
            if profile not in self.PROFILES:
                raise ValueError(
                    f"Unknown IMU profile '{profile}'. Available: {list(self.PROFILES)}"
                )
            p = self.PROFILES[profile]
            gyro_noise_std = p["gyro_noise_std"]
            accel_noise_std = p["accel_noise_std"]
            gyro_bias_walk_std = p["gyro_bias_walk_std"]
            accel_bias_walk_std = p["accel_bias_walk_std"]

        # Noise spectral densities
        self._N_g = gyro_noise_std
        self._N_a = accel_noise_std
        # Bias random-walk rate
        self._K_g = gyro_bias_walk_std
        self._K_a = accel_bias_walk_std

        # Running bias state
        self.gyro_bias: float = 0.0
        self.accel_bias: np.ndarray = np.zeros(2)

        # Previous-step state for finite differences
        if state is not None:
            s = np.asarray(state).ravel()
            self._prev_pos: np.ndarray = s[:2].copy()
            self._prev_theta: float = float(s[2])
        else:
            self._prev_pos = np.zeros(2)
            self._prev_theta = 0.0
        self._prev_vel_world: np.ndarray = np.zeros(2)

        # Outputs (initialised to zero)
        self.angular_velocity: float = 0.0
        self.linear_acceleration: np.ndarray = np.zeros(2)

        # Visualisation / compatibility stubs
        self.parent: ObjectBase | None = None
        self.plot_patch_list: list = []
        self.plot_line_list: list = []
        self.plot_text_list: list = []

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def step(self, state: np.ndarray) -> None:
        """Update IMU measurements from the current parent state.

        Args:
            state (np.ndarray): Current [x, y, theta] from the parent object.
        """
        s = np.asarray(state).ravel()
        pos = s[:2]
        theta = float(s[2])
        dt = self.step_time

        # --- Ground-truth signals ---
        omega_true = (theta - self._prev_theta) / dt

        vel_world = (pos - self._prev_pos) / dt
        dv_world = vel_world - self._prev_vel_world
        accel_world = dv_world / dt

        # Rotate world-frame acceleration to body frame
        c, s_theta = np.cos(theta), np.sin(theta)
        R_inv = np.array([[c, s_theta], [-s_theta, c]])  # R(-theta)
        accel_body = R_inv @ accel_world

        # --- Advance bias random walk ---
        if self.noise:
            self.gyro_bias += self._K_g * np.sqrt(dt) * float(rng.standard_normal())
            self.accel_bias += self._K_a * np.sqrt(dt) * rng.standard_normal(2)

            # --- Add white noise ---
            sigma_g = self._N_g / np.sqrt(dt)
            sigma_a = self._N_a / np.sqrt(dt)
            omega_meas = (
                omega_true + self.gyro_bias + sigma_g * float(rng.standard_normal())
            )
            accel_meas = accel_body + self.accel_bias + sigma_a * rng.standard_normal(2)
        else:
            omega_meas = omega_true
            accel_meas = accel_body

        # Store outputs
        self.angular_velocity = omega_meas
        self.linear_acceleration = accel_meas

        # Advance history
        self._prev_pos = pos.copy()
        self._prev_theta = theta
        self._prev_vel_world = vel_world.copy()

    def get_measurement(self) -> dict:
        """Return the most recent IMU reading as a dictionary.

        Returns:
            dict: Keys ``angular_velocity`` (float, rad/s) and
            ``linear_acceleration`` (np.ndarray shape (2,), m/s²).
        """
        return {
            "angular_velocity": self.angular_velocity,
            "linear_acceleration": self.linear_acceleration.copy(),
        }

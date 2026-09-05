"""Simulated IMU sensor with IEEE 517-style noise model.

Implements white noise (angle/velocity random walk) and bias random walk
on both gyroscope and accelerometer axes, consistent with the Moussaid-Helbing
2009 parameterisation used elsewhere in IR-SIM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    def __init__(
        self,
        state: np.ndarray | None = None,
        obj_id: int = 0,
        gyro_noise_std: float = 0.005,
        accel_noise_std: float = 0.05,
        gyro_bias_walk_std: float = 0.0001,
        accel_bias_walk_std: float = 0.001,
        step_time: float = 0.1,
        noise: bool = True,
        **kwargs,
    ) -> None:
        self.sensor_type = "imu"
        self.obj_id = obj_id
        self.noise = noise
        self.step_time = step_time

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
            omega_meas = omega_true + self.gyro_bias + sigma_g * float(
                rng.standard_normal()
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

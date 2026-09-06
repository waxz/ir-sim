"""Pure inverse- and forward-kinematics functions for individual wheel layouts.

All inputs/outputs are SI (rad/s for wheel angular velocity, rad for angle,
m/s for linear speed).  These functions are stateless; call them from
WheelLayout subclasses inside irsim/lib/handler/wheel_handler.py.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Differential drive — 2 drive wheels
# ---------------------------------------------------------------------------


def diff_inv_kin(
    v: float, omega: float, wheel_radius: float, track: float
) -> tuple[float, float]:
    """Body [v, omega] → (omega_left, omega_right) in rad/s.

    Args:
        v: Forward linear speed (m/s).
        omega: Yaw rate (rad/s), positive = CCW.
        wheel_radius: Wheel radius r (m).
        track: Lateral distance between wheel centres d (m).

    Returns:
        (omega_left, omega_right) wheel angular velocities (rad/s).
    """
    half_d = track / 2.0
    omega_l = (v - omega * half_d) / wheel_radius
    omega_r = (v + omega * half_d) / wheel_radius
    return omega_l, omega_r


def diff_fwd_kin(
    omega_l: float, omega_r: float, wheel_radius: float, track: float
) -> tuple[float, float]:
    """(omega_left, omega_right) → (v, omega).

    Args:
        omega_l: Left wheel angular velocity (rad/s).
        omega_r: Right wheel angular velocity (rad/s).
        wheel_radius: Wheel radius r (m).
        track: Track width d (m).

    Returns:
        (v, omega) body-frame linear speed and yaw rate.
    """
    v = wheel_radius * (omega_l + omega_r) / 2.0
    omega = wheel_radius * (omega_r - omega_l) / track
    return v, omega


# ---------------------------------------------------------------------------
# Mecanum 4-wheel (±45° passive rollers)
# ---------------------------------------------------------------------------


def mecanum_inv_kin(
    vx: float,
    vy: float,
    omega_z: float,
    wheel_radius: float,
    half_length: float,
    half_width: float,
) -> tuple[float, float, float, float]:
    """Body [vx, vy, omega_z] → (omega_FL, omega_FR, omega_RL, omega_RR).

    Wheel arrangement (top view, x = forward, y = left)::

        FL(+L,+W) ──── FR(+L,-W)
            |                |
        RL(-L,+W) ──── RR(-L,-W)

    Rollers are at +45° on FL/RR and -45° on FR/RL (standard H-drive layout).

    Args:
        vx: Forward body speed (m/s).
        vy: Lateral body speed (m/s), positive left.
        omega_z: Yaw rate (rad/s), positive CCW.
        wheel_radius: Wheel radius r (m).
        half_length: Distance from body centre to wheel axle along x (m).
        half_width: Distance from body centre to wheel axle along y (m).

    Returns:
        (omega_FL, omega_FR, omega_RL, omega_RR) in rad/s.
    """
    lw = half_length + half_width
    inv_r = 1.0 / wheel_radius
    omega_fl = inv_r * (vx - vy - lw * omega_z)
    omega_fr = inv_r * (vx + vy + lw * omega_z)
    omega_rl = inv_r * (vx + vy - lw * omega_z)
    omega_rr = inv_r * (vx - vy + lw * omega_z)
    return omega_fl, omega_fr, omega_rl, omega_rr


def mecanum_fwd_kin(
    omega_fl: float,
    omega_fr: float,
    omega_rl: float,
    omega_rr: float,
    wheel_radius: float,
    half_length: float,
    half_width: float,
) -> tuple[float, float, float]:
    """4-wheel mecanum speeds → (vx, vy, omega_z)."""
    r = wheel_radius
    lw = half_length + half_width
    vx = r * (omega_fl + omega_fr + omega_rl + omega_rr) / 4.0
    vy = r * (-omega_fl + omega_fr + omega_rl - omega_rr) / 4.0
    omega_z = r * (-omega_fl + omega_fr - omega_rl + omega_rr) / (4.0 * lw)
    return vx, vy, omega_z


# ---------------------------------------------------------------------------
# Ackermann (4-wheel, bicycle model + per-wheel correction)
# ---------------------------------------------------------------------------


def acker_steer_angles(
    psi: float, wheelbase: float, track: float
) -> tuple[float, float]:
    """Compute per-wheel Ackermann steering angles for a given bicycle angle.

    For a right turn (psi > 0): left wheel is outer (larger radius, smaller
    angle) and right wheel is inner (smaller radius, larger angle).

    Args:
        psi: Bicycle-model steering angle (rad), positive = left/CCW turn.
        wheelbase: Distance between front and rear axles L (m).
        track: Distance between left and right wheel centres T (m).

    Returns:
        (delta_left, delta_right) front-wheel steering angles (rad).
    """
    if abs(psi) < 1e-9:
        return 0.0, 0.0
    R_turn = wheelbase / np.tan(psi)
    delta_left = np.arctan2(wheelbase, R_turn - track / 2.0)
    delta_right = np.arctan2(wheelbase, R_turn + track / 2.0)
    return float(delta_left), float(delta_right)


def acker_inv_kin(
    v: float, psi: float, wheel_radius: float, wheelbase: float, track: float
) -> dict[str, float]:
    """Ackermann IK: [v, psi] → rear wheel speeds + front steer angles.

    Args:
        v: Body forward speed (m/s).
        psi: Bicycle-model front steering angle (rad).
        wheel_radius: Wheel radius r (m).
        wheelbase: Axle-to-axle distance L (m).
        track: Track width T (m).

    Returns:
        Dict with keys ``omega_rl``, ``omega_rr``, ``delta_fl``, ``delta_fr``.
    """
    if abs(psi) < 1e-9:
        omega_rw = v / wheel_radius
        return {
            "omega_rl": omega_rw,
            "omega_rr": omega_rw,
            "delta_fl": 0.0,
            "delta_fr": 0.0,
        }
    R_turn = wheelbase / np.tan(psi)
    r = wheel_radius
    omega_rl = v * (R_turn - track / 2.0) / (r * R_turn)
    omega_rr = v * (R_turn + track / 2.0) / (r * R_turn)
    delta_fl, delta_fr = acker_steer_angles(psi, wheelbase, track)
    return {
        "omega_rl": float(omega_rl),
        "omega_rr": float(omega_rr),
        "delta_fl": delta_fl,
        "delta_fr": delta_fr,
    }


# ---------------------------------------------------------------------------
# Forklift (2 front drive-wheels + 1 rear steer+drive wheel)
# ---------------------------------------------------------------------------


def forklift_inv_kin(
    v: float,
    omega: float,
    wheel_radius: float,
    half_wheelbase: float,
    track: float,
) -> dict[str, float]:
    """Forklift IK: [v, omega] → front pair speeds + rear steer+speed.

    The front two wheels share a differential drive axle at +half_wheelbase
    from the body centre; the single rear wheel is at -half_wheelbase.

    Args:
        v: Forward body speed (m/s).
        omega: Yaw rate (rad/s), positive CCW.
        wheel_radius: All-wheel radius r (m).
        half_wheelbase: Half the wheelbase (body centre to each axle) (m).
        track: Front-axle track width T (m).

    Returns:
        Dict with keys ``omega_fl``, ``omega_fr``, ``delta_rc``, ``omega_rc``.
    """
    # Front differential pair
    omega_fl, omega_fr = diff_inv_kin(v, omega, wheel_radius, track)

    # Rear centre wheel velocity in body frame:
    # Point at (x_w = -half_wheelbase, y_w = 0):
    #   v_point_x = v  (y_w = 0 → no lateral contribution from rotation)
    #   v_point_y = omega * (-half_wheelbase) = -omega * half_wheelbase
    v_rc_x = v
    v_rc_y = -omega * half_wheelbase
    speed_rc = np.hypot(v_rc_x, v_rc_y)
    if speed_rc > 1e-9:
        delta_rc = float(np.arctan2(v_rc_y, v_rc_x))
        omega_rc = speed_rc / wheel_radius
    else:
        delta_rc = 0.0
        omega_rc = 0.0

    return {
        "omega_fl": float(omega_fl),
        "omega_fr": float(omega_fr),
        "delta_rc": delta_rc,
        "omega_rc": float(omega_rc),
    }


# ---------------------------------------------------------------------------
# Swerve drive (N wheels, each independently steered and driven)
# ---------------------------------------------------------------------------


def swerve_wheel_cmd(
    vx: float,
    vy: float,
    omega_z: float,
    wheel_x: float,
    wheel_y: float,
    wheel_radius: float,
) -> tuple[float, float]:
    """IK for a single swerve module at body position (wheel_x, wheel_y).

    Args:
        vx: Body forward speed (m/s).
        vy: Body lateral speed (m/s).
        omega_z: Body yaw rate (rad/s).
        wheel_x: Wheel x-position in body frame (m).
        wheel_y: Wheel y-position in body frame (m).
        wheel_radius: Wheel radius r (m).

    Returns:
        (delta, omega_wheel) — steering angle (rad) and drive speed (rad/s).
    """
    # Velocity of the wheel contact patch in body frame
    vwx = vx - omega_z * wheel_y
    vwy = vy + omega_z * wheel_x
    speed = np.hypot(vwx, vwy)
    delta = float(np.arctan2(vwy, vwx)) if speed > 1e-9 else 0.0
    omega_w = speed / wheel_radius
    return delta, float(omega_w)


def swerve_inv_kin(
    vx: float,
    vy: float,
    omega_z: float,
    wheel_radius: float,
    wheel_positions: np.ndarray,
) -> np.ndarray:
    """IK for N swerve modules.

    Args:
        vx: Body forward speed (m/s).
        vy: Body lateral speed (m/s).
        omega_z: Body yaw rate (rad/s).
        wheel_radius: All-wheel radius r (m).
        wheel_positions: ``(N, 2)`` array of [x, y] positions in body frame.

    Returns:
        ``(N, 2)`` array of ``[delta, omega_wheel]`` per wheel.
    """
    wheel_positions = np.asarray(wheel_positions, dtype=float)
    n = len(wheel_positions)
    cmds = np.zeros((n, 2))
    for i in range(n):
        cmds[i] = swerve_wheel_cmd(
            vx, vy, omega_z, wheel_positions[i, 0], wheel_positions[i, 1], wheel_radius
        )
    return cmds

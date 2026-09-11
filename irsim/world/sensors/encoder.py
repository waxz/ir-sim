"""Encoder sensor that reads wheel state from a parent object's wheel layout.

When a named ``profile`` is given the sensor auto-creates and attaches a
``DiffWheelLayout`` to the parent's kinematics handler on the first step if no
wheel layout is already present, making the YAML configuration self-contained:

.. code-block:: yaml

    sensors:
      - name: encoder
        profile: dynamixel_xl430
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from irsim.world.object_base import ObjectBase


class Encoder:
    """Wheel encoder sensor with optional named motor+encoder profiles.

    Reads per-wheel encoder data from the parent object's ``wheel_layout``
    (via ``kf.wheel_layout``).  When a ``profile`` is given the sensor knows
    the expected CPR and motor preset, and will auto-attach a
    :class:`~irsim.lib.handler.wheel_handler.DiffWheelLayout` to the parent's
    kinematics handler on the first :meth:`step` call if no layout exists yet.

    Named profiles map to real motor+encoder combinations (all values are at
    the *wheel shaft* after gearbox, matching the ``MOTOR_PRESETS`` table in
    :mod:`irsim.lib.handler.wheel_handler`):

    =================== ============================== ============
    Profile             Motor                          CPR
    =================== ============================== ============
    ``small_dc``        46:1 brushed gearmotor         2048
    ``agv_hub_motor``   BLDC hub motor (direct drive)  1024
    ``forklift_drive``  Heavy brush motor, 20:1 chain  500
    ``dynamixel_xl430`` ROBOTIS XL430-W250-T (46.13:1) 4096
    ``pololu_37d_50``   Pololu 37D 50:1 gearmotor       3200
    ``maxon_ec45_43``   Maxon EC45 + GP42C 43:1         4096
    =================== ============================== ============

    Args:
        state: Initial [x, y, theta] state (unused; kept for factory API parity).
        obj_id: ID of the associated object.
        profile: Named encoder+motor preset.  ``None`` means use explicit params.
        motor: Motor preset name forwarded to the auto-created wheel layout when
            ``profile`` is not given.  Ignored when ``profile`` is set.
        encoder_cpr: Encoder counts per revolution for the auto-created layout.
            Ignored when ``profile`` is set.
        **kwargs: Ignored extra keyword arguments passed by SensorFactory.

    Attr:
        sensor_type (str): ``"encoder"``.
        profile (str | None): Active profile name.
        motor (str): Motor preset name used when auto-creating a layout.
        encoder_cpr (int): CPR used when auto-creating a layout.
        parent (ObjectBase | None): Set by the owning object after construction.
        data (dict): Latest encoder readings keyed by wheel name.
    """

    sensor_type: str = "encoder"

    # Named motor + encoder presets — mirrors the motor comments in wheel_handler.py.
    # Sources: MOTOR_PRESETS docstring recommendations for encoder_cpr.
    PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        # 46:1 brushed planetary gearmotor; 48-64 CPR motor shaft x 46 = ~2200 CPR wheel
        "small_dc": {"motor": "small_dc", "encoder_cpr": 2048},
        # BLDC hub motor, direct drive (FOC); 512-4096 pulse/rev magnetic encoder
        "agv_hub_motor": {"motor": "agv_hub_motor", "encoder_cpr": 1024},
        # Heavy brush motor, 20:1 chain reduction; industrial resolver or disk encoder
        "forklift_drive": {"motor": "forklift_drive", "encoder_cpr": 500},
        # ROBOTIS XL430-W250-T; 46.13:1 planetary; 12-bit absolute encoder = 4096 CPR
        "dynamixel_xl430": {"motor": "dynamixel_xl430", "encoder_cpr": 4096},
        # Pololu 37D 50:1 gearmotor; 64 CPR motor x 50 = 3200 CPR at wheel
        "pololu_37d_50": {"motor": "pololu_37d_50", "encoder_cpr": 3200},
        # Maxon EC45 flat + GP42C 43:1; 2048 CPR motor-shaft encoder
        "maxon_ec45_43": {"motor": "maxon_ec45_43", "encoder_cpr": 4096},
    }

    def __init__(
        self,
        state=None,
        obj_id: int = 0,
        profile: str | None = None,
        motor: str = "small_dc",
        encoder_cpr: int = 0,
        **kwargs: Any,
    ) -> None:
        self.obj_id = obj_id
        self.parent: ObjectBase | None = None
        self.data: dict[str, dict[str, Any]] = {}

        if profile is not None:
            if profile not in self.PROFILES:
                raise ValueError(
                    f"Unknown encoder profile {profile!r}. "
                    f"Available: {list(self.PROFILES)}"
                )
            p = self.PROFILES[profile]
            self.motor: str = p["motor"]
            self.encoder_cpr: int = p["encoder_cpr"]
        else:
            self.motor = motor
            self.encoder_cpr = int(encoder_cpr)

        self.profile: str | None = profile
        self._layout_attached: bool = False

    def _ensure_layout(self) -> None:
        """Auto-attach a DiffWheelLayout to the parent if none exists yet."""
        if self._layout_attached:
            return
        self._layout_attached = True
        if self.parent is None:
            return
        kf = getattr(self.parent, "kf", None)
        if kf is None or kf.wheel_layout is not None:
            return
        from irsim.lib.handler.wheel_handler import DiffWheelLayout

        kf.attach_wheel_layout(
            DiffWheelLayout(motor=self.motor, encoder_cpr=self.encoder_cpr)
        )

    def step(self, state) -> None:
        """Update encoder data from the parent's wheel layout.

        On the first call (when a profile or explicit motor is configured) a
        :class:`~irsim.lib.handler.wheel_handler.DiffWheelLayout` is
        auto-attached to the parent's kinematics handler if none exists yet.

        Args:
            state: Current [x, y, theta] state of the parent (unused here).
        """
        self._ensure_layout()
        if self.parent is None:
            return
        readings = getattr(self.parent, "encoder_readings", None)
        if readings is not None:
            self.data = readings

    def get_measurement(self) -> dict[str, dict[str, Any]]:
        """Return the latest encoder readings.

        Returns:
            Dict keyed by wheel name, each containing:
            ``{"theta_enc": float, "ticks": int, "omega_actual": float}``.
        """
        return dict(self.data)

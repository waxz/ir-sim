"""Encoder sensor that reads wheel state from a parent object's wheel layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from irsim.world.object_base import ObjectBase


class Encoder:
    """Wheel encoder sensor that exposes cumulative angle, tick count, and speed.

    Reads from the parent object's ``wheel_layout`` (via ``kf.wheel_layout``) on
    each :meth:`step` call.  Attach a wheel layout to the robot's kinematics
    handler (e.g. ``diff``, ``forklift``, ``dual_steer``) for this sensor to
    return non-empty data.

    Args:
        state: Initial [x, y, theta] state (unused; kept for factory API parity).
        obj_id: ID of the associated object.
        **kwargs: Ignored extra keyword arguments passed by SensorFactory.

    Attr:
        sensor_type (str): ``"encoder"``.
        parent (ObjectBase | None): Set by the owning object after construction.
        data (dict): Latest encoder readings keyed by wheel name.
    """

    sensor_type: str = "encoder"

    def __init__(self, state=None, obj_id: int = 0, **kwargs: Any) -> None:
        self.obj_id = obj_id
        self.parent: ObjectBase | None = None
        self.data: dict[str, dict[str, Any]] = {}

    def step(self, state) -> None:
        """Update encoder data from the parent's wheel layout.

        Args:
            state: Current [x, y, theta] state of the parent (unused).
        """
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

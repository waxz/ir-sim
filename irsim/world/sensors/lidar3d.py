"""3D LiDAR sensor backed by an Embree Scene3D raycaster.

The sensor delegates ray-casting to :class:`irsim.world.env3d.scene3d.Scene3D`
and requires a reference to that scene, set via the ``scene`` attribute after
construction.  When no scene is attached the sensor returns an empty array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from irsim.world.object_base import ObjectBase


class Lidar3D:
    """Simulated 3D LiDAR sensor using Embree BVH ray-casting.

    Args:
        state: Initial [x, y, theta] state of the parent object.
        obj_id: ID of the associated object.
        profile: Sensor profile name: ``"vlp16"``, ``"os64"``, or ``"os128"``.
        range_max: Maximum detection range in metres.
        sensor_height: Height of the sensor above the floor plane (m).
        offset: Sensor offset [x, y, z] from the object's position (m).
        **kwargs: Ignored extra keyword arguments passed by SensorFactory.

    Attr:
        sensor_type (str): ``"lidar3d"``.
        scene: Reference to a :class:`~irsim.world.env3d.scene3d.Scene3D`
            instance.  Must be set externally before the sensor is stepped.
        scan (np.ndarray): Latest scan points, shape ``(N, 4)`` columns
            ``(x, y, z, distance)`` in the world frame.
        parent (ObjectBase | None): Set by the owning object after construction.
    """

    sensor_type: str = "lidar3d"

    def __init__(
        self,
        state=None,
        obj_id: int = 0,
        profile: str = "vlp16",
        range_max: float = 50.0,
        sensor_height: float = 0.3,
        offset: list | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        self.obj_id = obj_id
        self.profile = profile
        self.range_max = float(range_max)
        self.sensor_height = float(sensor_height)
        self.offset = np.asarray(
            offset if offset is not None else [0.0, 0.0, 0.0], dtype=float
        )
        self.scene = None
        self.scan: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self.parent: ObjectBase | None = None

    def step(self, state: np.ndarray) -> None:
        """Cast a full 3D scan from the current sensor position.

        Args:
            state: Current [x, y, theta] state of the parent object (world frame).
        """
        if self.scene is None:
            return

        x = float(state[0])
        y = float(state[1])
        z = self.sensor_height + self.offset[2]
        origin = [x + self.offset[0], y + self.offset[1], z]

        self.scan = self.scene.cast_3d_lidar(origin, self.profile, self.range_max)

    def get_scan(self) -> np.ndarray:
        """Return the latest scan.

        Returns:
            Array of shape ``(N, 4)`` — columns ``(x, y, z, distance)``.
        """
        return self.scan

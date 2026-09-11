"""3D LiDAR sensor backed by open3d Embree raycasting (via Scene3D).

Requires ``pip install ir-sim[lidar3d]`` (open3d with Embree support).

The sensor delegates every scan to
:meth:`irsim.world.env3d.scene3d.Scene3D.cast_3d_lidar` and requires that a
:class:`~irsim.world.env3d.scene3d.Scene3D` instance be assigned to the
``scene`` attribute after the sensor is created::

    lidar = Lidar3D(state, obj_id, profile="vlp16", range_max=30.0)
    lidar.scene = my_scene3d_instance   # set before stepping
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from irsim.world.env3d.scene3d import Scene3D
    from irsim.world.object_base import ObjectBase


class Lidar3D:
    """Simulated 3D LiDAR sensor using open3d Embree BVH ray-casting.

    Wraps :meth:`Scene3D.cast_3d_lidar` for seamless integration with
    IR-SIM's sensor pipeline.  Three spinning-LiDAR profiles are available,
    matching the ``PROFILES`` table in
    :class:`~irsim.world.env3d.scene3d.Scene3D`:

    ========== ====== ======= =====================
    Profile    Beams  Azimuth Elevation
    ========== ====== ======= =====================
    ``vlp16``  16 ch  1 800   -15 to +15 deg
    ``os64``   64 ch  1 024   -45 to +45 deg
    ``os128``  128 ch 2 048   -45 to +45 deg
    ========== ====== ======= =====================

    Args:
        state: Initial [x, y, theta] state of the parent object (unused).
        obj_id: ID of the associated object.
        profile: Sensor beam pattern: ``"vlp16"``, ``"os64"``, or ``"os128"``.
        range_max: Maximum detection range in metres.
        sensor_height: Height of the sensor above the floor plane (m).
        offset: Sensor offset [x, y, z] from the object's XY position (m).
        **kwargs: Ignored extra keyword arguments passed by SensorFactory.

    Attr:
        sensor_type (str): ``"lidar3d"``.
        scene (Scene3D | None): Reference to a built
            :class:`~irsim.world.env3d.scene3d.Scene3D` instance.  Assign
            this before stepping; returns empty scan when ``None``.
        scan (np.ndarray): Latest scan, shape ``(N, 4)`` — columns
            ``(x, y, z, distance)`` in the world frame.
        parent (ObjectBase | None): Set by the owning object after construction.
    """

    sensor_type: str = "lidar3d"

    # Mirrors Scene3D.PROFILES: (n_vertical, n_horizontal, elev_min_deg, elev_max_deg)
    PROFILES: ClassVar[dict[str, tuple[int, int, float, float]]] = {
        "vlp16": (16, 1800, -15.0, 15.0),
        "os64": (64, 1024, -45.0, 45.0),
        "os128": (128, 2048, -45.0, 45.0),
    }

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
        if profile not in self.PROFILES:
            raise ValueError(
                f"Unknown lidar3d profile {profile!r}. Available: {list(self.PROFILES)}"
            )
        self.obj_id = obj_id
        self.profile = profile
        self.range_max = float(range_max)
        self.sensor_height = float(sensor_height)
        self.offset = np.asarray(
            offset if offset is not None else [0.0, 0.0, 0.0], dtype=float
        )
        self.scene: Scene3D | None = None
        self.scan: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self.parent: ObjectBase | None = None

    def step(self, state: np.ndarray) -> None:
        """Cast a full 3D scan from the current sensor position.

        Does nothing if :attr:`scene` has not been assigned.

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
        """Return the latest scan array.

        Returns:
            Array of shape ``(N, 4)`` — columns ``(x, y, z, distance)``.
            Empty ``(0, 4)`` array when no scene is attached or no hits.
        """
        return self.scan

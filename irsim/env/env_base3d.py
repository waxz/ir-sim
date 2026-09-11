from __future__ import annotations

from typing import Any

from irsim.config import env_param
from irsim.env import EnvBase
from irsim.world.object_factory import ObjectFactory
from irsim.world.world3d import World3D

from .env_plot3d import EnvPlot3D


class EnvBase3D(EnvBase):
    """
    This class is the 3D version of the environment class. It inherits from the :py:class:`.EnvBase` class to provide the 3D plot environment.
    """

    def __init__(self, world_name: str | None, **kwargs: Any):
        """Initialize a 3D environment with world and objects parsed from YAML.

        Args:
            world_name (str | None): Path to the world configuration YAML.
            **kwargs: Additional environment options forwarded to ``EnvBase``.
        """
        super().__init__(world_name, **kwargs)

        object_factory = ObjectFactory()

        self._world = World3D(
            world_name,
            world_param_instance=self._world_param,
            **self.env_config.parse["world"],
        )

        self._restart_object_ids()

        self._robot_collection = object_factory.create_from_parse(
            self.env_config.parse["robot"], "robot"
        )
        self._obstacle_collection = object_factory.create_from_parse(
            self.env_config.parse["obstacle"], "obstacle"
        )
        self._map_collection = object_factory.create_from_map(
            self._world.obstacle_positions,
            self._world.reso,
            grid_map=self._world.grid_map,
            world_offset=self._world.offset[:2],
        )

        if self._env_plot is not None:
            self._env_plot.close()
            self._env_plot = EnvPlot3D(
                self._world, self.objects, **self._world.plot_parse
            )

        self._init_scene3d()

        env_param.objects = self.objects

    def _init_scene3d(self) -> None:
        """Build the open3d raycasting scene from the ``scene3d`` YAML block.

        If a ``scene3d`` key is present in the config, a
        :class:`~irsim.world.env3d.scene3d.Scene3D` is built and stored on
        ``self._world.scene``.  All :class:`~irsim.world.sensors.lidar3d.Lidar3D`
        sensors on every robot and obstacle are then assigned that scene
        automatically so they can start scanning without manual wiring.
        """
        scene3d_cfg = self.env_config.parse.get("scene3d") or []
        if not scene3d_cfg:
            return

        from irsim.world.env3d.scene3d import Scene3D

        self._world.scene = Scene3D.from_config(scene3d_cfg)

        for obj in self._robot_collection + self._obstacle_collection:
            for sensor in getattr(obj, "sensors", []):
                if getattr(sensor, "sensor_type", None) == "lidar3d":
                    sensor.scene = self._world.scene

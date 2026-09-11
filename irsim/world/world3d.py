from __future__ import annotations

from typing import TYPE_CHECKING, Any

from irsim.world.world import World

if TYPE_CHECKING:
    from irsim.world.env3d.scene3d import Scene3D


class World3D(World):
    """3D world wrapper that extends :class:`~irsim.world.world.World` with z range.

    Attr:
        scene (Scene3D | None): Open3D raycasting scene built from the YAML
            ``scene3d`` block.  Set by :class:`~irsim.env.env_base3d.EnvBase3D`
            after construction; ``None`` until then.
    """

    def __init__(
        self,
        name: str,
        depth: float = 10.0,
        offset: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a 3D world extending the 2D world with depth.

        Args:
            name (str): World name or YAML file path.
            depth (float): Z-depth of the world (range in z). Default 10.0.
            offset (list[float] | None): [x, y, z] world offset. If a 2D
                [x, y] is provided, z defaults to 0.
            **kwargs: Forwarded to the base ``World`` constructor.
        """
        super().__init__(name=name, **kwargs)

        self.depth = depth
        self.scene: Scene3D | None = None

        if offset is None:
            offset = [0, 0, 0]
        self.offset = offset if len(offset) == 3 else [offset[0], offset[1], 0]

        self.z_range = [self.offset[2], self.offset[2] + self.depth]

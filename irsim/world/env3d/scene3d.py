"""
Scene3D — rich 3D environment builder for IR-SIM using open3d Embree.

Primitives
----------
- Wall    : thin box from start to end XY, with optional door / window openings
- Box     : axis-aligned or yaw-rotated box (crates, pillars, furniture)
- Car     : two-box sedan model (chassis + cabin)
- Ground  : flat or height-function terrain mesh

Sensors
-------
- cast_3d_lidar : spinning LiDAR (VLP-16, OS-64, OS-128 profiles)
- cast_2d_lidar : horizontal slice at given z (SICK TiM profile)

Both sensors use Intel Embree via open3d.t.geometry.RaycastingScene — real-time
for all common sensor profiles.

Requires: pip install ir-sim[lidar3d]
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

try:
    import open3d as o3d
    import open3d.t.geometry as otg

    _OPEN3D = True
except ImportError:  # pragma: no cover
    _OPEN3D = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_open3d() -> None:
    if not _OPEN3D:
        raise ImportError(
            "open3d is required for Scene3D.  "
            "Install it with:  pip install ir-sim[lidar3d]"
        )


def _legacy_to_tensor(mesh: o3d.geometry.TriangleMesh) -> otg.TriangleMesh:
    """Convert a legacy open3d TriangleMesh to the tensor variant."""
    v = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    t = o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.uint32))
    return otg.TriangleMesh(v, t)


def _make_box(
    cx: float,
    cy: float,
    cz: float,
    lx: float,
    ly: float,
    lz: float,
    yaw: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    """Axis-aligned box centered at (cx,cy,cz) with extents (lx,ly,lz), rotated by yaw."""
    mesh = o3d.geometry.TriangleMesh.create_box(lx, ly, lz)
    mesh.translate([-lx / 2, -ly / 2, -lz / 2])  # center at origin
    if abs(yaw) > 1e-9:
        R = mesh.get_rotation_matrix_from_xyz((0, 0, yaw))
        mesh.rotate(R, center=(0, 0, 0))
    mesh.translate([cx, cy, cz])
    return mesh


def _ground_mesh(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    resolution: int,
    height_fn: Callable[[float, float], float] | None,
) -> o3d.geometry.TriangleMesh:
    """Create a terrain mesh from a grid with optional height function."""
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xs, ys)

    Z = np.vectorize(height_fn)(X, Y) if height_fn is not None else np.zeros_like(X)

    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)

    faces = []
    n = resolution
    for j in range(n - 1):
        for i in range(n - 1):
            tl = j * n + i
            tr = tl + 1
            bl = (j + 1) * n + i
            br = bl + 1
            faces += [[tl, bl, tr], [tr, bl, br]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Primitive data-classes (for export / visualisation)
# ---------------------------------------------------------------------------


@dataclass
class _BoxRecord:
    """A single box in world frame (after opening carving)."""

    cx: float
    cy: float
    cz: float
    lx: float
    ly: float
    lz: float
    yaw: float
    color: tuple[float, float, float]
    label: str


@dataclass
class _GroundRecord:
    vertices: np.ndarray  # (N,3) float32
    faces: np.ndarray  # (M,3) int32
    color: tuple[float, float, float]


# ---------------------------------------------------------------------------
# Scene3D
# ---------------------------------------------------------------------------


class Scene3D:
    """
    3D simulation environment built from typed primitives.

    Workflow::

        scene = Scene3D()
        scene.add_ground(-20, 20, -20, 20)
        scene.add_wall([0, 0], [10, 0], height=3.0,
                       openings=[("door", 3.0, 5.0)])
        scene.add_car(center_xy=[5, 5], yaw=0.3)
        scene.add_box(center=[2, -3, 0.5], size=[1, 1, 1])

        # build once (triggers Embree BVH)
        scene.build()

        pts3d = scene.cast_3d_lidar([0, 0, 1.5], profile="vlp16")
        pts2d = scene.cast_2d_lidar([0, 0], z_height=1.2, n_beams=1500)

        data = scene.export()   # JSON-serialisable dict
    """

    # Sensor profile table: (n_vertical, n_horizontal, elev_min_deg, elev_max_deg)
    # Keep in sync with Lidar3D.PROFILES in irsim/world/sensors/lidar3d.py
    PROFILES: ClassVar[dict[str, tuple[int, int, float, float]]] = {
        # ── Velodyne ──────────────────────────────────────────────────────────
        "vlp16": (16, 1800, -15.0, 15.0),  # VLP-16 / Puck
        "vlp32c": (32, 1800, -25.0, 15.0),  # VLP-32C
        "hdl64e": (64, 1800, -24.9, 2.0),  # HDL-64E
        "vls128": (128, 1800, -25.0, 15.0),  # VLS-128 / Alpha Prime
        # ── Ouster OS0  (90° vertical FoV) ────────────────────────────────────
        "os0_32": (32, 1024, -45.0, 45.0),
        "os0_64": (64, 1024, -45.0, 45.0),
        "os0_128": (128, 2048, -45.0, 45.0),
        # ── Ouster OS1  (45° vertical FoV) ────────────────────────────────────
        "os1_32": (32, 1024, -22.5, 22.5),
        "os1_64": (64, 1024, -22.5, 22.5),
        "os1_128": (128, 2048, -22.5, 22.5),
        # ── Hesai ─────────────────────────────────────────────────────────────
        "xt32": (32, 1800, -15.0, 15.0),  # Hesai Pandar XT32
        "qt128": (128, 3600, -52.1, 52.1),  # Hesai QT128C2X
        # ── Robosense ─────────────────────────────────────────────────────────
        "rs16": (16, 1800, -15.0, 15.0),  # RS-LiDAR-16
        "rs32": (32, 1800, -15.0, 15.0),  # RS-LiDAR-32
        # ── Backward-compatibility aliases ────────────────────────────────────
        "os64": (64, 1024, -45.0, 45.0),  # legacy → os0_64
        "os128": (128, 2048, -45.0, 45.0),  # legacy → os0_128
    }

    def __init__(self) -> None:
        _require_open3d()
        self._boxes: list[_BoxRecord] = []
        self._grounds: list[_GroundRecord] = []
        self._meshes: list[o3d.geometry.TriangleMesh] = []
        self._mesh_files: list[
            dict
        ] = []  # parallel to _meshes; tracks source paths for Foxglove export
        self._rc: otg.RaycastingScene | None = None

    # ------------------------------------------------------------------
    # Primitive adders
    # ------------------------------------------------------------------

    def add_wall(
        self,
        start_xy: list | np.ndarray,
        end_xy: list | np.ndarray,
        height: float = 3.0,
        thickness: float = 0.15,
        openings: list[tuple] | None = None,
        color: tuple[float, float, float] = (0.72, 0.72, 0.72),
        label: str = "wall",
    ) -> Scene3D:
        """
        Add a wall segment with optional door/window openings.

        Parameters
        ----------
        openings : list of ("door"|"window", u_start, u_end)
            u is distance along wall from start.
            Doors span z=[0, 2.1].  Windows span z=[0.9, 2.1].
            Pass explicit z with ("opening", u_start, u_end, z_start, z_end).
        """
        s = np.asarray(start_xy, dtype=float)
        e = np.asarray(end_xy, dtype=float)
        vec2d = e - s
        L = float(np.linalg.norm(vec2d))
        if L < 1e-6:
            return self
        yaw = math.atan2(float(vec2d[1]), float(vec2d[0]))

        # Normalise openings to (u0, u1, z0, z1)
        norm_open: list[tuple[float, float, float, float]] = []
        for op in openings or []:
            if op[0] == "door":
                _, u0, u1 = op[:3]
                norm_open.append((float(u0), float(u1), 0.0, 2.1))
            elif op[0] == "window":
                _, u0, u1 = op[:3]
                norm_open.append((float(u0), float(u1), 0.9, 2.1))
            else:  # ("opening", u0, u1, z0, z1)
                _, u0, u1, z0, z1 = op
                norm_open.append((float(u0), float(u1), float(z0), float(z1)))

        # u breakpoints
        u_breaks = sorted({0.0, L} | {x for op in norm_open for x in [op[0], op[1]]})

        for k in range(len(u_breaks) - 1):
            u0, u1 = u_breaks[k], u_breaks[k + 1]
            u_mid = (u0 + u1) / 2

            blocked_z = [
                (oz0, oz1) for (ou0, ou1, oz0, oz1) in norm_open if ou0 <= u_mid <= ou1
            ]

            z_breaks = sorted({0.0, height} | {z for bz in blocked_z for z in bz})

            for m in range(len(z_breaks) - 1):
                z0, z1 = z_breaks[m], z_breaks[m + 1]
                z_mid = (z0 + z1) / 2
                if any(bz0 <= z_mid <= bz1 for bz0, bz1 in blocked_z):
                    continue

                seg_len = u1 - u0
                seg_h = z1 - z0
                u_c = (u0 + u1) / 2
                z_c = (z0 + z1) / 2

                # World-frame center: start + u_c along wall direction + z
                cx = float(s[0]) + u_c * math.cos(yaw)
                cy = float(s[1]) + u_c * math.sin(yaw)
                cz = z_c

                rec = _BoxRecord(
                    cx, cy, cz, seg_len, thickness, seg_h, yaw, color, label
                )
                self._boxes.append(rec)
                self._meshes.append(
                    _make_box(cx, cy, cz, seg_len, thickness, seg_h, yaw)
                )

        return self

    def add_box(
        self,
        center: list | np.ndarray,
        size: list | np.ndarray,
        yaw: float = 0.0,
        color: tuple[float, float, float] = (0.45, 0.32, 0.22),
        label: str = "box",
    ) -> Scene3D:
        """Add a generic box obstacle.  center=[x,y,z], size=[w,d,h]."""
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        lx, ly, lz = float(size[0]), float(size[1]), float(size[2])
        rec = _BoxRecord(cx, cy, cz, lx, ly, lz, yaw, color, label)
        self._boxes.append(rec)
        self._meshes.append(_make_box(cx, cy, cz, lx, ly, lz, yaw))
        return self

    def add_car(
        self,
        center_xy: list | np.ndarray,
        yaw: float = 0.0,
        model: str = "sedan",
        color: tuple[float, float, float] = (0.18, 0.36, 0.62),
        label: str = "car",
    ) -> Scene3D:
        """Add a two-box sedan model.  center_xy = [x, y] (ground plane)."""
        cx, cy = float(center_xy[0]), float(center_xy[1])

        if model == "sedan":
            # Chassis/body: 4.5 x 1.9 x 1.5 m
            bl, bw, bh = 4.5, 1.9, 1.5
            # Cabin: central 55%, narrower, on top at z=1.5
            rl, rw, rh = 2.5, 1.7, 0.65

        elif model == "suv":
            bl, bw, bh = 4.8, 2.0, 1.8
            rl, rw, rh = 2.8, 1.8, 0.0  # flush roof, skip roof box

        else:  # van / truck
            bl, bw, bh = 5.5, 2.2, 2.4
            rl, rw, rh = 0.0, 0.0, 0.0

        # Body centered at (cx, cy, bh/2)
        self.add_box([cx, cy, bh / 2], [bl, bw, bh], yaw, color, label)

        # Cabin centered at (cx, cy, bh + rh/2) — offset backward 0.3 m
        if rh > 0.01:
            cab_cx = cx - 0.3 * math.cos(yaw)
            cab_cy = cy - 0.3 * math.sin(yaw)
            roof_color = tuple(max(0.0, c - 0.08) for c in color)
            self.add_box(
                [cab_cx, cab_cy, bh + rh / 2],
                [rl, rw, rh],
                yaw,
                roof_color,
                label + "_cabin",
            )

        return self

    def add_ground(
        self,
        xmin: float = -20.0,
        xmax: float = 20.0,
        ymin: float = -20.0,
        ymax: float = 20.0,
        height_fn: Callable[[float, float], float] | None = None,
        resolution: int = 40,
        color: tuple[float, float, float] = (0.38, 0.45, 0.33),
    ) -> Scene3D:
        """Add a terrain ground.  height_fn(x,y) → z; None means flat at z=0."""
        mesh = _ground_mesh(xmin, xmax, ymin, ymax, resolution, height_fn)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        self._grounds.append(_GroundRecord(verts, faces, color))
        self._meshes.append(mesh)
        return self

    def load_mesh(
        self,
        path: str | os.PathLike,
        scale: float = 1.0,
        position: list | np.ndarray | None = None,
        yaw: float = 0.0,
        color: tuple[float, float, float] | None = None,
        label: str = "",
    ) -> Scene3D:
        """
        Load an external mesh file into the scene (GLB, OBJ, PLY, STL, FBX).

        The mesh is added to the Embree BVH on the next ``build()`` call.

        Parameters
        ----------
        path : str or PathLike
            Path to the mesh file.  Supported formats depend on the Open3D
            build (ASSIMP backend): GLB, GLTF, OBJ, PLY, STL, FBX, OFF.
        scale : float
            Uniform scale factor applied before rotation and translation.
        position : (3,) array-like or None
            World-frame translation [x, y, z] applied after scale and rotation.
        yaw : float
            Rotation around the Z-axis (radians), applied after scale.
        color : (R, G, B) tuple of floats in [0, 1] or None
            Paint the mesh with a uniform color.  None keeps the file's colors.
        label : str
            Human-readable label (unused by the raycaster; for bookkeeping).
        """
        _require_open3d()
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
        if len(mesh.vertices) == 0:
            raise ValueError(
                f"load_mesh: no geometry loaded from '{path}'.  "
                "Check that the file exists and the format is supported."
            )
        if scale != 1.0:
            mesh.scale(scale, center=(0.0, 0.0, 0.0))
        if abs(yaw) > 1e-9:
            R = mesh.get_rotation_matrix_from_xyz((0.0, 0.0, yaw))
            mesh.rotate(R, center=(0.0, 0.0, 0.0))
        if position is not None:
            mesh.translate(np.asarray(position, dtype=float))
        if color is not None:
            mesh.paint_uniform_color(list(color))
        mesh.compute_vertex_normals()
        self._mesh_files.append(
            {
                "path": str(path),
                "scale": scale,
                "position": list(position) if position is not None else None,
                "yaw": yaw,
                "label": label,
            }
        )
        self._meshes.append(mesh)
        return self

    # ------------------------------------------------------------------
    # Build Embree scene
    # ------------------------------------------------------------------

    def build(self) -> Scene3D:
        """Compile all primitives into the Embree raycasting scene."""
        rc = otg.RaycastingScene()
        for mesh in self._meshes:
            tm = _legacy_to_tensor(mesh)
            rc.add_triangles(tm)
        self._rc = rc
        return self

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _require_built(self) -> None:
        if self._rc is None:
            self.build()

    def cast_3d_lidar(
        self,
        origin: list | np.ndarray,
        profile: str = "vlp16",
        range_max: float = 50.0,
    ) -> np.ndarray:
        """
        Cast a full 3D LiDAR scan and return hit points.

        Parameters
        ----------
        origin : (3,) sensor origin in world frame
        profile : "vlp16" | "os64" | "os128"
        range_max : maximum range in metres

        Returns
        -------
        pts : (N, 4) array — columns (x, y, z, distance); only valid hits
        """
        self._require_built()
        n_v, n_h, e_min, e_max = self.PROFILES[profile]
        origin = np.asarray(origin, dtype=np.float32)

        elev = np.linspace(math.radians(e_min), math.radians(e_max), n_v)
        az = np.linspace(-math.pi, math.pi, n_h, endpoint=False)
        E, A = np.meshgrid(elev, az, indexing="ij")
        dx = np.cos(E) * np.cos(A)
        dy = np.cos(E) * np.sin(A)
        dz = np.sin(E)
        dirs = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3).astype(np.float32)

        n = len(dirs)
        origs = np.tile(origin, (n, 1))
        rays = o3d.core.Tensor(
            np.concatenate([origs, dirs], axis=1), dtype=o3d.core.Dtype.Float32
        )
        result = self._rc.cast_rays(rays)
        t_hit = result["t_hit"].numpy()

        valid = np.isfinite(t_hit) & (t_hit < range_max)
        t_valid = t_hit[valid]
        d_valid = dirs[valid]
        pts = origin + t_valid[:, None] * d_valid
        dists = t_valid.astype(np.float32)
        return np.concatenate([pts, dists[:, None]], axis=1).astype(np.float32)

    def cast_2d_lidar(
        self,
        origin_xy: list | np.ndarray,
        z_height: float = 1.2,
        n_beams: int = 1500,
        range_max: float = 20.0,
    ) -> np.ndarray:
        """
        Cast a horizontal 2D LiDAR slice at a given z height.

        Returns
        -------
        pts : (N, 4) — (x, y, z, distance); only valid hits
        """
        self._require_built()
        origin = np.array(
            [float(origin_xy[0]), float(origin_xy[1]), z_height], dtype=np.float32
        )

        az = np.linspace(-math.pi, math.pi, n_beams, endpoint=False).astype(np.float32)
        dx = np.cos(az)
        dy = np.sin(az)
        dz = np.zeros(n_beams, dtype=np.float32)
        dirs = np.stack([dx, dy, dz], axis=1)

        n = n_beams
        origs = np.tile(origin, (n, 1))
        rays = o3d.core.Tensor(
            np.concatenate([origs, dirs], axis=1), dtype=o3d.core.Dtype.Float32
        )
        result = self._rc.cast_rays(rays)
        t_hit = result["t_hit"].numpy()

        valid = np.isfinite(t_hit) & (t_hit < range_max)
        t_valid = t_hit[valid]
        d_valid = dirs[valid]
        pts = origin + t_valid[:, None] * d_valid
        dists = t_valid.astype(np.float32)
        return np.concatenate([pts, dists[:, None]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Config-driven construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, items: list[dict]) -> Scene3D:
        """Build a :class:`Scene3D` from a list of shape configuration dicts.

        Each dict must have a ``type`` key selecting which primitive to add.
        Accepted types and their keyword arguments mirror the ``add_*`` /
        ``load_mesh`` method signatures:

        =========== ============================================================
        Type        Keys
        =========== ============================================================
        ``ground``  ``xmin``, ``xmax``, ``ymin``, ``ymax``; opt: ``resolution``,
                    ``color``
        ``box``     ``center`` [x,y,z], ``size`` [w,d,h]; opt: ``yaw``,
                    ``color``, ``label``
        ``wall``    ``start_xy`` [x,y], ``end_xy`` [x,y]; opt: ``height``,
                    ``thickness``, ``openings``, ``color``, ``label``
        ``car``     ``center_xy`` [x,y]; opt: ``yaw``, ``model``, ``color``,
                    ``label``
        ``mesh``    ``path``; opt: ``scale``, ``position`` [x,y,z], ``yaw``,
                    ``color``, ``label``
        =========== ============================================================

        Example YAML block::

            scene3d:
              - type: ground
                xmin: -20
                xmax: 20
                ymin: -20
                ymax: 20
              - type: box
                center: [2, 3, 0.5]
                size: [1, 1, 1]
              - type: wall
                start_xy: [0, 0]
                end_xy: [10, 0]
                height: 3.0
              - type: mesh
                path: obstacles/building.ply

        Args:
            items: List of shape configuration dicts, each with a ``type`` key.

        Returns:
            A new :class:`Scene3D` with all primitives added (not yet built).

        Raises:
            ValueError: If a dict carries an unknown ``type``.
        """
        scene = cls()
        _DISPATCH = {
            "ground": scene.add_ground,
            "box": scene.add_box,
            "wall": scene.add_wall,
            "car": scene.add_car,
            "mesh": scene.load_mesh,
        }
        for raw in items:
            item = dict(raw)
            shape_type = item.pop("type", None)
            method = _DISPATCH.get(shape_type)  # type: ignore[arg-type]
            if method is None:
                raise ValueError(
                    f"Unknown scene3d shape type: {shape_type!r}. "
                    f"Available: {list(_DISPATCH)}"
                )
            method(**item)
        return scene

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        lidar3d_origin: list | np.ndarray | None = None,
        lidar3d_profile: str = "vlp16",
        lidar2d_origin_xy: list | np.ndarray | None = None,
        lidar2d_z: float = 1.2,
        range_max_3d: float = 50.0,
        range_max_2d: float = 20.0,
    ) -> dict:
        """
        Return a JSON-serialisable dict of geometry + optional LiDAR data.
        Calls build() if not already built.
        """
        self._require_built()

        boxes_out = []
        for b in self._boxes:
            boxes_out.append(
                {
                    "cx": round(b.cx, 4),
                    "cy": round(b.cy, 4),
                    "cz": round(b.cz, 4),
                    "lx": round(b.lx, 4),
                    "ly": round(b.ly, 4),
                    "lz": round(b.lz, 4),
                    "yaw": round(b.yaw, 6),
                    "color": f"#{int(b.color[0] * 255):02x}{int(b.color[1] * 255):02x}{int(b.color[2] * 255):02x}",
                    "label": b.label,
                }
            )

        grounds_out = []
        for g in self._grounds:
            grounds_out.append(
                {
                    "vertices": g.vertices.tolist(),
                    "faces": g.faces.tolist(),
                    "color": f"#{int(g.color[0] * 255):02x}{int(g.color[1] * 255):02x}{int(g.color[2] * 255):02x}",
                }
            )

        out: dict = {"boxes": boxes_out, "grounds": grounds_out}

        if lidar3d_origin is not None:
            o3 = list(map(float, lidar3d_origin))
            pts3 = self.cast_3d_lidar(o3, lidar3d_profile, range_max_3d)
            out["lidar3d"] = {
                "origin": o3,
                "profile": lidar3d_profile,
                "points": pts3[:, :3].tolist(),
                "distances": pts3[:, 3].tolist(),
                "range_max": range_max_3d,
            }

        if lidar2d_origin_xy is not None:
            o2 = list(map(float, lidar2d_origin_xy))
            pts2 = self.cast_2d_lidar(o2, lidar2d_z, 1500, range_max_2d)
            out["lidar2d"] = {
                "origin": [*o2, lidar2d_z],
                "points": pts2[:, :3].tolist(),
                "distances": pts2[:, 3].tolist(),
                "range_max": range_max_2d,
            }

        return out

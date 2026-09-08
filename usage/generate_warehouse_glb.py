"""
Generate a procedural warehouse GLB model for LiDAR testing.

The warehouse contains:
- Concrete floor (60 x 40 m)
- Perimeter walls (4.5 m tall, 0.25 m thick)
- Four interior support columns (1 x 1 x 4.5 m)
- Six double-sided shelving racks (0.8 x 8 x 2.4 m, on each side)
- Twelve stacked cargo boxes (random sizes ~0.8-1.2 m)

Output: usage/warehouse.glb

Requirements::

    pip install ir-sim[lidar3d]   # includes trimesh

Run::

    python usage/generate_warehouse_glb.py
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

try:
    import trimesh
    import trimesh.creation as tc
except ImportError as exc:
    raise SystemExit(
        "trimesh is required.  Install with:  pip install ir-sim[lidar3d]"
    ) from exc


def _box(
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    yaw: float = 0.0,
    color: tuple[int, int, int, int] = (180, 180, 180, 255),
) -> trimesh.Trimesh:
    mesh = tc.box(extents=size)
    if abs(yaw) > 1e-9:
        R = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
        mesh.apply_transform(R)
    mesh.apply_translation(pos)
    mesh.visual.face_colors = color
    return mesh


def build_warehouse() -> trimesh.Scene:
    parts: list[trimesh.Trimesh] = []

    # --- Floor ---------------------------------------------------------------
    parts.append(_box((60.0, 40.0, 0.1), (0.0, 0.0, -0.05), color=(140, 130, 120, 255)))

    # --- Perimeter walls (thickness 0.25 m, height 4.5 m) -------------------
    wall_h = 4.5
    wall_t = 0.25
    wall_color = (160, 155, 150, 255)
    # North / South (along X)
    for y_sign in (-1, 1):
        parts.append(
            _box(
                (60.0, wall_t, wall_h),
                (0.0, y_sign * 20.0, wall_h / 2),
                color=wall_color,
            )
        )
    # East / West (along Y)
    for x_sign in (-1, 1):
        parts.append(
            _box(
                (wall_t, 40.0, wall_h),
                (x_sign * 30.0, 0.0, wall_h / 2),
                color=wall_color,
            )
        )

    # --- Support columns (1 x 1 x 4.5) at four interior points --------------
    col_color = (100, 95, 90, 255)
    for cx, cy in [(-12.0, -8.0), (-12.0, 8.0), (12.0, -8.0), (12.0, 8.0)]:
        parts.append(_box((1.0, 1.0, wall_h), (cx, cy, wall_h / 2), color=col_color))

    # --- Shelving racks: two rows of 6, 8 m long, 2.4 m tall ----------------
    rack_color = (60, 100, 160, 255)
    shelf_panel_color = (200, 180, 120, 255)
    rack_w = 0.8
    rack_d = 8.0
    rack_h = 2.4
    n_bays = 4  # shelf panels per rack
    shelf_t = 0.04

    rack_positions = [
        (-20.0, -14.0, 0.0, 0.0),
        (-20.0, -5.5, 0.0, 0.0),
        (-20.0, 5.5, 0.0, 0.0),
        (20.0, -14.0, 0.0, 0.0),
        (20.0, -5.5, 0.0, 0.0),
        (20.0, 5.5, 0.0, 0.0),
    ]

    for rx, ry, _rz, _ryaw in rack_positions:
        # Upright frame posts (two 0.1 x 0.1 x rack_h columns)
        for dy in (-rack_d / 2, rack_d / 2):
            parts.append(
                _box((0.1, 0.1, rack_h), (rx, ry + dy, rack_h / 2), color=rack_color)
            )
        # Horizontal shelf panels
        for level in range(1, n_bays + 1):
            z_shelf = level * (rack_h / (n_bays + 1))
            parts.append(
                _box(
                    (rack_w, rack_d, shelf_t),
                    (rx, ry, z_shelf),
                    color=shelf_panel_color,
                )
            )

    # --- Cargo boxes scattered on the floor ----------------------------------
    rng = np.random.default_rng(42)
    box_positions = [
        (-5.0, -15.0, 0.0),
        (-3.0, -15.0, 0.0),
        (-5.0, -13.5, 0.0),
        (5.0, 12.0, 0.0),
        (6.5, 12.0, 0.0),
        (5.0, 13.5, 0.0),
        (0.0, -8.0, 0.0),
        (1.5, -8.0, 0.0),
        (-1.0, 6.0, 0.0),
        (2.0, 6.0, 0.0),
        (-8.0, 2.0, 0.0),
        (-8.0, -2.0, 0.0),
    ]
    box_colors = [
        (200, 160, 80, 255),
        (80, 140, 200, 255),
        (160, 200, 80, 255),
        (220, 100, 80, 255),
    ]
    for i, (bx, by, bz) in enumerate(box_positions):
        sz = float(rng.uniform(0.8, 1.2))
        sh = float(rng.uniform(0.6, 1.0))
        col = box_colors[i % len(box_colors)]
        yaw_b = float(rng.uniform(0, math.pi / 4))
        parts.append(_box((sz, sz, sh), (bx, by, bz + sh / 2), yaw=yaw_b, color=col))

    scene = trimesh.scene.scene.Scene()
    for i, p in enumerate(parts):
        scene.add_geometry(p, geom_name=f"part_{i:03d}")
    return scene


def main() -> None:
    out_path = pathlib.Path(__file__).parent / "warehouse.glb"
    print("Building warehouse …")
    scene = build_warehouse()
    scene.export(str(out_path))
    print(f"Saved → {out_path}")
    print(f"  Meshes : {len(scene.geometry)}")
    total_faces = sum(len(g.faces) for g in scene.geometry.values())
    print(f"  Faces  : {total_faces:,}")


if __name__ == "__main__":
    main()

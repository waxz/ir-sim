"""
Generate a simple differential-drive robot GLB model using trimesh.

The model is centred at the origin and consists of:
  - Chassis: flat box (0.28 m x 0.16 m x 0.09 m)  -- dark grey
  - Left wheel: cylinder, radius 0.033 m, width 0.018 m -- black
  - Right wheel: identical, mirrored             -- black
  - Caster sphere: radius 0.016 m                -- charcoal
  - LiDAR dome: hemisphere, radius 0.035 m       -- white

The coordinate convention is right-hand, Z-up (matches Foxglove / ROS):
  +X = forward, +Y = left, +Z = up.

Output: usage/robot.glb
"""

import numpy as np
import trimesh
import trimesh.creation


def _chassis() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[0.28, 0.16, 0.09])
    box.apply_translation([0.0, 0.0, 0.045])
    box.visual.vertex_colors = np.full(
        (len(box.vertices), 4), [55, 55, 60, 255], dtype=np.uint8
    )
    return box


def _wheel(side: float) -> trimesh.Trimesh:
    """side = +1 (left) or -1 (right)"""
    cyl = trimesh.creation.cylinder(radius=0.033, height=0.018, sections=24)
    # cylinder axis is Z; rotate to lie along Y
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    cyl.apply_transform(rot)
    cyl.apply_translation([0.0, side * 0.089, 0.033])
    cyl.visual.vertex_colors = np.full(
        (len(cyl.vertices), 4), [20, 20, 20, 255], dtype=np.uint8
    )
    return cyl


def _caster() -> trimesh.Trimesh:
    sph = trimesh.creation.icosphere(subdivisions=2, radius=0.016)
    sph.apply_translation([-0.10, 0.0, 0.016])
    sph.visual.vertex_colors = np.full(
        (len(sph.vertices), 4), [70, 70, 70, 255], dtype=np.uint8
    )
    return sph


def _lidar_dome() -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.035)
    # Keep only upper hemisphere (Z >= 0 after centering at lidar position)
    mask = sphere.vertices[:, 2] >= 0
    # Build new mesh from faces where all 3 vertices are in mask
    keep = np.all(mask[sphere.faces], axis=1)
    faces = sphere.faces[keep]
    verts = sphere.vertices
    dome = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    dome.apply_translation([0.04, 0.0, 0.09 + 0.035])
    dome.visual.vertex_colors = np.full(
        (len(dome.vertices), 4), [230, 230, 230, 255], dtype=np.uint8
    )
    return dome


def build_robot_glb(output_path: str = "robot.glb") -> None:
    parts = [_chassis(), _wheel(+1.0), _wheel(-1.0), _caster(), _lidar_dome()]
    scene = trimesh.scene.scene.Scene()
    for i, part in enumerate(parts):
        scene.add_geometry(part, node_name=f"part_{i}")
    scene.export(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).parent / "robot.glb"
    build_robot_glb(str(out))

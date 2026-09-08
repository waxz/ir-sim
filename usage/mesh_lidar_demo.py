"""
LiDAR verification demo for Scene3D.load_mesh().

Loads the procedural warehouse GLB (generate it first with
``python usage/generate_warehouse_glb.py``), casts 2D and 3D LiDAR from
the centre of the building, and saves two matplotlib figures:

- usage/lidar2d_warehouse.png  — top-down view of the 2D horizontal scan
- usage/lidar3d_warehouse.png  — 3D scatter plot coloured by height

Requirements::

    pip install ir-sim[lidar3d]

Run::

    python usage/generate_warehouse_glb.py   # once
    python usage/mesh_lidar_demo.py
"""

from __future__ import annotations

import math
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from irsim.world.env3d import Scene3D
except ImportError as exc:
    raise SystemExit(
        "ir-sim[lidar3d] is required.  Install with:  pip install ir-sim[lidar3d]"
    ) from exc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).parent
GLB_PATH = HERE / "warehouse.glb"

if not GLB_PATH.exists():
    raise SystemExit(
        f"Warehouse model not found at {GLB_PATH}.\n"
        "Run:  python usage/generate_warehouse_glb.py"
    )

OUT_2D = HERE / "lidar2d_warehouse.png"
OUT_3D = HERE / "lidar3d_warehouse.png"

# ---------------------------------------------------------------------------
# Build scene
# ---------------------------------------------------------------------------

print("Loading warehouse mesh …")
scene = Scene3D()
scene.load_mesh(GLB_PATH, label="warehouse")
scene.build()
print("  Embree BVH ready.")

# ---------------------------------------------------------------------------
# Cast 2D LiDAR (horizontal slice at z = 1.5 m)
# ---------------------------------------------------------------------------

ORIGIN_XY = [0.0, 0.0]
Z_HEIGHT = 1.5
N_BEAMS = 1440
RANGE_2D = 40.0

print(f"Casting 2D LiDAR ({N_BEAMS} beams, z={Z_HEIGHT} m) …")
pts2d = scene.cast_2d_lidar(
    ORIGIN_XY, z_height=Z_HEIGHT, n_beams=N_BEAMS, range_max=RANGE_2D
)
print(f"  {len(pts2d):,} valid hits")

fig2d, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("#1a1a2e")
fig2d.patch.set_facecolor("#1a1a2e")

# Draw scan rays (thinned out)
ox, oy = ORIGIN_XY
for pt in pts2d[::4]:
    ax.plot([ox, pt[0]], [oy, pt[1]], color="#2a6496", linewidth=0.3, alpha=0.4)

# Draw hit points
ax.scatter(pts2d[:, 0], pts2d[:, 1], s=1.2, c="#4fc3f7", linewidths=0, zorder=3)
ax.scatter([ox], [oy], s=60, c="#ff6b6b", zorder=5, label="sensor")

ax.set_xlim(-32, 32)
ax.set_ylim(-22, 22)
ax.set_aspect("equal")
ax.tick_params(colors="#aaa")
ax.spines[:].set_color("#444")
ax.set_xlabel("X (m)", color="#aaa")
ax.set_ylabel("Y (m)", color="#aaa")
ax.set_title("2D LiDAR — warehouse (z = 1.5 m slice)", color="white", fontsize=12)
ax.legend(loc="upper right", facecolor="#222", labelcolor="white", framealpha=0.7)

fig2d.tight_layout()
fig2d.savefig(OUT_2D, dpi=150, bbox_inches="tight")
plt.close(fig2d)
print(f"  Saved → {OUT_2D}")

# ---------------------------------------------------------------------------
# Cast 3D LiDAR (VLP-16 profile)
# ---------------------------------------------------------------------------

ORIGIN_3D = [0.0, 0.0, 1.8]
RANGE_3D = 40.0

print(f"Casting 3D LiDAR (VLP-16, origin={ORIGIN_3D}) …")
pts3d = scene.cast_3d_lidar(ORIGIN_3D, profile="vlp16", range_max=RANGE_3D)
print(f"  {len(pts3d):,} valid hits")

fig3d = plt.figure(figsize=(11, 7))
fig3d.patch.set_facecolor("#0d1117")
ax3 = fig3d.add_subplot(111, projection="3d")
ax3.set_facecolor("#0d1117")

z_vals = pts3d[:, 2]
z_min, z_max = float(z_vals.min()), float(z_vals.max())
z_norm = (z_vals - z_min) / max(z_max - z_min, 1e-6)

cmap = plt.get_cmap("plasma")
colors = cmap(z_norm)

ax3.scatter(
    pts3d[:, 0],
    pts3d[:, 1],
    pts3d[:, 2],
    c=colors,
    s=0.8,
    linewidths=0,
    depthshade=True,
)

ax3.scatter(*ORIGIN_3D, s=80, c="#ff6b6b", zorder=10)

ax3.set_xlim(-32, 32)
ax3.set_ylim(-22, 22)
ax3.set_zlim(0, 5)
ax3.set_xlabel("X (m)", color="#aaa")
ax3.set_ylabel("Y (m)", color="#aaa")
ax3.set_zlabel("Z (m)", color="#aaa")
ax3.tick_params(colors="#aaa")
ax3.set_title("3D LiDAR VLP-16 — warehouse", color="white", fontsize=12)

# Colorbar by height
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
sm.set_array([])
cbar = fig3d.colorbar(sm, ax=ax3, shrink=0.5, pad=0.1)
cbar.set_label("Height (m)", color="#aaa")
cbar.ax.yaxis.set_tick_params(color="#aaa")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#aaa")

fig3d.tight_layout()
fig3d.savefig(OUT_3D, dpi=150, bbox_inches="tight")
plt.close(fig3d)
print(f"  Saved → {OUT_3D}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n--- Summary ---")
print(f"2D scan : {len(pts2d):,} hits from {N_BEAMS} beams")
print(f"3D scan : {len(pts3d):,} hits (VLP-16: 16 x 1800 rays)")

az = np.arctan2(pts2d[:, 1] - oy, pts2d[:, 0] - ox)
unique_walls = int(np.sum(np.diff(np.sort(az)) > math.radians(5)))
print(f"  Approx structural features detected: {unique_walls}")

z_bands = {
    "floor (< 0.1 m)": int(np.sum(z_vals < 0.1)),
    "low (0.1-1 m)": int(np.sum((z_vals >= 0.1) & (z_vals < 1.0))),
    "mid (1-2.5 m)": int(np.sum((z_vals >= 1.0) & (z_vals < 2.5))),
    "high (> 2.5 m)": int(np.sum(z_vals >= 2.5)),
}
for band, count in z_bands.items():
    print(f"  {band}: {count:,} pts")
print("Done.")

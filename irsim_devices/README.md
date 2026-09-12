# irsim-devices

Standalone sensor and actuator simulation models extracted from
[ir-sim](https://github.com/hanruihua/ir-sim).  Each module operates
independently: no dependency on ObjectBase or KinematicsFactory.

## Modules

| Module | Description | Extra |
|--------|-------------|-------|
| `sensors.IMU` | 3-DOF IMU with IEEE 517 / Gaussian noise models | — |
| `sensors.Lidar2D` | 2D LiDAR with hardware profiles, AVX2 ray casting | `lidar2d` |
| `sensors.Lidar3D` | 3D spinning LiDAR via open3d Embree BVH | `lidar3d` |
| `sensors.Encoder` | Wheel encoder with named motor profiles | — |
| `actuators.Motor` | DC motor physics with built-in PID & encoder | — |
| `actuators.MotorDiffChassis` | Dual-motor differential drive chassis | — |

## Install

```bash
pip install irsim-devices           # core (IMU, Encoder, Motor)
pip install "irsim-devices[lidar2d]" # adds 2D LiDAR (shapely + matplotlib)
pip install "irsim-devices[lidar3d]" # adds 3D LiDAR (open3d)
pip install "irsim-devices[all]"     # everything
```

## Quick start

```python
from irsim_devices.actuators import Motor, MotorDiffChassis

chassis = MotorDiffChassis(wheel_radius=0.05, wheel_base=0.30,
                            motor_profile="small_dc")
for _ in range(1000):
    state = chassis.step([0.8, 0.8], dt=0.001)

from irsim_devices.sensors import IMU
imu = IMU(noise_model="ieee517")
imu.step(state_3dof, dt=0.01)
print(imu.angular_velocity, imu.linear_acceleration)
```

## World model interface

`Lidar2D` and `Lidar3D` accept any object satisfying the lightweight
protocols in `irsim_devices.core.world_model`:

```python
from irsim_devices.core.world_model import GeometryObject2D, Scene3DProtocol
```

These protocols make it straightforward to use these sensors outside of
ir-sim with any simulation framework that provides 2D Shapely geometry or
an open3d `RaycastingScene`.

## C extensions

Fast ray casting is provided by a C+OpenMP kernel (`csrc/ray_casting_omp.c`)
and the IMU sub-stepping by `csrc/imu_c_ext.c`.  Both are optional
accelerators; the package falls back to NumPy when they are absent.

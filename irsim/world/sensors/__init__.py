"""
Sensor classes for IR-SIM simulation.

This package contains:
- encoder: Wheel encoder sensor
- fmcw_lidar2d: 2D FMCW LiDAR sensor implementation
- lidar2d: 2D LiDAR sensor implementation
- lidar3d: 3D LiDAR sensor backed by Embree BVH ray-casting
- sensor_factory: Sensor factory for creating sensors
"""

from .encoder import Encoder
from .fmcw_lidar2d import FMCWLidar2D
from .imu import IMU
from .lidar2d import Lidar2D
from .lidar3d import Lidar3D
from .sensor_factory import SensorFactory

__all__ = ["IMU", "Encoder", "FMCWLidar2D", "Lidar2D", "Lidar3D", "SensorFactory"]

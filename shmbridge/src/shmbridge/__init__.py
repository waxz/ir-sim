"""
shmbridge - zero-dependency POSIX shared-memory bridge for Python ↔ C++ robot control.

Quick start::

    # Python sim side
    from shmbridge import ShmBridge
    bridge = ShmBridge()
    bridge.open()
    while not env.done():
        env.step(bridge.read_cmd())
        bridge.write_state_from_robot(env.robot_list[0], step, t)
    bridge.close()

    // C++ controller side
    #include <shmbridge/shmbridge.h>
    auto *blk = shmbridge_attach("/irsim_bridge_v2");
    IrsimState s; irsim_read_state(&blk->states[0], &s);
    IrsimCmd   c = {0.5f, 0.0f, 1, 1};
    irsim_write_cmd(&blk->cmds[0], &c);
"""

from ._types import EXT_SHM_NAME_DEFAULT, MAGIC, SCHEMA_VERSION, SHM_NAME_DEFAULT
from .bridge import RobotCmd, RobotState, ShmBridge
from .bridge_ext import ExtShmBridge

__version__ = "2.0.0"
__all__ = [
    "EXT_SHM_NAME_DEFAULT",
    "MAGIC",
    "SCHEMA_VERSION",
    "SHM_NAME_DEFAULT",
    "ExtShmBridge",
    "RobotCmd",
    "RobotState",
    "ShmBridge",
]

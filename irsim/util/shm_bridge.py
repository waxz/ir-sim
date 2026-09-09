"""
shm_bridge.py - compatibility shim; real implementation is in the shmbridge package.

Install:  pip install ./shmbridge   (from the repo root)
"""

from __future__ import annotations

try:
    from shmbridge._libc import _libc
    from shmbridge._platform import O_CREAT, O_EXCL, O_RDWR
    from shmbridge._types import (
        SHM_NAME_DEFAULT as SHM_NAME,
    )
    from shmbridge._types import (
        _IrsimBlock,  # used by bench_bidir.py for round-trip test
        _IrsimCmdSlot,
        _IrsimStateSlot,
    )
    from shmbridge.bridge import RobotCmd, RobotState, ShmBridge
except ImportError as _e:
    raise ImportError(
        "shmbridge package not installed. "
        "Run: pip install ./shmbridge  (from the ir-sim repo root)"
    ) from _e

# Legacy size constant (v1 had 512 B, v2 is page-aligned 4096 B for n=1)
SHM_SIZE = 4096

__all__ = [
    "O_CREAT",
    "O_EXCL",
    "O_RDWR",
    "SHM_NAME",
    "SHM_SIZE",
    "RobotCmd",
    "RobotState",
    "ShmBridge",
    "_IrsimBlock",
    "_IrsimCmdSlot",
    "_IrsimStateSlot",
    "_libc",
]

"""
shm_bridge_ext.py - compatibility shim; real implementation is in the shmbridge package.
"""

from __future__ import annotations

try:
    from shmbridge._types import EXT_SHM_NAME_DEFAULT as EXT_SHM_NAME
    from shmbridge.bridge_ext import ExtShmBridge
except ImportError as _e:
    raise ImportError(
        "shmbridge package not installed. "
        "Run: pip install ./shmbridge  (from the ir-sim repo root)"
    ) from _e

__all__ = ["EXT_SHM_NAME", "ExtShmBridge"]

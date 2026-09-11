"""Build the optional C+OpenMP ray-casting extension.

setuptools compiles ``_ray_casting_omp`` on supported platforms.
The extension is *optional*: if it cannot be built (missing compiler,
missing OpenMP), the package still installs cleanly and the Python
fallback in ``ray_casting_2d_omp.py`` is used instead.

Platform flags
--------------
Linux / POSIX : gcc  -O3 -march=native -mavx2 -fopenmp
                (-march=native already enables AVX2 on capable CPUs;
                 -mavx2 is added explicitly so the C guard _IRSIM_AVX2
                 is always defined when the hardware supports it)
macOS         : clang -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
                (falls back to serial -O3 when libomp is not installed;
                 -march=native enables AVX2 on Intel Macs automatically,
                 AArch64/Apple-Silicon has no AVX2 so only OMP path fires)
Windows       : cl.exe /O2 /openmp [/arch:AVX2]
                Set IRSIM_ENABLE_AVX2=1 to add /arch:AVX2 (requires an
                AVX2-capable CPU; the resulting .pyd will crash at startup
                on older hardware if set incorrectly).
"""

from __future__ import annotations

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# LIBOMP_PREFIX is set by cibuildwheel's environment config to point at a
# conda-forge llvm-openmp install whose macOS deployment target matches the
# wheel target.  Local Homebrew paths are checked as fallbacks.
_LIBOMP_PREFIX = os.environ.get("LIBOMP_PREFIX", "")
_LIBOMP_ROOTS = [
    *([_LIBOMP_PREFIX] if _LIBOMP_PREFIX else []),
    "/opt/homebrew/opt/libomp",  # Apple Silicon Homebrew
    "/usr/local/opt/libomp",  # Intel Homebrew
]


class _OmpBuildExt(build_ext):
    """Inject per-platform OpenMP compiler/linker flags at build time."""

    def build_extension(self, ext: Extension) -> None:
        if ext.name in (
            "irsim.lib.algorithm._ray_casting_omp",
            "irsim.lib.algorithm._imu_c_ext",
        ):
            self._apply_omp_flags(ext)
        try:
            super().build_extension(ext)
        except Exception:
            # optional=True on the Extension lets the overall build continue
            if getattr(ext, "optional", False):
                return
            raise

    def _apply_omp_flags(self, ext: Extension) -> None:
        if sys.platform == "win32":
            avx2_args = ["/arch:AVX2"] if os.environ.get("IRSIM_ENABLE_AVX2") else []
            ext.extra_compile_args = ["/O2", "/openmp", *avx2_args]
            ext.extra_link_args = []
        elif sys.platform == "darwin":
            root = next((r for r in _LIBOMP_ROOTS if os.path.isdir(r)), None)
            if root:
                ext.extra_compile_args = [
                    "-O3",
                    "-march=native",
                    "-Xpreprocessor",
                    "-fopenmp",
                    f"-I{root}/include",
                ]
                ext.extra_link_args = [
                    f"-L{root}/lib",
                    "-lomp",
                    # Embed rpath so delocate can resolve @rpath/libomp.dylib
                    f"-Wl,-rpath,{root}/lib",
                ]
            else:
                # Build without OpenMP: still correct, just serial
                ext.extra_compile_args = ["-O3", "-march=native"]
                ext.extra_link_args = []
        else:
            # Linux / other POSIX with gcc/clang + libgomp.
            # -mavx2 is explicit so _IRSIM_AVX2 is always defined when the CPU
            # supports it; -march=native subsumes it on capable hosts but the
            # explicit flag ensures the preprocessor guard fires even when a
            # cross-compile or toolchain sets -march to something lower.
            ext.extra_compile_args = ["-O3", "-march=native", "-mavx2", "-fopenmp"]
            ext.extra_link_args = ["-fopenmp"]


setup(
    ext_modules=[
        Extension(
            "irsim.lib.algorithm._ray_casting_omp",
            sources=[
                "irsim/lib/algorithm/_ray_casting_omp_module.c",
                "irsim/lib/algorithm/ray_casting_omp.c",
            ],
            optional=True,
        ),
        Extension(
            "irsim.lib.algorithm._imu_c_ext",
            sources=[
                "irsim/lib/algorithm/_imu_c_ext_module.c",
                "irsim/lib/algorithm/imu_c_ext.c",
            ],
            optional=True,
        ),
    ],
    cmdclass={"build_ext": _OmpBuildExt},
)

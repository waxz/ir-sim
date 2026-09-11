"""Build the optional C+OpenMP ray-casting extension.

setuptools compiles ``_ray_casting_omp`` on supported platforms.
The extension is *optional*: if it cannot be built (missing compiler,
missing OpenMP), the package still installs cleanly and the Python
fallback in ``ray_casting_2d_omp.py`` is used instead.

AVX2 detection
--------------
``_build_cpu_has_avx2()`` probes the build host at setup time and returns
True only when the CPU actually supports AVX2.  The probe is free of
third-party dependencies and never executes foreign binaries:

* Linux   — asks gcc/clang which macros ``-march=native`` would define;
            falls back to parsing ``/proc/cpuinfo``.
* macOS   — ``sysctl hw.optional.avx2_0``; Intel-only (arm64 always False).
* Windows — ``IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE=40)``
            via ctypes (no compiler required).

When AVX2 is detected the matching compiler flag is added automatically:
``-mavx2`` (gcc/clang) or ``/arch:AVX2`` (MSVC).  No environment variable
or manual opt-in is needed.

Platform flags (AVX2-capable host)
-----------------------------------
Linux / POSIX : gcc  -O3 -march=native -mavx2 -fopenmp
macOS Intel   : clang -O3 -march=native -mavx2 -Xpreprocessor -fopenmp ...
macOS arm64   : clang -O3 -march=native         -Xpreprocessor -fopenmp ...
Windows AVX2  : cl.exe /O2 /openmp /arch:AVX2
Windows no AVX2: cl.exe /O2 /openmp
"""

from __future__ import annotations

import os
import subprocess
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


def _build_cpu_has_avx2() -> bool:
    """Return True when the build-host CPU supports AVX2.

    Uses OS/compiler facilities only — no compiled probe binary, no
    third-party packages, no network access.  Returns False on any
    non-x86 architecture (arm64, riscv, …) or on any error.
    """
    import platform

    if platform.machine().lower() not in ("x86_64", "amd64", "i386", "i686"):
        return False  # AVX2 is x86-only

    # ── Windows ──────────────────────────────────────────────────────────────
    if sys.platform == "win32":
        try:
            import ctypes

            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40  (winnt.h)
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        except Exception:
            return False

    # ── macOS ────────────────────────────────────────────────────────────────
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["sysctl", "hw.optional.avx2_0"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Output: "hw.optional.avx2_0: 1"
            return r.returncode == 0 and r.stdout.strip().endswith("1")
        except Exception:
            return False

    # ── Linux / other POSIX ──────────────────────────────────────────────────
    # Primary: ask gcc/clang which macros -march=native would emit.
    # This is accurate even inside containers where /proc/cpuinfo may be
    # filtered, and it confirms the *toolchain* can generate AVX2 code.
    for cc in ("gcc", "cc", "clang"):
        try:
            r = subprocess.run(
                [cc, "-march=native", "-dM", "-E", "-x", "c", "-"],
                input=b"",
                capture_output=True,
                timeout=10,
            )
            if r.returncode == 0 and b"__AVX2__" in r.stdout:
                return True
            if r.returncode == 0:
                # Compiler found but AVX2 not in native macros → CPU lacks it.
                return False
        except FileNotFoundError:
            continue  # try next compiler name
        except Exception:
            break

    # Fallback: /proc/cpuinfo (Linux; always present on bare metal/VMs).
    try:
        with open("/proc/cpuinfo") as fh:
            return "avx2" in fh.read()
    except OSError:
        return False


_AVX2_ON_BUILD_HOST: bool = _build_cpu_has_avx2()


class _OmpBuildExt(build_ext):
    """Inject per-platform OpenMP + optional AVX2 flags at build time."""

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
        avx2 = _AVX2_ON_BUILD_HOST
        if sys.platform == "win32":
            avx2_flag = ["/arch:AVX2"] if avx2 else []
            ext.extra_compile_args = ["/O2", "/openmp", *avx2_flag]
            ext.extra_link_args = []
        elif sys.platform == "darwin":
            avx2_flag = ["-mavx2"] if avx2 else []
            root = next((r for r in _LIBOMP_ROOTS if os.path.isdir(r)), None)
            if root:
                ext.extra_compile_args = [
                    "-O3",
                    "-march=native",
                    *avx2_flag,
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
                ext.extra_compile_args = ["-O3", "-march=native", *avx2_flag]
                ext.extra_link_args = []
        else:
            # Linux / other POSIX with gcc/clang + libgomp
            avx2_flag = ["-mavx2"] if avx2 else []
            ext.extra_compile_args = ["-O3", "-march=native", *avx2_flag, "-fopenmp"]
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

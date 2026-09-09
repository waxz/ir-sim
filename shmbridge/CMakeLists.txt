cmake_minimum_required(VERSION 3.18)
project(shmbridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── pybind11 ──────────────────────────────────────────────────────────────────
# scikit-build-core installs pybind11 as a build dep; find it via CMake config.
find_package(pybind11 CONFIG REQUIRED)

# ── Extension module ──────────────────────────────────────────────────────────
pybind11_add_module(_core csrc/py_bindings.cpp)

target_include_directories(_core PRIVATE include)

# Aggressive inlining for the header-only seqlock paths
target_compile_options(_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -fvisibility=hidden>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# Install into the shmbridge Python package directory
install(TARGETS _core DESTINATION shmbridge)

cmake_minimum_required(VERSION 3.18)
project(shmbridge_irsim_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find shmbridge headers (installed or from source tree).
find_package(shmbridge CONFIG QUIET)

if(NOT shmbridge_FOUND)
    # Fallback: use the include directory from the repo source tree.
    set(SHMBRIDGE_INCLUDE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shmbridge/include"
        CACHE PATH "Path to shmbridge include directory")
    message(STATUS "shmbridge package not found; using source headers: ${SHMBRIDGE_INCLUDE}")
    add_library(shmbridge::shmbridge INTERFACE IMPORTED)
    target_include_directories(shmbridge::shmbridge INTERFACE "${SHMBRIDGE_INCLUDE}")
endif()

add_executable(ctrl_cpp controller_cpp.cpp)
target_link_libraries(ctrl_cpp PRIVATE shmbridge::shmbridge)

# shm_open / shm_unlink live in librt on Linux glibc < 2.17; on newer glibc
# they moved to libc but manylinux CPython builds still skip it, so always
# link explicitly on Linux.
if(UNIX AND NOT APPLE)
    target_link_libraries(ctrl_cpp PRIVATE rt pthread)
endif()

target_compile_options(ctrl_cpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

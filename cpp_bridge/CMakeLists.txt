cmake_minimum_required(VERSION 3.16)
project(irsim_cpp_bridge C CXX)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)

# ── controller binary ────────────────────────────────────────────────────────
add_executable(controller controller.cpp)
target_include_directories(controller PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# shm_open lives in -lrt on Linux, built into libc on macOS
if(UNIX AND NOT APPLE)
    target_link_libraries(controller PRIVATE rt)
endif()

# Optional: size / layout checker (run once to validate struct sizes)
add_executable(check_layout check_layout.c)
target_include_directories(check_layout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

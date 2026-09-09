cmake_minimum_required(VERSION 3.16)
project(shmbridge_controller CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Header-only shmbridge library – just point at include/
add_library(shmbridge INTERFACE)
target_include_directories(shmbridge INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(controller controller.cpp)
target_link_libraries(controller PRIVATE shmbridge)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(controller PRIVATE rt)
endif()

# Detect AArch64 and enable barrier-aware compilation
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    target_compile_definitions(controller PRIVATE SHMBRIDGE_AARCH64=1)
    message(STATUS "shmbridge: AArch64 detected – using dmb ish barriers")
endif()

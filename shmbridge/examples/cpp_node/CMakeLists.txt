cmake_minimum_required(VERSION 3.18)
project(shmbridge_examples LANGUAGES CXX)

# Find shmbridge — either via installed CMake package or as a subdirectory
if(NOT TARGET shmbridge::shmbridge)
    find_package(shmbridge CONFIG QUIET)
endif()
if(NOT TARGET shmbridge::shmbridge)
    # Fall back: source tree layout
    add_subdirectory(../.. shmbridge_build EXCLUDE_FROM_ALL)
endif()

foreach(ex talker listener)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE shmbridge::shmbridge)
    target_compile_options(${ex} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
    )
endforeach()

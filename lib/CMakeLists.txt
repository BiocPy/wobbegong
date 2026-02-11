cmake_minimum_required(VERSION 3.24)

project(wobbegong
    VERSION 1.0.0
    DESCRIPTION "Building the wobbegong cpp library"
    LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    lz4
    GIT_REPOSITORY https://github.com/lz4/lz4.git
    GIT_TAG        v1.10.0
    SOURCE_SUBDIR  build/cmake
)

set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_MakeAvailable(lz4)

find_package(pybind11 CONFIG)

pybind11_add_module(wobbegong
    src/reader.cpp
)

set_property(TARGET wobbegong PROPERTY CXX_STANDARD 17)

find_package(ZLIB REQUIRED)

target_link_libraries(wobbegong PRIVATE lz4_static pybind11::pybind11 ZLIB::ZLIB)

# Ensure the output name matches what Python expects
set_target_properties(wobbegong PROPERTIES
    OUTPUT_NAME lib_wobbegong
    PREFIX ""
)
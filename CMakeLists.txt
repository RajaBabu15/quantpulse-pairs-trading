cmake_minimum_required(VERSION 3.12)
project(QuantPulseCore)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Platform-specific settings
if(APPLE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Xpreprocessor -fopenmp")
    set(CMAKE_SHARED_LINKER_FLAGS "-lomp")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    set(CMAKE_SHARED_LINKER_FLAGS "-fopenmp")
endif()

# Check for AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
    add_definitions(-DHAVE_AVX2)
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(quantpulse_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Source files
set(SOURCES
    csrc/parallel_cv.cpp
    csrc/simd_ops.cpp
    csrc/optimization_cache.cpp
)

# Headers
set(HEADERS
    csrc/quantpulse_core.h
)

# Include directory
include_directories(csrc)

# Create shared library
add_library(quantpulse_core SHARED ${SOURCES} ${HEADERS})

# Link libraries
if(APPLE)
    find_library(OMP_LIB omp HINTS /opt/homebrew/lib /usr/local/lib)
    if(OMP_LIB)
        target_link_libraries(quantpulse_core ${OMP_LIB})
    endif()
else()
    target_link_libraries(quantpulse_core gomp)
endif()

# Set properties
set_target_properties(quantpulse_core PROPERTIES
    VERSION 1.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS quantpulse_core 
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} DESTINATION include)

# Generate pkg-config file
configure_file(
    ${CMAKE_SOURCE_DIR}/quantpulse-core.pc.in
    ${CMAKE_BINARY_DIR}/quantpulse-core.pc
    @ONLY
)

install(FILES ${CMAKE_BINARY_DIR}/quantpulse-core.pc
    DESTINATION lib/pkgconfig
)

# Testing (optional)
option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks (optional)
option(BUILD_BENCHMARKS "Build benchmark suite" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "AVX2 support: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")

"""
Unified setup script for HFT Core trading engine modules.

This builds both the baseline hft_core and the NEON-accelerated hft_core_neon
extensions for ultra-low latency trading on Apple M4.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

# Common compiler flags for maximum performance
common_flags = [
    '-O3',                    # Maximum optimization
    '-ffast-math',           # Aggressive math optimizations  
    '-march=native',         # Optimize for target CPU
    '-mtune=native',         # Tune for target CPU
    '-funroll-loops',        # Loop unrolling
    '-finline-functions',    # Aggressive inlining
    '-fomit-frame-pointer',  # Remove frame pointers
    '-DNDEBUG'               # Disable debug checks
]

# Apple Silicon specific flags
apple_flags = [
    '-mcpu=apple-m4',        # Target Apple M4
]

# NEON SIMD flags (ARM64 has NEON enabled by default)
neon_flags = [
    '-ftree-vectorize',      # Auto-vectorization
    '-fvectorize',           # Enable vectorization
]

# Baseline HFT Core extension
hft_core_ext = Pybind11Extension(
    "hft_core",
    sources=[
        "csrc/hft_core.cpp"
    ],
    include_dirs=[
        "csrc/",
        pybind11.get_cmake_dir() + "/../../../include"
    ],
    cxx_std=17,
    extra_compile_args=common_flags + apple_flags,
    extra_link_args=['-O3'],
    define_macros=[
        ('VERSION_INFO', '"production"'),
        ('HFT_CORE_BASELINE', '1')
    ]
)

# NEON-accelerated HFT Core extension
hft_core_neon_ext = Pybind11Extension(
    "hft_core_neon", 
    sources=[
        "csrc/hft_core_neon.cpp"
    ],
    include_dirs=[
        "csrc/",
        pybind11.get_cmake_dir() + "/../../../include"
    ],
    cxx_std=17,
    extra_compile_args=common_flags + apple_flags + neon_flags,
    extra_link_args=['-O3'],
    define_macros=[
        ('VERSION_INFO', '"neon-accelerated"'),
        ('HFT_CORE_NEON', '1')
    ]
)

setup(
    name="hft-core",
    version="1.0.0",
    author="Ultra-HFT Team",
    author_email="dev@ultra-hft.com", 
    description="Ultra-low latency HFT trading engine core",
    long_description=__doc__,
    long_description_content_type="text/markdown",
    ext_modules=[hft_core_ext, hft_core_neon_ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pybind11>=2.6.0"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Operating System :: MacOS :: MacOS X",
        "Intended Audience :: Developers",
    ],
    keywords="hft trading low-latency finance neon simd apple-silicon",
    zip_safe=False,
)

#!/usr/bin/env python3
"""
Setup script for building C++ extensions for the pairs trading system.
Builds three optimized modules: hft_core, hft_core_neon, and hft_strategies.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11
import platform
import os

def get_compile_args():
    """Get platform-specific compile arguments"""
    args = [
        "-O3",
        "-DWITH_THREAD",
        "-std=c++17",
        "-ffast-math",
        "-march=native",
        "-mtune=native"
    ]
    
    # Platform-specific optimizations
    machine = platform.machine().lower()
    if 'arm' in machine or 'aarch64' in machine:
        # ARM64 specific
        args.extend([
            "-mcpu=native",
            "-D__ARM_NEON__"
        ])
    elif 'x86_64' in machine or 'amd64' in machine:
        # x86_64 specific  
        args.extend([
            "-msse4.2",
            "-mavx2",
            "-mfma"
        ])
    
    return args

def get_link_args():
    """Get platform-specific link arguments"""
    args = []
    
    # Platform-specific linking
    if platform.system() == "Darwin":  # macOS
        args.extend(["-stdlib=libc++"])
    
    return args

# Define extensions
extensions = [
    Pybind11Extension(
        "hft_core",
        sources=["csrc/hft_core.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir(),
            "csrc/"
        ],
        cxx_std=17,
        language="c++",
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args()
    ),
    Pybind11Extension(
        "hft_core_neon", 
        sources=["csrc/hft_core_neon.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir(),
            "csrc/"
        ],
        cxx_std=17,
        language="c++",
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args()
    ),
    Pybind11Extension(
        "hft_strategies",
        sources=["csrc/hft_strategies.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir(), 
            "csrc/"
        ],
        cxx_std=17,
        language="c++",
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args()
    ),
]

setup(
    name="quantpulse-pairs-trading",
    version="1.0.0",
    description="High-performance pairs trading system with C++ acceleration",
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0", 
        "yfinance>=0.1.63",
        "matplotlib>=3.3.0"
    ],
    zip_safe=False,
)

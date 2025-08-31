#!/usr/bin/env python3

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11

ext_modules = [
    Pybind11Extension(
        "nanoext_runloop_neon",
        ["csrc/nanoext_runloop_neon.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir(),
            "csrc",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            "-funroll-loops",
            "-finline-functions", 
            "-DNDEBUG",
            # ARM64/NEON specific optimizations
            "-mcpu=apple-m1",  # Enable Apple Silicon optimizations
            "-mtune=apple-m1",
        ],
    ),
]

setup(
    name="nanoext_runloop_neon",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)

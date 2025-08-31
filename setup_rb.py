from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11
import sys
import os

try:
    import numpy as np
except Exception as e:
    print("NumPy is required to build nano_rb.")
    raise

# Compiler-specific optimization flags
extra_compile_args = [
    "-O3", "-DNDEBUG", "-march=native", "-mtune=native",
    "-fno-math-errno", "-ffast-math", "-fvisibility=hidden", 
    "-std=c++14", "-funroll-loops"
]

extra_link_args = []

if sys.platform == "darwin":
    # macOS specific optimizations
    extra_compile_args.extend(["-mmacosx-version-min=10.15", "-stdlib=libc++"])
    extra_link_args.extend(["-mmacosx-version-min=10.15"])
elif sys.platform.startswith("linux"):
    # Linux specific optimizations
    extra_compile_args.extend(["-fopenmp"])
    extra_link_args.extend(["-fopenmp"])

# Check for AVX2 support (most modern CPUs)
if os.environ.get("ENABLE_AVX2", "1") == "1":
    extra_compile_args.extend(["-mavx2", "-mfma"])

nano_rb_ext = Pybind11Extension(
    "nano_rb",
    sources=["csrc/rb_ext.cpp"],
    include_dirs=[
        pybind11.get_cmake_dir() + "/../../../include",
        np.get_include()
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    cxx_std=14,
)

setup(
    name="nano_rb",
    version="0.1.0",
    description="Ultra-fast ring buffer updates for HFT",
    ext_modules=[nano_rb_ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
    install_requires=["pybind11>=2.6", "numpy"],
)

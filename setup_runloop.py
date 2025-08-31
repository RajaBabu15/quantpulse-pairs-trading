from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11
import sys
import os

try:
    import numpy as np
except Exception as e:
    print("NumPy is required to build nanoext_runloop.")
    raise

# Aggressive optimization flags for maximum performance
extra_compile_args = [
    "-O3", "-DNDEBUG", "-march=native", "-mtune=native",
    "-fno-math-errno", "-ffast-math", "-fvisibility=hidden", 
    "-std=c++17", "-funroll-loops", "-fomit-frame-pointer",
    "-finline-functions", "-fno-signed-zeros", "-frename-registers"
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

# Check for SIMD support
if os.environ.get("ENABLE_AVX2", "0") == "1":
    extra_compile_args.extend(["-mavx2", "-mfma"])

# For ARM64 (Apple Silicon), use NEON optimizations
if sys.platform == "darwin" and os.uname().machine == "arm64":
    extra_compile_args.extend(["-mcpu=native"])

runloop_ext = Pybind11Extension(
    "nanoext_runloop",
    sources=["csrc/nanoext_runloop.cpp"],
    include_dirs=[
        pybind11.get_cmake_dir() + "/../../../include",
        np.get_include()
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    cxx_std=17,
)

setup(
    name="nanoext_runloop",
    version="0.1.0",
    description="Ultra-optimized C++ main loop for HFT with RDTSC timing and XorShift128+ PRNG",
    ext_modules=[runloop_ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
    install_requires=["pybind11>=2.6", "numpy"],
)

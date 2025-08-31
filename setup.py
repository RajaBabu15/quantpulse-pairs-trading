from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11

# Maximum performance compiler flags
optim_flags = [
    '-O3', '-ffast-math', '-march=native', '-mtune=native', 
    '-funroll-loops', '-finline-functions', '-fomit-frame-pointer', 
    '-DNDEBUG', '-mcpu=apple-m4', '-ftree-vectorize'
]

# HFT Core extensions
hft_core_ext = Pybind11Extension(
    "hft_core",
    sources=["csrc/hft_core.cpp"],
    include_dirs=["csrc/"],
    cxx_std=17,
    extra_compile_args=optim_flags,
    extra_link_args=['-O3']
)

hft_core_neon_ext = Pybind11Extension(
    "hft_core_neon", 
    sources=["csrc/hft_core_neon.cpp"],
    include_dirs=["csrc/"],
    cxx_std=17,
    extra_compile_args=optim_flags,
    extra_link_args=['-O3']
)

setup(
    name="hft-core",
    version="1.0.0",
    ext_modules=[hft_core_ext, hft_core_neon_ext],
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy", "pybind11"],
    zip_safe=False
)

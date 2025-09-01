#!/usr/bin/env python3
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11
import platform
__version__ = "2.1.0"
__author__ = "QuantPulse Trading Systems"
__description__ = "High-Performance Pairs Trading with Native C++ Acceleration"
def get_optimization_flags():
    system = platform.system()
    machine = platform.machine().lower()
    flags = ['-O3', '-ffast-math', '-funroll-loops', '-finline-functions', '-fomit-frame-pointer', '-DNDEBUG', '-ftree-vectorize']
    if system == "Darwin":
        if 'arm64' in machine or 'aarch64' in machine:
            flags.extend(['-mcpu=apple-m4', '-march=native'])
        else:
            flags.extend(['-march=native', '-mtune=native'])
    elif system == "Linux":
        flags.extend(['-march=native', '-mtune=native'])
        if 'aarch64' in machine or 'arm64' in machine:
            flags.append('-mcpu=native')
    return flags
optim_flags = get_optimization_flags()
quantpulse_core_ext = Pybind11Extension("quantpulse_core_py", sources=["csrc/quantpulse_core_py.cpp", "csrc/simd_ops.cpp", "csrc/parallel_cv.cpp", "csrc/optimization_cache.cpp"], include_dirs=["csrc/"], cxx_std=17, extra_compile_args=optim_flags, extra_link_args=['-O3'])

setup(name="quantpulse-pairs-trading", version=__version__, author=__author__, description=__description__, long_description=open('README.md', 'r').read() if __file__ == '__main__' else __description__, long_description_content_type='text/markdown', url='https://github.com/quantpulse/pairs-trading', ext_modules=[quantpulse_core_ext], cmdclass={"build_ext": build_ext}, install_requires=["numpy>=1.20.0", "pybind11>=2.10.0", "pandas>=1.3.0", "matplotlib>=3.3.0", "scikit-learn>=1.0.0", "scipy>=1.7.0"], python_requires='>=3.7', classifiers=['Development Status :: 4 - Beta', 'Intended Audience :: Financial and Insurance Industry', 'License :: OSI Approved :: MIT License', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.7', 'Programming Language :: Python :: 3.8', 'Programming Language :: Python :: 3.9', 'Programming Language :: Python :: 3.10', 'Programming Language :: Python :: 3.11', 'Programming Language :: C++', 'Topic :: Office/Business :: Financial :: Investment', 'Topic :: Scientific/Engineering :: Mathematics'], keywords='trading, pairs-trading, quantitative-finance, algorithmic-trading, optimization, cpp-acceleration', zip_safe=False)

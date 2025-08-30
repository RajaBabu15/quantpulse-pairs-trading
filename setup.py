from setuptools import setup, Extension
import sys
import os

try:
    import numpy as np
except Exception as e:
    print("NumPy is required to build nanoext.")
    raise

extra_compile_args = [
    "-O3", "-DNDEBUG", "-march=native", "-mtune=native",
    "-fno-math-errno", "-ffast-math", "-fvisibility=hidden", "-std=c11"
]

if sys.platform == "darwin":
    # Allow use of C11 atomics on macOS clang
    extra_compile_args.append("-mmacosx-version-min=10.15")

nanoext = Extension(
    name="nanoext",
    sources=["csrc/nanoext.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(
    name="nanoext",
    version="0.1.0",
    description="Nanosecond HFT native extensions (SPSC queue + ring-buffer zscore)",
    ext_modules=[nanoext],
)


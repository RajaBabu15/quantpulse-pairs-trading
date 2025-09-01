#!/usr/bin/env python3
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

# Define the C++ extension
quantpulse_core_ext = Extension(
    'quantpulse_core',
    sources=['csrc/quantpulse_core_py.cpp', 
             'csrc/simd_ops.cpp', 
             'csrc/parallel_cv.cpp', 
             'csrc/optimization_cache.cpp'],
    include_dirs=['/usr/local/include', numpy.get_include()],
    library_dirs=['/usr/local/lib'],
    libraries=['openblas'],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3', '-march=native', '-ffast-math']
)

setup(
    name='quantpulse-pairs-trading',
    version='2.0.0',
    description='Professional quantitative trading system with advanced pairs trading, statistical arbitrage, and comprehensive analytics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='QuantPulse Team',
    author_email='team@quantpulse.com',
    url='https://github.com/quantpulse/quantpulse-pairs-trading',
    
    # Package configuration
    packages=find_packages(),
    package_dir={'quantpulse': 'quantpulse'},
    
    # Dependencies
    install_requires=[
        # Core scientific computing
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        
        # Machine learning and statistics
        'statsmodels>=0.13.0',
        'pykalman>=0.9.5',
        'arch>=5.0.0',  # For GARCH models
        
        # Data providers and APIs
        'yfinance>=0.1.70',
        'alpha_vantage>=2.3.0',
        'pandas-datareader>=0.10.0',
        
        # Performance and optimization
        'numba>=0.56.0',
        'cython>=0.29.0',
        'joblib>=1.1.0',
        
        # Visualization
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        
        # Configuration and utilities
        'pyyaml>=6.0',
        'psutil>=5.8.0',
        'tqdm>=4.62.0',
        
        # Database and caching
        'sqlalchemy>=1.4.0',
        'redis>=4.0.0',
        
        # Testing and development
        'pytest>=6.2.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.910',
    ],
    
    # Optional dependencies
    extras_require={
        'full': [
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
            'ipywidgets>=7.6.0',
            'bokeh>=2.4.0',
            'dash>=2.0.0',
            'streamlit>=1.0.0',
        ],
        'dev': [
            'pytest-xdist>=2.4.0',
            'pre-commit>=2.15.0',
            'sphinx>=4.2.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'ml': [
            'tensorflow>=2.7.0',
            'torch>=1.10.0',
            'xgboost>=1.5.0',
            'lightgbm>=3.3.0',
        ]
    },
    
    # C++ extensions
    ext_modules=[quantpulse_core_ext],
    
    # Include additional files
    include_package_data=True,
    package_data={
        'quantpulse': ['config/*.yaml', 'templates/*.html'],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Entry points
    entry_points={
        'console_scripts': [
            'quantpulse=quantpulse.cli:main',
            'quantpulse-backtest=quantpulse.cli:backtest_command',
            'quantpulse-optimize=quantpulse.cli:optimize_command',
            'quantpulse-stream=quantpulse.cli:stream_command',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # Keywords
    keywords='quantitative-finance pairs-trading algorithmic-trading statistical-arbitrage machine-learning risk-management portfolio-optimization backtesting',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/quantpulse/quantpulse-pairs-trading/issues',
        'Source': 'https://github.com/quantpulse/quantpulse-pairs-trading',
        'Documentation': 'https://quantpulse-pairs-trading.readthedocs.io/',
        'Changelog': 'https://github.com/quantpulse/quantpulse-pairs-trading/blob/main/CHANGELOG.md',
    },
)
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

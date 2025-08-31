#!/usr/bin/env python3
"""
QuantPulse Optimized Pairs Trading Setup
========================================

Enhanced setup script for installing the high-performance pairs trading system
with native C++ acceleration, automated dependency management, and environment
configuration.

Features:
- Automatic C++ library compilation
- Dependency resolution and installation
- Environment validation
- Performance optimization settings
- Optional data download and preparation

Author: QuantPulse Trading Systems
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext as _build_ext
import warnings

# Suppress warnings during setup
warnings.filterwarnings('ignore')

# Version and metadata
__version__ = "2.1.0"
__author__ = "QuantPulse Trading Systems"
__description__ = "High-Performance Pairs Trading with Native C++ Acceleration"

class CustomBuildExt(_build_ext):
    """Custom build extension for compiling C++ native libraries."""
    
    def run(self):
        """Override the build process to compile native libraries first."""
        print("Building native C++ acceleration libraries...")
        
        # Try to build the native library
        success = self.build_native_library()
        
        if success:
            print("✓ Native C++ libraries built successfully")
        else:
            print("⚠ Native library build failed, Python fallback will be used")
        
        # Continue with standard Python extension building
        try:
            super().run()
        except Exception as e:
            print(f"Standard extension build failed: {e}")
    
    def build_native_library(self):
        """Build the native C++ library using available tools."""
        try:
            # Check if build script exists and is executable
            build_script = Path("build.sh")
            if build_script.exists():
                print("Using build.sh for native compilation...")
                result = subprocess.run(["bash", str(build_script)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return True
                else:
                    print(f"Build script failed: {result.stderr}")
            
            # Fallback to manual compilation
            return self.manual_build()
            
        except Exception as e:
            print(f"Native build failed: {e}")
            return False
    
    def manual_build(self):
        """Manual compilation fallback."""
        try:
            # Detect compiler
            compiler = None
            for cmd in ['g++', 'clang++', 'c++']:
                if shutil.which(cmd):
                    compiler = cmd
                    break
            
            if not compiler:
                print("No suitable C++ compiler found")
                return False
            
            print(f"Using compiler: {compiler}")
            
            # Source files
            source_dir = Path("csrc")
            if not source_dir.exists():
                print("Source directory 'csrc' not found")
                return False
            
            source_files = [
                "csrc/parallel_cv.cpp",
                "csrc/simd_ops.cpp", 
                "csrc/optimization_cache.cpp"
            ]
            
            # Check all source files exist
            missing_files = [src for src in source_files if not Path(src).exists()]
            if missing_files:
                print(f"Missing source files: {missing_files}")
                return False
            
            # Compile command
            flags = ["-std=c++17", "-O3", "-fPIC", "-shared", "-march=native"]
            
            # Platform-specific adjustments
            if platform.system() == "Darwin":
                flags.extend(["-Xpreprocessor", "-fopenmp"])
                output_name = "quantpulse_core.dylib"
                link_flags = ["-lomp"]
            elif platform.system() == "Windows":
                output_name = "quantpulse_core.dll"
                link_flags = ["-fopenmp"]
            else:
                flags.append("-fopenmp")
                output_name = "quantpulse_core.so"
                link_flags = ["-fopenmp"]
            
            # Try to add AVX2 support
            try:
                test_cmd = [compiler, "-mavx2", "-x", "c++", "-E", "-"]
                subprocess.run(test_cmd, input="", text=True, 
                             capture_output=True, check=True)
                flags.extend(["-mavx2", "-DHAVE_AVX2"])
                print("AVX2 support enabled")
            except:
                print("AVX2 not available")
            
            # Build command
            cmd = [compiler] + flags + ["-I", "csrc"] + source_files + \
                  ["-o", output_name] + link_flags
            
            print(f"Compile command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully compiled {output_name}")
                return True
            else:
                print(f"Compilation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Manual build failed: {e}")
            return False

def check_dependencies():
    """Check and install required dependencies."""
    required_packages = [
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'optuna>=2.10.0',
        'yfinance>=0.1.70',
        'requests>=2.25.0',
        'tqdm>=4.60.0'
    ]
    
    print("Checking dependencies...")
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        try:
            __import__(package_name.replace('-', '_'))
            print(f"✓ {package_name} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package_name} is missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✓ All dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False
    
    return True

def validate_environment():
    """Validate the Python environment and system requirements."""
    print("Validating environment...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print(f"✗ Python 3.7+ required, found {sys.version}")
        return False
    else:
        print(f"✓ Python {sys.version.split()[0]} is supported")
    
    # Check available memory (rough estimate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            print(f"⚠ Low memory detected ({memory_gb:.1f}GB), performance may be limited")
        else:
            print(f"✓ Sufficient memory available ({memory_gb:.1f}GB)")
    except ImportError:
        print("⚠ Cannot check memory (psutil not available)")
    
    # Check CPU cores
    import multiprocessing
    cores = multiprocessing.cpu_count()
    print(f"✓ {cores} CPU cores detected")
    
    # Check platform
    system = platform.system()
    machine = platform.machine()
    print(f"✓ Platform: {system} {machine}")
    
    return True

def setup_data_directory():
    """Set up data directory structure."""
    directories = [
        "data",          # For CSV files and data
        "performance",   # For JSON performance reports
        "static",        # For images and charts
        "data_cache",    # For cached data
        "logs",          # For log files
        "build"          # For compiled libraries
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def create_example_script():
    """Create a native-only example usage script."""
    example_content = '''#!/usr/bin/env python3
"""
QuantPulse Native Optimization Example
=====================================

Demonstrates native-only ElasticNet+KL+RMSprop optimization using the
QuantPulse native library.
"""

import numpy as np
import pandas as pd
from quantpulse_native import (
    is_native_available,
    create_native_elasticnet_optimizer,
    NativeElasticNetKLPortfolioOptimizer,
    generate_native_performance_report,
)


def make_synthetic_pairs(pairs, n_days=1000, seed=42):
    """Create synthetic price data for the given pairs."""
    np.random.seed(seed)
    data = {}
    for s1, s2 in pairs:
        r1 = np.random.normal(0.0005, 0.02, n_days)
        r2 = np.random.normal(0.0005, 0.02, n_days)
        p1 = 100 * np.cumprod(1 + r1)
        p2 = 100 * np.cumprod(1 + r2)
        df = pd.DataFrame({'price1': p1, 'price2': p2})
        data[f"{s1}-{s2}"] = df
    return data


def try_fetch_pairs(pairs, start='2023-01-01', end='2023-12-31'):
    """Try fetching data via run.PairsTrader; return {} if unavailable."""
    try:
        from run import PairsTrader
    except Exception:
        return {}
    data = {}
    for s1, s2 in pairs:
        try:
            df = PairsTrader(s1, s2).get_data(start, end)
            data[f"{s1}-{s2}"] = df
        except Exception:
            continue
    return data


def main():
    print("QuantPulse Native Optimization Example")
    print("=" * 50)
    if not is_native_available():
        print("Native library not available. Please build the native library first (./build.sh).")
        return

    # Define pairs to optimize
    pairs = [("XOM", "CVX"), ("KO", "PEP")]

    # Attempt to fetch real data; fall back to synthetic for any missing pairs
    pairs_data = try_fetch_pairs(pairs)
    if len(pairs_data) < len(pairs):
        missing = [p for p in pairs if f"{p[0]}-{p[1]}" not in pairs_data]
        if missing:
            print("Using synthetic data for:", missing)
            synthetic = make_synthetic_pairs(missing)
            pairs_data.update(synthetic)

    # Native portfolio optimization
    portfolio = NativeElasticNetKLPortfolioOptimizer(l1_ratio=0.7, alpha=0.02, kl_weight=0.15)
    results = portfolio.optimize_all(pairs_data, n_splits=3, max_iterations=15)

    # Generate and save report
    report = generate_native_performance_report(results, 'native_optimization_report.json')

    print("\nPortfolio Summary")
    print("=" * 50)
    summary = report.get('summary', {})
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nDone. Report saved to performance/native_optimization_report.json")


if __name__ == "__main__":
    main()
'''

    example_file = Path("example_usage.py")
    if not example_file.exists():
        with open(example_file, 'w') as f:
            f.write(example_content)
        os.chmod(example_file, 0o755)
        print("✓ Created native example usage script: example_usage.py")

def main_setup():
    """Main setup function."""
    print("QuantPulse Optimized Pairs Trading Setup")
    print("=" * 50)
    
    # Validate environment
    if not validate_environment():
        print("✗ Environment validation failed")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("✗ Dependency check failed")
        return False
    
    # Set up directories and files
    setup_data_directory()
    create_example_script()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: ./build.sh (to compile native libraries)")
    print("2. Test: python3 example_usage.py")
    print("3. Run native optimization example: python3 example_usage.py")
    
    return True

# Setup configuration
setup_config = {
    'name': 'quantpulse-pairs-trading',
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'long_description': open('README.md', 'r').read() if os.path.exists('README.md') else __description__,
    'long_description_content_type': 'text/markdown',
    'url': 'https://github.com/quantpulse/pairs-trading',
    'packages': [],
    'py_modules': [
        'quantpulse_native',
        'run',
        'setup'
    ],
    'install_requires': [
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'optuna>=2.10.0',
        'yfinance>=0.1.70',
        'requests>=2.25.0',
        'tqdm>=4.60.0'
    ],
    'python_requires': '>=3.7',
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    'keywords': 'trading, pairs-trading, quantitative-finance, algorithmic-trading, optimization, cpp-acceleration',
    'cmdclass': {'build_ext': CustomBuildExt},
}

if __name__ == '__main__':
    # If run directly, perform setup
    if len(sys.argv) == 1 or sys.argv[1] not in ['build', 'install', 'develop', 'build_ext']:
        main_setup()
    else:
        # Standard setuptools installation
        setup(**setup_config)

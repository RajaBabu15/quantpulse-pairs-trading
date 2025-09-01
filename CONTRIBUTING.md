# Contributing to QuantPulse Pairs Trading System

Thank you for your interest in contributing to QuantPulse! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** thoroughly
6. **Submit** a pull request

## 🛠️ Development Setup

### Prerequisites

```bash
# macOS
brew install cmake python3 libomp

# Ubuntu/Debian  
sudo apt-get install cmake python3-dev libomp-dev build-essential

# Windows (using vcpkg)
vcpkg install pybind11 openmp
```

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/quantpulse-pairs-trading.git
cd quantpulse-pairs-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Build C++ extensions
python setup.py build_ext --inplace --debug

# Verify installation
python -c "import native_interface; print('✅ Setup successful!')"
```

## 📋 Development Guidelines

### Code Style

#### Python Code
- **Formatter**: Black with 88-character line length
- **Import Sorting**: isort with profile "black"
- **Type Hints**: All public functions must have type annotations
- **Docstrings**: Google-style docstrings for all functions

```python
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: Array of portfolio returns
        risk_free_rate: Annual risk-free rate (default 0.02)
        
    Returns:
        Sharpe ratio as a float
        
    Raises:
        ValueError: If returns array is empty
    """
    pass
```

#### C++ Code
- **Standard**: C++17 with modern practices
- **Style**: Google C++ Style Guide
- **Memory**: RAII, smart pointers, no raw `new/delete`
- **Performance**: ARM64 SIMD where applicable

```cpp
// Good: Modern C++ with RAII
class BacktestEngine {
public:
    explicit BacktestEngine(const TradingParameters& params) 
        : params_(params) {}
        
private:
    TradingParameters params_;
    std::unique_ptr<Cache> cache_;
};
```

### Testing Requirements

All contributions must include appropriate tests:

#### Python Tests
```bash
# Run full test suite
python -m pytest tests/ -v --cov=.

# Run specific test file
python -m pytest tests/test_native_interface.py -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

#### C++ Tests (if applicable)
```bash
# Build and run C++ tests
mkdir build && cd build
cmake ..
make test
```

#### Performance Tests
```bash
# Run benchmark suite
python benchmarks/performance_test.py

# Profile specific functions
python -m cProfile -s tottime your_script.py
```

## 🐛 Bug Reports

When reporting bugs, please include:

### Bug Report Template
```markdown
**Environment:**
- OS: [e.g., macOS 13.0, Ubuntu 22.04]
- Python Version: [e.g., 3.11.5]
- QuantPulse Version: [e.g., 2.1.0]
- Hardware: [e.g., Apple M3 Pro, Intel i7]

**Bug Description:**
Clear description of the issue

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Expected vs Actual result

**Error Messages:**
```
Include full stack trace
```

**Code Sample:**
```python
# Minimal reproducible example
```
```

## ✨ Feature Requests

### Feature Request Template
```markdown
**Feature Summary:**
Brief description of the proposed feature

**Use Case:**
Explain the problem this feature would solve

**Proposed Implementation:**
Technical approach (if you have ideas)

**Acceptance Criteria:**
- [ ] Criterion 1
- [ ] Criterion 2

**Performance Impact:**
Expected impact on system performance
```

## 📝 Pull Request Process

### Before Submitting

1. **Code Quality Checks**
```bash
# Format code
black .
isort .

# Lint code
flake8 .
pylint quantpulse/

# Type checking
mypy .
```

2. **Run Tests**
```bash
# Full test suite
python -m pytest tests/ -v --cov=.

# Performance regression tests
python benchmarks/regression_test.py
```

3. **Update Documentation**
- Update docstrings for new/modified functions
- Add examples to README if needed
- Update CHANGELOG.md

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass  
- [ ] Performance tests pass
- [ ] Manual testing completed

## Performance Impact
- Benchmark results (if applicable)
- Memory usage impact
- Breaking changes (if any)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No compiler warnings
```

## 🏗️ Architecture Guidelines

### Adding New Features

1. **Python Interface**: Start with Python API design
2. **C++ Implementation**: Implement performance-critical parts in C++
3. **Testing**: Add comprehensive tests
4. **Documentation**: Update all relevant docs

### Performance Considerations

- **Profiling**: Always profile before optimizing
- **SIMD**: Use ARM64 NEON for vectorizable operations
- **Memory**: Minimize allocations in hot paths
- **Caching**: Leverage existing cache infrastructure

### Code Organization

```
quantpulse-pairs-trading/
├── 🐍 Python Frontend
│   ├── __init__.py           # Package initialization
│   ├── chart_generator.py    # Visualization layer
│   ├── portfolio_manager.py  # Business logic
│   └── native_interface.py   # C++ interface
├── ⚡ C++ Backend  
│   ├── include/              # Header files
│   ├── src/                  # Implementation
│   └── bindings/             # Python bindings
├── 🧪 Tests
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── performance/          # Performance tests
└── 📚 Documentation
    ├── api/                  # API documentation
    ├── examples/             # Usage examples
    └── architecture/         # Design docs
```

## 🚦 Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
1. Update version numbers in `setup.py`
2. Update `CHANGELOG.md` with new features
3. Run full test suite on multiple platforms
4. Update documentation
5. Create GitHub release with compiled binaries

## 🤝 Community Guidelines

### Code of Conduct
- **Be respectful** and inclusive
- **Constructive feedback** only
- **Focus on the code**, not the person
- **Help others learn** and grow

### Communication
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

## 🎯 Areas for Contribution

### High Priority
- **Performance Optimization**: SIMD improvements, memory efficiency
- **Testing**: Increase coverage, add edge cases
- **Documentation**: Examples, tutorials, API docs
- **Platform Support**: Windows compatibility, additional ARM variants

### Medium Priority  
- **New Features**: Additional trading strategies, risk metrics
- **Visualization**: Enhanced charting capabilities
- **Data Sources**: Additional market data providers
- **Cloud Integration**: Docker, Kubernetes deployment

### Good First Issues
Look for issues labeled `good-first-issue` and `help-wanted` on GitHub.

## 📚 Resources

- **Python Style**: [PEP 8](https://pep8.org/)
- **C++ Style**: [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- **Testing**: [pytest Documentation](https://pytest.org/)
- **PyBind11**: [PyBind11 Documentation](https://pybind11.readthedocs.io/)
- **ARM64**: [ARM Neon Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

## 💬 Getting Help

- **Documentation**: Check the README and docs/ folder first
- **GitHub Issues**: Search existing issues before creating new ones
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: Reach out to maintainers for sensitive issues

---

**Thank you for contributing to QuantPulse! 🚀**

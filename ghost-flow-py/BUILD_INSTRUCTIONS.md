# Building GhostFlow Python Bindings

## Prerequisites

1. **Rust** (already installed)
2. **Python 3.8+**
3. **maturin** (Python build tool)

## Setup

### Install maturin

```bash
pip install maturin
```

### Build for Development

```bash
cd ghost-flow-py

# Build and install in development mode
maturin develop

# Or with release optimizations
maturin develop --release
```

### Build Wheel for Distribution

```bash
# Build wheel
maturin build --release

# Wheels will be in target/wheels/
```

### Test the Package

```bash
# After maturin develop
python examples/basic_usage.py
```

## Publishing to PyPI

### Prerequisites

1. Create account on https://pypi.org
2. Get API token from https://pypi.org/manage/account/token/

### Publish

```bash
# Build wheels for multiple Python versions
maturin build --release --strip

# Publish to PyPI
maturin publish --username __token__ --password YOUR_PYPI_TOKEN
```

Or use GitHub Actions for automated builds (see .github/workflows/python.yml)

## Performance Notes

- **Development builds** (`maturin develop`): Fast compilation, slower runtime
- **Release builds** (`maturin develop --release`): Slower compilation, maximum performance
- **Always use release builds for benchmarks!**

## Troubleshooting

### "maturin not found"
```bash
pip install --upgrade maturin
```

### "Python.h not found"
Install Python development headers:
- **Windows**: Included with Python installer
- **Ubuntu**: `sudo apt install python3-dev`
- **macOS**: `brew install python`

### Build fails
```bash
# Clean and rebuild
cargo clean
maturin develop --release
```

## Cross-Platform Builds

Maturin can build wheels for multiple platforms:

```bash
# Build for current platform
maturin build --release

# Build for specific Python versions
maturin build --release --interpreter python3.8 python3.9 python3.10 python3.11 python3.12
```

## CI/CD Integration

See `.github/workflows/python.yml` for automated wheel building on:
- Linux (x86_64, aarch64)
- Windows (x86_64)
- macOS (x86_64, aarch64)

Wheels are automatically uploaded to PyPI on release tags.

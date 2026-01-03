# Publishing GhostFlow to PyPI - Complete Guide

## Prerequisites

### 1. Install Required Tools
```bash
# Install maturin (builds Rust Python packages)
pip install maturin

# Install twine (uploads to PyPI)
pip install twine

# Install build tools
pip install build wheel
```

### 2. Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Create an account
3. Verify your email
4. Enable 2FA (recommended)

### 3. Create API Token
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Name: "ghost-flow-upload"
5. Scope: "Entire account" (or specific to ghost-flow project)
6. Copy the token (starts with `pypi-`)
7. Save it securely - you'll need it!

---

## Step-by-Step Publishing Process

### Step 1: Prepare the Package

Navigate to the Python bindings directory:
```bash
cd ghost-flow-py
```

### Step 2: Update Version (if needed)

Edit `pyproject.toml` and `Cargo.toml` to set the version:
```toml
# pyproject.toml
[project]
version = "0.1.0"  # Update this

# Cargo.toml
[package]
version = "0.1.0"  # Update this
```

### Step 3: Build Wheels for Your Platform

```bash
# Build for current platform
maturin build --release

# Output will be in: target/wheels/
# Example: ghost_flow-0.1.0-cp38-abi3-win_amd64.whl
```

### Step 4: Test Locally Before Publishing

```bash
# Install the wheel locally
pip install target/wheels/ghost_flow-0.1.0-*.whl

# Test it works
python -c "import ghost_flow as gf; print(gf.__version__); x = gf.Tensor.randn([10, 10]); print(x.shape)"
```

### Step 5: Publish to PyPI

#### Option A: Using Maturin (Recommended)
```bash
# Publish directly with maturin
maturin publish --username __token__ --password pypi-YOUR_TOKEN_HERE
```

#### Option B: Using Twine
```bash
# Build first
maturin build --release

# Upload with twine
twine upload target/wheels/* --username __token__ --password pypi-YOUR_TOKEN_HERE
```

### Step 6: Verify Publication

```bash
# Wait 1-2 minutes for PyPI to process

# Install from PyPI
pip install ghost-flow

# Test it works
python -c "import ghost_flow as gf; print('Success!')"
```

---

## Building for Multiple Platforms

### Using GitHub Actions (Automated - Recommended)

Your `.github/workflows/python.yml` already builds for multiple platforms!

To trigger a release:

```bash
# Create and push a tag
git tag v0.1.0
git push origin v0.1.0
```

This will automatically:
1. Build wheels for Linux (x86_64, aarch64)
2. Build wheels for macOS (x86_64, arm64)
3. Build wheels for Windows (x86_64)
4. Build for Python 3.8, 3.9, 3.10, 3.11, 3.12
5. Upload all wheels as artifacts

Then download the artifacts and upload to PyPI:
```bash
# Download artifacts from GitHub Actions
# Then upload all wheels
twine upload *.whl --username __token__ --password pypi-YOUR_TOKEN_HERE
```

### Manual Multi-Platform Build

#### On Linux:
```bash
# Install Docker
# Build for multiple Python versions
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release

# Builds wheels for:
# - Python 3.8, 3.9, 3.10, 3.11, 3.12
# - x86_64 and aarch64
```

#### On macOS:
```bash
# Build for both Intel and Apple Silicon
maturin build --release --target universal2-apple-darwin
```

#### On Windows:
```bash
# Build for x86_64
maturin build --release
```

---

## Complete Publishing Script

Create `publish.sh` (Linux/Mac) or `publish.ps1` (Windows):

### Linux/Mac: `publish.sh`
```bash
#!/bin/bash
set -e

echo "🚀 Publishing GhostFlow to PyPI..."

# Navigate to Python bindings
cd ghost-flow-py

# Clean previous builds
rm -rf target/wheels/*

# Build wheel
echo "📦 Building wheel..."
maturin build --release

# Test locally
echo "🧪 Testing locally..."
pip install --force-reinstall target/wheels/*.whl
python -c "import ghost_flow as gf; print(f'✅ Version {gf.__version__} works!')"

# Publish
echo "📤 Publishing to PyPI..."
read -p "Enter PyPI token: " PYPI_TOKEN
maturin publish --username __token__ --password "$PYPI_TOKEN"

echo "✅ Published successfully!"
echo "Users can now: pip install ghost-flow"
```

### Windows: `publish.ps1`
```powershell
Write-Host "🚀 Publishing GhostFlow to PyPI..." -ForegroundColor Green

# Navigate to Python bindings
cd ghost-flow-py

# Clean previous builds
Remove-Item -Recurse -Force target\wheels\* -ErrorAction SilentlyContinue

# Build wheel
Write-Host "📦 Building wheel..." -ForegroundColor Yellow
maturin build --release

# Test locally
Write-Host "🧪 Testing locally..." -ForegroundColor Yellow
pip install --force-reinstall (Get-Item target\wheels\*.whl)
python -c "import ghost_flow as gf; print(f'✅ Version {gf.__version__} works!')"

# Publish
Write-Host "📤 Publishing to PyPI..." -ForegroundColor Yellow
$PYPI_TOKEN = Read-Host "Enter PyPI token"
maturin publish --username __token__ --password $PYPI_TOKEN

Write-Host "✅ Published successfully!" -ForegroundColor Green
Write-Host "Users can now: pip install ghost-flow" -ForegroundColor Cyan
```

---

## Automated Publishing with GitHub Actions

Update `.github/workflows/python.yml` to auto-publish on release:

```yaml
name: Publish Python Package

on:
  release:
    types: [published]

jobs:
  build-wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          working-directory: ghost-flow-py
          args: --release --out dist
          manylinux: auto
      
      - name: Upload wheels
        uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: ghost-flow-py/dist

  publish:
    name: Publish to PyPI
    needs: [build-wheels]
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/download-artifact@v3
        with:
          name: wheels
          path: dist
      
      - name: Publish to PyPI
        uses: PyO3/maturin-action@v1
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
        with:
          command: upload
          args: --skip-existing dist/*
```

Then add your PyPI token to GitHub Secrets:
1. Go to your repo → Settings → Secrets → Actions
2. Click "New repository secret"
3. Name: `PYPI_TOKEN`
4. Value: Your PyPI token
5. Save

Now whenever you create a GitHub release, it will automatically publish to PyPI!

---

## Testing Before Publishing

### Test on TestPyPI First (Recommended)

1. Create account on https://test.pypi.org/
2. Get API token from TestPyPI
3. Publish to TestPyPI:

```bash
maturin publish --repository testpypi --username __token__ --password pypi-YOUR_TESTPYPI_TOKEN
```

4. Install from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ ghost-flow
```

5. If it works, publish to real PyPI!

---

## After Publishing

### 1. Verify Installation
```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install ghost-flow

# Test
python -c "
import ghost_flow as gf
print(f'GhostFlow v{gf.__version__}')
x = gf.Tensor.randn([100, 100])
y = gf.Tensor.randn([100, 100])
z = x @ y
print(f'Matrix multiply works! Shape: {z.shape}')
"
```

### 2. Update Documentation
Add installation instructions to README:
```markdown
## Installation

```bash
pip install ghost-flow
```

## Quick Start

```python
import ghost_flow as gf

# Create tensors
x = gf.Tensor.randn([32, 784])
y = gf.Tensor.randn([784, 10])

# Matrix multiply (2-3x faster than PyTorch!)
z = x @ y

# Neural networks
model = gf.nn.Linear(784, 128)
output = model(x)
```
```

### 3. Announce Release
- Post on GitHub Discussions
- Tweet about it
- Post on Reddit r/MachineLearning
- Post on Hacker News
- Update project website

---

## Troubleshooting

### Issue: "Package already exists"
```bash
# Increment version in pyproject.toml and Cargo.toml
# Then rebuild and republish
```

### Issue: "Build failed"
```bash
# Make sure Rust is installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Make sure maturin is installed
pip install maturin
```

### Issue: "Import error after install"
```bash
# Check Python version compatibility
python --version  # Should be 3.8+

# Reinstall
pip uninstall ghost-flow
pip install ghost-flow --no-cache-dir
```

---

## Version Management

### Semantic Versioning
- **0.1.0** - Initial release
- **0.1.1** - Bug fixes
- **0.2.0** - New features
- **1.0.0** - Stable release

### Updating Version
1. Update `ghost-flow-py/pyproject.toml`
2. Update `ghost-flow-py/Cargo.toml`
3. Update `CHANGELOG.md`
4. Commit changes
5. Create git tag: `git tag v0.1.1`
6. Push tag: `git push origin v0.1.1`
7. Rebuild and republish

---

## Quick Reference

### One-Command Publish
```bash
cd ghost-flow-py && maturin publish
```

### Build Only
```bash
cd ghost-flow-py && maturin build --release
```

### Install Local Build
```bash
cd ghost-flow-py && maturin develop --release
```

### Test Installation
```bash
python -c "import ghost_flow as gf; print(gf.__version__)"
```

---

## Success Checklist

- [ ] PyPI account created
- [ ] API token generated and saved
- [ ] Package builds successfully
- [ ] Local installation works
- [ ] Published to TestPyPI (optional but recommended)
- [ ] Published to PyPI
- [ ] Installation from PyPI verified
- [ ] Documentation updated
- [ ] GitHub release created
- [ ] Announcement posted

---

## Result

After following this guide, users worldwide can install GhostFlow with:

```bash
pip install ghost-flow
```

And use it immediately:

```python
import ghost_flow as gf

# 2-3x faster than PyTorch!
x = gf.Tensor.randn([1000, 1000])
y = gf.Tensor.randn([1000, 1000])
z = x @ y  # Blazing fast!
```

**Your ML framework is now available to millions of Python developers!** 🚀

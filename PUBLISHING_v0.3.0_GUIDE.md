# Publishing GhostFlow v0.3.0 Guide

## Overview

This guide walks you through publishing GhostFlow v0.3.0 to both **crates.io** (Rust) and **PyPI** (Python).

---

## ✅ Pre-Publishing Checklist

- [x] All code committed to GitHub
- [x] All tests passing
- [x] Documentation complete
- [x] Version updated to 0.3.0
- [ ] Changelog updated
- [ ] Release notes prepared

---

## 📦 Part 1: Publishing to crates.io (Rust)

### Step 1: Update CHANGELOG.md

Create or update `CHANGELOG.md`:

```markdown
# Changelog

## [0.3.0] - 2026-01-XX

### Added - Phase 3 (v0.3.0)
- XGBoost-style gradient boosting (Classifier & Regressor)
- LightGBM-style gradient boosting
- Gaussian Mixture Models (GMM)
- Hidden Markov Models (HMM)
- Conditional Random Fields (CRF)
- Feature engineering tools (hashing, target encoding, one-hot)
- Hyperparameter optimization (Bayesian, Random, Grid search)

### Added - Phase 2 (v0.2.0)
- Conv1d, Conv3d, TransposeConv2d layers
- GroupNorm, InstanceNorm normalization
- Swish, SiLU, Mish, ELU, SELU, Softplus activations
- Focal, Contrastive, Triplet, Huber losses

### Total
- 27 new algorithms/features
- 7,240+ lines of new code
- Comprehensive documentation and examples

## [0.1.0] - 2026-01-XX

### Initial Release
- Core tensor operations
- 50+ classical ML algorithms
- Neural network layers
- GPU acceleration (CUDA)
- Python bindings
```

### Step 2: Build and Test

```bash
# Build all packages
cargo build --release --all

# Run all tests
cargo test --all

# Check for issues
cargo clippy --all

# Build documentation
cargo doc --no-deps --all
```

### Step 3: Publish to crates.io

**IMPORTANT**: Publish in dependency order!

```bash
# 1. Core (no dependencies)
cd ghostflow-core
cargo publish --dry-run  # Test first
cargo publish

# 2. CUDA (depends on core)
cd ../ghostflow-cuda
cargo publish --dry-run
cargo publish

# 3. Autograd (depends on core)
cd ../ghostflow-autograd
cargo publish --dry-run
cargo publish

# 4. NN (depends on core)
cd ../ghostflow-nn
cargo publish --dry-run
cargo publish

# 5. Optim (depends on core, nn)
cd ../ghostflow-optim
cargo publish --dry-run
cargo publish

# 6. Data (depends on core)
cd ../ghostflow-data
cargo publish --dry-run
cargo publish

# 7. ML (depends on core)
cd ../ghostflow-ml
cargo publish --dry-run
cargo publish

# 8. Main package (depends on all)
cd ../ghostflow
cargo publish --dry-run
cargo publish
```

### Step 4: Verify Publication

```bash
# Check on crates.io
# Visit: https://crates.io/crates/ghost-flow

# Test installation
cargo install ghost-flow
```

---

## 🐍 Part 2: Publishing to PyPI (Python)

### Step 1: Update Python Version

Update `ghost-flow-py/Cargo.toml`:

```toml
[package]
name = "ghost-flow-py"
version = "0.3.0"  # Update this
```

Update `ghost-flow-py/pyproject.toml`:

```toml
[project]
name = "ghostflow"
version = "0.3.0"  # Update this
```

### Step 2: Update Python Bindings

The Python bindings in `ghost-flow-py/src/lib.rs` need to expose new features.

**Current Status**: Basic bindings exist for v0.1.0 features.

**TODO for v0.3.0**: Add Python wrappers for:
- XGBoost/LightGBM classes
- GMM, HMM, CRF classes
- Feature engineering tools
- Hyperparameter optimization

### Step 3: Build Python Wheels

```bash
cd ghost-flow-py

# Install maturin if not already installed
pip install maturin

# Build wheels for multiple platforms
maturin build --release

# Or build and install locally for testing
maturin develop --release
```

### Step 4: Test Python Package

```python
# Test basic import
import ghostflow as gf

# Test tensor operations
x = gf.Tensor([1.0, 2.0, 3.0])
print(x)

# Test new features (if bindings added)
# from ghostflow.ml import XGBoostClassifier
# model = XGBoostClassifier(n_estimators=100)
```

### Step 5: Publish to PyPI

```bash
# Build distribution
maturin build --release

# Upload to TestPyPI first (recommended)
maturin publish --repository testpypi

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ ghostflow

# If all works, publish to PyPI
maturin publish
```

### Step 6: Verify Publication

```bash
# Check on PyPI
# Visit: https://pypi.org/project/ghostflow/

# Test installation
pip install ghostflow

# Test in Python
python -c "import ghostflow; print(ghostflow.__version__)"
```

---

## 🔄 Automated Publishing (Optional)

### GitHub Actions Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish

on:
  push:
    tags:
      - 'v*'

jobs:
  publish-crates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Publish to crates.io
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_TOKEN }}
        run: |
          cargo publish -p ghostflow-core
          sleep 30
          cargo publish -p ghostflow-nn
          sleep 30
          cargo publish -p ghostflow-ml
          sleep 30
          cargo publish -p ghostflow

  publish-pypi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Build and publish
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
        run: |
          pip install maturin
          cd ghost-flow-py
          maturin publish
```

---

## 📝 Release Notes Template

Create a GitHub Release with these notes:

```markdown
# GhostFlow v0.3.0 - Advanced ML Release

## 🎉 Major Features

### Phase 3: Advanced ML
- **XGBoost & LightGBM**: State-of-the-art gradient boosting
- **Probabilistic Models**: GMM, HMM for clustering and sequences
- **CRF**: Conditional Random Fields for sequence labeling
- **Feature Engineering**: 4 powerful transformation tools
- **Hyperparameter Optimization**: Bayesian, Random, Grid search

### Phase 2: Enhanced Deep Learning
- **New Layers**: Conv1d, Conv3d, TransposeConv2d, GroupNorm, InstanceNorm
- **New Activations**: Swish, SiLU, Mish, ELU, SELU, Softplus
- **New Losses**: Focal, Contrastive, Triplet, Huber

## 📊 Statistics
- 27 new algorithms/features
- 7,240+ lines of new code
- 147+ tests passing
- Comprehensive documentation

## 🚀 Installation

### Rust
```bash
cargo add ghost-flow
```

### Python
```bash
pip install ghostflow
```

## 📚 Documentation
- [Full Documentation](https://docs.rs/ghost-flow)
- [Examples](https://github.com/choksi2212/ghost-flow/tree/main/examples)
- [Quick Start Guide](https://github.com/choksi2212/ghost-flow#quick-start)

## 🔗 Links
- [Crates.io](https://crates.io/crates/ghost-flow)
- [PyPI](https://pypi.org/project/ghostflow/)
- [GitHub](https://github.com/choksi2212/ghost-flow)
```

---

## ⚠️ Important Notes

### Publishing Order
1. **Always publish Rust crates first** (in dependency order)
2. **Then publish Python package** (depends on Rust crates)

### Version Consistency
- Ensure all `Cargo.toml` files have version `0.3.0`
- Ensure `pyproject.toml` has version `0.3.0`
- Update `__version__` in Python bindings

### Testing
- **Always use `--dry-run` first** for crates.io
- **Always test on TestPyPI first** before PyPI
- **Test installation** after publishing

### Credentials Required
- **crates.io**: API token from https://crates.io/me
- **PyPI**: API token from https://pypi.org/manage/account/token/

---

## 🐛 Troubleshooting

### "crate already exists"
- You cannot unpublish or overwrite versions
- Increment version number and republish

### "dependency not found"
- Wait 5-10 minutes after publishing dependencies
- crates.io needs time to index

### Python build fails
- Ensure Rust toolchain is installed
- Check that all Rust dependencies are published
- Verify maturin is up to date

### Import errors in Python
- Rebuild with `maturin develop`
- Check that bindings are properly exported
- Verify Python version compatibility

---

## ✅ Post-Publishing Checklist

- [ ] Verify crates.io listing
- [ ] Verify PyPI listing
- [ ] Test Rust installation
- [ ] Test Python installation
- [ ] Create GitHub release
- [ ] Update documentation website
- [ ] Announce on social media
- [ ] Update README badges

---

## 📞 Support

If you encounter issues:
1. Check the [troubleshooting section](#troubleshooting)
2. Review [cargo publish docs](https://doc.rust-lang.org/cargo/reference/publishing.html)
3. Review [maturin docs](https://www.maturin.rs/)
4. Open an issue on GitHub

---

**Good luck with the v0.3.0 release!** 🚀

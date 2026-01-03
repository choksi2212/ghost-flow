# Quick Start: Publish to PyPI in 5 Minutes

## TL;DR - Fastest Way

```bash
# 1. Install maturin
pip install maturin

# 2. Go to Python bindings
cd ghost-flow-py

# 3. Run publish script
./publish.ps1  # Windows
# OR
./publish.sh   # Linux/Mac

# 4. Enter your PyPI token when prompted

# Done! Users can now: pip install ghost-flow
```

---

## Step-by-Step (First Time)

### 1. Create PyPI Account (2 minutes)
1. Go to https://pypi.org/account/register/
2. Sign up with email
3. Verify email

### 2. Get API Token (1 minute)
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Name it "ghost-flow"
5. Copy the token (starts with `pypi-`)
6. **Save it somewhere safe!**

### 3. Install Maturin (30 seconds)
```bash
pip install maturin
```

### 4. Publish (1 minute)
```bash
cd ghost-flow-py

# Windows
.\publish.ps1

# Linux/Mac
chmod +x publish.sh
./publish.sh
```

Enter your token when prompted.

### 5. Verify (30 seconds)
```bash
# Wait 1-2 minutes for PyPI to process

# Install from PyPI
pip install ghost-flow

# Test
python -c "import ghost_flow as gf; print('Success!')"
```

---

## Manual Method (If Scripts Don't Work)

```bash
cd ghost-flow-py

# Build
maturin build --release

# Publish
maturin publish --username __token__ --password pypi-YOUR_TOKEN_HERE
```

---

## What Happens After Publishing?

### Anyone in the world can now:

```bash
pip install ghost-flow
```

### And use it:

```python
import ghost_flow as gf

# Create tensors (2-3x faster than PyTorch!)
x = gf.Tensor.randn([1000, 1000])
y = gf.Tensor.randn([1000, 1000])

# Matrix multiply
z = x @ y

# Neural networks
model = gf.nn.Linear(1000, 500)
output = model(x)

print(f"GhostFlow v{gf.__version__} - Blazing fast! 🚀")
```

---

## Updating Version

When you want to release a new version:

1. Update version in `ghost-flow-py/pyproject.toml`:
```toml
[project]
version = "0.1.1"  # Increment this
```

2. Update version in `ghost-flow-py/Cargo.toml`:
```toml
[package]
version = "0.1.1"  # Increment this
```

3. Run publish script again:
```bash
./publish.ps1  # or ./publish.sh
```

---

## Troubleshooting

### "Package already exists"
- Increment version in `pyproject.toml` and `Cargo.toml`
- Rebuild and republish

### "Build failed"
- Make sure Rust is installed: https://rustup.rs/
- Make sure maturin is installed: `pip install maturin`

### "Import error"
- Check Python version: `python --version` (need 3.8+)
- Reinstall: `pip install ghost-flow --no-cache-dir`

---

## Test on TestPyPI First (Optional but Recommended)

```bash
# 1. Create account on https://test.pypi.org/

# 2. Get token from TestPyPI

# 3. Publish to TestPyPI
maturin publish --repository testpypi --username __token__ --password YOUR_TESTPYPI_TOKEN

# 4. Test install
pip install --index-url https://test.pypi.org/simple/ ghost-flow

# 5. If it works, publish to real PyPI!
```

---

## After Publishing

### Update README
Add installation instructions:
```markdown
## Installation

```bash
pip install ghost-flow
```
```

### Announce
- GitHub Discussions
- Twitter/X
- Reddit r/MachineLearning
- Hacker News

### Monitor
- Check PyPI page: https://pypi.org/project/ghost-flow/
- Monitor downloads
- Respond to issues

---

## Success!

Once published, your package will be available at:
- **PyPI:** https://pypi.org/project/ghost-flow/
- **Install:** `pip install ghost-flow`
- **Docs:** Auto-generated from your README

**Millions of Python developers can now use GhostFlow!** 🎉

# Testing on TestPyPI - Step by Step Guide

## What is TestPyPI?

TestPyPI is a separate instance of PyPI for testing. It's the safe way to:
- Test your package metadata
- Test installation process
- Make sure everything works before going live
- Practice without consequences

**Important**: TestPyPI and PyPI are completely separate! You need separate accounts and tokens.

---

## Step 1: Create TestPyPI Account

1. Go to: https://test.pypi.org/account/register/
2. Fill in your details (can use same email as PyPI)
3. Click "Create account"
4. Check your email and verify

---

## Step 2: Create TestPyPI API Token

1. Go to: https://test.pypi.org/manage/account/
2. Scroll down to "API tokens" section
3. Click "Add API token"
4. Fill in:
   - Token name: `ghost-flow-test`
   - Scope: `Entire account`
5. Click "Add token"
6. **COPY THE TOKEN NOW** (you'll only see it once!)
   - It starts with `pypi-`
   - Save it somewhere safe

---

## Step 3: Publish to TestPyPI

Open PowerShell in the GHOSTFLOW directory and run:

```powershell
cd ghost-flow-py
python -m maturin publish --repository testpypi
```

When prompted:
- **Username**: Enter `__token__` (exactly like that, with double underscores)
- **Password**: Paste your TestPyPI token (the one starting with `pypi-`)

Or use the automated script:
```powershell
.\ghost-flow-py\publish_test.ps1
```

---

## Step 4: Test Installation from TestPyPI

### Create a fresh test environment:
```powershell
python -m venv test_env
test_env\Scripts\activate
```

### Install from TestPyPI:
```powershell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ghost-flow
```

**Note**: We use `--extra-index-url https://pypi.org/simple/` because dependencies (like numpy) are on real PyPI, not TestPyPI.

### Test it works:
```powershell
python -c "import ghost_flow as gf; print(f'GhostFlow v{gf.__version__}'); x = gf.Tensor.randn([10, 10]); print(f'Shape: {x.shape}')"
```

### Run a more complete test:
```python
python
>>> import ghost_flow as gf
>>> 
>>> # Test tensor creation
>>> x = gf.Tensor.randn([100, 100])
>>> print(f"Created tensor: {x.shape}")
>>> 
>>> # Test operations
>>> y = gf.Tensor.randn([100, 100])
>>> z = x @ y  # Matrix multiply
>>> print(f"Matrix multiply works: {z.shape}")
>>> 
>>> # Test neural network
>>> model = gf.nn.Linear(100, 50)
>>> output = model(x)
>>> print(f"Neural network works: {output.shape}")
>>> 
>>> print("✅ All tests passed!")
```

---

## Step 5: If Everything Works - Publish to Real PyPI

If all tests pass, you're ready for the real thing!

```powershell
# Deactivate test environment
deactivate

# Go back to ghost-flow-py directory
cd ghost-flow-py

# Publish to real PyPI
python -m maturin publish
```

When prompted:
- **Username**: `__token__`
- **Password**: Your **real PyPI token** (not the TestPyPI one!)

---

## Troubleshooting

### Issue: "Package already exists"
This can happen if you've already published to TestPyPI. Options:
1. Increment version in `pyproject.toml` (e.g., 0.1.0 → 0.1.1)
2. Or just skip to publishing to real PyPI if tests passed

### Issue: "Invalid credentials"
- Make sure you're using the TestPyPI token (from test.pypi.org)
- Username must be exactly `__token__` (with double underscores)
- Token should start with `pypi-`

### Issue: "Installation fails from TestPyPI"
- Make sure you included `--extra-index-url https://pypi.org/simple/`
- This allows pip to get dependencies from real PyPI

### Issue: "Import error after install"
- Check Python version: `python --version` (should be 3.8+)
- Try reinstalling: `pip uninstall ghost-flow && pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ghost-flow`

---

## Quick Command Reference

### Publish to TestPyPI:
```powershell
cd ghost-flow-py
python -m maturin publish --repository testpypi
```

### Install from TestPyPI:
```powershell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ghost-flow
```

### Publish to Real PyPI:
```powershell
cd ghost-flow-py
python -m maturin publish
```

### Install from Real PyPI:
```powershell
pip install ghost-flow
```

---

## What to Check Before Publishing to Real PyPI

- ✅ Package installs successfully from TestPyPI
- ✅ Import works: `import ghost_flow as gf`
- ✅ Version number is correct: `gf.__version__`
- ✅ Basic operations work (tensor creation, math ops)
- ✅ Neural network modules work
- ✅ No import errors or missing dependencies
- ✅ Package metadata looks good on TestPyPI page

---

## Next Steps After Real PyPI

Once published to real PyPI:

1. **Test real installation**:
   ```powershell
   pip install ghost-flow
   python -c "import ghost_flow as gf; print('Success!')"
   ```

2. **Update documentation** with installation instructions

3. **Create GitHub release** linking to PyPI package

4. **Announce it**:
   - GitHub Discussions
   - Reddit r/MachineLearning
   - Twitter/X
   - Hacker News

---

## Summary

**TestPyPI Flow**:
1. Create TestPyPI account → Get token
2. Publish: `python -m maturin publish --repository testpypi`
3. Test install: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ghost-flow`
4. Verify it works

**Real PyPI Flow**:
1. Use your real PyPI token
2. Publish: `python -m maturin publish`
3. Install: `pip install ghost-flow`
4. Celebrate! 🎉

---

**You're ready to go! Start with TestPyPI to be safe.** 🚀

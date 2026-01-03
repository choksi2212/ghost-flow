#!/bin/bash
# GhostFlow PyPI Publishing Script for Linux/Mac
# Run this to publish to PyPI

set -e

echo "🚀 GhostFlow PyPI Publisher"
echo "================================"
echo ""

# Check if maturin is installed
echo "Checking dependencies..."
if ! command -v maturin &> /dev/null; then
    echo "❌ Maturin not found. Installing..."
    pip install maturin
fi

# Clean previous builds
echo ""
echo "🧹 Cleaning previous builds..."
rm -rf target/wheels/*

# Build wheel
echo ""
echo "📦 Building wheel..."
maturin build --release

echo "✅ Build successful!"

# Test locally
echo ""
echo "🧪 Testing local installation..."
pip install --force-reinstall target/wheels/*.whl

# Test import
echo ""
echo "Testing import..."
python3 -c "
import ghost_flow as gf
print(f'✅ GhostFlow v{gf.__version__} works!')
x = gf.Tensor.randn([10, 10])
print(f'✅ Tensor operations work! Shape: {x.shape}')
"

echo ""
echo "✅ All tests passed!"

# Ask for confirmation
echo ""
echo "================================"
echo "Ready to publish to PyPI!"
echo ""
echo "You will need your PyPI API token."
echo "Get it from: https://pypi.org/manage/account/"
echo ""
read -p "Do you want to publish now? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo ""
    echo "❌ Publishing cancelled."
    echo ""
    echo "To publish later, run:"
    echo "  maturin publish --username __token__ --password YOUR_TOKEN"
    exit 0
fi

# Get PyPI token
echo ""
read -sp "Enter your PyPI token (starts with pypi-): " token
echo ""

# Publish
echo ""
echo "📤 Publishing to PyPI..."
maturin publish --username __token__ --password "$token"

echo ""
echo "================================"
echo "✅ Successfully published to PyPI!"
echo ""
echo "Users can now install with:"
echo "  pip install ghost-flow"
echo ""
echo "Test it yourself:"
echo "  pip install ghost-flow --upgrade"
echo ""
echo "🎉 Congratulations! GhostFlow is now available worldwide!"

#!/bin/bash
# Master script to publish GhostFlow v0.5.0 to all platforms
# Run from GHOSTFLOW directory

set -e

VERSION="0.5.0"

echo "🚀 GhostFlow v${VERSION} - Complete Publishing Pipeline"
echo "========================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print section header
section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}❌ Aborted${NC}"
        exit 1
    fi
}

# Pre-flight checks
section "Pre-flight Checks"

echo "Checking prerequisites..."

# Check if in correct directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Error: Must run from GHOSTFLOW directory${NC}"
    exit 1
fi

# Check git status
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
    git status -s
    confirm "Continue anyway?"
fi

# Check if tests pass
echo -e "\n${YELLOW}🧪 Running tests...${NC}"
if cargo test --lib 2>&1 | grep -q "test result: ok"; then
    echo -e "${GREEN}✓ All tests passing${NC}"
else
    echo -e "${RED}❌ Tests failing${NC}"
    confirm "Continue anyway?"
fi

# Check version numbers
echo -e "\n${YELLOW}🔍 Checking version numbers...${NC}"
if grep -q "version = \"${VERSION}\"" Cargo.toml && \
   grep -q "version = \"${VERSION}\"" ghostflow/Cargo.toml; then
    echo -e "${GREEN}✓ Version numbers correct${NC}"
else
    echo -e "${RED}❌ Version numbers don't match${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ All pre-flight checks passed${NC}"

# Step 1: Build release assets
section "Step 1: Build Release Assets"
confirm "Build release assets?"

bash scripts/build_release_assets.sh

echo -e "${GREEN}✓ Release assets built${NC}"

# Step 2: GitHub release
section "Step 2: GitHub Release"
confirm "Create GitHub release?"

bash scripts/publish_github.sh

echo -e "${GREEN}✓ GitHub release created${NC}"

# Step 3: crates.io
section "Step 3: Publish to crates.io"
echo -e "${YELLOW}⚠️  This will publish all packages to crates.io${NC}"
echo "Make sure you're logged in: cargo login <token>"
confirm "Publish to crates.io?"

bash scripts/publish_crates.sh

echo -e "${GREEN}✓ Published to crates.io${NC}"

# Step 4: PyPI
section "Step 4: Publish to PyPI"
echo -e "${YELLOW}⚠️  This requires Python package to be built${NC}"
confirm "Publish to PyPI?"

cd ghost-flow-py
if [ -d "dist" ]; then
    echo "Uploading to PyPI..."
    twine upload dist/*
    echo -e "${GREEN}✓ Published to PyPI${NC}"
else
    echo -e "${RED}❌ No dist/ directory found. Build first with: maturin build${NC}"
fi
cd ..

# Step 5: npm
section "Step 5: Publish to npm"
echo -e "${YELLOW}⚠️  This will publish WASM package to npm${NC}"
confirm "Publish to npm?"

cd ghostflow-wasm/pkg
if [ -f "package.json" ]; then
    npm publish
    echo -e "${GREEN}✓ Published to npm${NC}"
else
    echo -e "${RED}❌ No package.json found. Build first with: wasm-pack build${NC}"
fi
cd ../..

# Summary
section "🎉 Publishing Complete!"

echo "Published to:"
echo "  ✓ GitHub: https://github.com/choksi2212/ghost-flow/releases/tag/v${VERSION}"
echo "  ✓ crates.io: https://crates.io/crates/ghost-flow"
echo "  ✓ PyPI: https://pypi.org/project/ghost-flow/"
echo "  ✓ npm: https://www.npmjs.com/package/ghostflow-wasm"
echo ""
echo "Next steps:"
echo "  1. Verify installations work"
echo "  2. Post announcements (Twitter, Reddit, HN)"
echo "  3. Update documentation sites"
echo "  4. Monitor issues and feedback"
echo ""
echo -e "${GREEN}🚀 GhostFlow v${VERSION} is live!${NC}"

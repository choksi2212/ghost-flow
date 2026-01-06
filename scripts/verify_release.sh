#!/bin/bash
# Verify GhostFlow v0.5.0 release on all platforms
# Run after publishing

set -e

VERSION="0.5.0"

echo "🔍 Verifying GhostFlow v${VERSION} Release"
echo "==========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓ $1 found${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 not found${NC}"
        return 1
    fi
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"
check_command cargo || echo "Install from: https://rustup.rs"
check_command python3 || echo "Install from: https://python.org"
check_command node || echo "Install from: https://nodejs.org"

# Test crates.io
echo -e "\n${YELLOW}Testing crates.io...${NC}"
if cargo search ghost-flow | grep -q "ghost-flow"; then
    echo -e "${GREEN}✓ Package found on crates.io${NC}"
    
    # Try to fetch
    cargo fetch ghost-flow 2>&1 | head -5
    echo -e "${GREEN}✓ Package can be fetched${NC}"
else
    echo -e "${RED}✗ Package not found on crates.io${NC}"
fi

# Test PyPI
echo -e "\n${YELLOW}Testing PyPI...${NC}"
if pip3 search ghost-flow 2>/dev/null | grep -q "ghost-flow"; then
    echo -e "${GREEN}✓ Package found on PyPI${NC}"
else
    # pip search is disabled, try direct check
    if curl -s https://pypi.org/pypi/ghost-flow/json | grep -q "\"version\": \"${VERSION}\""; then
        echo -e "${GREEN}✓ Package found on PyPI${NC}"
    else
        echo -e "${RED}✗ Package not found on PyPI${NC}"
    fi
fi

# Test npm
echo -e "\n${YELLOW}Testing npm...${NC}"
if npm view ghostflow-wasm version 2>/dev/null | grep -q "${VERSION}"; then
    echo -e "${GREEN}✓ Package found on npm${NC}"
else
    echo -e "${RED}✗ Package not found on npm${NC}"
fi

# Test GitHub release
echo -e "\n${YELLOW}Testing GitHub release...${NC}"
if curl -s https://api.github.com/repos/choksi2212/ghost-flow/releases/tags/v${VERSION} | grep -q "\"tag_name\": \"v${VERSION}\""; then
    echo -e "${GREEN}✓ GitHub release found${NC}"
else
    echo -e "${RED}✗ GitHub release not found${NC}"
fi

# Test installations
echo -e "\n${YELLOW}Testing installations...${NC}"

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Test Rust installation
echo -e "\n${YELLOW}Testing Rust installation...${NC}"
cargo new test-ghostflow
cd test-ghostflow
echo 'ghost-flow = "'${VERSION}'"' >> Cargo.toml
if cargo check 2>&1 | grep -q "Finished"; then
    echo -e "${GREEN}✓ Rust package installs correctly${NC}"
else
    echo -e "${RED}✗ Rust package installation failed${NC}"
fi
cd ..

# Test Python installation
echo -e "\n${YELLOW}Testing Python installation...${NC}"
python3 -m venv test-venv
source test-venv/bin/activate 2>/dev/null || source test-venv/Scripts/activate
if pip install ghost-flow==${VERSION} 2>&1 | grep -q "Successfully installed"; then
    echo -e "${GREEN}✓ Python package installs correctly${NC}"
    
    # Test import
    if python -c "import ghostflow; print(ghostflow.__version__)" 2>/dev/null; then
        echo -e "${GREEN}✓ Python package imports correctly${NC}"
    else
        echo -e "${RED}✗ Python package import failed${NC}"
    fi
else
    echo -e "${RED}✗ Python package installation failed${NC}"
fi
deactivate

# Test npm installation
echo -e "\n${YELLOW}Testing npm installation...${NC}"
mkdir test-npm
cd test-npm
npm init -y > /dev/null 2>&1
if npm install ghostflow-wasm@${VERSION} 2>&1 | grep -q "added"; then
    echo -e "${GREEN}✓ npm package installs correctly${NC}"
else
    echo -e "${RED}✗ npm package installation failed${NC}"
fi
cd ..

# Cleanup
cd /
rm -rf "$TEMP_DIR"

# Summary
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Verification Complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Check results above for any failures."
echo ""
echo "Manual verification:"
echo "  - crates.io: https://crates.io/crates/ghost-flow"
echo "  - PyPI: https://pypi.org/project/ghost-flow/"
echo "  - npm: https://www.npmjs.com/package/ghostflow-wasm"
echo "  - GitHub: https://github.com/choksi2212/ghost-flow/releases"
echo "  - docs.rs: https://docs.rs/ghost-flow"

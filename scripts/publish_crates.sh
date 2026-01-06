#!/bin/bash
# Publish all GhostFlow packages to crates.io
# Run from GHOSTFLOW directory

set -e

echo "🚀 Publishing GhostFlow v0.5.0 to crates.io"
echo "============================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to publish a package
publish_package() {
    local package=$1
    echo -e "\n${YELLOW}📦 Publishing $package...${NC}"
    cd "$package"
    
    # Dry run first
    cargo publish --dry-run
    
    # Actual publish
    cargo publish
    
    cd ..
    echo -e "${GREEN}✓ $package published${NC}"
    
    # Wait for crates.io to index
    echo "⏳ Waiting 30 seconds for crates.io to index..."
    sleep 30
}

# Verify we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Must run from GHOSTFLOW directory"
    exit 1
fi

# Check if logged in
if ! cargo login --help > /dev/null 2>&1; then
    echo "❌ Error: cargo not found"
    exit 1
fi

echo "⚠️  Make sure you're logged in to crates.io:"
echo "   cargo login <your-api-token>"
read -p "Press Enter to continue..."

# Publish in dependency order
publish_package "ghostflow-core"
publish_package "ghostflow-autograd"
publish_package "ghostflow-nn"
publish_package "ghostflow-optim"
publish_package "ghostflow-data"
publish_package "ghostflow-ml"
publish_package "ghostflow-wasm"
publish_package "ghostflow-ffi"
publish_package "ghostflow-serve"
publish_package "ghostflow"

echo -e "\n${GREEN}🎉 All packages published successfully!${NC}"
echo ""
echo "Verify at: https://crates.io/crates/ghost-flow"
echo ""
echo "Test installation:"
echo "  cargo install ghost-flow"

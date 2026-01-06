#!/bin/bash
# Build release assets for GitHub release
# Run from GHOSTFLOW directory

set -e

VERSION="0.5.0"
ASSETS_DIR="release-assets"

echo "🔨 Building Release Assets for v${VERSION}"
echo "==========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create assets directory
mkdir -p "${ASSETS_DIR}"

echo -e "\n${YELLOW}📦 Building FFI library...${NC}"
cd ghostflow-ffi

# Build for current platform
cargo build --release

# Copy library
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp ../target/release/libghostflow_ffi.so "../${ASSETS_DIR}/"
    echo -e "${GREEN}✓ Linux library built${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp ../target/release/libghostflow_ffi.dylib "../${ASSETS_DIR}/"
    echo -e "${GREEN}✓ macOS library built${NC}"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    cp ../target/release/ghostflow_ffi.dll "../${ASSETS_DIR}/"
    echo -e "${GREEN}✓ Windows library built${NC}"
fi

# Copy header file
cp ghostflow.h "../${ASSETS_DIR}/"
echo -e "${GREEN}✓ Header file copied${NC}"

cd ..

echo -e "\n${YELLOW}🌐 Building WASM package...${NC}"
cd ghostflow-wasm

# Install wasm-pack if not present
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build for web
wasm-pack build --target web --release

# Create tarball
tar -czf "../${ASSETS_DIR}/ghostflow-wasm-${VERSION}.tar.gz" pkg/

echo -e "${GREEN}✓ WASM package built${NC}"

cd ..

echo -e "\n${YELLOW}📄 Creating documentation archive...${NC}"
tar -czf "${ASSETS_DIR}/ghostflow-docs-${VERSION}.tar.gz" \
    README.md \
    CHANGELOG.md \
    QUICK_START_GUIDE.md \
    COMPLETE_IMPLEMENTATION_SUMMARY.md \
    V0.5.0_ECOSYSTEM_COMPLETE.md \
    PUBLISHING_v0.5.0_GUIDE.md

echo -e "${GREEN}✓ Documentation archived${NC}"

echo -e "\n${YELLOW}📝 Creating checksums...${NC}"
cd "${ASSETS_DIR}"
sha256sum * > checksums.txt
cd ..

echo -e "${GREEN}✓ Checksums created${NC}"

echo -e "\n${GREEN}🎉 Release assets built successfully!${NC}"
echo ""
echo "Assets location: ${ASSETS_DIR}/"
ls -lh "${ASSETS_DIR}/"
echo ""
echo "Upload these files to the GitHub release"

#!/bin/bash
# Create GitHub release for GhostFlow v0.5.0
# Run from GHOSTFLOW directory

set -e

VERSION="0.5.0"
TAG="v${VERSION}"

echo "🚀 Creating GitHub Release for GhostFlow v${VERSION}"
echo "===================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if git is clean
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Stage and commit all changes
echo -e "\n${YELLOW}📝 Committing changes...${NC}"
git add .
git commit -m "Release v${VERSION}: Ecosystem Features

- Add WebAssembly support (ghostflow-wasm)
- Add C FFI bindings (ghostflow-ffi)
- Add REST API server (ghostflow-serve)
- Add ONNX export/import
- Add inference optimization
- Add performance profiling
- Fix all ML tests (250+ passing)
- Update documentation

See CHANGELOG.md for full details."

echo -e "${GREEN}✓ Changes committed${NC}"

# Push to GitHub
echo -e "\n${YELLOW}⬆️  Pushing to GitHub...${NC}"
git push origin main
echo -e "${GREEN}✓ Pushed to main${NC}"

# Create and push tag
echo -e "\n${YELLOW}🏷️  Creating tag ${TAG}...${NC}"
git tag -a "${TAG}" -m "GhostFlow v${VERSION} - Ecosystem Features

Major Features:
- WebAssembly support for browser deployment
- C FFI bindings for multi-language integration
- REST API server for model serving
- ONNX export/import
- Inference optimization with operator fusion
- Performance profiling and optimization
- 250+ tests passing

Platforms: Web, Mobile, Desktop, Server, Embedded
Languages: Rust, JavaScript, C, C++, Python, Go, Java, Ruby

Full changelog: https://github.com/choksi2212/ghost-flow/blob/main/CHANGELOG.md"

git push origin "${TAG}"
echo -e "${GREEN}✓ Tag created and pushed${NC}"

echo -e "\n${GREEN}🎉 GitHub release prepared!${NC}"
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/choksi2212/ghost-flow/releases/new"
echo "2. Select tag: ${TAG}"
echo "3. Title: v${VERSION} - Ecosystem Features 🌐"
echo "4. Copy description from PUBLISHING_v0.5.0_GUIDE.md"
echo "5. Attach release assets (FFI libraries)"
echo "6. Click 'Publish release'"
echo ""
echo "Or use GitHub CLI:"
echo "  gh release create ${TAG} --title 'v${VERSION} - Ecosystem Features' --notes-file CHANGELOG.md"

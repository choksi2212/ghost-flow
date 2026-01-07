# GhostFlow WASM Publishing Status

## Current Status

The WASM package (`ghostflow-wasm`) is **not yet published** to npm for v0.5.0.

## Why?

The WASM bindings depend on the v0.5.0 API of the core GhostFlow packages, but only v0.1.0 is available on crates.io. This creates a dependency mismatch.

## Options to Publish WASM

### Option 1: Publish Core Packages First (Recommended)
1. Publish all core packages (ghostflow-core, ghostflow-nn, ghostflow-ml) to crates.io at v0.5.0
2. Update ghostflow-wasm to use these published versions
3. Build with `wasm-pack build --target web --release`
4. Publish to npm with `wasm-pack publish`

### Option 2: Standalone Build
1. Keep path dependencies in Cargo.toml
2. Build with `wasm-pack build --target web --release`
3. Manually publish the generated `pkg/` directory to npm

### Option 3: Separate Repository
Create a separate npm package that wraps the Rust WASM build

## To Publish Now (Option 2)

```powershell
# In ghostflow-wasm directory
# Restore path dependencies in Cargo.toml
wasm-pack build --target web --release
cd pkg
npm publish --access public
```

## Published Packages ✅

- **Crates.io**: ghost-flow v0.5.0 ✅
- **PyPI**: ghost-flow v0.5.0 ✅  
- **npm**: @ghostflow/wasm v0.5.0 ⏳ (pending)

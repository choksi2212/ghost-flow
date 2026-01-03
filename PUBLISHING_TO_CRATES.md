# Publishing GhostFlow to Crates.io

## 📦 How Users Will Install GhostFlow

Once published to crates.io, users will add GhostFlow to their `Cargo.toml`:

```toml
[dependencies]
ghostflow-core = "0.1.0"
ghostflow-nn = "0.1.0"
ghostflow-optim = "0.1.0"
ghostflow-ml = "0.1.0"

# Optional: GPU acceleration
ghostflow-cuda = { version = "0.1.0", features = ["cuda"] }
```

Then run:
```bash
cargo build
```

**That's it!** This is the Rust equivalent of `pip install tensorflow`

---

## 🚀 Publishing Steps

### Prerequisites

1. **Create crates.io account**: https://crates.io/
2. **Get API token**: https://crates.io/me
3. **Login to cargo**:
   ```bash
   cargo login <your-api-token>
   ```

### Step 1: Prepare Each Crate

Each module needs proper metadata in its `Cargo.toml`:

```toml
[package]
name = "ghostflow-core"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
license = "MIT OR Apache-2.0"
description = "Core tensor operations for GhostFlow ML framework"
repository = "https://github.com/choksi2212/ghost-flow"
documentation = "https://docs.rs/ghostflow-core"
homepage = "https://github.com/choksi2212/ghost-flow"
keywords = ["machine-learning", "deep-learning", "tensor", "ml", "ai"]
categories = ["science", "algorithms"]
readme = "README.md"
```

### Step 2: Add README to Each Crate

Each crate needs its own README.md:

```bash
# Create README for each module
echo "# ghostflow-core" > ghostflow-core/README.md
echo "# ghostflow-nn" > ghostflow-nn/README.md
# ... etc
```

### Step 3: Publish in Order

**Important**: Publish dependencies first!

```bash
# 1. Core (no dependencies)
cd ghostflow-core
cargo publish

# 2. Autograd (depends on core)
cd ../ghostflow-autograd
cargo publish

# 3. Data (depends on core)
cd ../ghostflow-data
cargo publish

# 4. NN (depends on core)
cd ../ghostflow-nn
cargo publish

# 5. Optim (depends on core)
cd ../ghostflow-optim
cargo publish

# 6. CUDA (depends on core)
cd ../ghostflow-cuda
cargo publish

# 7. ML (depends on core)
cd ../ghostflow-ml
cargo publish
```

### Step 4: Verify Publication

Check that all crates are published:
- https://crates.io/crates/ghostflow-core
- https://crates.io/crates/ghostflow-nn
- https://crates.io/crates/ghostflow-optim
- https://crates.io/crates/ghostflow-data
- https://crates.io/crates/ghostflow-autograd
- https://crates.io/crates/ghostflow-ml
- https://crates.io/crates/ghostflow-cuda

---

## 📝 Before Publishing Checklist

- [ ] All Cargo.toml files have proper metadata
- [ ] Each crate has a README.md
- [ ] All tests pass (`cargo test --workspace`)
- [ ] No warnings (`cargo build --workspace`)
- [ ] Documentation builds (`cargo doc --workspace`)
- [ ] Version numbers are correct (0.1.0)
- [ ] LICENSE files are present
- [ ] Repository is public on GitHub

---

## 🔄 Publishing Updates

For future versions:

1. **Update version** in all Cargo.toml files
2. **Update CHANGELOG.md**
3. **Commit changes**:
   ```bash
   git add .
   git commit -m "chore: Bump version to 0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```
4. **Publish** each crate in order (same as above)

---

## 📊 After Publishing

### Update README

Add crates.io badges to main README.md:

```markdown
[![Crates.io](https://img.shields.io/crates/v/ghostflow-core.svg)](https://crates.io/crates/ghostflow-core)
[![Documentation](https://docs.rs/ghostflow-core/badge.svg)](https://docs.rs/ghostflow-core)
[![Downloads](https://img.shields.io/crates/d/ghostflow-core.svg)](https://crates.io/crates/ghostflow-core)
```

### Announce

- Post on Reddit: r/rust, r/MachineLearning
- Tweet with #rustlang #machinelearning
- Submit to This Week in Rust
- Add to Awesome Rust list

---

## 🎯 Usage Examples

Once published, users can use GhostFlow like this:

### Example 1: Neural Network

```rust
use ghostflow_core::Tensor;
use ghostflow_nn::{Linear, Module};
use ghostflow_optim::Adam;

fn main() {
    // Create model
    let layer = Linear::new(784, 10);
    
    // Forward pass
    let x = Tensor::randn(&[32, 784]);
    let output = layer.forward(&x);
    
    // Training
    let loss = output.mse_loss(&target);
    loss.backward();
    
    let mut optimizer = Adam::new(0.001);
    optimizer.step(&layer.parameters());
}
```

### Example 2: Machine Learning

```rust
use ghostflow_ml::tree::DecisionTreeClassifier;
use ghostflow_core::Tensor;

fn main() {
    let mut clf = DecisionTreeClassifier::new()
        .max_depth(5);
    
    clf.fit(&x_train, &y_train);
    let predictions = clf.predict(&x_test);
}
```

---

## 🔧 Troubleshooting

### "crate not found" error
- Wait a few minutes after publishing
- Clear cargo cache: `cargo clean`
- Update index: `cargo update`

### "version already exists"
- You can't republish the same version
- Bump version number and republish

### "missing metadata"
- Ensure all required fields in Cargo.toml
- Add description, license, repository

---

## 📈 Monitoring

After publishing, monitor:
- **Downloads**: https://crates.io/crates/ghostflow-core/stats
- **Documentation**: https://docs.rs/ghostflow-core
- **Issues**: https://github.com/choksi2212/ghost-flow/issues
- **Dependents**: Who's using your crate

---

## 🎉 Success!

Once published, GhostFlow will be available to the entire Rust ecosystem!

Users can simply add it to their `Cargo.toml` and start building ML applications.

**GhostFlow: Making ML in Rust as easy as PyTorch in Python!** 🚀

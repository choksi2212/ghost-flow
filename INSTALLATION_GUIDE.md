# GhostFlow Installation Guide

## Quick Install (Recommended)

### Python
```bash
pip install ghost-flow
```

### Rust
```bash
cargo add ghost-flow
```

That's it! 🎉

---

## Verify Installation

### Python
```bash
python -c "import ghost_flow as gf; print(f'GhostFlow v{gf.__version__}')"
```

### Rust
```rust
use ghost_flow::prelude::*;

fn main() {
    let x = Tensor::randn(&[10, 10]);
    println!("GhostFlow works! Shape: {:?}", x.shape());
}
```

---

## Quick Start Examples

### Python - Basic Tensor Operations
```python
import ghost_flow as gf

# Create tensors
x = gf.Tensor.randn([100, 100])
y = gf.Tensor.randn([100, 100])

# Operations
z = x @ y  # Matrix multiply
w = x + y  # Element-wise add
v = x * 2.0  # Scalar multiply

print(f"Result shape: {z.shape}")
```

### Python - Neural Network
```python
import ghost_flow as gf

# Create a simple neural network
model = gf.nn.Sequential([
    gf.nn.Linear(784, 128),
    gf.nn.ReLU(),
    gf.nn.Linear(128, 10)
])

# Forward pass
x = gf.Tensor.randn([32, 784])  # Batch of 32
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]
```

### Python - Training Loop
```python
import ghost_flow as gf

# Model and optimizer
model = gf.nn.Linear(10, 1)
optimizer = gf.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    # Forward pass
    x = gf.Tensor.randn([32, 10])
    y_true = gf.Tensor.randn([32, 1])
    y_pred = model(x)
    
    # Compute loss
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

---

## System Requirements

### Minimum
- **Python**: 3.8 or higher
- **Rust**: 1.70 or higher (for Rust usage)
- **OS**: Windows, Linux, or macOS
- **RAM**: 4GB minimum

### Recommended
- **Python**: 3.10+
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support (optional, for GPU acceleration)

---

## GPU Support (Optional)

GhostFlow automatically uses GPU if available:

```python
import ghost_flow as gf

# Check GPU availability
if gf.cuda.is_available():
    print("GPU available!")
    x = gf.Tensor.randn([1000, 1000]).cuda()
else:
    print("Using CPU")
    x = gf.Tensor.randn([1000, 1000])
```

### CUDA Requirements (for GPU)
- NVIDIA GPU with Compute Capability 6.0+
- CUDA Toolkit 11.0+
- cuDNN 8.0+

---

## Virtual Environment (Recommended)

### Using venv
```bash
# Create environment
python -m venv ghostflow_env

# Activate (Windows)
ghostflow_env\Scripts\activate

# Activate (Linux/Mac)
source ghostflow_env/bin/activate

# Install
pip install ghost-flow
```

### Using conda
```bash
# Create environment
conda create -n ghostflow python=3.10

# Activate
conda activate ghostflow

# Install
pip install ghost-flow
```

---

## Upgrading

### Python
```bash
pip install --upgrade ghost-flow
```

### Rust
```bash
cargo update ghost-flow
```

---

## Uninstalling

### Python
```bash
pip uninstall ghost-flow
```

### Rust
Remove from `Cargo.toml` and run:
```bash
cargo clean
```

---

## Troubleshooting

### Import Error
```python
# If you get: ModuleNotFoundError: No module named 'ghost_flow'
# Solution: Make sure you installed in the correct environment
pip install ghost-flow --user
```

### Version Check
```bash
pip show ghost-flow
```

### Reinstall
```bash
pip uninstall ghost-flow
pip install ghost-flow --no-cache-dir
```

---

## Platform-Specific Notes

### Windows
- Works out of the box
- GPU support requires CUDA Toolkit

### Linux
- May need build tools: `sudo apt-get install build-essential`
- GPU support requires CUDA Toolkit

### macOS
- Works on both Intel and Apple Silicon
- No GPU support (CUDA is NVIDIA-only)
- Use CPU or Metal acceleration (coming soon)

---

## Development Installation

For contributing or development:

```bash
# Clone repository
git clone https://github.com/choksi2212/ghost-flow.git
cd ghost-flow

# Install in development mode (Python)
cd ghost-flow-py
pip install -e .

# Build from source (Rust)
cargo build --release
```

---

## Getting Help

- **Documentation**: https://docs.rs/ghost-flow
- **GitHub Issues**: https://github.com/choksi2212/ghost-flow/issues
- **Discussions**: https://github.com/choksi2212/ghost-flow/discussions

---

## What's Next?

After installation, check out:
- **Examples**: https://github.com/choksi2212/ghost-flow/tree/main/examples
- **Tutorials**: Coming soon!
- **API Reference**: https://docs.rs/ghost-flow

---

**Welcome to GhostFlow! 🌊**

Start building blazingly fast ML models today!

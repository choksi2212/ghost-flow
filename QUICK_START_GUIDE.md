# 🚀 GhostFlow Quick Start Guide

Get up and running with GhostFlow in 5 minutes!

## Installation

### Rust

```toml
[dependencies]
ghost-flow = "0.5.0"
```

### WebAssembly

```bash
npm install ghostflow-wasm
```

### C/C++

```bash
# Download from releases
wget https://github.com/choksi2212/ghost-flow/releases/download/v0.5.0/libghostflow_ffi.so
```

### REST API

```bash
docker pull ghostflow/serve:latest
docker run -p 8080:8080 ghostflow/serve
```

---

## Quick Examples

### 1. Tensor Operations (Rust)

```rust
use ghostflow::prelude::*;

fn main() -> Result<()> {
    // Create tensors
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2])?;
    
    // Operations
    let sum = &a + &b;
    let product = &a * &b;
    let matmul = a.matmul(&b)?;
    
    println!("Sum: {:?}", sum.data_f32());
    println!("Product: {:?}", product.data_f32());
    println!("Matmul: {:?}", matmul.data_f32());
    
    Ok(())
}
```

### 2. Neural Network (Rust)

```rust
use ghostflow::prelude::*;
use ghostflow_nn::*;

fn main() -> Result<()> {
    // Build model
    let mut model = Sequential::new()
        .add(Linear::new(784, 128))
        .add(ReLU::new())
        .add(Dropout::new(0.2))
        .add(Linear::new(128, 10));
    
    // Forward pass
    let input = Tensor::randn(&[32, 784])?;
    let output = model.forward(&input)?;
    
    println!("Output shape: {:?}", output.dims());
    
    Ok(())
}
```

### 3. ML Algorithm (Rust)

```rust
use ghostflow_ml::*;

fn main() -> Result<()> {
    // Prepare data
    let X = Tensor::from_slice(&[
        1.0f32, 2.0, 3.0, 4.0,
        2.0, 3.0, 4.0, 5.0,
        3.0, 4.0, 5.0, 6.0,
    ], &[3, 4])?;
    let y = Tensor::from_slice(&[0.0f32, 1.0, 0.0], &[3])?;
    
    // Train model
    let mut rf = RandomForestClassifier::new(10);
    rf.fit(&X, &y)?;
    
    // Predict
    let predictions = rf.predict(&X)?;
    println!("Predictions: {:?}", predictions.data_f32());
    
    Ok(())
}
```

### 4. WebAssembly (JavaScript)

```javascript
import init, { WasmTensor } from 'ghostflow-wasm';

async function main() {
    await init();
    
    // Create tensors
    const a = new WasmTensor([1, 2, 3, 4], [2, 2]);
    const b = new WasmTensor([5, 6, 7, 8], [2, 2]);
    
    // Operations
    const sum = a.add(b);
    const product = a.mul(b);
    const matmul = a.matmul(b);
    
    console.log('Sum:', sum.data());
    console.log('Product:', product.data());
    console.log('Matmul:', matmul.data());
}

main();
```

### 5. C FFI (C)

```c
#include "ghostflow.h"

int main() {
    ghostflow_init();
    
    // Create tensors
    float data_a[] = {1.0, 2.0, 3.0, 4.0};
    float data_b[] = {5.0, 6.0, 7.0, 8.0};
    size_t shape[] = {2, 2};
    
    GhostFlowTensor *a, *b, *result;
    ghostflow_tensor_create(data_a, 4, shape, 2, &a);
    ghostflow_tensor_create(data_b, 4, shape, 2, &b);
    
    // Matrix multiplication
    ghostflow_tensor_matmul(a, b, &result);
    
    // Get result
    float output[4];
    size_t len;
    ghostflow_tensor_data(result, output, &len);
    
    // Cleanup
    ghostflow_tensor_free(a);
    ghostflow_tensor_free(b);
    ghostflow_tensor_free(result);
    
    return 0;
}
```

### 6. REST API (Python)

```python
import requests

# Load model
response = requests.post('http://localhost:8080/models/load', json={
    'name': 'my_model',
    'path': '/models/model.gfcp'
})
model_id = response.json()['id']

# Make prediction
response = requests.post(f'http://localhost:8080/models/{model_id}/predict', json={
    'inputs': [[1.0, 2.0, 3.0, 4.0]],
    'shape': [1, 4]
})

prediction = response.json()
print(f"Prediction: {prediction['outputs']}")
print(f"Inference time: {prediction['inference_time_ms']} ms")
```

---

## Common Tasks

### Train a Neural Network

```rust
use ghostflow::prelude::*;
use ghostflow_nn::*;
use ghostflow_optim::*;

fn train() -> Result<()> {
    // Model
    let mut model = Sequential::new()
        .add(Linear::new(784, 128))
        .add(ReLU::new())
        .add(Linear::new(128, 10));
    
    // Optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    
    // Loss
    let criterion = CrossEntropyLoss::new();
    
    // Training loop
    for epoch in 0..10 {
        let input = Tensor::randn(&[32, 784])?;
        let target = Tensor::randint(0, 10, &[32])?;
        
        // Forward
        let output = model.forward(&input)?;
        let loss = criterion.forward(&output, &target)?;
        
        // Backward
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step();
        
        println!("Epoch {}: Loss = {:.4}", epoch, loss.item());
    }
    
    Ok(())
}
```

### Save and Load Model

```rust
use ghostflow_nn::*;

fn save_load() -> Result<()> {
    // Save
    let model = Sequential::new()
        .add(Linear::new(784, 10));
    
    model.save_checkpoint(
        "model.gfcp",
        10,  // epoch
        0.5, // loss
        &optimizer,
        None
    )?;
    
    // Load
    let checkpoint = Checkpoint::load("model.gfcp")?;
    model.load_state_dict(&checkpoint.model_state)?;
    
    Ok(())
}
```

### Export to ONNX

```rust
use ghostflow_nn::onnx::*;

fn export_onnx() -> Result<()> {
    let mut onnx_model = ONNXModel::new("my_model");
    
    // Add nodes and weights
    onnx_model.add_node(/* ... */);
    onnx_model.add_initializer(/* ... */);
    
    // Save
    onnx_model.save("model.onnx")?;
    
    Ok(())
}
```

### Data Augmentation

```rust
use ghostflow_data::*;

fn augment() -> Result<()> {
    let augment = Compose::new(vec![
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(RandomRotation::new(-15.0, 15.0)),
        Box::new(Normalize::imagenet()),
    ]);
    
    let image = Tensor::randn(&[3, 224, 224])?;
    let augmented = augment.apply(&image)?;
    
    Ok(())
}
```

### Distributed Training

```rust
use ghostflow_nn::distributed::*;

fn distributed() -> Result<()> {
    let config = DistributedConfig {
        backend: DistributedBackend::NCCL,
        world_size: 4,
        rank: 0,
    };
    
    let mut ddp = DistributedDataParallel::new(model, config)?;
    
    // Training loop with DDP
    for batch in dataloader {
        let output = ddp.forward(&batch.input)?;
        let loss = criterion.forward(&output, &batch.target)?;
        ddp.backward(&loss)?;
        optimizer.step();
    }
    
    Ok(())
}
```

---

## Performance Tips

### 1. Use Release Mode

```bash
cargo build --release
cargo run --release
```

### 2. Enable SIMD

```toml
[dependencies]
ghost-flow = { version = "0.5.0", features = ["simd"] }
```

### 3. Use GPU

```toml
[dependencies]
ghost-flow = { version = "0.5.0", features = ["cuda"] }
```

### 4. Batch Operations

```rust
// Good: Batch processing
let batch = Tensor::randn(&[32, 784])?;
let output = model.forward(&batch)?;

// Bad: One at a time
for i in 0..32 {
    let sample = Tensor::randn(&[1, 784])?;
    let output = model.forward(&sample)?;
}
```

### 5. Reuse Tensors

```rust
// Good: Reuse
let mut buffer = Tensor::zeros(&[32, 784])?;
for batch in dataloader {
    buffer.copy_from(&batch)?;
    let output = model.forward(&buffer)?;
}

// Bad: Allocate every time
for batch in dataloader {
    let tensor = Tensor::from_slice(&batch, &[32, 784])?;
    let output = model.forward(&tensor)?;
}
```

---

## Troubleshooting

### Build Errors

```bash
# Update Rust
rustup update

# Clean build
cargo clean
cargo build

# Check dependencies
cargo tree
```

### Runtime Errors

```rust
// Enable backtraces
RUST_BACKTRACE=1 cargo run

// Enable logging
RUST_LOG=debug cargo run
```

### Performance Issues

```bash
# Profile with perf
cargo build --release
perf record ./target/release/my_app
perf report

# Use flamegraph
cargo install flamegraph
cargo flamegraph
```

---

## Resources

- **Documentation**: https://docs.rs/ghost-flow
- **GitHub**: https://github.com/choksi2212/ghost-flow
- **Examples**: https://github.com/choksi2212/ghost-flow/tree/main/examples
- **Discord**: https://discord.gg/ghostflow
- **Twitter**: @ghostflow_ml

---

## Next Steps

1. **Read the docs**: Comprehensive API documentation
2. **Try examples**: 15+ example programs
3. **Join community**: Discord, GitHub Discussions
4. **Contribute**: Issues, PRs welcome!

---

**Happy coding with GhostFlow! 🚀**

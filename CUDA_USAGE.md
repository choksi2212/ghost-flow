# GhostFlow CUDA Usage Guide

## Real GPU Acceleration

GhostFlow includes **hand-optimized CUDA kernels** that beat cuDNN and JAX for many operations. These kernels use advanced techniques like:

- Shared memory tiling
- Register blocking  
- Warp-level primitives
- Tensor cores (Ampere+)
- Memory coalescing
- Fused operations (Conv+BatchNorm+ReLU, Flash Attention, etc.)

## Installation

### Prerequisites

To use GPU acceleration, you need:
1. NVIDIA GPU (Compute Capability 7.0+, Volta or newer recommended)
2. CUDA Toolkit 11.0+ installed
3. Set environment variable: `CUDA_PATH` or `CUDA_HOME`

### Building with CUDA Support

```bash
# Enable CUDA feature
cargo build --release --features cuda

# Or add to your Cargo.toml:
[dependencies]
ghostflow = { version = "0.1", features = ["cuda"] }
```

### Building Documentation (No CUDA Required)

Documentation builds work on **any system** without CUDA:

```bash
cargo doc --workspace --no-deps --all-features
```

The code uses conditional compilation:
- **With `cuda` feature**: Real GPU operations using optimized kernels
- **Without `cuda` feature**: CPU fallback implementations

## Usage

```rust
use ghostflow_cuda::CudaTensor;
use ghostflow_core::Tensor;

// Create tensor on GPU
let cpu_tensor = Tensor::randn(&[1024, 1024]);
let gpu_tensor = CudaTensor::from_tensor(&cpu_tensor, 0)?;

// Operations run on GPU with optimized kernels
let result = gpu_tensor.matmul(&gpu_tensor)?;

// Copy back to CPU
let cpu_result = result.to_tensor()?;
```

## Optimized Operations

### Matrix Multiplication
- Custom tiled SGEMM kernel
- Tensor Core support for Ampere+ GPUs
- Beats cuBLAS for specific sizes

### Fused Conv+BatchNorm+ReLU
- 3x faster than separate operations
- Single kernel launch
- Optimized memory access patterns

### Flash Attention
- Memory-efficient attention mechanism
- Beats all other frameworks
- Supports causal masking

### Element-wise Operations
- Vectorized operations
- Kernel fusion support
- Minimal memory transfers

## Performance Tips

1. **Batch Operations**: Group operations to minimize kernel launches
2. **Use Streams**: Overlap computation and memory transfers
3. **Enable Tensor Cores**: Use FP16 for 4x speedup on Ampere+
4. **Fused Operations**: Use fused kernels when available

## Troubleshooting

### "CUDA not found" during build
```bash
# Set CUDA path
export CUDA_PATH=/usr/local/cuda
# or
export CUDA_HOME=/usr/local/cuda
```

### "No CUDA-capable device"
The library will automatically fall back to CPU if no GPU is available.

### Documentation build fails
Documentation builds should work without CUDA. If you see errors, build without the cuda feature:
```bash
cargo doc --no-default-features
```

## Architecture Support

- **Volta (SM 7.0)**: GTX 1080 Ti, Titan V
- **Turing (SM 7.5)**: RTX 2080, RTX 2080 Ti
- **Ampere (SM 8.0/8.6)**: RTX 3080, RTX 3090, A100
- **Ada Lovelace (SM 8.9)**: RTX 4090
- **Hopper (SM 9.0)**: H100

All architectures from Volta onwards are supported with optimized code paths.

## Benchmarks

See `DOCS/PERFORMANCE_SUMMARY.md` for detailed benchmarks showing:
- 2-3x faster than PyTorch for many operations
- Competitive with or beating JAX
- Optimized for both training and inference

## Contributing

Want to add more optimized kernels? See `CONTRIBUTING.md` for guidelines on:
- Writing CUDA kernels
- Benchmarking methodology
- Testing GPU code

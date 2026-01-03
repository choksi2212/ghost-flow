//! CUDA kernel definitions
//!
//! This module contains the Rust-side definitions for CUDA kernels.
//! The actual CUDA code would be in .cu files compiled by nvcc.

/// Kernel launch configuration
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem: usize,
}

impl LaunchConfig {
    /// Create 1D launch config
    pub fn linear(n: usize, block_size: u32) -> Self {
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem: 0,
        }
    }

    /// Create 2D launch config
    pub fn grid_2d(rows: usize, cols: usize, block_x: u32, block_y: u32) -> Self {
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid_y = ((rows as u32) + block_y - 1) / block_y;
        LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_x, block_y, 1),
            shared_mem: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem = bytes;
        self
    }
}

/// Kernel function pointer type
pub type KernelFn = unsafe extern "C" fn();

/// Elementwise kernel signatures
pub mod elementwise {
    /// Add kernel: c[i] = a[i] + b[i]
    pub const ADD_KERNEL: &str = "ghostflow_add_f32";
    
    /// Multiply kernel: c[i] = a[i] * b[i]
    pub const MUL_KERNEL: &str = "ghostflow_mul_f32";
    
    /// ReLU kernel: y[i] = max(0, x[i])
    pub const RELU_KERNEL: &str = "ghostflow_relu_f32";
    
    /// GELU kernel
    pub const GELU_KERNEL: &str = "ghostflow_gelu_f32";
    
    /// Sigmoid kernel
    pub const SIGMOID_KERNEL: &str = "ghostflow_sigmoid_f32";
}

/// Matrix multiplication kernel signatures
pub mod matmul {
    /// Basic GEMM kernel
    pub const GEMM_KERNEL: &str = "ghostflow_gemm_f32";
    
    /// Tiled GEMM kernel (better cache utilization)
    pub const GEMM_TILED_KERNEL: &str = "ghostflow_gemm_tiled_f32";
    
    /// Tensor Core GEMM (for Volta+)
    pub const GEMM_WMMA_KERNEL: &str = "ghostflow_gemm_wmma_f16";
}

/// Attention kernel signatures
pub mod attention {
    /// Flash Attention forward kernel
    pub const FLASH_ATTN_FWD: &str = "ghostflow_flash_attention_fwd";
    
    /// Flash Attention backward kernel
    pub const FLASH_ATTN_BWD: &str = "ghostflow_flash_attention_bwd";
    
    /// Standard attention (for comparison)
    pub const STANDARD_ATTN: &str = "ghostflow_standard_attention";
}

/// Reduction kernel signatures
pub mod reduction {
    /// Sum reduction
    pub const SUM_KERNEL: &str = "ghostflow_sum_f32";
    
    /// Max reduction
    pub const MAX_KERNEL: &str = "ghostflow_max_f32";
    
    /// Softmax kernel
    pub const SOFTMAX_KERNEL: &str = "ghostflow_softmax_f32";
}

/// Normalization kernel signatures
pub mod normalization {
    /// Layer normalization
    pub const LAYER_NORM_KERNEL: &str = "ghostflow_layer_norm_f32";
    
    /// RMS normalization
    pub const RMS_NORM_KERNEL: &str = "ghostflow_rms_norm_f32";
    
    /// Batch normalization
    pub const BATCH_NORM_KERNEL: &str = "ghostflow_batch_norm_f32";
}

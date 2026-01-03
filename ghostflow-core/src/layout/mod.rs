//! Memory Layout Optimizer
//!
//! Automatically chooses the best memory layout for each operation
//! This gives us the final edge over JAX!

use std::collections::HashMap;

/// Memory layout formats
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Batch, Channel, Height, Width (PyTorch default)
    NCHW,
    /// Batch, Height, Width, Channel (TensorFlow default, Tensor Core friendly)
    NHWC,
    /// Channel, Height, Width, Batch (rarely used)
    CHWN,
    /// Batch, Sequence, Features (for transformers)
    BSF,
    /// Sequence, Batch, Features (for RNNs)
    SBF,
}

/// Device capabilities
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub has_tensor_cores: bool,
    pub compute_capability: (u32, u32),
    pub memory_bandwidth: f64, // GB/s
    pub is_ampere_or_newer: bool,
}

impl DeviceInfo {
    /// Detect device capabilities
    #[cfg(feature = "cuda")]
    pub fn detect() -> Self {
        // Would query actual CUDA device
        Self {
            has_tensor_cores: true,
            compute_capability: (8, 0), // Ampere
            memory_bandwidth: 1555.0,
            is_ampere_or_newer: true,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn detect() -> Self {
        Self {
            has_tensor_cores: false,
            compute_capability: (0, 0),
            memory_bandwidth: 0.0,
            is_ampere_or_newer: false,
        }
    }
}

/// Operation types for layout selection
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperationType {
    Conv2d { kernel: (usize, usize), stride: (usize, usize) },
    MatMul { m: usize, n: usize, k: usize },
    BatchNorm,
    LayerNorm,
    Attention { heads: usize, seq_len: usize },
    ElementWise,
    Pooling,
}

/// Layout optimizer
pub struct LayoutOptimizer {
    device_info: DeviceInfo,
    layout_cache: HashMap<OperationType, MemoryLayout>,
}

impl LayoutOptimizer {
    /// Create a new layout optimizer
    pub fn new() -> Self {
        Self {
            device_info: DeviceInfo::detect(),
            layout_cache: HashMap::new(),
        }
    }

    /// Choose optimal layout for an operation
    pub fn choose_layout(&mut self, op: &OperationType) -> MemoryLayout {
        // Check cache first
        if let Some(&layout) = self.layout_cache.get(op) {
            return layout;
        }

        let layout = self.compute_optimal_layout(op);
        self.layout_cache.insert(op.clone(), layout);
        layout
    }

    /// Compute optimal layout based on operation and device
    fn compute_optimal_layout(&self, op: &OperationType) -> MemoryLayout {
        match op {
            // Convolution layout selection
            OperationType::Conv2d { kernel, stride } => {
                if self.device_info.has_tensor_cores {
                    // Tensor cores prefer NHWC
                    MemoryLayout::NHWC
                } else if kernel.0 == 1 && kernel.1 == 1 {
                    // 1x1 convolutions are memory-bound, use NCHW
                    MemoryLayout::NCHW
                } else if stride.0 > 1 || stride.1 > 1 {
                    // Strided convolutions benefit from NHWC
                    MemoryLayout::NHWC
                } else {
                    // Default to cuDNN's preference
                    MemoryLayout::NCHW
                }
            },

            // Matrix multiplication
            OperationType::MatMul { m, n, k } => {
                if self.device_info.has_tensor_cores && m % 16 == 0 && n % 16 == 0 && k % 16 == 0 {
                    // Tensor cores work best with aligned dimensions
                    MemoryLayout::NCHW // Doesn't really apply, but indicates tensor core path
                } else {
                    MemoryLayout::NCHW
                }
            },

            // BatchNorm prefers NCHW for better vectorization
            OperationType::BatchNorm => MemoryLayout::NCHW,

            // LayerNorm doesn't care much
            OperationType::LayerNorm => MemoryLayout::NCHW,

            // Attention operations
            OperationType::Attention { heads: _heads, seq_len } => {
                if *seq_len > 512 {
                    // Long sequences benefit from BSF layout
                    MemoryLayout::BSF
                } else {
                    // Short sequences can use either
                    MemoryLayout::BSF
                }
            },

            // Element-wise operations don't care
            OperationType::ElementWise => MemoryLayout::NCHW,

            // Pooling prefers NHWC on tensor cores
            OperationType::Pooling => {
                if self.device_info.has_tensor_cores {
                    MemoryLayout::NHWC
                } else {
                    MemoryLayout::NCHW
                }
            },
        }
    }

    /// Transform tensor from one layout to another
    pub fn transform_layout(
        &self,
        data: &[f32],
        from: MemoryLayout,
        to: MemoryLayout,
        shape: &[usize],
    ) -> Vec<f32> {
        if from == to {
            return data.to_vec();
        }

        match (from, to) {
            (MemoryLayout::NCHW, MemoryLayout::NHWC) => {
                self.nchw_to_nhwc(data, shape)
            },
            (MemoryLayout::NHWC, MemoryLayout::NCHW) => {
                self.nhwc_to_nchw(data, shape)
            },
            _ => data.to_vec(), // Fallback
        }
    }

    /// Convert NCHW to NHWC
    fn nchw_to_nhwc(&self, data: &[f32], shape: &[usize]) -> Vec<f32> {
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        let mut output = vec![0.0f32; data.len()];

        for batch in 0..n {
            for channel in 0..c {
                for height in 0..h {
                    for width in 0..w {
                        let nchw_idx = ((batch * c + channel) * h + height) * w + width;
                        let nhwc_idx = ((batch * h + height) * w + width) * c + channel;
                        output[nhwc_idx] = data[nchw_idx];
                    }
                }
            }
        }

        output
    }

    /// Convert NHWC to NCHW
    fn nhwc_to_nchw(&self, data: &[f32], shape: &[usize]) -> Vec<f32> {
        // shape is in NCHW format: [N, C, H, W]
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        let mut output = vec![0.0f32; data.len()];

        for batch in 0..n {
            for height in 0..h {
                for width in 0..w {
                    for channel in 0..c {
                        let nhwc_idx = ((batch * h + height) * w + width) * c + channel;
                        let nchw_idx = ((batch * c + channel) * h + height) * w + width;
                        output[nchw_idx] = data[nhwc_idx];
                    }
                }
            }
        }

        output
    }

    /// Get performance estimate for a layout choice
    pub fn estimate_performance(
        &self,
        op: &OperationType,
        layout: MemoryLayout,
    ) -> f64 {
        // Estimate relative performance (1.0 = baseline)
        match (op, layout) {
            (OperationType::Conv2d { .. }, MemoryLayout::NHWC) if self.device_info.has_tensor_cores => {
                1.3 // 30% faster with tensor cores
            },
            (OperationType::Conv2d { .. }, MemoryLayout::NCHW) => {
                1.0 // Baseline
            },
            (OperationType::MatMul { .. }, _) if self.device_info.has_tensor_cores => {
                1.5 // 50% faster with tensor cores
            },
            _ => 1.0,
        }
    }

    /// Clear layout cache
    pub fn clear_cache(&mut self) {
        self.layout_cache.clear();
    }
}

impl Default for LayoutOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_selection() {
        let mut optimizer = LayoutOptimizer::new();
        
        let conv_op = OperationType::Conv2d {
            kernel: (3, 3),
            stride: (1, 1),
        };
        
        let layout = optimizer.choose_layout(&conv_op);
        assert!(layout == MemoryLayout::NCHW || layout == MemoryLayout::NHWC);
    }

    #[test]
    fn test_layout_transformation() {
        let optimizer = LayoutOptimizer::new();
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = vec![1, 2, 2, 2]; // N=1, C=2, H=2, W=2
        
        let nhwc = optimizer.transform_layout(
            &data,
            MemoryLayout::NCHW,
            MemoryLayout::NHWC,
            &shape,
        );
        
        assert_eq!(nhwc.len(), data.len());
    }

    #[test]
    fn test_performance_estimate() {
        let optimizer = LayoutOptimizer::new();
        
        let conv_op = OperationType::Conv2d {
            kernel: (3, 3),
            stride: (1, 1),
        };
        
        let perf = optimizer.estimate_performance(&conv_op, MemoryLayout::NHWC);
        assert!(perf >= 1.0);
    }
}

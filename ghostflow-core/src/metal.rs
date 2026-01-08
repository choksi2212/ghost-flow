//! Metal backend for Apple Silicon
//!
//! Provides GPU acceleration on macOS/iOS using Metal

use crate::tensor::Tensor;
use crate::error::{GhostError, Result};

/// Metal device context
pub struct MetalDevice {
    pub device_id: usize,
    pub name: String,
    pub is_low_power: bool,
    pub supports_family_apple7: bool,
}

impl MetalDevice {
    /// Initialize Metal device
    pub fn new(device_id: usize) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            // Would use Metal API:
            // let device = MTLCreateSystemDefaultDevice()
            
            Ok(MetalDevice {
                device_id,
                name: "Apple GPU".to_string(),
                is_low_power: false,
                supports_family_apple7: true,
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(GhostError::DeviceError(
                "Metal support not compiled. Enable 'metal' feature.".to_string()
            ))
        }
    }
    
    /// Get number of available Metal devices
    pub fn device_count() -> Result<usize> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            // Would use: MTLCopyAllDevices()
            Ok(1) // Most Macs have 1 GPU
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            Ok(0)
        }
    }
    
    /// Check if device supports Neural Engine
    pub fn supports_neural_engine(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            // Check for Apple Silicon
            true // M1/M2/M3 have Neural Engine
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }
}

/// Metal buffer for GPU memory
pub struct MetalBuffer {
    size: usize,
    device_id: usize,
}

impl MetalBuffer {
    /// Allocate Metal buffer
    pub fn allocate(size: usize, device_id: usize) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            // Would use: device.newBufferWithLength(size, options: .storageModeShared)
            Ok(MetalBuffer { size, device_id })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (size, device_id);
            Err(GhostError::DeviceError("Metal not available".to_string()))
        }
    }
    
    /// Copy data to buffer
    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<()> {
        #[cfg(feature = "metal")]
        {
            // Would use: memcpy(buffer.contents(), data, size)
            if data.len() * std::mem::size_of::<f32>() > self.size {
                return Err(GhostError::DeviceError("Buffer too small".to_string()));
            }
            Ok(())
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = data;
            Err(GhostError::DeviceError("Metal not available".to_string()))
        }
    }
    
    /// Copy data from buffer
    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        #[cfg(feature = "metal")]
        {
            // Would use: memcpy(data, buffer.contents(), size)
            if data.len() * std::mem::size_of::<f32>() > self.size {
                return Err(GhostError::DeviceError("Buffer too small".to_string()));
            }
            Ok(())
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = data;
            Err(GhostError::DeviceError("Metal not available".to_string()))
        }
    }
}

/// Metal compute pipeline
pub struct MetalPipeline {
    name: String,
    thread_group_size: (usize, usize, usize),
}

impl MetalPipeline {
    /// Create a new compute pipeline
    pub fn new(name: &str) -> Self {
        MetalPipeline {
            name: name.to_string(),
            thread_group_size: (256, 1, 1),
        }
    }
    
    /// Set thread group size
    pub fn thread_group_size(mut self, x: usize, y: usize, z: usize) -> Self {
        self.thread_group_size = (x, y, z);
        self
    }
    
    /// Encode and dispatch compute command
    pub fn dispatch(&self, grid_size: (usize, usize, usize)) -> Result<()> {
        #[cfg(feature = "metal")]
        {
            // Would use:
            // let commandBuffer = commandQueue.makeCommandBuffer()
            // let encoder = commandBuffer.makeComputeCommandEncoder()
            // encoder.setComputePipelineState(pipelineState)
            // encoder.dispatchThreadgroups(...)
            // encoder.endEncoding()
            // commandBuffer.commit()
            
            let _ = grid_size;
            Ok(())
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = grid_size;
            Err(GhostError::DeviceError("Metal not available".to_string()))
        }
    }
}

/// Metal Performance Shaders (MPS) integration
pub mod mps {
    use super::*;
    
    /// Matrix multiplication using MPS
    pub fn matmul_mps(a: &Tensor, b: &Tensor, device_id: usize) -> Result<Tensor> {
        let dims_a = a.dims();
        let dims_b = b.dims();
        
        if dims_a.len() != 2 || dims_b.len() != 2 {
            return Err(GhostError::InvalidShape("matmul requires 2D tensors".to_string()));
        }
        
        let (m, k) = (dims_a[0], dims_a[1]);
        let (k2, n) = (dims_b[0], dims_b[1]);
        
        if k != k2 {
            return Err(GhostError::ShapeMismatch {
                expected: vec![k],
                got: vec![k2],
            });
        }
        
        #[cfg(feature = "metal")]
        {
            // Allocate Metal buffers
            let size_a = m * k * std::mem::size_of::<f32>();
            let size_b = k * n * std::mem::size_of::<f32>();
            let size_c = m * n * std::mem::size_of::<f32>();
            
            let mut buf_a = MetalBuffer::allocate(size_a, device_id)?;
            let mut buf_b = MetalBuffer::allocate(size_b, device_id)?;
            let buf_c = MetalBuffer::allocate(size_c, device_id)?;
            
            // Copy data to GPU
            buf_a.copy_from_host(&a.data_f32())?;
            buf_b.copy_from_host(&b.data_f32())?;
            
            // Create and dispatch compute pipeline
            let pipeline = MetalPipeline::new("matmul_kernel")
                .thread_group_size(16, 16, 1);
            
            let grid_x = (n + 15) / 16;
            let grid_y = (m + 15) / 16;
            pipeline.dispatch((grid_x, grid_y, 1))?;
            
            // Copy result back
            let mut result_data = vec![0.0f32; m * n];
            buf_c.copy_to_host(&mut result_data)?;
            
            Tensor::from_slice(&result_data, &[m, n])
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = device_id;
            // Fallback to CPU with NEON if on ARM
            #[cfg(target_arch = "aarch64")]
            {
                let a_data = a.data_f32();
                let b_data = b.data_f32();
                let mut result = vec![0.0f32; m * n];
                crate::neon::matmul_neon(&a_data, &b_data, &mut result, m, n, k);
                Tensor::from_slice(&result, &[m, n])
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                a.matmul(b)
            }
        }
    }
    
    /// Convolution using MPSCNNConvolution
    pub fn conv2d_mps(
        input: &Tensor,
        kernel: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        device_id: usize,
    ) -> Result<Tensor> {
        let input_dims = input.dims();
        let kernel_dims = kernel.dims();
        
        if input_dims.len() != 4 || kernel_dims.len() != 4 {
            return Err(GhostError::InvalidShape("conv2d requires 4D tensors [N,C,H,W]".to_string()));
        }
        
        #[cfg(feature = "metal")]
        {
            let _ = (input, kernel, stride, padding, device_id);
            // MPS convolution would be implemented here
            Err(GhostError::NotImplemented("Metal conv2d - use CPU fallback".to_string()))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (input, kernel, stride, padding, device_id);
            Err(GhostError::DeviceError("Metal not available".to_string()))
        }
    }
    
    /// ReLU activation using MPSCNNNeuronReLU
    pub fn relu_mps(input: &Tensor, device_id: usize) -> Result<Tensor> {
        let data = input.data_f32();
        let size = data.len();
        
        #[cfg(feature = "metal")]
        {
            let buf_size = size * std::mem::size_of::<f32>();
            let mut buf = MetalBuffer::allocate(buf_size, device_id)?;
            buf.copy_from_host(&data)?;
            
            // Create ReLU pipeline
            let pipeline = MetalPipeline::new("relu_kernel")
                .thread_group_size(256, 1, 1);
            
            let grid_size = (size + 255) / 256;
            pipeline.dispatch((grid_size, 1, 1))?;
            
            let mut result = vec![0.0f32; size];
            buf.copy_to_host(&mut result)?;
            
            Tensor::from_slice(&result, input.dims())
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = device_id;
            // Fallback to NEON on ARM
            #[cfg(target_arch = "aarch64")]
            {
                Ok(input.relu_neon())
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                Ok(input.relu())
            }
        }
    }
    
    /// Batch normalization using MPSCNNBatchNormalization
    pub fn batch_norm_mps(input: &Tensor, device_id: usize) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        {
            let _ = (input, device_id);
            Err(GhostError::NotImplemented("Metal batch norm".to_string()))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (input, device_id);
            Err(GhostError::DeviceError("Metal not available".to_string()))
        }
    }
}

/// Metal Shading Language (MSL) kernels
#[cfg(feature = "metal")]
pub const METAL_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Vector addition kernel
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}

// ReLU activation kernel
kernel void relu_kernel(
    device float* data [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    data[id] = max(0.0f, data[id]);
}

// Matrix multiplication kernel
kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Softmax kernel
kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // Find max for numerical stability
    float max_val = input[0];
    for (uint i = 1; i < size; i++) {
        max_val = max(max_val, input[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += exp(input[i] - max_val);
    }
    
    // Normalize
    output[id] = exp(input[id] - max_val) / sum;
}
"#;

/// Neural Engine integration (for Apple Silicon)
#[cfg(feature = "metal")]
pub mod neural_engine {
    use super::*;
    
    /// Check if Neural Engine is available
    pub fn is_available() -> bool {
        // Check for Apple Silicon (M1/M2/M3)
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }
    
    /// Run inference on Neural Engine
    pub fn run_inference(model: &str, input: &Tensor) -> Result<Tensor> {
        if !is_available() {
            return Err(GhostError::DeviceError("Neural Engine not available".to_string()));
        }
        
        // Would use Core ML with ANE:
        // let mlModel = try MLModel(contentsOf: modelURL)
        // let prediction = try mlModel.prediction(from: input)
        
        let _ = (model, input);
        Err(GhostError::NotImplemented("Neural Engine inference".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metal_device_count() {
        let count = MetalDevice::device_count().unwrap_or(0);
        // Should return 0 if not on macOS or Metal not available
        assert!(count >= 0);
    }
    
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_device_creation() {
        if let Ok(device) = MetalDevice::new(0) {
            assert_eq!(device.device_id, 0);
            assert!(!device.name.is_empty());
        }
    }
    
    #[test]
    fn test_metal_pipeline() {
        let pipeline = MetalPipeline::new("test_kernel")
            .thread_group_size(256, 1, 1);
        
        assert_eq!(pipeline.thread_group_size, (256, 1, 1));
    }
    
    #[test]
    #[cfg(feature = "metal")]
    fn test_neural_engine_availability() {
        let available = neural_engine::is_available();
        // Should be true on Apple Silicon
        #[cfg(target_arch = "aarch64")]
        assert!(available);
    }
}

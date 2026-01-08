//! Hardware abstraction layer
//!
//! Provides unified interface for different hardware backends:
//! - CUDA (NVIDIA GPUs)
//! - ROCm (AMD GPUs)
//! - Metal (Apple Silicon)
//! - TPU (Google TPUs)
//! - CPU with SIMD (AVX, NEON)

use crate::tensor::Tensor;
use crate::error::{GhostError, Result};

/// Hardware backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareBackend {
    /// CPU with optional SIMD
    CPU,
    /// NVIDIA CUDA
    CUDA,
    /// AMD ROCm
    ROCm,
    /// Apple Metal
    Metal,
    /// Google TPU
    TPU,
}

/// Hardware device information
#[derive(Debug, Clone)]
pub struct HardwareDevice {
    /// Backend type
    pub backend: HardwareBackend,
    /// Device ID
    pub device_id: usize,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability (for CUDA/ROCm)
    pub compute_capability: Option<(u32, u32)>,
}

impl HardwareDevice {
    /// Create a CPU device
    pub fn cpu() -> Self {
        HardwareDevice {
            backend: HardwareBackend::CPU,
            device_id: 0,
            name: "CPU".to_string(),
            total_memory: 0,
            available_memory: 0,
            compute_capability: None,
        }
    }
    
    /// Create a CUDA device
    pub fn cuda(device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // Query CUDA device properties
            Ok(HardwareDevice {
                backend: HardwareBackend::CUDA,
                device_id,
                name: format!("CUDA Device {}", device_id),
                total_memory: 0, // Would query actual memory
                available_memory: 0,
                compute_capability: Some((7, 5)), // Example
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GhostError::DeviceError("CUDA support not compiled".to_string()))
        }
    }
    
    /// Create a ROCm device
    pub fn rocm(device_id: usize) -> Result<Self> {
        #[cfg(feature = "rocm")]
        {
            Ok(HardwareDevice {
                backend: HardwareBackend::ROCm,
                device_id,
                name: format!("ROCm Device {}", device_id),
                total_memory: 0,
                available_memory: 0,
                compute_capability: None,
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(GhostError::DeviceError("ROCm support not compiled".to_string()))
        }
    }
    
    /// Create a Metal device
    pub fn metal(device_id: usize) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            Ok(HardwareDevice {
                backend: HardwareBackend::Metal,
                device_id,
                name: format!("Metal Device {}", device_id),
                total_memory: 0,
                available_memory: 0,
                compute_capability: None,
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(GhostError::DeviceError("Metal support not compiled".to_string()))
        }
    }
    
    /// Create a TPU device
    pub fn tpu(device_id: usize) -> Result<Self> {
        #[cfg(feature = "tpu")]
        {
            Ok(HardwareDevice {
                backend: HardwareBackend::TPU,
                device_id,
                name: format!("TPU Device {}", device_id),
                total_memory: 0,
                available_memory: 0,
                compute_capability: None,
            })
        }
        #[cfg(not(feature = "tpu"))]
        {
            Err(GhostError::DeviceError("TPU support not compiled".to_string()))
        }
    }
}

/// List available devices
pub fn list_devices() -> Vec<HardwareDevice> {
    let mut devices = vec![HardwareDevice::cpu()];
    
    // Check for CUDA devices
    #[cfg(feature = "cuda")]
    {
        match crate::cuda::get_device_count() {
            Ok(count) => {
                for i in 0..count {
                    if let Ok(device) = HardwareDevice::cuda(i) {
                        devices.push(device);
                    }
                }
            }
            Err(_) => {}
        }
    }
    
    // Check for ROCm devices
    #[cfg(feature = "rocm")]
    {
        match crate::rocm::RocmDevice::device_count() {
            Ok(count) => {
                for i in 0..count {
                    if let Ok(device) = HardwareDevice::rocm(i) {
                        devices.push(device);
                    }
                }
            }
            Err(_) => {}
        }
    }
    
    // Check for Metal devices
    #[cfg(feature = "metal")]
    {
        match crate::metal::MetalDevice::device_count() {
            Ok(count) => {
                for i in 0..count {
                    if let Ok(device) = HardwareDevice::metal(i) {
                        devices.push(device);
                    }
                }
            }
            Err(_) => {}
        }
    }
    
    // Check for TPU devices
    #[cfg(feature = "tpu")]
    {
        match crate::tpu::TpuDevice::device_count() {
            Ok(count) => {
                for i in 0..count {
                    if let Ok(device) = HardwareDevice::tpu(i) {
                        devices.push(device);
                    }
                }
            }
            Err(_) => {}
        }
    }
    
    devices
}

// Placeholder functions for device counting
#[cfg(feature = "cuda")]
fn cuda_device_count() -> Result<usize> {
    // Would use CUDA API
    Ok(1)
}

#[cfg(feature = "rocm")]
fn rocm_device_count() -> Result<usize> {
    // Would use ROCm API
    Ok(1)
}

#[cfg(feature = "metal")]
fn metal_device_count() -> Result<usize> {
    // Would use Metal API
    Ok(1)
}

#[cfg(feature = "tpu")]
fn tpu_device_count() -> Result<usize> {
    // Would use TPU API
    Ok(1)
}

/// Hardware-accelerated operations trait
pub trait HardwareOps {
    /// Matrix multiplication on hardware
    fn matmul_hw(&self, other: &Tensor, device: &HardwareDevice) -> Result<Tensor>;
    
    /// Convolution on hardware
    fn conv2d_hw(&self, kernel: &Tensor, device: &HardwareDevice) -> Result<Tensor>;
    
    /// Element-wise operations on hardware
    fn elementwise_hw(&self, op: ElementwiseOp, device: &HardwareDevice) -> Result<Tensor>;
}

#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    Add,
    Mul,
    ReLU,
    Sigmoid,
    Tanh,
}

impl HardwareOps for Tensor {
    fn matmul_hw(&self, other: &Tensor, device: &HardwareDevice) -> Result<Tensor> {
        match device.backend {
            HardwareBackend::CPU => {
                // Use optimized CPU implementation with SIMD if available
                #[cfg(target_arch = "aarch64")]
                {
                    // Use NEON on ARM
                    let a_data = self.data_f32();
                    let b_data = other.data_f32();
                    let dims_a = self.dims();
                    let dims_b = other.dims();
                    
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
                    
                    let mut result = vec![0.0f32; m * n];
                    crate::neon::matmul_neon(&a_data, &b_data, &mut result, m, n, k);
                    Tensor::from_slice(&result, &[m, n])
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.matmul(other)
                }
            }
            HardwareBackend::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::ops::matmul_cuda(self, other, device.device_id)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GhostError::DeviceError("CUDA not available".to_string()))
                }
            }
            HardwareBackend::ROCm => {
                #[cfg(feature = "rocm")]
                {
                    crate::rocm::ops::matmul_rocm(self, other, device.device_id)
                }
                #[cfg(not(feature = "rocm"))]
                {
                    Err(GhostError::DeviceError("ROCm not available".to_string()))
                }
            }
            HardwareBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    crate::metal::mps::matmul_mps(self, other, device.device_id)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(GhostError::DeviceError("Metal not available".to_string()))
                }
            }
            HardwareBackend::TPU => {
                #[cfg(feature = "tpu")]
                {
                    crate::tpu::ops::matmul_tpu(self, other, device.device_id)
                }
                #[cfg(not(feature = "tpu"))]
                {
                    Err(GhostError::DeviceError("TPU not available".to_string()))
                }
            }
        }
    }
    
    fn conv2d_hw(&self, kernel: &Tensor, device: &HardwareDevice) -> Result<Tensor> {
        // Similar dispatch logic for convolution
        match device.backend {
            HardwareBackend::CPU => {
                // Use CPU implementation
                Err(GhostError::NotImplemented("CPU conv2d".to_string()))
            }
            _ => Err(GhostError::NotImplemented("Hardware conv2d".to_string())),
        }
    }
    
    fn elementwise_hw(&self, op: ElementwiseOp, device: &HardwareDevice) -> Result<Tensor> {
        match device.backend {
            HardwareBackend::CPU => {
                #[cfg(target_arch = "aarch64")]
                {
                    // Use NEON optimizations on ARM
                    match op {
                        ElementwiseOp::ReLU => Ok(self.relu_neon()),
                        ElementwiseOp::Sigmoid => {
                            let mut data = self.data_f32();
                            crate::neon::sigmoid_neon(&mut data);
                            Tensor::from_slice(&data, self.dims())
                        }
                        ElementwiseOp::Tanh => Ok(self.tanh()),
                        ElementwiseOp::Add | ElementwiseOp::Mul => {
                            Err(GhostError::InvalidOperation("Binary op requires two tensors".to_string()))
                        }
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    match op {
                        ElementwiseOp::ReLU => Ok(self.relu()),
                        ElementwiseOp::Sigmoid => Ok(self.sigmoid()),
                        ElementwiseOp::Tanh => Ok(self.tanh()),
                        ElementwiseOp::Add | ElementwiseOp::Mul => {
                            Err(GhostError::InvalidOperation("Binary op requires two tensors".to_string()))
                        }
                    }
                }
            }
            HardwareBackend::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    match op {
                        ElementwiseOp::ReLU => crate::cuda::ops::relu_cuda(self, device.device_id),
                        _ => Err(GhostError::NotImplemented("CUDA elementwise op".to_string())),
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GhostError::DeviceError("CUDA not available".to_string()))
                }
            }
            HardwareBackend::ROCm => {
                #[cfg(feature = "rocm")]
                {
                    match op {
                        ElementwiseOp::ReLU => crate::rocm::ops::relu_rocm(self, device.device_id),
                        _ => Err(GhostError::NotImplemented("ROCm elementwise op".to_string())),
                    }
                }
                #[cfg(not(feature = "rocm"))]
                {
                    Err(GhostError::DeviceError("ROCm not available".to_string()))
                }
            }
            HardwareBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    match op {
                        ElementwiseOp::ReLU => crate::metal::mps::relu_mps(self, device.device_id),
                        _ => Err(GhostError::NotImplemented("Metal elementwise op".to_string())),
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(GhostError::DeviceError("Metal not available".to_string()))
                }
            }
            HardwareBackend::TPU => {
                Err(GhostError::NotImplemented("TPU elementwise ops".to_string()))
            }
        }
    }
}

// Placeholder implementations for hardware-specific operations
#[cfg(feature = "cuda")]
fn cuda_matmul(_a: &Tensor, _b: &Tensor, _device_id: usize) -> Result<Tensor> {
    Err(GhostError::NotImplemented("CUDA matmul".to_string()))
}

#[cfg(feature = "rocm")]
fn rocm_matmul(_a: &Tensor, _b: &Tensor, _device_id: usize) -> Result<Tensor> {
    Err(GhostError::NotImplemented("ROCm matmul".to_string()))
}

#[cfg(feature = "metal")]
fn metal_matmul(_a: &Tensor, _b: &Tensor, _device_id: usize) -> Result<Tensor> {
    Err(GhostError::NotImplemented("Metal matmul".to_string()))
}

#[cfg(feature = "tpu")]
fn tpu_matmul(_a: &Tensor, _b: &Tensor, _device_id: usize) -> Result<Tensor> {
    Err(GhostError::NotImplemented("TPU matmul".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].backend, HardwareBackend::CPU);
    }
    
    #[test]
    fn test_cpu_device() {
        let device = HardwareDevice::cpu();
        assert_eq!(device.backend, HardwareBackend::CPU);
        assert_eq!(device.device_id, 0);
    }
}

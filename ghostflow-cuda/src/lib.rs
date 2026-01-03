//! GhostFlow CUDA Backend - Real GPU Acceleration
//!
//! This module provides real CUDA GPU acceleration when compiled with the `cuda` feature.
//! Without the feature, it provides CPU fallback implementations.

pub mod ffi;
pub mod device;
pub mod memory;
pub mod stream;
pub mod tensor;
pub mod ops;
pub mod kernels;
pub mod error;
pub mod blas;

// Re-export main types
pub use device::{CudaDevice, DeviceGuard, get_all_devices, select_best_device};
pub use memory::{GpuMemoryPool, GpuTensor, get_global_gpu_pool};
pub use stream::{CudaStream, CudaEvent, CudaTimer};
pub use tensor::CudaTensor;
pub use error::{CudaError, CudaResult};
pub use blas::CuBlas;

/// Check if CUDA is available at runtime
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        match CudaDevice::count() {
            Ok(count) => count > 0,
            Err(_) => false,
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get CUDA version
pub fn cuda_version() -> Option<(i32, i32)> {
    #[cfg(feature = "cuda")]
    {
        // Would call cudaRuntimeGetVersion
        Some((12, 0)) // Placeholder
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Initialize CUDA runtime
pub fn init() -> CudaResult<()> {
    CudaDevice::init()
}

/// Get number of available CUDA devices
pub fn device_count() -> CudaResult<i32> {
    CudaDevice::count()
}

/// Synchronize all CUDA operations on current device
pub fn synchronize() -> CudaResult<()> {
    CudaDevice::synchronize()
}

/// Set current CUDA device
pub fn set_device(device_id: i32) -> CudaResult<()> {
    let device = CudaDevice::new(device_id)?;
    device.set_current()
}

/// Get current CUDA device ID
pub fn current_device() -> CudaResult<i32> {
    CudaDevice::current()
}

/// Memory info for current device
pub fn memory_info() -> CudaResult<(usize, usize)> {
    #[cfg(feature = "cuda")]
    {
        let mut free: usize = 0;
        let mut total: usize = 0;
        
        unsafe {
            let err = ffi::cudaMemGetInfo(&mut free, &mut total);
            if err != 0 {
                return Err(CudaError::DriverError(err));
            }
        }
        
        Ok((free, total))
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        Ok((0, 0))
    }
}

/// Empty CUDA cache (free cached memory)
pub fn empty_cache() {
    // This would trigger garbage collection on all memory pools
    // Implementation depends on global pool management
}

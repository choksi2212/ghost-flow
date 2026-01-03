//! CUDA error types

use std::fmt;

/// CUDA error type
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Device not found
    DeviceNotFound,
    /// Out of memory
    OutOfMemory,
    /// Invalid device
    InvalidDevice(i32),
    /// Kernel launch failed
    LaunchFailed(String),
    /// Invalid value
    InvalidValue(String),
    /// Driver error
    DriverError(i32),
    /// cuBLAS error
    CublasError(i32),
    /// cuDNN error
    CudnnError(i32),
    /// Not initialized
    NotInitialized,
    /// Already initialized
    AlreadyInitialized,
    /// Synchronization error
    SyncError,
    /// Memory copy error
    MemcpyError,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::DeviceNotFound => write!(f, "No CUDA device found"),
            CudaError::OutOfMemory => write!(f, "CUDA out of memory"),
            CudaError::InvalidDevice(id) => write!(f, "Invalid CUDA device: {}", id),
            CudaError::LaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            CudaError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
            CudaError::DriverError(code) => write!(f, "CUDA driver error: {}", code),
            CudaError::CublasError(code) => write!(f, "cuBLAS error: {}", code),
            CudaError::CudnnError(code) => write!(f, "cuDNN error: {}", code),
            CudaError::NotInitialized => write!(f, "CUDA not initialized"),
            CudaError::AlreadyInitialized => write!(f, "CUDA already initialized"),
            CudaError::SyncError => write!(f, "CUDA synchronization error"),
            CudaError::MemcpyError => write!(f, "CUDA memory copy error"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Result type for CUDA operations
pub type CudaResult<T> = Result<T, CudaError>;

/// Convert CUDA error code to CudaError
pub fn check_cuda_error(code: i32) -> CudaResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError::DriverError(code))
    }
}

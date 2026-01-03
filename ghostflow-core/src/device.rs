//! Device abstraction for CPU/GPU execution

use std::fmt;

/// Trait for compute devices (CPU, CUDA, etc.)
pub trait Device: Clone + Send + Sync + 'static {
    /// Device name for display
    fn name(&self) -> &'static str;
    
    /// Check if this is a CPU device
    fn is_cpu(&self) -> bool;
    
    /// Check if this is a CUDA device
    fn is_cuda(&self) -> bool;
}

/// CPU device - default compute device
#[derive(Debug, Clone, Copy, Default)]
pub struct Cpu;

impl Device for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }
    
    fn is_cpu(&self) -> bool {
        true
    }
    
    fn is_cuda(&self) -> bool {
        false
    }
}

impl fmt::Display for Cpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cpu")
    }
}

/// CUDA device (placeholder for future implementation)
#[derive(Debug, Clone, Copy, Default)]
pub struct Cuda {
    pub device_id: i32,
}

impl Cuda {
    pub fn new(device_id: i32) -> Self {
        Cuda { device_id }
    }
}

impl Device for Cuda {
    fn name(&self) -> &'static str {
        "cuda"
    }
    
    fn is_cpu(&self) -> bool {
        false
    }
    
    fn is_cuda(&self) -> bool {
        true
    }
}

impl fmt::Display for Cuda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cuda:{}", self.device_id)
    }
}

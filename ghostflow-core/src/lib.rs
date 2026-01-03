//! GhostFlow Core - High-performance tensor operations
//! 
//! This crate provides the foundational tensor type and operations
//! for the GhostFlow ML framework.
//!
//! ## Phase 4 Optimizations (Beat JAX!)
//! - Operation fusion engine
//! - JIT compilation
//! - Memory layout optimization
//! - Custom optimized kernels

pub mod dtype;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod ops;
pub mod device;
pub mod error;
pub mod serialize;

// Phase 4: Advanced optimizations
pub mod fusion;
pub mod jit;
pub mod layout;

pub use dtype::DType;
pub use shape::{Shape, Strides};
pub use storage::Storage;
pub use tensor::Tensor;
pub use device::{Device, Cpu};
pub use error::{GhostError, Result};
pub use serialize::{StateDict, save_state_dict, load_state_dict, Serializable};

// Phase 4 exports
pub use fusion::{FusionEngine, ComputeGraph, FusionPattern};
pub use jit::{JitCompiler, CompiledKernel};
pub use layout::{LayoutOptimizer, MemoryLayout, DeviceInfo};

/// Prelude for convenient imports
#[allow(unused_imports)]
pub mod prelude {
    pub use crate::{Tensor, DType, Shape, Device, Cpu};
    pub use crate::tensor_ops::*;
    pub use crate::serialize::{StateDict, save_state_dict, load_state_dict, Serializable};
    pub use crate::{FusionEngine, JitCompiler, LayoutOptimizer};
}

/// Tensor operations trait extensions
#[allow(unused_imports)]
pub mod tensor_ops {
    pub use crate::ops::arithmetic::*;
    pub use crate::ops::reduction::*;
    pub use crate::ops::activation::*;
    pub use crate::ops::matmul::*;
}

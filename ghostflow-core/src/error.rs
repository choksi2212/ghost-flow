//! Error types for GhostFlow

use thiserror::Error;

/// Main error type for GhostFlow operations
#[derive(Error, Debug)]
pub enum GhostError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Dimension out of bounds: dim {dim} for tensor with {ndim} dimensions")]
    DimOutOfBounds { dim: usize, ndim: usize },

    #[error("Index out of bounds: index {index} for dimension of size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Data type mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: String, got: String },

    #[error("Device mismatch: tensors on different devices")]
    DeviceMismatch,

    #[error("Cannot broadcast shapes {a:?} and {b:?}")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },

    #[error("Operation requires gradient tracking")]
    NoGradient,

    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("IO error: {0}")]
    IOError(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Result type alias for GhostFlow operations
pub type Result<T> = std::result::Result<T, GhostError>;

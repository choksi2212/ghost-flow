//! # GhostFlow - Complete Machine Learning Framework in Rust
//!
//! GhostFlow is a production-ready machine learning framework built entirely in Rust,
//! designed to rival PyTorch and TensorFlow in both performance and ease of use.
//!
//! ## Features
//!
//! - **Tensor Operations**: Multi-dimensional arrays with SIMD optimization
//! - **Automatic Differentiation**: Full autograd engine with computational graph
//! - **Neural Networks**: CNN, RNN, LSTM, GRU, Transformer, Attention
//! - **50+ ML Algorithms**: Decision trees, random forests, SVM, clustering, and more
//! - **GPU Acceleration**: Hand-optimized CUDA kernels (optional)
//! - **Production Ready**: Zero warnings, comprehensive tests, full documentation
//!
//! ## Quick Start
//!
//! ```rust
//! use ghostflow::prelude::*;
//!
//! // Create tensors
//! let x = Tensor::randn(&[32, 784]);
//! let y = Tensor::randn(&[32, 10]);
//!
//! // Tensor operations
//! let z = x.matmul(&y.transpose(0, 1).unwrap()).unwrap();
//! ```
//!
//! ## Installation
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! ghostflow = "0.1"
//! ```
//!
//! With GPU support:
//!
//! ```toml
//! [dependencies]
//! ghostflow = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! ## Modules
//!
//! - [`core`] - Core tensor operations and data structures
//! - [`nn`] - Neural network layers and building blocks
//! - [`ml`] - Classical machine learning algorithms
//! - [`autograd`] - Automatic differentiation
//! - [`optim`] - Optimizers (SGD, Adam, AdamW)
//! - [`data`] - Data loading and preprocessing utilities
//! - [`cuda`] - GPU acceleration (optional)

// Re-export core (always available)
pub use ghostflow_core as core;
pub use ghostflow_core::*;

// Re-export optional modules
#[cfg(feature = "nn")]
pub use ghostflow_nn as nn;

#[cfg(feature = "ml")]
pub use ghostflow_ml as ml;

#[cfg(feature = "autograd")]
pub use ghostflow_autograd as autograd;

#[cfg(feature = "optim")]
pub use ghostflow_optim as optim;

#[cfg(feature = "data")]
pub use ghostflow_data as data;

#[cfg(feature = "cuda")]
pub use ghostflow_cuda as cuda;

/// Prelude module for convenient imports
///
/// Import everything you need with:
/// ```
/// use ghostflow::prelude::*;
/// ```
pub mod prelude {
    pub use crate::core::{Tensor, Shape, DType};
    
    #[cfg(feature = "nn")]
    pub use crate::nn::{Linear, Conv2d, ReLU, Sigmoid, Softmax};
    
    #[cfg(feature = "autograd")]
    pub use crate::autograd::*;
    
    #[cfg(feature = "optim")]
    pub use crate::optim::{SGD, Adam, AdamW};
    
    #[cfg(feature = "data")]
    pub use crate::data::{Dataset, DataLoader};
}

//! GhostFlow Autograd - Automatic Differentiation
//!
//! Tape-based reverse-mode automatic differentiation.

pub mod tape;
pub mod backward;
pub mod dynamic_graph;

pub use tape::GradTape;
pub use backward::backward;
pub use dynamic_graph::{DynamicGraph, DynamicContext, DynamicTensor, GraphNode};

/// Enable gradient computation for a tensor
pub fn enable_grad() {
    // Global flag to enable gradient tracking
}

/// Disable gradient computation (inference mode)
pub fn no_grad() {
    // Global flag to disable gradient tracking
}

//! GhostFlow Neural Network Layers
//!
//! High-level building blocks for neural networks.

pub mod module;
pub mod linear;
pub mod conv;
pub mod norm;
pub mod activation;
pub mod dropout;
pub mod loss;
pub mod init;
pub mod attention;
pub mod transformer;
pub mod embedding;
pub mod pooling;

pub use module::Module;
pub use linear::Linear;
pub use conv::{Conv1d, Conv2d};
pub use norm::{BatchNorm1d, BatchNorm2d, LayerNorm};
pub use activation::*;
pub use dropout::Dropout;
pub use loss::*;
pub use attention::{MultiHeadAttention, scaled_dot_product_attention};
pub use transformer::{
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoderLayer, FeedForward,
    PositionalEncoding, RotaryEmbedding,
};
pub use embedding::Embedding;
pub use pooling::*;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{Module, Linear, Conv2d, BatchNorm2d, LayerNorm, Dropout};
    pub use crate::activation::*;
    pub use crate::loss::*;
    pub use crate::attention::MultiHeadAttention;
    pub use crate::transformer::{TransformerEncoder, TransformerEncoderLayer};
    pub use crate::embedding::Embedding;
}

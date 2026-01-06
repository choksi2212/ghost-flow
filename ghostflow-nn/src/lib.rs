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
pub mod rnn;
pub mod quantization;
pub mod distributed;

pub use module::Module;
pub use linear::Linear;
pub use conv::{Conv1d, Conv2d, Conv3d, TransposeConv2d};
pub use norm::{BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm};
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
pub use rnn::{LSTM, LSTMCell, GRU, GRUCell};
pub use quantization::{
    QuantizedTensor, QuantizationConfig, QuantizationScheme,
    QuantizationAwareTraining, DynamicQuantization,
};
pub use distributed::{
    DistributedConfig, DistributedBackend, DataParallel, ModelParallel,
    GradientAccumulator, DistributedDataParallel, PipelineParallel,
};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{Module, Linear, Conv1d, Conv2d, Conv3d, TransposeConv2d};
    pub use crate::{BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm, Dropout};
    pub use crate::activation::*;
    pub use crate::loss::*;
    pub use crate::attention::MultiHeadAttention;
    pub use crate::transformer::{TransformerEncoder, TransformerEncoderLayer};
    pub use crate::embedding::Embedding;
    pub use crate::rnn::{LSTM, GRU};
}

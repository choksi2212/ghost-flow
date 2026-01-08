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
pub mod serialization;
pub mod onnx;
pub mod inference;
pub mod gnn;
pub mod rl;
pub mod federated;
pub mod differential_privacy;
pub mod adversarial;
pub mod vision_transformer;
pub mod bert;
pub mod gpt;

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
pub use serialization::{
    ModelCheckpoint, ModelMetadata, save_model, load_model,
};
pub use gnn::{
    Graph, GCNLayer, GATLayer, GraphSAGELayer, MPNNLayer, AggregatorType,
};
pub use rl::{
    ReplayBuffer, Experience, DQNAgent, QNetwork,
    PolicyNetwork, REINFORCEAgent, ActorCriticAgent, ValueNetwork, PPOAgent,
};
pub use federated::{
    FederatedClient, FederatedServer, AggregationStrategy,
    SecureAggregation, DifferentialPrivacy,
};
pub use onnx::{
    ONNXModel, ONNXNode, ONNXTensor, ONNXDataType, ONNXAttribute,
    tensor_to_onnx, onnx_to_tensor,
};
pub use inference::{
    InferenceConfig, InferenceOptimizer, InferenceSession,
    BatchInference, warmup_model,
};
pub use differential_privacy::{
    DPConfig, PrivacyAccountant, DPSGDOptimizer, PATEEnsemble, LocalDP,
};
pub use adversarial::{
    AttackConfig, AttackType, AdversarialAttack, AdversarialTrainingConfig,
    AdversarialTrainer, RandomizedSmoothing,
};
// pub use vision_transformer::{
//     VisionTransformer, ViTConfig, PatchEmbedding,
// };

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

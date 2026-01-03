//! Deep Learning Architectures
//!
//! This module contains implementations of major deep learning architectures.

pub mod cnn;
pub mod transformer;
pub mod gan;
pub mod object_detection;
pub mod segmentation;
pub mod rnn;
pub mod vae;
pub mod diffusion;
pub mod gnn;
pub mod specialized;
pub mod attention;
pub mod rl_meta;
pub mod efficient;
pub mod pooling;
pub mod nas;
pub mod multimodal;
pub mod self_supervised;
pub mod normalization;
pub mod activation_nets;
pub mod feedforward;
pub mod autoencoder;
pub mod capsule;
pub mod spiking;
pub mod neural_ode;

// Re-export major architectures
pub use cnn::{
    // ResNet family
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    ResidualBlock, BottleneckBlock,
    
    // ResNeXt
    ResNeXt50, ResNeXtBlock,
    
    // Wide ResNet
    WideResNet, WideResidualBlock,
    
    // VGG family
    VGG16, VGG19,
    
    // Inception family
    GoogLeNet, InceptionModule,
    
    // DenseNet family
    DenseNet121, DenseBlock, TransitionLayer,
    
    // MobileNet family
    MobileNetV1, MobileNetV2,
    DepthwiseSeparableConv, InvertedResidual,
    
    // EfficientNet family
    EfficientNetB0, MBConvBlock, SEBlock,
    
    // SqueezeNet
    SqueezeNet, FireModule,
};

pub use transformer::{
    // BERT family
    BERTBase, RoBERTa, ALBERT, DistilBERT,
    BERTEncoderLayer,
    
    // GPT family
    GPT2, GPTDecoderLayer,
    
    // Vision Transformers
    VisionTransformer, DeiT, SwinTransformer,
    PatchEmbedding, SwinTransformerBlock,
};

pub use gan::{
    // DCGAN
    DCGANGenerator, DCGANDiscriminator,
    
    // StyleGAN
    StyleGANGenerator, StyleGANMappingNetwork,
    StyleGANSynthesisBlock, AdaIN,
    
    // WGAN
    WGANCritic,
    
    // CycleGAN
    CycleGANGenerator, CycleGANDiscriminator,
    
    // Pix2Pix
    Pix2PixGenerator,
    
    // Conditional GAN
    ConditionalGANGenerator,
};

pub use object_detection::{
    // YOLO family
    YOLOv3, YOLOv3DetectionLayer, Darknet53,
    YOLOv4, CSPDarknet53,
    
    // Faster R-CNN
    FasterRCNN, RegionProposalNetwork, FasterRCNNHead,
    ROIPooling,
    
    // RetinaNet
    RetinaNet, RetinaNetHead, FeaturePyramidNetwork,
    
    // SSD
    SSDExtraLayers,
    
    // EfficientDet
    EfficientDet,
    
    // DETR
    DETR,
};

pub use segmentation::{
    // U-Net
    UNet, UNetEncoderBlock, UNetDecoderBlock,
    
    // DeepLab
    DeepLabV3Plus, ASPP,
    
    // PSPNet
    PyramidPoolingModule,
};

pub use rnn::{
    // LSTM
    LSTM, LSTMCell,
    
    // GRU
    GRU, GRUCell,
    
    // Bidirectional
    BiLSTM,
    
    // Seq2Seq
    Seq2Seq, Seq2SeqEncoder, Seq2SeqDecoder,
    
    // Attention
    Attention,
};

pub use vae::{
    // Standard VAE
    VAE, VAEEncoder, VAEDecoder,
    
    // Beta-VAE
    BetaVAE,
    
    // Conditional VAE
    ConditionalVAE,
    
    // VQ-VAE
    VQVAE,
};

pub use diffusion::{
    // DDPM
    DDPM, DiffusionUNet,
    
    // DDIM
    DDIM,
    
    // Latent Diffusion
    LatentDiffusion,
    
    // Components
    TimeEmbedding, ResidualBlock as DiffusionResidualBlock,
};

pub use gnn::{
    // GCN
    GCN, GCNLayer,
    
    // GAT
    GAT, GATLayer,
    
    // GraphSAGE
    GraphSAGE, GraphSAGELayer,
    
    // GIN
    GIN, GINLayer,
    
    // MPNN
    MPNN, MPNNLayer,
};


pub use specialized::{
    // 3D CNNs
    C3D, I3D, Conv3d,
    
    // Video Models
    SlowFast,
    
    // Audio Models
    WaveNet,
    
    // Point Cloud
    PointNet, PointNetPlusPlus,
    
    // Text-to-Speech
    Tacotron2,
};


pub use attention::{
    // Attention Mechanisms
    SelfAttention,
    CrossAttention,
    SEBlock,
    CBAM,
};


pub use rl_meta::{
    // Reinforcement Learning
    DQN,
    DuelingDQN,
    ActorCritic,
    PPOActor,
    PPOCritic,
    
    // Meta-Learning
    MAML,
    PrototypicalNetwork,
    MatchingNetwork,
    RelationNetwork,
    SNAIL,
};


pub use nas::{
    // Neural Architecture Search
    NASNetMobile,
    DARTS,
    ENAS,
    AmoebaNet,
    ProxylessNAS,
    FBNet,
};

pub use efficient::{
    // Efficient Models
    GhostNet,
    MnasNet,
    MobileNetV3,
    RegNet,
};


pub use multimodal::{
    // Multi-Modal Models
    CLIP,
    ALIGN,
    ViLT,
    BLIP,
    Flamingo,
};


pub use self_supervised::{
    // Self-Supervised Learning
    SimCLR,
    MoCo,
    BYOL,
    SwAV,
    DINO,
    MAE,
};


pub use normalization::{
    // Normalization Techniques
    BatchNorm,
    LayerNormalization,
    InstanceNorm,
    GroupNorm,
    WeightNorm,
    SpectralNorm,
};



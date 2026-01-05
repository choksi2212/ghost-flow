# GhostFlow v0.2.0 Quick Reference Guide

Quick reference for all new features in v0.2.0. Copy-paste ready examples!

## 🚀 Quick Start

```rust
use ghostflow_nn::prelude::*;
use ghostflow_core::Tensor;
```

## 📦 New Convolutional Layers

### Conv1d - 1D Convolution
```rust
// For sequence data (audio, time series, text)
let conv1d = Conv1d::new(
    3,      // in_channels
    16,     // out_channels
    3,      // kernel_size
    1,      // stride
    1       // padding
);

let input = Tensor::randn(&[2, 3, 32]);  // [batch, channels, length]
let output = conv1d.forward(&input);      // [2, 16, 32]
```

### Conv3d - 3D Convolution
```rust
// For volumetric data (video, medical imaging)
let conv3d = Conv3d::new(
    3,              // in_channels
    16,             // out_channels
    (3, 3, 3),      // kernel_size (depth, height, width)
    (1, 1, 1),      // stride
    (1, 1, 1)       // padding
);

let input = Tensor::randn(&[2, 3, 8, 8, 8]);  // [batch, channels, D, H, W]
let output = conv3d.forward(&input);           // [2, 16, 8, 8, 8]
```

### TransposeConv2d - Deconvolution/Upsampling
```rust
// For image generation (GANs, autoencoders)
let tconv = TransposeConv2d::new(
    16,         // in_channels
    3,          // out_channels
    (4, 4),     // kernel_size
    (2, 2),     // stride (upsampling factor)
    (1, 1),     // padding
    (0, 0)      // output_padding
);

let input = Tensor::randn(&[2, 16, 8, 8]);
let output = tconv.forward(&input);  // [2, 3, 16, 16] - 2x upsampled!
```

## 🔄 New Normalization Layers

### GroupNorm - Group Normalization
```rust
// Better than BatchNorm for small batches
let group_norm = GroupNorm::new(
    4,      // num_groups
    16      // num_channels (must be divisible by num_groups)
);

let input = Tensor::randn(&[2, 16, 8, 8]);
let output = group_norm.forward(&input);  // Same shape
```

### InstanceNorm - Instance Normalization
```rust
// For style transfer and GANs
let instance_norm = InstanceNorm::new(16);  // num_channels

let input = Tensor::randn(&[2, 16, 8, 8]);
let output = instance_norm.forward(&input);  // Same shape
```

## ⚡ New Activation Functions

### Swish (Parameterized)
```rust
let swish = Swish::new(1.0);  // beta parameter
let output = swish.forward(&input);
```

### SiLU (Swish with beta=1)
```rust
let silu = SiLU::new();
let output = silu.forward(&input);
```

### Mish
```rust
let mish = Mish::new();
let output = mish.forward(&input);
```

### ELU (Exponential Linear Unit)
```rust
let elu = ELU::new(1.0);  // alpha parameter
let output = elu.forward(&input);
```

### SELU (Scaled ELU)
```rust
let selu = SELU::new();  // Uses standard parameters
let output = selu.forward(&input);
```

### Softplus
```rust
let softplus = Softplus::default();  // beta=1.0, threshold=20.0
let output = softplus.forward(&input);

// Or with custom parameters
let softplus = Softplus::new(1.0, 20.0);
```

## 📉 New Loss Functions

### Focal Loss - For Class Imbalance
```rust
use ghostflow_nn::loss::focal_loss;

let logits = Tensor::randn(&[32, 10]);    // [batch, num_classes]
let targets = Tensor::randn(&[32]);        // [batch] - class indices

let loss = focal_loss(
    &logits,
    &targets,
    1.0,    // alpha (weighting factor)
    2.0     // gamma (focusing parameter)
);
```

### Contrastive Loss - For Metric Learning
```rust
use ghostflow_nn::loss::contrastive_loss;

let x1 = Tensor::randn(&[32, 128]);       // First embeddings
let x2 = Tensor::randn(&[32, 128]);       // Second embeddings
let labels = Tensor::randn(&[32]);         // 1.0 for similar, 0.0 for dissimilar

let loss = contrastive_loss(
    &x1,
    &x2,
    &labels,
    1.0     // margin
);
```

### Triplet Loss - For Face Recognition
```rust
use ghostflow_nn::loss::triplet_margin_loss;

let anchor = Tensor::randn(&[32, 128]);
let positive = Tensor::randn(&[32, 128]);
let negative = Tensor::randn(&[32, 128]);

let loss = triplet_margin_loss(
    &anchor,
    &positive,
    &negative,
    0.5     // margin
);
```

### Huber Loss - Robust Regression
```rust
use ghostflow_nn::loss::huber_loss;

let predictions = Tensor::randn(&[32, 1]);
let targets = Tensor::randn(&[32, 1]);

let loss = huber_loss(
    &predictions,
    &targets,
    1.0     // delta (threshold between MSE and MAE)
);
```

## 🏗️ Common Architectures

### Simple CNN with New Features
```rust
use ghostflow_nn::prelude::*;

// Feature extractor
let conv1 = Conv2d::new(3, 64, 3, 1, 1);
let gn1 = GroupNorm::new(8, 64);
let silu = SiLU::new();

let conv2 = Conv2d::new(64, 128, 3, 2, 1);
let gn2 = GroupNorm::new(8, 128);

// Forward pass
let x = conv1.forward(&input);
let x = gn1.forward(&x);
let x = silu.forward(&x);

let x = conv2.forward(&x);
let x = gn2.forward(&x);
let x = silu.forward(&x);
```

### Autoencoder with TransposeConv2d
```rust
// Encoder
let enc1 = Conv2d::new(3, 64, 4, 2, 1);    // Downsample
let enc2 = Conv2d::new(64, 128, 4, 2, 1);  // Downsample

// Decoder
let dec1 = TransposeConv2d::new(128, 64, 4, 2, 1, 0);  // Upsample
let dec2 = TransposeConv2d::new(64, 3, 4, 2, 1, 0);    // Upsample

// Forward
let z = enc2.forward(&enc1.forward(&input));
let recon = dec2.forward(&dec1.forward(&z));
```

### 3D Medical Imaging Network
```rust
let conv3d_1 = Conv3d::new(1, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1));
let in1 = InstanceNorm::new(32);
let mish = Mish::new();

let conv3d_2 = Conv3d::new(32, 64, (3, 3, 3), (2, 2, 2), (1, 1, 1));
let in2 = InstanceNorm::new(64);

// Forward
let x = conv3d_1.forward(&input);
let x = in1.forward(&x);
let x = mish.forward(&x);

let x = conv3d_2.forward(&x);
let x = in2.forward(&x);
let x = mish.forward(&x);
```

### Siamese Network with Contrastive Loss
```rust
// Shared encoder
let encoder = |x: &Tensor| {
    let conv1 = Conv2d::new(3, 64, 3, 1, 1);
    let conv2 = Conv2d::new(64, 128, 3, 2, 1);
    let linear = Linear::new(128 * 16 * 16, 256);
    
    let x = conv2.forward(&conv1.forward(x));
    linear.forward(&x.flatten())
};

// Training
let emb1 = encoder(&img1);
let emb2 = encoder(&img2);
let loss = contrastive_loss(&emb1, &emb2, &labels, 1.0);
```

## 🎯 Use Case Cheat Sheet

| Task | Layers | Activation | Loss |
|------|--------|------------|------|
| **Image Classification** | Conv2d, GroupNorm | SiLU/Mish | Focal Loss |
| **Image Generation** | TransposeConv2d, InstanceNorm | Mish | MSE/BCE |
| **Face Recognition** | Conv2d, BatchNorm | ELU | Triplet Loss |
| **Style Transfer** | Conv2d, InstanceNorm | ReLU | MSE |
| **Audio Processing** | Conv1d, BatchNorm | GELU | MSE |
| **Video Analysis** | Conv3d, GroupNorm | SiLU | CrossEntropy |
| **Metric Learning** | Conv2d, Linear | SELU | Contrastive |
| **Robust Regression** | Linear | ReLU | Huber Loss |

## 💡 Pro Tips

### When to Use GroupNorm vs BatchNorm
```rust
// Use BatchNorm for large batches (>32)
let bn = BatchNorm2d::new(64);

// Use GroupNorm for small batches or when batch size varies
let gn = GroupNorm::new(8, 64);  // 8 groups of 8 channels each
```

### When to Use InstanceNorm
```rust
// Use InstanceNorm for:
// - Style transfer
// - GANs
// - When each sample should be normalized independently
let in_norm = InstanceNorm::new(64);
```

### Choosing Activation Functions
```rust
// General purpose: SiLU/Swish (modern, smooth)
let silu = SiLU::new();

// Self-normalizing networks: SELU
let selu = SELU::new();

// When you need smooth gradients: Mish
let mish = Mish::new();

// When you want negative saturation: ELU
let elu = ELU::new(1.0);
```

### Loss Function Selection
```rust
// Imbalanced classification: Focal Loss
let loss = focal_loss(&logits, &targets, 1.0, 2.0);

// Learning embeddings: Triplet or Contrastive
let loss = triplet_margin_loss(&anchor, &pos, &neg, 0.5);

// Regression with outliers: Huber Loss
let loss = huber_loss(&pred, &target, 1.0);
```

## 🔍 Debugging Tips

### Check Tensor Shapes
```rust
println!("Input shape: {:?}", input.dims());
let output = layer.forward(&input);
println!("Output shape: {:?}", output.dims());
```

### Verify Normalization
```rust
let output = norm_layer.forward(&input);
let mean = output.mean();
let std = output.std();
println!("Mean: {:.4}, Std: {:.4}", mean, std);
// Should be close to 0 and 1 respectively
```

### Monitor Loss Values
```rust
let loss = loss_fn(&pred, &target);
println!("Loss: {:.4}", loss.data_f32()[0]);
// Should decrease over training
```

## 📚 Further Reading

- **GroupNorm Paper**: https://arxiv.org/abs/1803.08494
- **Focal Loss Paper**: https://arxiv.org/abs/1708.02002
- **Swish Paper**: https://arxiv.org/abs/1710.05941
- **Mish Paper**: https://arxiv.org/abs/1908.08681

## 🆘 Common Issues

### Issue: GroupNorm assertion failure
```rust
// ❌ Wrong: channels not divisible by groups
let gn = GroupNorm::new(3, 16);  // 16 % 3 != 0

// ✅ Correct: channels divisible by groups
let gn = GroupNorm::new(4, 16);  // 16 % 4 == 0
```

### Issue: TransposeConv2d output size
```rust
// Output size formula:
// H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding

// For 2x upsampling:
let tconv = TransposeConv2d::new(64, 32, (4, 4), (2, 2), (1, 1), (0, 0));
// 8x8 -> 16x16
```

### Issue: Loss not decreasing
```rust
// Check:
// 1. Learning rate (might be too high/low)
// 2. Loss function matches task
// 3. Data normalization
// 4. Gradient flow (use simpler activations first)
```

---

**Happy coding with GhostFlow v0.2.0!** 🚀

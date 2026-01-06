//! Normalization-based Architectures - Batch Norm variants, Layer Norm, Group Norm, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense, BatchNorm2d};
use crate::deep::activations::ReLU;

/// Batch Renormalization Network
pub struct BatchRenorm {
    conv_layers: Vec<Conv2d>,
    renorm_layers: Vec<BatchRenormLayer>,
    fc: Dense,
}

struct BatchRenormLayer {
    bn: BatchNorm2d,
    r_max: f32,
    d_max: f32,
}

impl BatchRenormLayer {
    fn new(num_features: usize) -> Self {
        BatchRenormLayer {
            bn: BatchNorm2d::new(num_features),
            r_max: 3.0,
            d_max: 5.0,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // Simplified batch renormalization
        self.bn.forward(x, training)
    }
}

impl BatchRenorm {
    pub fn new(num_classes: usize) -> Self {
        BatchRenorm {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
                Conv2d::new(128, 256, (3, 3)).padding((1, 1)),
            ],
            renorm_layers: vec![
                BatchRenormLayer::new(64),
                BatchRenormLayer::new(128),
                BatchRenormLayer::new(256),
            ],
            fc: Dense::new(256, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, renorm) in self.conv_layers.iter_mut().zip(self.renorm_layers.iter_mut()) {
            out = conv.forward(&out, training);
            out = renorm.forward(&out, training);
            out = ReLU::new().forward(&out);
        }
        
        self.fc.forward(&out, training)
    }
}

/// Group Normalization Network
pub struct GroupNorm {
    conv_layers: Vec<Conv2d>,
    gn_layers: Vec<GroupNormLayer>,
    fc: Dense,
}

struct GroupNormLayer {
    num_groups: usize,
    num_channels: usize,
    eps: f32,
}

impl GroupNormLayer {
    fn new(num_groups: usize, num_channels: usize) -> Self {
        GroupNormLayer {
            num_groups,
            num_channels,
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let channels_per_group = channels / self.num_groups;
        let mut result = vec![0.0f32; data.len()];
        
        for b in 0..batch {
            for g in 0..self.num_groups {
                // Compute mean and variance for this group
                let mut sum = 0.0f32;
                let mut count = 0;
                
                for c in (g * channels_per_group)..((g + 1) * channels_per_group) {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            sum += data[idx];
                            count += 1;
                        }
                    }
                }
                
                let mean = sum / count as f32;
                
                let mut var_sum = 0.0f32;
                for c in (g * channels_per_group)..((g + 1) * channels_per_group) {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            var_sum += (data[idx] - mean).powi(2);
                        }
                    }
                }
                
                let variance = var_sum / count as f32;
                let std = (variance + self.eps).sqrt();
                
                // Normalize
                for c in (g * channels_per_group)..((g + 1) * channels_per_group) {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            result[idx] = (data[idx] - mean) / std;
                        }
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, dims).unwrap()
    }
}

impl GroupNorm {
    pub fn new(num_classes: usize) -> Self {
        GroupNorm {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
                Conv2d::new(128, 256, (3, 3)).padding((1, 1)),
            ],
            gn_layers: vec![
                GroupNormLayer::new(8, 64),
                GroupNormLayer::new(8, 128),
                GroupNormLayer::new(8, 256),
            ],
            fc: Dense::new(256, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, gn) in self.conv_layers.iter_mut().zip(self.gn_layers.iter()) {
            out = conv.forward(&out, training);
            out = gn.forward(&out);
            out = ReLU::new().forward(&out);
        }
        
        self.fc.forward(&out, training)
    }
}

/// Instance Normalization Network
pub struct InstanceNorm {
    conv_layers: Vec<Conv2d>,
    in_layers: Vec<InstanceNormLayer>,
    fc: Dense,
}

struct InstanceNormLayer {
    eps: f32,
}

impl InstanceNormLayer {
    fn new() -> Self {
        InstanceNormLayer {
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let mut result = vec![0.0f32; data.len()];
        
        for b in 0..batch {
            for c in 0..channels {
                // Compute mean for this instance
                let mut sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        sum += data[idx];
                    }
                }
                let mean = sum / (height * width) as f32;
                
                // Compute variance
                let mut var_sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        var_sum += (data[idx] - mean).powi(2);
                    }
                }
                let variance = var_sum / (height * width) as f32;
                let std = (variance + self.eps).sqrt();
                
                // Normalize
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        result[idx] = (data[idx] - mean) / std;
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, dims).unwrap()
    }
}

impl InstanceNorm {
    pub fn new(num_classes: usize) -> Self {
        InstanceNorm {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
            ],
            in_layers: vec![
                InstanceNormLayer::new(),
                InstanceNormLayer::new(),
            ],
            fc: Dense::new(128, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, in_layer) in self.conv_layers.iter_mut().zip(self.in_layers.iter()) {
            out = conv.forward(&out, training);
            out = in_layer.forward(&out);
            out = ReLU::new().forward(&out);
        }
        
        self.fc.forward(&out, training)
    }
}

/// Layer Normalization Network
pub struct LayerNormNet {
    layers: Vec<Dense>,
    ln_layers: Vec<LayerNormLayer>,
}

struct LayerNormLayer {
    normalized_shape: usize,
    eps: f32,
}

impl LayerNormLayer {
    fn new(normalized_shape: usize) -> Self {
        LayerNormLayer {
            normalized_shape,
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let batch_size = x.dims()[0];
        let dim = x.dims()[1];
        
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            let offset = b * dim;
            
            // Compute mean
            let mut sum = 0.0f32;
            for i in 0..dim {
                sum += data[offset + i];
            }
            let mean = sum / dim as f32;
            
            // Compute variance
            let mut var_sum = 0.0f32;
            for i in 0..dim {
                var_sum += (data[offset + i] - mean).powi(2);
            }
            let variance = var_sum / dim as f32;
            let std = (variance + self.eps).sqrt();
            
            // Normalize
            for i in 0..dim {
                result.push((data[offset + i] - mean) / std);
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

impl LayerNormNet {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        LayerNormNet {
            layers: vec![
                Dense::new(input_dim, hidden_dim),
                Dense::new(hidden_dim, hidden_dim),
                Dense::new(hidden_dim, output_dim),
            ],
            ln_layers: vec![
                LayerNormLayer::new(hidden_dim),
                LayerNormLayer::new(hidden_dim),
            ],
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.layers[0].forward(x, training);
        out = self.ln_layers[0].forward(&out);
        out = ReLU::new().forward(&out);
        
        out = self.layers[1].forward(&out, training);
        out = self.ln_layers[1].forward(&out);
        out = ReLU::new().forward(&out);
        
        self.layers[2].forward(&out, training)
    }
}

/// Switchable Normalization
pub struct SwitchableNorm {
    conv_layers: Vec<Conv2d>,
    sn_layers: Vec<SwitchableNormLayer>,
    fc: Dense,
}

struct SwitchableNormLayer {
    bn: BatchNorm2d,
    in_layer: InstanceNormLayer,
    ln_weight: f32,
    in_weight: f32,
    bn_weight: f32,
}

impl SwitchableNormLayer {
    fn new(num_features: usize) -> Self {
        SwitchableNormLayer {
            bn: BatchNorm2d::new(num_features),
            in_layer: InstanceNormLayer::new(),
            ln_weight: 0.33,
            in_weight: 0.33,
            bn_weight: 0.34,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let bn_out = self.bn.forward(x, training);
        let in_out = self.in_layer.forward(x);
        let ln_out = self.layer_norm(x);
        
        self.combine_norms(&bn_out, &in_out, &ln_out)
    }

    fn layer_norm(&self, x: &Tensor) -> Tensor {
        // Simplified layer norm
        x.clone()
    }

    fn combine_norms(&self, bn: &Tensor, in_norm: &Tensor, ln: &Tensor) -> Tensor {
        let bn_data = bn.data_f32();
        let in_data = in_norm.data_f32();
        let ln_data = ln.data_f32();
        
        let result: Vec<f32> = (0..bn_data.len())
            .map(|i| {
                self.bn_weight * bn_data[i] + 
                self.in_weight * in_data[i] + 
                self.ln_weight * ln_data[i]
            })
            .collect();
        
        Tensor::from_slice(&result, bn.dims()).unwrap()
    }
}

impl SwitchableNorm {
    pub fn new(num_classes: usize) -> Self {
        SwitchableNorm {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
            ],
            sn_layers: vec![
                SwitchableNormLayer::new(64),
                SwitchableNormLayer::new(128),
            ],
            fc: Dense::new(128, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, sn) in self.conv_layers.iter_mut().zip(self.sn_layers.iter_mut()) {
            out = conv.forward(&out, training);
            out = sn.forward(&out, training);
            out = ReLU::new().forward(&out);
        }
        
        self.fc.forward(&out, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_renorm() {
        let mut model = BatchRenorm::new(10);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 3 * 32 * 32], &[2, 3, 32, 32]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }

    #[test]
    fn test_group_norm() {
        let mut model = GroupNorm::new(10);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 3 * 32 * 32], &[2, 3, 32, 32]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }

    #[test]
    fn test_layer_norm_net() {
        let mut model = LayerNormNet::new(784, 256, 10);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 784], &[2, 784]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }
}



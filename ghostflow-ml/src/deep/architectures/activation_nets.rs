//! Activation-based Architectures - PReLU, ELU, Swish, Mish networks, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense, BatchNorm2d};

/// PReLU Network (Parametric ReLU)
pub struct PReLUNet {
    conv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
    prelu_layers: Vec<PReLU>,
    fc: Dense,
}

struct PReLU {
    alpha: Vec<f32>,
}

impl PReLU {
    fn new(num_parameters: usize) -> Self {
        PReLU {
            alpha: vec![0.25f32; num_parameters],
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .enumerate()
            .map(|(i, &val)| {
                if val > 0.0 {
                    val
                } else {
                    self.alpha[i % self.alpha.len()] * val
                }
            })
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

impl PReLUNet {
    pub fn new(num_classes: usize) -> Self {
        PReLUNet {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
                Conv2d::new(128, 256, (3, 3)).padding((1, 1)),
            ],
            bn_layers: vec![
                BatchNorm2d::new(64),
                BatchNorm2d::new(128),
                BatchNorm2d::new(256),
            ],
            prelu_layers: vec![
                PReLU::new(64),
                PReLU::new(128),
                PReLU::new(256),
            ],
            fc: Dense::new(256, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for ((conv, bn), prelu) in self.conv_layers.iter_mut()
            .zip(self.bn_layers.iter_mut())
            .zip(self.prelu_layers.iter()) {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = prelu.forward(&out);
        }
        
        self.fc.forward(&out, training)
    }
}

/// ELU Network (Exponential Linear Unit)
pub struct ELUNet {
    conv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
    alpha: f32,
    fc: Dense,
}

impl ELUNet {
    pub fn new(num_classes: usize, alpha: f32) -> Self {
        ELUNet {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
            ],
            bn_layers: vec![
                BatchNorm2d::new(64),
                BatchNorm2d::new(128),
            ],
            alpha,
            fc: Dense::new(128, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, bn) in self.conv_layers.iter_mut().zip(self.bn_layers.iter_mut()) {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = self.elu(&out);
        }
        
        self.fc.forward(&out, training)
    }

    fn elu(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| {
                if val > 0.0 {
                    val
                } else {
                    self.alpha * (val.exp() - 1.0)
                }
            })
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Swish Network (Self-Gated activation)
pub struct SwishNet {
    conv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
    beta: f32,
    fc: Dense,
}

impl SwishNet {
    pub fn new(num_classes: usize, beta: f32) -> Self {
        SwishNet {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
                Conv2d::new(128, 256, (3, 3)).padding((1, 1)),
            ],
            bn_layers: vec![
                BatchNorm2d::new(64),
                BatchNorm2d::new(128),
                BatchNorm2d::new(256),
            ],
            beta,
            fc: Dense::new(256, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, bn) in self.conv_layers.iter_mut().zip(self.bn_layers.iter_mut()) {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = self.swish(&out);
        }
        
        self.fc.forward(&out, training)
    }

    fn swish(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| {
                val / (1.0 + (-self.beta * val).exp())
            })
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Mish Network
pub struct MishNet {
    conv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
    fc: Dense,
}

impl MishNet {
    pub fn new(num_classes: usize) -> Self {
        MishNet {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
                Conv2d::new(128, 256, (3, 3)).padding((1, 1)),
            ],
            bn_layers: vec![
                BatchNorm2d::new(64),
                BatchNorm2d::new(128),
                BatchNorm2d::new(256),
            ],
            fc: Dense::new(256, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, bn) in self.conv_layers.iter_mut().zip(self.bn_layers.iter_mut()) {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = self.mish(&out);
        }
        
        self.fc.forward(&out, training)
    }

    fn mish(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| {
                val * ((1.0 + val.exp()).ln()).tanh()
            })
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// GELU Network (Gaussian Error Linear Unit)
pub struct GELUNet {
    conv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
    fc: Dense,
}

impl GELUNet {
    pub fn new(num_classes: usize) -> Self {
        GELUNet {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
            ],
            bn_layers: vec![
                BatchNorm2d::new(64),
                BatchNorm2d::new(128),
            ],
            fc: Dense::new(128, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, bn) in self.conv_layers.iter_mut().zip(self.bn_layers.iter_mut()) {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = self.gelu(&out);
        }
        
        self.fc.forward(&out, training)
    }

    fn gelu(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * val * (1.0 + (sqrt_2_over_pi * (val + 0.044715 * val.powi(3))).tanh())
            })
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// SELU Network (Scaled Exponential Linear Unit)
pub struct SELUNet {
    conv_layers: Vec<Conv2d>,
    fc: Dense,
    alpha: f32,
    scale: f32,
}

impl SELUNet {
    pub fn new(num_classes: usize) -> Self {
        SELUNet {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
            ],
            fc: Dense::new(128, num_classes),
            alpha: 1.6732632423543772848170429916717,
            scale: 1.0507009873554804934193349852946,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for conv in &mut self.conv_layers {
            out = conv.forward(&out, training);
            out = self.selu(&out);
        }
        
        self.fc.forward(&out, training)
    }

    fn selu(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| {
                if val > 0.0 {
                    self.scale * val
                } else {
                    self.scale * self.alpha * (val.exp() - 1.0)
                }
            })
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Maxout Network
pub struct MaxoutNet {
    conv_layers: Vec<Conv2d>,
    maxout_layers: Vec<MaxoutLayer>,
    fc: Dense,
}

struct MaxoutLayer {
    num_pieces: usize,
}

impl MaxoutLayer {
    fn new(num_pieces: usize) -> Self {
        MaxoutLayer { num_pieces }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let out_channels = channels / self.num_pieces;
        let mut result = Vec::new();
        
        for b in 0..batch {
            for c in 0..out_channels {
                for h in 0..height {
                    for w in 0..width {
                        let mut max_val = f32::MIN;
                        
                        for p in 0..self.num_pieces {
                            let idx = ((b * channels + c * self.num_pieces + p) * height + h) * width + w;
                            max_val = max_val.max(data[idx]);
                        }
                        
                        result.push(max_val);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch, out_channels, height, width]).unwrap()
    }
}

impl MaxoutNet {
    pub fn new(num_classes: usize) -> Self {
        MaxoutNet {
            conv_layers: vec![
                Conv2d::new(3, 128, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 256, (3, 3)).padding((1, 1)),
            ],
            maxout_layers: vec![
                MaxoutLayer::new(2),
                MaxoutLayer::new(2),
            ],
            fc: Dense::new(128, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, maxout) in self.conv_layers.iter_mut().zip(self.maxout_layers.iter()) {
            out = conv.forward(&out, training);
            out = maxout.forward(&out);
        }
        
        self.fc.forward(&out, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelu_net() {
        let mut model = PReLUNet::new(10);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }

    #[test]
    fn test_swish_net() {
        let mut model = SwishNet::new(10, 1.0);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }

    #[test]
    fn test_mish_net() {
        let mut model = MishNet::new(10);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }
}



//! Pooling-based Architectures and Variants

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense, BatchNorm2d};
use crate::deep::activations::ReLU;

/// Spatial Pyramid Pooling (SPP)
pub struct SPP {
    pyramid_levels: Vec<usize>,
    fc: Dense,
}

impl SPP {
    pub fn new(pyramid_levels: Vec<usize>, in_channels: usize, num_classes: usize) -> Self {
        let total_bins: usize = pyramid_levels.iter().map(|&l| l * l).sum();
        SPP {
            pyramid_levels,
            fc: Dense::new(in_channels * total_bins, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let pooled_features = self.spatial_pyramid_pool(x);
        self.fc.forward(&pooled_features, training)
    }

    fn spatial_pyramid_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let mut all_features = Vec::new();
        
        for &level in &self.pyramid_levels {
            let bin_h = height / level;
            let bin_w = width / level;
            
            for b in 0..batch {
                for c in 0..channels {
                    for i in 0..level {
                        for j in 0..level {
                            let mut max_val = f32::MIN;
                            
                            for h in (i * bin_h)..((i + 1) * bin_h).min(height) {
                                for w in (j * bin_w)..((j + 1) * bin_w).min(width) {
                                    let idx = ((b * channels + c) * height + h) * width + w;
                                    max_val = max_val.max(data[idx]);
                                }
                            }
                            
                            all_features.push(max_val);
                        }
                    }
                }
            }
        }
        
        let total_bins: usize = self.pyramid_levels.iter().map(|&l| l * l).sum();
        Tensor::from_slice(&all_features, &[batch, channels * total_bins]).unwrap()
    }
}

/// Stochastic Pooling Network
pub struct StochasticPooling {
    conv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
    fc: Dense,
}

impl StochasticPooling {
    pub fn new(num_classes: usize) -> Self {
        StochasticPooling {
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
        
        for (i, (conv, bn)) in self.conv_layers.iter_mut().zip(self.bn_layers.iter_mut()).enumerate() {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = ReLU::new().forward(&out);
            
            if training {
                out = self.stochastic_pool(&out);
            } else {
                out = self.avg_pool(&out);
            }
        }
        
        self.fc.forward(&out, training)
    }

    fn stochastic_pool(&self, x: &Tensor) -> Tensor {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let new_h = height / 2;
        let new_w = width / 2;
        let mut result = Vec::new();
        
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..new_h {
                    for w in 0..new_w {
                        // Compute probabilities
                        let mut sum = 0.0f32;
                        let mut values = Vec::new();
                        
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let idx = ((b * channels + c) * height + h * 2 + dh) * width + w * 2 + dw;
                                let val = data[idx].max(0.0);
                                values.push(val);
                                sum += val;
                            }
                        }
                        
                        // Sample based on probabilities
                        if sum > 0.0 {
                            let rand_val: f32 = rng.gen::<f32>() * sum;
                            let mut cumsum = 0.0f32;
                            let mut selected = values[0];
                            
                            for val in values {
                                cumsum += val;
                                if cumsum >= rand_val {
                                    selected = val;
                                    break;
                                }
                            }
                            result.push(selected);
                        } else {
                            result.push(0.0);
                        }
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels, new_h, new_w]).unwrap()
    }

    fn avg_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let new_h = height / 2;
        let new_w = width / 2;
        let mut result = Vec::new();
        
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..new_h {
                    for w in 0..new_w {
                        let mut sum = 0.0f32;
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let idx = ((b * channels + c) * height + h * 2 + dh) * width + w * 2 + dw;
                                sum += data[idx];
                            }
                        }
                        result.push(sum / 4.0);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels, new_h, new_w]).unwrap()
    }
}

/// Mixed Pooling Network
pub struct MixedPooling {
    conv_layers: Vec<Conv2d>,
    fc: Dense,
    alpha: f32,
}

impl MixedPooling {
    pub fn new(num_classes: usize, alpha: f32) -> Self {
        MixedPooling {
            conv_layers: vec![
                Conv2d::new(3, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).padding((1, 1)),
            ],
            fc: Dense::new(128, num_classes),
            alpha,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for conv in &mut self.conv_layers {
            out = conv.forward(&out, training);
            out = ReLU::new().forward(&out);
            out = self.mixed_pool(&out);
        }
        
        self.fc.forward(&out, training)
    }

    fn mixed_pool(&self, x: &Tensor) -> Tensor {
        let max_pooled = self.max_pool(x);
        let avg_pooled = self.avg_pool(x);
        
        self.mix_tensors(&max_pooled, &avg_pooled, self.alpha)
    }

    fn max_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let new_h = height / 2;
        let new_w = width / 2;
        let mut result = Vec::new();
        
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..new_h {
                    for w in 0..new_w {
                        let mut max_val = f32::MIN;
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let idx = ((b * channels + c) * height + h * 2 + dh) * width + w * 2 + dw;
                                max_val = max_val.max(data[idx]);
                            }
                        }
                        result.push(max_val);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels, new_h, new_w]).unwrap()
    }

    fn avg_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let new_h = height / 2;
        let new_w = width / 2;
        let mut result = Vec::new();
        
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..new_h {
                    for w in 0..new_w {
                        let mut sum = 0.0f32;
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let idx = ((b * channels + c) * height + h * 2 + dh) * width + w * 2 + dw;
                                sum += data[idx];
                            }
                        }
                        result.push(sum / 4.0);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels, new_h, new_w]).unwrap()
    }

    fn mix_tensors(&self, max_pool: &Tensor, avg_pool: &Tensor, alpha: f32) -> Tensor {
        let max_data = max_pool.data_f32();
        let avg_data = avg_pool.data_f32();
        
        let result: Vec<f32> = max_data.iter()
            .zip(avg_data.iter())
            .map(|(&m, &a)| alpha * m + (1.0 - alpha) * a)
            .collect();
        
        Tensor::from_slice(&result, max_pool.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spp() {
        let mut spp = SPP::new(vec![1, 2, 4], 256, 1000);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 256 * 13 * 13], &[1, 256, 13, 13]).unwrap();
        let output = spp.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_stochastic_pooling() {
        let mut model = StochasticPooling::new(10);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
     
   }
}



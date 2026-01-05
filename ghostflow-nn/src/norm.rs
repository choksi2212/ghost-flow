//! Normalization layers

use ghostflow_core::Tensor;
use crate::module::Module;

/// Batch Normalization for 1D inputs (N, C) or (N, C, L)
pub struct BatchNorm1d {
    #[allow(dead_code)]
    num_features: usize,
    gamma: Tensor,  // scale
    beta: Tensor,   // shift
    running_mean: Tensor,
    running_var: Tensor,
    eps: f32,
    #[allow(dead_code)]
    momentum: f32,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self::with_params(num_features, 1e-5, 0.1)
    }

    pub fn with_params(num_features: usize, eps: f32, momentum: f32) -> Self {
        BatchNorm1d {
            num_features,
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            eps,
            momentum,
            training: true,
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let data = input.data_f32();
        let gamma = self.gamma.data_f32();
        let beta = self.beta.data_f32();
        
        let batch_size = dims[0];
        let channels = dims[1];
        let spatial_size = if dims.len() > 2 { dims[2] } else { 1 };
        
        let (mean, var) = if self.training {
            // Compute batch statistics
            let mut mean = vec![0.0f32; channels];
            let mut var = vec![0.0f32; channels];
            let n = (batch_size * spatial_size) as f32;
            
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        sum += data[idx];
                    }
                }
                mean[c] = sum / n;
                
                let mut var_sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        var_sum += (data[idx] - mean[c]).powi(2);
                    }
                }
                var[c] = var_sum / n;
            }
            
            (mean, var)
        } else {
            (self.running_mean.data_f32(), self.running_var.data_f32())
        };
        
        // Normalize
        let mut output = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            for c in 0..channels {
                let std = (var[c] + self.eps).sqrt();
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    output[idx] = gamma[c] * (data[idx] - mean[c]) / std + beta[c];
                }
            }
        }
        
        Tensor::from_slice(&output, dims).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// Batch Normalization for 2D inputs (N, C, H, W)
pub struct BatchNorm2d {
    #[allow(dead_code)]
    num_features: usize,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f32,
    #[allow(dead_code)]
    momentum: f32,
    training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self::with_params(num_features, 1e-5, 0.1)
    }

    pub fn with_params(num_features: usize, eps: f32, momentum: f32) -> Self {
        BatchNorm2d {
            num_features,
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            eps,
            momentum,
            training: true,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let data = input.data_f32();
        let gamma = self.gamma.data_f32();
        let beta = self.beta.data_f32();
        
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let spatial_size = height * width;
        
        let (mean, var) = if self.training {
            let mut mean = vec![0.0f32; channels];
            let mut var = vec![0.0f32; channels];
            let n = (batch_size * spatial_size) as f32;
            
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * channels * spatial_size + c * spatial_size + h * width + w;
                            sum += data[idx];
                        }
                    }
                }
                mean[c] = sum / n;
                
                let mut var_sum = 0.0f32;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * channels * spatial_size + c * spatial_size + h * width + w;
                            var_sum += (data[idx] - mean[c]).powi(2);
                        }
                    }
                }
                var[c] = var_sum / n;
            }
            
            (mean, var)
        } else {
            (self.running_mean.data_f32(), self.running_var.data_f32())
        };
        
        let mut output = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            for c in 0..channels {
                let std = (var[c] + self.eps).sqrt();
                for h in 0..height {
                    for w in 0..width {
                        let idx = b * channels * spatial_size + c * spatial_size + h * width + w;
                        output[idx] = gamma[c] * (data[idx] - mean[c]) / std + beta[c];
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, dims).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// Layer Normalization
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
    training: bool,
}

impl LayerNorm {
    pub fn new(normalized_shape: &[usize]) -> Self {
        Self::with_eps(normalized_shape, 1e-5)
    }

    pub fn with_eps(normalized_shape: &[usize], eps: f32) -> Self {
        let numel: usize = normalized_shape.iter().product();
        
        LayerNorm {
            normalized_shape: normalized_shape.to_vec(),
            gamma: Tensor::ones(&[numel]),
            beta: Tensor::zeros(&[numel]),
            eps,
            training: true,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let data = input.data_f32();
        let gamma = self.gamma.data_f32();
        let beta = self.beta.data_f32();
        
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = data.len() / norm_size;
        
        let mut output = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &data[start..end];
            
            // Compute mean
            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
            
            // Compute variance
            let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;
            let std = (var + self.eps).sqrt();
            
            // Normalize
            for i in 0..norm_size {
                output[start + i] = gamma[i] * (slice[i] - mean) / std + beta[i];
            }
        }
        
        Tensor::from_slice(&output, dims).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// Group Normalization
/// Divides channels into groups and normalizes within each group
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
    training: bool,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self::with_eps(num_groups, num_channels, 1e-5)
    }

    pub fn with_eps(num_groups: usize, num_channels: usize, eps: f32) -> Self {
        assert!(num_channels % num_groups == 0, "num_channels must be divisible by num_groups");
        
        GroupNorm {
            num_groups,
            num_channels,
            gamma: Tensor::ones(&[num_channels]),
            beta: Tensor::zeros(&[num_channels]),
            eps,
            training: true,
        }
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let data = input.data_f32();
        let gamma = self.gamma.data_f32();
        let beta = self.beta.data_f32();
        
        let batch_size = dims[0];
        let channels = dims[1];
        let spatial_size: usize = dims[2..].iter().product();
        
        assert_eq!(channels, self.num_channels, "Input channels must match num_channels");
        
        let channels_per_group = channels / self.num_groups;
        let mut output = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            for g in 0..self.num_groups {
                // Calculate mean and variance for this group
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                let group_size = (channels_per_group * spatial_size) as f32;
                
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        let val = data[idx];
                        sum += val;
                        sum_sq += val * val;
                    }
                }
                
                let mean = sum / group_size;
                let variance = (sum_sq / group_size) - (mean * mean);
                let std = (variance + self.eps).sqrt();
                
                // Normalize and apply affine transformation
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        let val = data[idx];
                        let normalized = (val - mean) / std;
                        output[idx] = gamma[channel_idx] * normalized + beta[channel_idx];
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, dims).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// Instance Normalization
/// Normalizes each channel independently for each sample
pub struct InstanceNorm {
    num_channels: usize,
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
    training: bool,
}

impl InstanceNorm {
    pub fn new(num_channels: usize) -> Self {
        Self::with_eps(num_channels, 1e-5)
    }

    pub fn with_eps(num_channels: usize, eps: f32) -> Self {
        InstanceNorm {
            num_channels,
            gamma: Tensor::ones(&[num_channels]),
            beta: Tensor::zeros(&[num_channels]),
            eps,
            training: true,
        }
    }
}

impl Module for InstanceNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let data = input.data_f32();
        let gamma = self.gamma.data_f32();
        let beta = self.beta.data_f32();
        
        let batch_size = dims[0];
        let channels = dims[1];
        let spatial_size: usize = dims[2..].iter().product();
        
        assert_eq!(channels, self.num_channels, "Input channels must match num_channels");
        
        let mut output = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            for c in 0..channels {
                // Calculate mean and variance for this channel in this sample
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let val = data[idx];
                    sum += val;
                    sum_sq += val * val;
                }
                
                let mean = sum / spatial_size as f32;
                let variance = (sum_sq / spatial_size as f32) - (mean * mean);
                let std = (variance + self.eps).sqrt();
                
                // Normalize and apply affine transformation
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let val = data[idx];
                    let normalized = (val - mean) / std;
                    output[idx] = gamma[c] * normalized + beta[c];
                }
            }
        }
        
        Tensor::from_slice(&output, dims).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batchnorm2d() {
        let bn = BatchNorm2d::new(16);
        let input = Tensor::randn(&[2, 16, 8, 8]);
        let output = bn.forward(&input);
        
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn test_layernorm() {
        let ln = LayerNorm::new(&[64]);
        let input = Tensor::randn(&[2, 10, 64]);
        let output = ln.forward(&input);
        
        assert_eq!(output.dims(), input.dims());
    }
}

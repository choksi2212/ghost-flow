//! Attention Mechanisms and Variants

use ghostflow_core::Tensor;
use crate::deep::layers::Dense;
use crate::deep::activations::ReLU;

/// Self-Attention Layer
pub struct SelfAttention {
    query: Dense,
    key: Dense,
    value: Dense,
    d_model: usize,
}

impl SelfAttention {
    pub fn new(d_model: usize) -> Self {
        SelfAttention {
            query: Dense::new(d_model, d_model),
            key: Dense::new(d_model, d_model),
            value: Dense::new(d_model, d_model),
            d_model,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let q = self.query.forward(x, training);
        let k = self.key.forward(x, training);
        let v = self.value.forward(x, training);
        
        self.scaled_dot_product_attention(&q, &k, &v)
    }

    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        // Simplified attention computation
        v.clone()
    }
}

/// Cross-Attention Layer
pub struct CrossAttention {
    query: Dense,
    key: Dense,
    value: Dense,
    d_model: usize,
}

impl CrossAttention {
    pub fn new(d_model: usize) -> Self {
        CrossAttention {
            query: Dense::new(d_model, d_model),
            key: Dense::new(d_model, d_model),
            value: Dense::new(d_model, d_model),
            d_model,
        }
    }

    pub fn forward(&mut self, query: &Tensor, context: &Tensor, training: bool) -> Tensor {
        let q = self.query.forward(query, training);
        let k = self.key.forward(context, training);
        let v = self.value.forward(context, training);
        
        self.scaled_dot_product_attention(&q, &k, &v)
    }

    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        v.clone()
    }
}

/// Squeeze-and-Excitation Block
pub struct SEBlock {
    fc1: Dense,
    fc2: Dense,
    reduction: usize,
}

impl SEBlock {
    pub fn new(channels: usize, reduction: usize) -> Self {
        SEBlock {
            fc1: Dense::new(channels, channels / reduction),
            fc2: Dense::new(channels / reduction, channels),
            reduction,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let squeezed = self.global_avg_pool(x);
        
        let mut out = self.fc1.forward(&squeezed, training);
        out = ReLU::new().forward(&out);
        
        out = self.fc2.forward(&out, training);
        out = self.sigmoid(&out);
        
        self.scale(x, &out)
    }

    fn global_avg_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let mut result = vec![0.0f32; batch * channels];
        
        for b in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        sum += data[idx];
                    }
                }
                result[b * channels + c] = sum / (height * width) as f32;
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels]).unwrap()
    }

    fn sigmoid(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn scale(&self, x: &Tensor, scale: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let scale_data = scale.data_f32();
        
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        
        let mut result = Vec::new();
        
        for b in 0..batch {
            for c in 0..channels {
                let s = scale_data[b * channels + c];
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        result.push(x_data[idx] * s);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, dims).unwrap()
    }
}

/// CBAM (Convolutional Block Attention Module)
pub struct CBAM {
    channel_attention: ChannelAttention,
    spatial_attention: SpatialAttention,
}

struct ChannelAttention {
    fc1: Dense,
    fc2: Dense,
}

impl ChannelAttention {
    fn new(channels: usize, reduction: usize) -> Self {
        ChannelAttention {
            fc1: Dense::new(channels, channels / reduction),
            fc2: Dense::new(channels / reduction, channels),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let avg_pool = self.global_avg_pool(x);
        let max_pool = self.global_max_pool(x);
        
        let mut avg_out = self.fc1.forward(&avg_pool, training);
        avg_out = ReLU::new().forward(&avg_out);
        avg_out = self.fc2.forward(&avg_out, training);
        
        let mut max_out = self.fc1.forward(&max_pool, training);
        max_out = ReLU::new().forward(&max_out);
        max_out = self.fc2.forward(&max_out, training);
        
        let combined = self.add_tensors(&avg_out, &max_out);
        self.sigmoid(&combined)
    }

    fn global_avg_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let mut result = vec![0.0f32; batch * channels];
        
        for b in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        sum += data[idx];
                    }
                }
                result[b * channels + c] = sum / (height * width) as f32;
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels]).unwrap()
    }

    fn global_max_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let mut result = vec![f32::MIN; batch * channels];
        
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        let out_idx = b * channels + c;
                        result[out_idx] = result[out_idx].max(data[idx]);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels]).unwrap()
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }

    fn sigmoid(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

struct SpatialAttention {
    conv: Dense, // Simplified as Dense
}

impl SpatialAttention {
    fn new() -> Self {
        SpatialAttention {
            conv: Dense::new(2, 1),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let avg_pool = self.avg_pool_spatial(x);
        let max_pool = self.max_pool_spatial(x);
        
        let concat = self.concatenate(&avg_pool, &max_pool);
        let out = self.conv.forward(&concat, training);
        self.sigmoid(&out)
    }

    fn avg_pool_spatial(&self, x: &Tensor) -> Tensor {
        x.clone() // Simplified
    }

    fn max_pool_spatial(&self, x: &Tensor) -> Tensor {
        x.clone() // Simplified
    }

    fn concatenate(&self, a: &Tensor, b: &Tensor) -> Tensor {
        a.clone() // Simplified
    }

    fn sigmoid(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

impl CBAM {
    pub fn new(channels: usize, reduction: usize) -> Self {
        CBAM {
            channel_attention: ChannelAttention::new(channels, reduction),
            spatial_attention: SpatialAttention::new(),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let channel_out = self.channel_attention.forward(x, training);
        let x_scaled = self.scale_channel(x, &channel_out);
        
        let spatial_out = self.spatial_attention.forward(&x_scaled, training);
        self.scale_spatial(&x_scaled, &spatial_out)
    }

    fn scale_channel(&self, x: &Tensor, scale: &Tensor) -> Tensor {
        x.clone() // Simplified
    }

    fn scale_spatial(&self, x: &Tensor, scale: &Tensor) -> Tensor {
        x.clone() // Simplified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_attention() {
        let mut attn = SelfAttention::new(512);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 10 * 512], &[2, 10, 512]).unwrap();
        let output = attn.forward(&input, false);
        assert_eq!(output.dims()[2], 512);
    }

    #[test]
    fn test_se_block() {
        let mut se = SEBlock::new(64, 16);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 64 * 32 * 32], &[1, 64, 32, 32]).unwrap();
        let output = se.forward(&input, false);
        assert_eq!(output.dims()[1], 64);
    }
}



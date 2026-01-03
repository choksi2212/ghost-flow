//! Transformer architecture components

use ghostflow_core::Tensor;
use crate::module::Module;
use crate::linear::Linear;
use crate::norm::LayerNorm;
use crate::dropout::Dropout;
use crate::attention::MultiHeadAttention;

/// Feed-Forward Network (FFN) used in Transformers
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
    activation: Activation,
    training: bool,
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    GELU,
    SiLU,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, dropout: f32) -> Self {
        Self::with_activation(d_model, d_ff, dropout, Activation::GELU)
    }

    pub fn with_activation(d_model: usize, d_ff: usize, dropout: f32, activation: Activation) -> Self {
        FeedForward {
            linear1: Linear::new(d_model, d_ff),
            linear2: Linear::new(d_ff, d_model),
            dropout: Dropout::new(dropout),
            activation,
            training: true,
        }
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.linear1.forward(input);
        let x = match self.activation {
            Activation::ReLU => x.relu(),
            Activation::GELU => x.gelu(),
            Activation::SiLU => x.silu(),
        };
        let x = if self.training {
            self.dropout.forward(&x)
        } else {
            x
        };
        self.linear2.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.linear1.parameters();
        params.extend(self.linear2.parameters());
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// Transformer Encoder Layer
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    pre_norm: bool,
    training: bool,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, nhead: usize, d_ff: usize, dropout: f32) -> Self {
        Self::with_config(d_model, nhead, d_ff, dropout, false)
    }

    pub fn with_config(d_model: usize, nhead: usize, d_ff: usize, dropout: f32, pre_norm: bool) -> Self {
        TransformerEncoderLayer {
            self_attn: MultiHeadAttention::new(d_model, nhead, dropout),
            ffn: FeedForward::new(d_model, d_ff, dropout),
            norm1: LayerNorm::new(&[d_model]),
            norm2: LayerNorm::new(&[d_model]),
            dropout: Dropout::new(dropout),
            pre_norm,
            training: true,
        }
    }

    pub fn forward_with_mask(&self, src: &Tensor, _mask: Option<&Tensor>) -> Tensor {
        if self.pre_norm {
            // Pre-LN: x + Attn(LN(x))
            let x = self.norm1.forward(src);
            let attn_out = self.self_attn.forward(&x);
            let x = src.add(&self.dropout.forward(&attn_out)).unwrap();
            
            let x2 = self.norm2.forward(&x);
            let ffn_out = self.ffn.forward(&x2);
            x.add(&self.dropout.forward(&ffn_out)).unwrap()
        } else {
            // Post-LN: LN(x + Attn(x))
            let attn_out = self.self_attn.forward(src);
            let x = self.norm1.forward(&src.add(&self.dropout.forward(&attn_out)).unwrap());
            
            let ffn_out = self.ffn.forward(&x);
            self.norm2.forward(&x.add(&self.dropout.forward(&ffn_out)).unwrap())
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.ffn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.ffn.eval();
    }

    fn is_training(&self) -> bool { self.training }
}

/// Transformer Decoder Layer
pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
    #[allow(dead_code)]
    pre_norm: bool,
    training: bool,
}

impl TransformerDecoderLayer {
    pub fn new(d_model: usize, nhead: usize, d_ff: usize, dropout: f32) -> Self {
        TransformerDecoderLayer {
            self_attn: MultiHeadAttention::new(d_model, nhead, dropout),
            cross_attn: MultiHeadAttention::new(d_model, nhead, dropout),
            ffn: FeedForward::new(d_model, d_ff, dropout),
            norm1: LayerNorm::new(&[d_model]),
            norm2: LayerNorm::new(&[d_model]),
            norm3: LayerNorm::new(&[d_model]),
            dropout: Dropout::new(dropout),
            pre_norm: false,
            training: true,
        }
    }

    pub fn forward_with_memory(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        _tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
    ) -> Tensor {
        // Self-attention on target
        let x = self.norm1.forward(&tgt.add(&self.dropout.forward(&self.self_attn.forward(tgt))).unwrap());
        
        // Cross-attention with encoder output
        let (cross_out, _, _, _) = self.cross_attn.forward_with_cache(&x, memory, memory, memory_mask, None, None);
        let x = self.norm2.forward(&x.add(&self.dropout.forward(&cross_out)).unwrap());
        
        // Feed-forward
        let ffn_out = self.ffn.forward(&x);
        self.norm3.forward(&x.add(&self.dropout.forward(&ffn_out)).unwrap())
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        // For standalone use, treat as self-attention only
        self.self_attn.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.cross_attn.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.cross_attn.train();
        self.ffn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.cross_attn.eval();
        self.ffn.eval();
    }

    fn is_training(&self) -> bool { self.training }
}

/// Transformer Encoder (stack of encoder layers)
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    norm: Option<LayerNorm>,
}

impl TransformerEncoder {
    pub fn new(d_model: usize, nhead: usize, d_ff: usize, num_layers: usize, dropout: f32) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, nhead, d_ff, dropout))
            .collect();
        
        TransformerEncoder {
            layers,
            norm: Some(LayerNorm::new(&[d_model])),
        }
    }

    pub fn forward_with_mask(&self, src: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let mut output = src.clone();
        
        for layer in &self.layers {
            output = layer.forward_with_mask(&output, mask);
        }
        
        if let Some(ref norm) = self.norm {
            output = norm.forward(&output);
        }
        
        output
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params: Vec<Tensor> = self.layers.iter()
            .flat_map(|l| l.parameters())
            .collect();
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.layers.first().map_or(false, |l| l.is_training())
    }
}

/// Positional Encoding (sinusoidal)
pub struct PositionalEncoding {
    encoding: Tensor,
    dropout: Dropout,
    #[allow(dead_code)]
    max_len: usize,
    d_model: usize,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, dropout: f32) -> Self {
        let encoding = Self::create_encoding(d_model, max_len);
        
        PositionalEncoding {
            encoding,
            dropout: Dropout::new(dropout),
            max_len,
            d_model,
        }
    }

    fn create_encoding(d_model: usize, max_len: usize) -> Tensor {
        let mut pe = vec![0.0f32; max_len * d_model];
        
        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / (10000.0f32).powf(2.0 * i as f32 / d_model as f32);
                pe[pos * d_model + 2 * i] = angle.sin();
                pe[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }
        
        Tensor::from_slice(&pe, &[max_len, d_model]).unwrap()
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.dims()[1];
        
        // Get positional encoding for this sequence length
        let pe_data = self.encoding.data_f32();
        let pe_slice: Vec<f32> = pe_data[..seq_len * self.d_model].to_vec();
        let pe = Tensor::from_slice(&pe_slice, &[seq_len, self.d_model]).unwrap();
        
        // Add positional encoding to input
        let result = input.add(&pe).unwrap();
        self.dropout.forward(&result)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![] // Positional encoding has no learnable parameters
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Rotary Position Embedding (RoPE) - used in modern LLMs
pub struct RotaryEmbedding {
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    max_seq_len: usize,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        let (cos_cache, sin_cache) = Self::compute_freqs(dim, max_seq_len, base);
        
        RotaryEmbedding {
            dim,
            max_seq_len,
            cos_cache,
            sin_cache,
        }
    }

    fn compute_freqs(dim: usize, max_seq_len: usize, base: f32) -> (Tensor, Tensor) {
        let half_dim = dim / 2;
        
        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
            .collect();
        
        // Compute position * inv_freq
        let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_data = vec![0.0f32; max_seq_len * half_dim];
        
        for pos in 0..max_seq_len {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }
        
        (
            Tensor::from_slice(&cos_data, &[max_seq_len, half_dim]).unwrap(),
            Tensor::from_slice(&sin_data, &[max_seq_len, half_dim]).unwrap(),
        )
    }

    /// Apply rotary embedding to query and key tensors
    pub fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> (Tensor, Tensor) {
        let seq_len = q.dims()[q.ndim() - 2];
        let head_dim = q.dims()[q.ndim() - 1];
        let half_dim = head_dim / 2;
        
        let cos_data = self.cos_cache.data_f32();
        let sin_data = self.sin_cache.data_f32();
        
        let apply_rope = |x: &Tensor| -> Tensor {
            let data = x.data_f32();
            let batch_heads: usize = x.dims()[..x.ndim()-2].iter().product();
            
            let mut result = vec![0.0f32; data.len()];
            
            for bh in 0..batch_heads {
                for s in 0..seq_len {
                    let pos = start_pos + s;
                    for i in 0..half_dim {
                        let cos_val = cos_data[pos * half_dim + i];
                        let sin_val = sin_data[pos * half_dim + i];
                        
                        let idx1 = bh * seq_len * head_dim + s * head_dim + i;
                        let idx2 = bh * seq_len * head_dim + s * head_dim + i + half_dim;
                        
                        let x1 = data[idx1];
                        let x2 = data[idx2];
                        
                        result[idx1] = x1 * cos_val - x2 * sin_val;
                        result[idx2] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
            
            Tensor::from_slice(&result, x.dims()).unwrap()
        };
        
        (apply_rope(q), apply_rope(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward() {
        let ffn = FeedForward::new(64, 256, 0.1);
        let input = Tensor::randn(&[2, 10, 64]);
        let output = ffn.forward(&input);
        
        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let layer = TransformerEncoderLayer::new(64, 8, 256, 0.1);
        let input = Tensor::randn(&[2, 10, 64]);
        let output = layer.forward(&input);
        
        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_transformer_encoder() {
        let encoder = TransformerEncoder::new(64, 8, 256, 6, 0.1);
        let input = Tensor::randn(&[2, 10, 64]);
        let output = encoder.forward(&input);
        
        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(64, 512, 0.1);
        let input = Tensor::randn(&[2, 10, 64]);
        let output = pe.forward(&input);
        
        assert_eq!(output.dims(), &[2, 10, 64]);
    }
}

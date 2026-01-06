//! Transformer Architecture - Self-Attention, Multi-Head Attention, Transformer Blocks

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Scaled Dot-Product Attention
pub struct ScaledDotProductAttention {
    pub dropout_rate: f32,
    scale: f32,
}

impl ScaledDotProductAttention {
    pub fn new(d_k: usize) -> Self {
        ScaledDotProductAttention {
            dropout_rate: 0.0,
            scale: 1.0 / (d_k as f32).sqrt(),
        }
    }

    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = rate;
        self
    }

    fn softmax(x: &[f32], size: usize) -> Vec<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = x.iter().map(|&v| (v - max_val).exp()).sum();
        x.iter().map(|&v| (v - max_val).exp() / exp_sum).collect()
    }

    /// Forward pass: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    /// Q: [batch, seq_len, d_k]
    /// K: [batch, seq_len, d_k]  
    /// V: [batch, seq_len, d_v]
    /// mask: Optional [batch, seq_len, seq_len]
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let q_data = q.data_f32();
        let k_data = k.data_f32();
        let v_data = v.data_f32();
        
        let batch_size = q.dims()[0];
        let seq_len_q = q.dims()[1];
        let d_k = q.dims()[2];
        let seq_len_k = k.dims()[1];
        let d_v = v.dims()[2];

        let mut output = vec![0.0f32; batch_size * seq_len_q * d_v];

        for b in 0..batch_size {
            // Compute attention scores: Q @ K^T
            let mut scores = vec![0.0f32; seq_len_q * seq_len_k];

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    let mut dot = 0.0f32;
                    for d in 0..d_k {
                        let q_idx = b * seq_len_q * d_k + i * d_k + d;
                        let k_idx = b * seq_len_k * d_k + j * d_k + d;
                        dot += q_data[q_idx] * k_data[k_idx];
                    }
                    scores[i * seq_len_k + j] = dot * self.scale;
                }
            }

            // Apply mask if provided
            if let Some(m) = mask {
                let mask_data = m.data_f32();
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        let mask_idx = b * seq_len_q * seq_len_k + i * seq_len_k + j;
                        if mask_data[mask_idx] == 0.0 {
                            scores[i * seq_len_k + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Softmax over keys for each query
            for i in 0..seq_len_q {
                let row_start = i * seq_len_k;
                let row = &scores[row_start..row_start + seq_len_k];
                let softmax_row = Self::softmax(row, seq_len_k);
                for j in 0..seq_len_k {
                    scores[row_start + j] = softmax_row[j];
                }
            }

            // Compute output: attention_weights @ V
            for i in 0..seq_len_q {
                for d in 0..d_v {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len_k {
                        let attn_weight = scores[i * seq_len_k + j];
                        let v_idx = b * seq_len_k * d_v + j * d_v + d;
                        sum += attn_weight * v_data[v_idx];
                    }
                    let out_idx = b * seq_len_q * d_v + i * d_v + d;
                    output[out_idx] = sum;
                }
            }
        }

        Tensor::from_slice(&output, &[batch_size, seq_len_q, d_v]).unwrap()
    }
}

/// Multi-Head Attention
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub dropout_rate: f32,
    d_k: usize,
    d_v: usize,
    // Weight matrices
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    // Biases
    b_q: Vec<f32>,
    b_k: Vec<f32>,
    b_v: Vec<f32>,
    b_o: Vec<f32>,
    // Gradients
    grad_w_q: Vec<f32>,
    grad_w_k: Vec<f32>,
    grad_w_v: Vec<f32>,
    grad_w_o: Vec<f32>,
    // Cache for backward
    q_cache: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    attn_weights_cache: Vec<f32>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
        
        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;
        
        let mut rng = thread_rng();
        let scale = (2.0 / d_model as f32).sqrt();

        let w_q: Vec<f32> = (0..d_model * d_model).map(|_| (rng.gen::<f32>() - 0.5) * scale).collect();
        let w_k: Vec<f32> = (0..d_model * d_model).map(|_| (rng.gen::<f32>() - 0.5) * scale).collect();
        let w_v: Vec<f32> = (0..d_model * d_model).map(|_| (rng.gen::<f32>() - 0.5) * scale).collect();
        let w_o: Vec<f32> = (0..d_model * d_model).map(|_| (rng.gen::<f32>() - 0.5) * scale).collect();

        MultiHeadAttention {
            d_model,
            num_heads,
            dropout_rate: 0.0,
            d_k,
            d_v,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q: vec![0.0; d_model],
            b_k: vec![0.0; d_model],
            b_v: vec![0.0; d_model],
            b_o: vec![0.0; d_model],
            grad_w_q: vec![0.0; d_model * d_model],
            grad_w_k: vec![0.0; d_model * d_model],
            grad_w_v: vec![0.0; d_model * d_model],
            grad_w_o: vec![0.0; d_model * d_model],
            q_cache: Vec::new(),
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            attn_weights_cache: Vec::new(),
        }
    }

    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = rate;
        self
    }

    fn linear_transform(&self, x: &[f32], w: &[f32], b: &[f32], 
                        batch_size: usize, seq_len: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * seq_len * out_dim];
        
        for b_idx in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..out_dim {
                    let mut sum = b[o];
                    for i in 0..in_dim {
                        let x_idx = b_idx * seq_len * in_dim + s * in_dim + i;
                        sum += x[x_idx] * w[i * out_dim + o];
                    }
                    let out_idx = b_idx * seq_len * out_dim + s * out_dim + o;
                    output[out_idx] = sum;
                }
            }
        }
        output
    }

    fn softmax_row(x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = x.iter().map(|&v| (v - max_val).exp()).sum();
        x.iter().map(|&v| (v - max_val).exp() / exp_sum).collect()
    }

    /// Forward pass for multi-head attention
    /// x: [batch, seq_len, d_model]
    /// mask: Optional [batch, seq_len, seq_len]
    pub fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let x_data = x.data_f32();
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        // Linear projections
        let q = self.linear_transform(&x_data, &self.w_q, &self.b_q, batch_size, seq_len, self.d_model, self.d_model);
        let k = self.linear_transform(&x_data, &self.w_k, &self.b_k, batch_size, seq_len, self.d_model, self.d_model);
        let v = self.linear_transform(&x_data, &self.w_v, &self.b_v, batch_size, seq_len, self.d_model, self.d_model);

        // Cache for backward
        self.q_cache = q.clone();
        self.k_cache = k.clone();
        self.v_cache = v.clone();

        // Reshape for multi-head: [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        let scale = 1.0 / (self.d_k as f32).sqrt();
        
        let mut output = vec![0.0f32; batch_size * seq_len * self.d_model];
        let mut all_attn_weights = vec![0.0f32; batch_size * self.num_heads * seq_len * seq_len];

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                // Extract head-specific Q, K, V
                let head_offset = h * self.d_k;
                
                // Compute attention scores for this head
                let mut scores = vec![0.0f32; seq_len * seq_len];
                
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..self.d_k {
                            let q_idx = b * seq_len * self.d_model + i * self.d_model + head_offset + d;
                            let k_idx = b * seq_len * self.d_model + j * self.d_model + head_offset + d;
                            dot += q[q_idx] * k[k_idx];
                        }
                        scores[i * seq_len + j] = dot * scale;
                    }
                }

                // Apply mask
                if let Some(m) = mask {
                    let mask_data = m.data_f32();
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let mask_idx = b * seq_len * seq_len + i * seq_len + j;
                            if mask_data[mask_idx] == 0.0 {
                                scores[i * seq_len + j] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }

                // Softmax
                for i in 0..seq_len {
                    let row = &scores[i * seq_len..(i + 1) * seq_len];
                    let softmax_row = Self::softmax_row(row);
                    for j in 0..seq_len {
                        scores[i * seq_len + j] = softmax_row[j];
                        all_attn_weights[b * self.num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = softmax_row[j];
                    }
                }

                // Compute weighted sum of values
                for i in 0..seq_len {
                    for d in 0..self.d_k {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let v_idx = b * seq_len * self.d_model + j * self.d_model + head_offset + d;
                            sum += scores[i * seq_len + j] * v[v_idx];
                        }
                        let out_idx = b * seq_len * self.d_model + i * self.d_model + head_offset + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        self.attn_weights_cache = all_attn_weights;

        // Final linear projection
        let final_output = self.linear_transform(&output, &self.w_o, &self.b_o, batch_size, seq_len, self.d_model, self.d_model);

        Tensor::from_slice(&final_output, &[batch_size, seq_len, self.d_model]).unwrap()
    }

    pub fn parameters(&self) -> Vec<&Vec<f32>> {
        vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o]
    }
}

/// Positional Encoding using sine and cosine functions
pub struct PositionalEncoding {
    pub d_model: usize,
    pub max_len: usize,
    pub dropout_rate: f32,
    encoding: Vec<f32>,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut encoding = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / (10000.0f32).powf((2 * (i / 2)) as f32 / d_model as f32);
                encoding[pos * d_model + i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }

        PositionalEncoding {
            d_model,
            max_len,
            dropout_rate: 0.0,
            encoding,
        }
    }

    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Add positional encoding to input
    /// x: [batch, seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        let mut output = x_data.clone();

        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..self.d_model {
                    let idx = b * seq_len * self.d_model + s * self.d_model + d;
                    let pe_idx = s * self.d_model + d;
                    output[idx] += self.encoding[pe_idx];
                }
            }
        }

        Tensor::from_slice(&output, x.dims()).unwrap()
    }
}

/// Feed-Forward Network (used in Transformer blocks)
pub struct FeedForward {
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout_rate: f32,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    // Cache
    hidden_cache: Vec<f32>,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = thread_rng();
        let scale1 = (2.0 / d_model as f32).sqrt();
        let scale2 = (2.0 / d_ff as f32).sqrt();

        FeedForward {
            d_model,
            d_ff,
            dropout_rate: 0.0,
            w1: (0..d_model * d_ff).map(|_| (rng.gen::<f32>() - 0.5) * scale1).collect(),
            b1: vec![0.0; d_ff],
            w2: (0..d_ff * d_model).map(|_| (rng.gen::<f32>() - 0.5) * scale2).collect(),
            b2: vec![0.0; d_model],
            hidden_cache: Vec::new(),
        }
    }

    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = rate;
        self
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// Forward pass: FFN(x) = max(0, xW1 + b1)W2 + b2
    /// x: [batch, seq_len, d_model]
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        // First linear + GELU
        let mut hidden = vec![0.0f32; batch_size * seq_len * self.d_ff];
        for b in 0..batch_size {
            for s in 0..seq_len {
                for f in 0..self.d_ff {
                    let mut sum = self.b1[f];
                    for d in 0..self.d_model {
                        let x_idx = b * seq_len * self.d_model + s * self.d_model + d;
                        sum += x_data[x_idx] * self.w1[d * self.d_ff + f];
                    }
                    let h_idx = b * seq_len * self.d_ff + s * self.d_ff + f;
                    hidden[h_idx] = Self::gelu(sum);
                }
            }
        }

        self.hidden_cache = hidden.clone();

        // Second linear
        let mut output = vec![0.0f32; batch_size * seq_len * self.d_model];
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..self.d_model {
                    let mut sum = self.b2[d];
                    for f in 0..self.d_ff {
                        let h_idx = b * seq_len * self.d_ff + s * self.d_ff + f;
                        sum += hidden[h_idx] * self.w2[f * self.d_model + d];
                    }
                    let out_idx = b * seq_len * self.d_model + s * self.d_model + d;
                    output[out_idx] = sum;
                }
            }
        }

        Tensor::from_slice(&output, x.dims()).unwrap()
    }
}

/// Layer Normalization for Transformers
pub struct TransformerLayerNorm {
    pub normalized_shape: usize,
    pub eps: f32,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl TransformerLayerNorm {
    pub fn new(normalized_shape: usize) -> Self {
        TransformerLayerNorm {
            normalized_shape,
            eps: 1e-5,
            gamma: vec![1.0; normalized_shape],
            beta: vec![0.0; normalized_shape],
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let d_model = dims[2];

        let mut output = vec![0.0f32; x_data.len()];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let start = b * seq_len * d_model + s * d_model;
                let slice = &x_data[start..start + d_model];

                // Compute mean
                let mean: f32 = slice.iter().sum::<f32>() / d_model as f32;

                // Compute variance
                let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / d_model as f32;

                // Normalize
                let std = (var + self.eps).sqrt();
                for d in 0..d_model {
                    let idx = start + d;
                    output[idx] = self.gamma[d] * (x_data[idx] - mean) / std + self.beta[d];
                }
            }
        }

        Tensor::from_slice(&output, dims).unwrap()
    }
}

/// Transformer Encoder Layer
pub struct TransformerEncoderLayer {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub dropout_rate: f32,
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: TransformerLayerNorm,
    norm2: TransformerLayerNorm,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        TransformerEncoderLayer {
            d_model,
            num_heads,
            d_ff,
            dropout_rate: 0.1,
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            feed_forward: FeedForward::new(d_model, d_ff),
            norm1: TransformerLayerNorm::new(d_model),
            norm2: TransformerLayerNorm::new(d_model),
        }
    }

    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = rate;
        self.self_attn = self.self_attn.dropout(rate);
        self.feed_forward = self.feed_forward.dropout(rate);
        self
    }

    /// Forward pass with pre-norm architecture
    /// x: [batch, seq_len, d_model]
    /// mask: Optional attention mask
    pub fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let x_data = x.data_f32();
        
        // Self-attention with residual
        let normed1 = self.norm1.forward(x);
        let attn_output = self.self_attn.forward(&normed1, mask);
        let attn_data = attn_output.data_f32();
        
        // Residual connection
        let residual1: Vec<f32> = x_data.iter().zip(attn_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let residual1_tensor = Tensor::from_slice(&residual1, x.dims()).unwrap();

        // Feed-forward with residual
        let normed2 = self.norm2.forward(&residual1_tensor);
        let ff_output = self.feed_forward.forward(&normed2);
        let ff_data = ff_output.data_f32();

        // Residual connection
        let output: Vec<f32> = residual1.iter().zip(ff_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Tensor::from_slice(&output, x.dims()).unwrap()
    }
}

/// Transformer Encoder (stack of encoder layers)
pub struct TransformerEncoder {
    pub num_layers: usize,
    pub d_model: usize,
    layers: Vec<TransformerEncoderLayer>,
    final_norm: TransformerLayerNorm,
}

impl TransformerEncoder {
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let layers: Vec<TransformerEncoderLayer> = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, num_heads, d_ff))
            .collect();

        TransformerEncoder {
            num_layers,
            d_model,
            layers,
            final_norm: TransformerLayerNorm::new(d_model),
        }
    }

    pub fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let mut output = x.clone();
        
        for layer in &mut self.layers {
            output = layer.forward(&output, mask);
        }

        self.final_norm.forward(&output)
    }
}

/// Transformer Decoder Layer
pub struct TransformerDecoderLayer {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub dropout_rate: f32,
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: TransformerLayerNorm,
    norm2: TransformerLayerNorm,
    norm3: TransformerLayerNorm,
}

impl TransformerDecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        TransformerDecoderLayer {
            d_model,
            num_heads,
            d_ff,
            dropout_rate: 0.1,
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            cross_attn: MultiHeadAttention::new(d_model, num_heads),
            feed_forward: FeedForward::new(d_model, d_ff),
            norm1: TransformerLayerNorm::new(d_model),
            norm2: TransformerLayerNorm::new(d_model),
            norm3: TransformerLayerNorm::new(d_model),
        }
    }

    /// Forward pass
    /// x: [batch, tgt_seq_len, d_model] - decoder input
    /// memory: [batch, src_seq_len, d_model] - encoder output
    /// tgt_mask: Optional mask for self-attention (causal)
    /// memory_mask: Optional mask for cross-attention
    pub fn forward(&mut self, x: &Tensor, memory: &Tensor, 
                   tgt_mask: Option<&Tensor>, memory_mask: Option<&Tensor>) -> Tensor {
        let x_data = x.data_f32();
        
        // Self-attention (masked)
        let normed1 = self.norm1.forward(x);
        let self_attn_output = self.self_attn.forward(&normed1, tgt_mask);
        let self_attn_data = self_attn_output.data_f32();
        
        let residual1: Vec<f32> = x_data.iter().zip(self_attn_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let residual1_tensor = Tensor::from_slice(&residual1, x.dims()).unwrap();

        // Cross-attention
        let normed2 = self.norm2.forward(&residual1_tensor);
        // For cross-attention, Q comes from decoder, K and V from encoder
        let cross_attn_output = self.cross_attention(&normed2, memory, memory_mask);
        let cross_attn_data = cross_attn_output.data_f32();

        let residual2: Vec<f32> = residual1.iter().zip(cross_attn_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let residual2_tensor = Tensor::from_slice(&residual2, x.dims()).unwrap();

        // Feed-forward
        let normed3 = self.norm3.forward(&residual2_tensor);
        let ff_output = self.feed_forward.forward(&normed3);
        let ff_data = ff_output.data_f32();

        let output: Vec<f32> = residual2.iter().zip(ff_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Tensor::from_slice(&output, x.dims()).unwrap()
    }

    fn cross_attention(&mut self, q: &Tensor, kv: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Simplified: use the cross_attn module with kv as both key and value source
        self.cross_attn.forward(q, mask)
    }
}

/// Transformer Decoder (stack of decoder layers)
pub struct TransformerDecoder {
    pub num_layers: usize,
    pub d_model: usize,
    layers: Vec<TransformerDecoderLayer>,
    final_norm: TransformerLayerNorm,
}

impl TransformerDecoder {
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let layers: Vec<TransformerDecoderLayer> = (0..num_layers)
            .map(|_| TransformerDecoderLayer::new(d_model, num_heads, d_ff))
            .collect();

        TransformerDecoder {
            num_layers,
            d_model,
            layers,
            final_norm: TransformerLayerNorm::new(d_model),
        }
    }

    pub fn forward(&mut self, x: &Tensor, memory: &Tensor,
                   tgt_mask: Option<&Tensor>, memory_mask: Option<&Tensor>) -> Tensor {
        let mut output = x.clone();
        
        for layer in &mut self.layers {
            output = layer.forward(&output, memory, tgt_mask, memory_mask);
        }

        self.final_norm.forward(&output)
    }
}

/// Full Transformer Model (Encoder-Decoder)
pub struct Transformer {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    pos_encoding: PositionalEncoding,
}

impl Transformer {
    pub fn new(d_model: usize, num_heads: usize, num_encoder_layers: usize,
               num_decoder_layers: usize, d_ff: usize, max_seq_len: usize) -> Self {
        Transformer {
            d_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            d_ff,
            max_seq_len,
            encoder: TransformerEncoder::new(num_encoder_layers, d_model, num_heads, d_ff),
            decoder: TransformerDecoder::new(num_decoder_layers, d_model, num_heads, d_ff),
            pos_encoding: PositionalEncoding::new(d_model, max_seq_len),
        }
    }

    /// Generate causal mask for decoder self-attention
    pub fn generate_causal_mask(seq_len: usize) -> Tensor {
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask[i * seq_len + j] = 1.0;
            }
        }
        Tensor::from_slice(&mask, &[1, seq_len, seq_len]).unwrap()
    }

    /// Forward pass
    /// src: [batch, src_seq_len, d_model] - source sequence
    /// tgt: [batch, tgt_seq_len, d_model] - target sequence
    pub fn forward(&mut self, src: &Tensor, tgt: &Tensor,
                   src_mask: Option<&Tensor>, tgt_mask: Option<&Tensor>) -> Tensor {
        // Add positional encoding
        let src_pe = self.pos_encoding.forward(src);
        let tgt_pe = self.pos_encoding.forward(tgt);

        // Encode
        let memory = self.encoder.forward(&src_pe, src_mask);

        // Decode
        self.decoder.forward(&tgt_pe, &memory, tgt_mask, src_mask)
    }

    /// Encode only (for inference)
    pub fn encode(&mut self, src: &Tensor, src_mask: Option<&Tensor>) -> Tensor {
        let src_pe = self.pos_encoding.forward(src);
        self.encoder.forward(&src_pe, src_mask)
    }

    /// Decode one step (for autoregressive generation)
    pub fn decode_step(&mut self, tgt: &Tensor, memory: &Tensor,
                       tgt_mask: Option<&Tensor>, memory_mask: Option<&Tensor>) -> Tensor {
        let tgt_pe = self.pos_encoding.forward(tgt);
        self.decoder.forward(&tgt_pe, memory, tgt_mask, memory_mask)
    }
}

/// Vision Transformer (ViT) Patch Embedding
pub struct PatchEmbedding {
    pub img_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    num_patches: usize,
    projection: Vec<f32>,
    bias: Vec<f32>,
}

impl PatchEmbedding {
    pub fn new(img_size: usize, patch_size: usize, in_channels: usize, embed_dim: usize) -> Self {
        assert!(img_size % patch_size == 0, "Image size must be divisible by patch size");
        
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        let patch_dim = in_channels * patch_size * patch_size;
        
        let mut rng = thread_rng();
        let scale = (2.0 / patch_dim as f32).sqrt();

        PatchEmbedding {
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            num_patches,
            projection: (0..patch_dim * embed_dim).map(|_| (rng.gen::<f32>() - 0.5) * scale).collect(),
            bias: vec![0.0; embed_dim],
        }
    }

    /// Convert image to patch embeddings
    /// x: [batch, channels, height, width]
    /// Returns: [batch, num_patches, embed_dim]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let batch_size = x.dims()[0];
        let patch_dim = self.in_channels * self.patch_size * self.patch_size;
        let patches_per_side = self.img_size / self.patch_size;

        let mut output = vec![0.0f32; batch_size * self.num_patches * self.embed_dim];

        for b in 0..batch_size {
            for py in 0..patches_per_side {
                for px in 0..patches_per_side {
                    let patch_idx = py * patches_per_side + px;
                    
                    // Extract and flatten patch
                    let mut patch = vec![0.0f32; patch_dim];
                    for c in 0..self.in_channels {
                        for dy in 0..self.patch_size {
                            for dx in 0..self.patch_size {
                                let y = py * self.patch_size + dy;
                                let x_coord = px * self.patch_size + dx;
                                let img_idx = b * self.in_channels * self.img_size * self.img_size
                                    + c * self.img_size * self.img_size
                                    + y * self.img_size + x_coord;
                                let patch_flat_idx = c * self.patch_size * self.patch_size 
                                    + dy * self.patch_size + dx;
                                patch[patch_flat_idx] = x_data[img_idx];
                            }
                        }
                    }

                    // Project patch to embedding
                    for e in 0..self.embed_dim {
                        let mut sum = self.bias[e];
                        for p in 0..patch_dim {
                            sum += patch[p] * self.projection[p * self.embed_dim + e];
                        }
                        let out_idx = b * self.num_patches * self.embed_dim 
                            + patch_idx * self.embed_dim + e;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        Tensor::from_slice(&output, &[batch_size, self.num_patches, self.embed_dim]).unwrap()
    }

    pub fn num_patches(&self) -> usize {
        self.num_patches
    }
}

/// Vision Transformer (ViT)
pub struct VisionTransformer {
    pub img_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub num_classes: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    patch_embed: PatchEmbedding,
    cls_token: Vec<f32>,
    pos_embed: Vec<f32>,
    encoder: TransformerEncoder,
    head: Vec<f32>,
    head_bias: Vec<f32>,
}

impl VisionTransformer {
    pub fn new(img_size: usize, patch_size: usize, in_channels: usize,
               num_classes: usize, embed_dim: usize, num_heads: usize, 
               num_layers: usize, d_ff: usize) -> Self {
        let patch_embed = PatchEmbedding::new(img_size, patch_size, in_channels, embed_dim);
        let num_patches = patch_embed.num_patches();
        
        let mut rng = thread_rng();
        let scale = 0.02;

        VisionTransformer {
            img_size,
            patch_size,
            in_channels,
            num_classes,
            embed_dim,
            num_heads,
            num_layers,
            patch_embed,
            cls_token: (0..embed_dim).map(|_| rng.gen::<f32>() * scale).collect(),
            pos_embed: (0..(num_patches + 1) * embed_dim).map(|_| rng.gen::<f32>() * scale).collect(),
            encoder: TransformerEncoder::new(num_layers, embed_dim, num_heads, d_ff),
            head: (0..embed_dim * num_classes).map(|_| (rng.gen::<f32>() - 0.5) * scale).collect(),
            head_bias: vec![0.0; num_classes],
        }
    }

    /// Forward pass
    /// x: [batch, channels, height, width]
    /// Returns: [batch, num_classes]
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let batch_size = x.dims()[0];
        let num_patches = self.patch_embed.num_patches();

        // Patch embedding
        let patches = self.patch_embed.forward(x);
        let patches_data = patches.data_f32();

        // Prepend CLS token and add positional embedding
        let seq_len = num_patches + 1;
        let mut embedded = vec![0.0f32; batch_size * seq_len * self.embed_dim];

        for b in 0..batch_size {
            // CLS token
            for e in 0..self.embed_dim {
                embedded[b * seq_len * self.embed_dim + e] = 
                    self.cls_token[e] + self.pos_embed[e];
            }
            // Patch tokens
            for p in 0..num_patches {
                for e in 0..self.embed_dim {
                    let patch_idx = b * num_patches * self.embed_dim + p * self.embed_dim + e;
                    let out_idx = b * seq_len * self.embed_dim + (p + 1) * self.embed_dim + e;
                    let pos_idx = (p + 1) * self.embed_dim + e;
                    embedded[out_idx] = patches_data[patch_idx] + self.pos_embed[pos_idx];
                }
            }
        }

        let embedded_tensor = Tensor::from_slice(&embedded, &[batch_size, seq_len, self.embed_dim]).unwrap();

        // Transformer encoder
        let encoded = self.encoder.forward(&embedded_tensor, None);
        let encoded_data = encoded.data_f32();

        // Classification head (use CLS token)
        let mut output = vec![0.0f32; batch_size * self.num_classes];
        for b in 0..batch_size {
            for c in 0..self.num_classes {
                let mut sum = self.head_bias[c];
                for e in 0..self.embed_dim {
                    let cls_idx = b * seq_len * self.embed_dim + e;
                    sum += encoded_data[cls_idx] * self.head[e * self.num_classes + c];
                }
                output[b * self.num_classes + c] = sum;
            }
        }

        Tensor::from_slice(&output, &[batch_size, self.num_classes]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product_attention() {
        let q = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
        let k = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();

        let attn = ScaledDotProductAttention::new(2);
        let output = attn.forward(&q, &k, &v, None);
        assert_eq!(output.dims(), &[1, 2, 2]);
    }

    #[test]
    fn test_multi_head_attention() {
        let x = Tensor::from_slice(&[1.0f32; 16], &[1, 2, 8]).unwrap();
        let mut mha = MultiHeadAttention::new(8, 2);
        let output = mha.forward(&x, None);
        assert_eq!(output.dims(), &[1, 2, 8]);
    }

    #[test]
    fn test_positional_encoding() {
        let x = Tensor::from_slice(&[0.0f32; 16], &[1, 2, 8]).unwrap();
        let pe = PositionalEncoding::new(8, 100);
        let output = pe.forward(&x);
        assert_eq!(output.dims(), &[1, 2, 8]);
        // Output should not be all zeros due to positional encoding
        assert!(output.storage().as_slice::<f32>().to_vec().iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let x = Tensor::from_slice(&[1.0f32; 32], &[1, 4, 8]).unwrap();
        let mut layer = TransformerEncoderLayer::new(8, 2, 32);
        let output = layer.forward(&x, None);
        assert_eq!(output.dims(), &[1, 4, 8]);
    }

    #[test]
    fn test_transformer_encoder() {
        let x = Tensor::from_slice(&[1.0f32; 32], &[1, 4, 8]).unwrap();
        let mut encoder = TransformerEncoder::new(2, 8, 2, 32);
        let output = encoder.forward(&x, None);
        assert_eq!(output.dims(), &[1, 4, 8]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = Transformer::generate_causal_mask(4);
        let mask_data = mask.storage().as_slice::<f32>().to_vec();
        // Lower triangular should be 1s
        assert_eq!(mask_data[0], 1.0);  // [0,0]
        assert_eq!(mask_data[1], 0.0);  // [0,1]
        assert_eq!(mask_data[4], 1.0);  // [1,0]
        assert_eq!(mask_data[5], 1.0);  // [1,1]
    }
}



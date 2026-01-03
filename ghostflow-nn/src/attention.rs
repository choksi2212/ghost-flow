//! Attention mechanisms

use ghostflow_core::Tensor;
use crate::module::Module;
use crate::linear::Linear;
use crate::dropout::Dropout;

/// Scaled Dot-Product Attention
/// 
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    dropout_p: f32,
    training: bool,
) -> (Tensor, Tensor) {
    let d_k = query.dims()[query.ndim() - 1] as f32;
    let scale = 1.0 / d_k.sqrt();
    
    // QK^T
    let key_t = key.transpose(key.ndim() - 2, key.ndim() - 1).unwrap();
    let scores = query.matmul(&key_t).unwrap();
    let scaled_scores = scores.mul_scalar(scale);
    
    // Apply mask if provided
    let masked_scores = if let Some(m) = mask {
        apply_attention_mask(&scaled_scores, m)
    } else {
        scaled_scores
    };
    
    // Softmax
    let attn_weights = masked_scores.softmax(-1);
    
    // Apply dropout during training
    let attn_weights = if training && dropout_p > 0.0 {
        let dropout = Dropout::new(dropout_p);
        dropout.forward(&attn_weights)
    } else {
        attn_weights
    };
    
    // Attention output
    let output = attn_weights.matmul(value).unwrap();
    
    (output, attn_weights)
}

fn apply_attention_mask(scores: &Tensor, mask: &Tensor) -> Tensor {
    // mask: 1 = keep, 0 = mask out
    let mask_data = mask.data_f32();
    let scores_data = scores.data_f32();
    
    let result: Vec<f32> = scores_data.iter()
        .zip(mask_data.iter().cycle())
        .map(|(&s, &m)| {
            if m > 0.5 { s } else { f32::NEG_INFINITY }
        })
        .collect();
    
    Tensor::from_slice(&result, scores.dims()).unwrap()
}

/// Multi-Head Attention
pub struct MultiHeadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    
    dropout_p: f32,
    training: bool,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert!(embed_dim.is_multiple_of(num_heads), "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        
        MultiHeadAttention {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            dropout_p: dropout,
            training: true,
        }
    }

    /// Forward pass with optional key-value caching
    pub fn forward_with_cache(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        past_key: Option<&Tensor>,
        past_value: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let batch_size = query.dims()[0];
        let seq_len = query.dims()[1];
        
        // Project Q, K, V
        let q = self.q_proj.forward(query);
        let mut k = self.k_proj.forward(key);
        let mut v = self.v_proj.forward(value);
        
        // Concatenate with past key-value if provided (for incremental decoding)
        if let (Some(pk), Some(pv)) = (past_key, past_value) {
            k = concat_tensors(pk, &k, 1);
            v = concat_tensors(pv, &v, 1);
        }
        
        let kv_seq_len = k.dims()[1];
        
        // Reshape to [batch, heads, seq, head_dim]
        let q = self.reshape_for_attention(&q, batch_size, seq_len);
        let k = self.reshape_for_attention(&k, batch_size, kv_seq_len);
        let v = self.reshape_for_attention(&v, batch_size, kv_seq_len);
        
        // Scaled dot-product attention
        let (attn_output, attn_weights) = scaled_dot_product_attention(
            &q, &k, &v, mask, self.dropout_p, self.training
        );
        
        // Reshape back to [batch, seq, embed_dim]
        let attn_output = self.reshape_from_attention(&attn_output, batch_size, seq_len);
        
        // Output projection
        let output = self.out_proj.forward(&attn_output);
        
        // Return output, weights, and current k/v for caching
        let k_cache = self.reshape_for_attention(&self.k_proj.forward(key), batch_size, key.dims()[1]);
        let v_cache = self.reshape_for_attention(&self.v_proj.forward(value), batch_size, value.dims()[1]);
        
        (output, attn_weights, k_cache, v_cache)
    }

    fn reshape_for_attention(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, seq, embed_dim] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let reshaped = x.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]).unwrap();
        reshaped.transpose(1, 2).unwrap()
    }

    fn reshape_from_attention(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim] -> [batch, seq, embed_dim]
        let transposed = x.transpose(1, 2).unwrap();
        transposed.reshape(&[batch_size, seq_len, self.embed_dim]).unwrap()
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Self-attention: Q = K = V = input
        let (output, _, _, _) = self.forward_with_cache(input, input, input, None, None, None);
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// Cross-Attention (for encoder-decoder models)
pub struct CrossAttention {
    mha: MultiHeadAttention,
}

impl CrossAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        CrossAttention {
            mha: MultiHeadAttention::new(embed_dim, num_heads, dropout),
        }
    }

    pub fn forward_cross(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let (output, _, _, _) = self.mha.forward_with_cache(query, key, value, mask, None, None);
        output
    }
}

impl Module for CrossAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.mha.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.mha.parameters()
    }

    fn train(&mut self) { self.mha.train(); }
    fn eval(&mut self) { self.mha.eval(); }
    fn is_training(&self) -> bool { self.mha.is_training() }
}

/// Helper function to concatenate tensors along a dimension
fn concat_tensors(a: &Tensor, b: &Tensor, dim: usize) -> Tensor {
    let a_dims = a.dims();
    let b_dims = b.dims();
    
    let mut new_dims = a_dims.to_vec();
    new_dims[dim] = a_dims[dim] + b_dims[dim];
    
    let a_data = a.data_f32();
    let b_data = b.data_f32();
    
    // Simple concatenation for dim=1 (sequence dimension)
    if dim == 1 {
        let batch = a_dims[0];
        let a_seq = a_dims[1];
        let b_seq = b_dims[1];
        let rest: usize = a_dims[2..].iter().product();
        
        let mut result = Vec::with_capacity(batch * (a_seq + b_seq) * rest);
        
        for b_idx in 0..batch {
            // Copy from a
            let a_start = b_idx * a_seq * rest;
            result.extend_from_slice(&a_data[a_start..a_start + a_seq * rest]);
            // Copy from b
            let b_start = b_idx * b_seq * rest;
            result.extend_from_slice(&b_data[b_start..b_start + b_seq * rest]);
        }
        
        Tensor::from_slice(&result, &new_dims).unwrap()
    } else {
        // Fallback: just return b (simplified)
        b.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product_attention() {
        let q = Tensor::randn(&[2, 4, 8]); // [batch, seq, dim]
        let k = Tensor::randn(&[2, 4, 8]);
        let v = Tensor::randn(&[2, 4, 8]);
        
        let (output, weights) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
        
        assert_eq!(output.dims(), &[2, 4, 8]);
        assert_eq!(weights.dims(), &[2, 4, 4]);
    }

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(64, 8, 0.1);
        let input = Tensor::randn(&[2, 10, 64]); // [batch, seq, embed_dim]
        
        let output = mha.forward(&input);
        
        assert_eq!(output.dims(), &[2, 10, 64]);
    }
}

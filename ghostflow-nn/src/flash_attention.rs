//! Flash Attention
//!
//! Implements memory-efficient attention computation:
//! - Flash Attention algorithm for reduced memory usage
//! - Tiling and recomputation strategies
//! - Support for causal and non-causal attention
//! - Optimized for long sequences

use ghostflow_core::Tensor;
use std::cmp;

/// Flash Attention configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for tiling (M)
    pub block_size_m: usize,
    /// Block size for tiling (N)
    pub block_size_n: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability
    pub dropout: f32,
    /// Scale factor (usually 1/sqrt(d_k))
    pub scale: f32,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        FlashAttentionConfig {
            block_size_m: 64,
            block_size_n: 64,
            causal: false,
            dropout: 0.0,
            scale: 1.0,
        }
    }
}

impl FlashAttentionConfig {
    /// Configuration for causal attention (GPT-style)
    pub fn causal(scale: f32) -> Self {
        FlashAttentionConfig {
            causal: true,
            scale,
            ..Default::default()
        }
    }
    
    /// Configuration for bidirectional attention (BERT-style)
    pub fn bidirectional(scale: f32) -> Self {
        FlashAttentionConfig {
            causal: false,
            scale,
            ..Default::default()
        }
    }
    
    /// Configuration for long sequences
    pub fn long_sequence(scale: f32) -> Self {
        FlashAttentionConfig {
            block_size_m: 128,
            block_size_n: 128,
            scale,
            ..Default::default()
        }
    }
}

/// Flash Attention implementation
pub struct FlashAttention {
    config: FlashAttentionConfig,
}

impl FlashAttention {
    /// Create new Flash Attention
    pub fn new(config: FlashAttentionConfig) -> Self {
        FlashAttention { config }
    }
    
    /// Forward pass with Flash Attention algorithm
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<Tensor, String> {
        let q_dims = query.dims();
        let k_dims = key.dims();
        let v_dims = value.dims();
        
        // Validate dimensions
        if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
            return Err("Expected 3D tensors [batch, seq_len, d_model]".to_string());
        }
        
        let batch_size = q_dims[0];
        let seq_len_q = q_dims[1];
        let seq_len_k = k_dims[1];
        let d_model = q_dims[2];
        
        if k_dims[2] != d_model || v_dims[2] != d_model {
            return Err("Key and Value must have same d_model as Query".to_string());
        }
        
        // Process each batch independently
        let mut batch_outputs = Vec::new();
        
        for b in 0..batch_size {
            let q_batch = self.extract_batch(query, b)?;
            let k_batch = self.extract_batch(key, b)?;
            let v_batch = self.extract_batch(value, b)?;
            
            let output = self.flash_attention_single_batch(&q_batch, &k_batch, &v_batch)?;
            batch_outputs.push(output);
        }
        
        // Concatenate batch outputs
        self.concatenate_batches(&batch_outputs, batch_size, seq_len_q, d_model)
    }
    
    /// Flash Attention for single batch
    fn flash_attention_single_batch(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<Tensor, String> {
        let q_data = query.data_f32();
        let k_data = key.data_f32();
        let v_data = value.data_f32();
        
        let seq_len_q = query.dims()[0];
        let seq_len_k = key.dims()[0];
        let d_model = query.dims()[1];
        
        let mut output = vec![0.0f32; seq_len_q * d_model];
        let mut row_max = vec![f32::NEG_INFINITY; seq_len_q];
        let mut row_sum = vec![0.0f32; seq_len_q];
        
        // Tile over sequence length
        let block_m = self.config.block_size_m;
        let block_n = self.config.block_size_n;
        
        for i in (0..seq_len_q).step_by(block_m) {
            let end_i = cmp::min(i + block_m, seq_len_q);
            
            for j in (0..seq_len_k).step_by(block_n) {
                let end_j = cmp::min(j + block_n, seq_len_k);
                
                // Skip if causal and j > i
                if self.config.causal && j >= end_i {
                    continue;
                }
                
                self.process_block(
                    &q_data, &k_data, &v_data,
                    &mut output, &mut row_max, &mut row_sum,
                    i, end_i, j, end_j,
                    seq_len_q, seq_len_k, d_model,
                )?;
            }
        }
        
        // Final normalization
        for i in 0..seq_len_q {
            if row_sum[i] > 0.0 {
                for d in 0..d_model {
                    output[i * d_model + d] /= row_sum[i];
                }
            }
        }
        
        Tensor::from_slice(&output, &[seq_len_q, d_model])
            .map_err(|e| format!("Failed to create output tensor: {:?}", e))
    }
    
    /// Process a single attention block
    fn process_block(
        &self,
        q_data: &[f32],
        k_data: &[f32],
        v_data: &[f32],
        output: &mut [f32],
        row_max: &mut [f32],
        row_sum: &mut [f32],
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
        _seq_len_q: usize,
        seq_len_k: usize,
        d_model: usize,
    ) -> Result<(), String> {
        // Compute attention scores for this block
        for i in i_start..i_end {
            let mut block_max = f32::NEG_INFINITY;
            let mut scores = Vec::new();
            
            // Compute scores Q_i @ K_j^T
            for j in j_start..j_end {
                // Apply causal mask
                if self.config.causal && j > i {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                
                let mut score = 0.0;
                for d in 0..d_model {
                    score += q_data[i * d_model + d] * k_data[j * d_model + d];
                }
                score *= self.config.scale;
                
                scores.push(score);
                block_max = block_max.max(score);
            }
            
            // Update global max and compute softmax
            let old_max = row_max[i];
            let new_max = old_max.max(block_max);
            row_max[i] = new_max;
            
            // Compute exponentials and sum
            let mut block_sum = 0.0;
            for score in &mut scores {
                if *score != f32::NEG_INFINITY {
                    *score = (*score - new_max).exp();
                    block_sum += *score;
                } else {
                    *score = 0.0;
                }
            }
            
            // Update running sum
            let correction = (old_max - new_max).exp();
            row_sum[i] = row_sum[i] * correction + block_sum;
            
            // Update output with weighted values
            for (idx, &score) in scores.iter().enumerate() {
                let j = j_start + idx;
                if j < seq_len_k {
                    for d in 0..d_model {
                        output[i * d_model + d] = output[i * d_model + d] * correction
                            + score * v_data[j * d_model + d];
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract single batch from tensor
    fn extract_batch(&self, tensor: &Tensor, batch_idx: usize) -> Result<Tensor, String> {
        let data = tensor.data_f32();
        let dims = tensor.dims();
        let seq_len = dims[1];
        let d_model = dims[2];
        
        let start = batch_idx * seq_len * d_model;
        let end = start + seq_len * d_model;
        
        Tensor::from_slice(&data[start..end], &[seq_len, d_model])
            .map_err(|e| format!("Failed to extract batch: {:?}", e))
    }
    
    /// Concatenate batch outputs
    fn concatenate_batches(
        &self,
        batches: &[Tensor],
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
    ) -> Result<Tensor, String> {
        let mut result = Vec::with_capacity(batch_size * seq_len * d_model);
        
        for batch in batches {
            result.extend_from_slice(&batch.data_f32());
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_len, d_model])
            .map_err(|e| format!("Failed to concatenate batches: {:?}", e))
    }
    
    /// Estimate memory usage compared to standard attention
    pub fn memory_usage_ratio(&self, seq_len: usize, _d_model: usize) -> f32 {
        // Standard attention: O(seq_len^2)
        let standard_memory = seq_len * seq_len;
        
        // Flash attention: O(block_size^2)
        let flash_memory = self.config.block_size_m * self.config.block_size_n;
        
        flash_memory as f32 / standard_memory as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.block_size_m, 64);
        assert!(!config.causal);
        
        let causal = FlashAttentionConfig::causal(0.125);
        assert!(causal.causal);
        assert_eq!(causal.scale, 0.125);
    }
    
    #[test]
    fn test_flash_attention_forward() {
        let config = FlashAttentionConfig::default();
        let flash_attn = FlashAttention::new(config);
        
        let batch_size = 2;
        let seq_len = 8;
        let d_model = 16;
        
        let query = Tensor::randn(&[batch_size, seq_len, d_model]);
        let key = Tensor::randn(&[batch_size, seq_len, d_model]);
        let value = Tensor::randn(&[batch_size, seq_len, d_model]);
        
        let output = flash_attn.forward(&query, &key, &value).unwrap();
        assert_eq!(output.dims(), &[batch_size, seq_len, d_model]);
    }
    
    #[test]
    fn test_causal_attention() {
        let config = FlashAttentionConfig::causal(1.0);
        let flash_attn = FlashAttention::new(config);
        
        let query = Tensor::randn(&[1, 4, 8]);
        let key = Tensor::randn(&[1, 4, 8]);
        let value = Tensor::randn(&[1, 4, 8]);
        
        let output = flash_attn.forward(&query, &key, &value).unwrap();
        assert_eq!(output.dims(), &[1, 4, 8]);
    }
    
    #[test]
    fn test_memory_usage_ratio() {
        let config = FlashAttentionConfig::default();
        let flash_attn = FlashAttention::new(config);
        
        let ratio = flash_attn.memory_usage_ratio(1024, 512);
        assert!(ratio < 1.0); // Flash attention should use less memory
        assert!(ratio > 0.0);
    }
}

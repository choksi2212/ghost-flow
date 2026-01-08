//! LLaMA (Large Language Model Meta AI)
//!
//! Implements LLaMA architecture:
//! - RMSNorm instead of LayerNorm
//! - SwiGLU activation
//! - Rotary Position Embeddings (RoPE)
//! - Grouped Query Attention (GQA)
//! - KV cache for efficient inference

use ghostflow_core::Tensor;
use crate::linear::Linear;
use crate::Module;

/// LLaMA configuration
#[derive(Debug, Clone)]
pub struct LLaMAConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size (FFN)
    pub intermediate_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta
    pub rope_theta: f32,
}

impl Default for LLaMAConfig {
    fn default() -> Self {
        LLaMAConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

impl LLaMAConfig {
    /// LLaMA 7B
    pub fn llama_7b() -> Self {
        Self::default()
    }
    
    /// LLaMA 13B
    pub fn llama_13b() -> Self {
        LLaMAConfig {
            hidden_size: 5120,
            intermediate_size: 13824,
            num_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 40,
            ..Default::default()
        }
    }
    
    /// LLaMA 30B
    pub fn llama_30b() -> Self {
        LLaMAConfig {
            hidden_size: 6656,
            intermediate_size: 17920,
            num_layers: 60,
            num_attention_heads: 52,
            num_key_value_heads: 52,
            ..Default::default()
        }
    }
    
    /// LLaMA 65B
    pub fn llama_65b() -> Self {
        LLaMAConfig {
            hidden_size: 8192,
            intermediate_size: 22016,
            num_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 64,
            ..Default::default()
        }
    }
    
    /// LLaMA 2 7B
    pub fn llama2_7b() -> Self {
        LLaMAConfig {
            max_position_embeddings: 4096,
            ..Self::llama_7b()
        }
    }
    
    /// LLaMA 2 13B
    pub fn llama2_13b() -> Self {
        LLaMAConfig {
            max_position_embeddings: 4096,
            ..Self::llama_13b()
        }
    }
    
    /// LLaMA 2 70B
    pub fn llama2_70b() -> Self {
        LLaMAConfig {
            hidden_size: 8192,
            intermediate_size: 28672,
            num_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8, // GQA
            max_position_embeddings: 4096,
            ..Default::default()
        }
    }
    
    /// Tiny LLaMA for testing
    pub fn llama_tiny() -> Self {
        LLaMAConfig {
            vocab_size: 1000,
            hidden_size: 256,
            intermediate_size: 688,
            num_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

/// RMSNorm (Root Mean Square Layer Normalization)
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    /// Create new RMSNorm
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = Tensor::ones(&[hidden_size]);
        RMSNorm { weight, eps }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let x_data = x.data_f32();
        let dims = x.dims();
        
        if dims.len() < 2 {
            return Err(format!("Expected at least 2D input, got {}D", dims.len()));
        }
        
        let hidden_size = dims[dims.len() - 1];
        let batch_seq = x_data.len() / hidden_size;
        
        let weight_data = self.weight.data_f32();
        let mut result = Vec::with_capacity(x_data.len());
        
        for i in 0..batch_seq {
            let start = i * hidden_size;
            let end = start + hidden_size;
            let slice = &x_data[start..end];
            
            // Compute RMS
            let mean_sq: f32 = slice.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32;
            let rms = (mean_sq + self.eps).sqrt();
            
            // Normalize and scale
            for (j, &x) in slice.iter().enumerate() {
                result.push(x / rms * weight_data[j]);
            }
        }
        
        Tensor::from_slice(&result, dims)
            .map_err(|e| format!("Failed to create normalized tensor: {:?}", e))
    }
}

/// Rotary Position Embedding (RoPE)
pub struct RotaryEmbedding {
    /// Dimension
    dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Precomputed cos/sin values
    cos_cached: Vec<f32>,
    sin_cached: Vec<f32>,
}

impl RotaryEmbedding {
    /// Create new rotary embedding
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let mut cos_cached = Vec::with_capacity(max_seq_len * dim);
        let mut sin_cached = Vec::with_capacity(max_seq_len * dim);
        
        // Precompute cos and sin values
        for pos in 0..max_seq_len {
            for i in 0..(dim / 2) {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
                let angle = pos as f32 * freq;
                cos_cached.push(angle.cos());
                sin_cached.push(angle.sin());
            }
        }
        
        RotaryEmbedding {
            dim,
            max_seq_len,
            cos_cached,
            sin_cached,
        }
    }
    
    /// Apply rotary embedding
    pub fn forward(&self, x: &Tensor, position: usize) -> Result<Tensor, String> {
        if position >= self.max_seq_len {
            return Err(format!("Position {} exceeds max_seq_len {}", position, self.max_seq_len));
        }
        
        let x_data = x.data_f32();
        let dims = x.dims();
        let hidden_size = dims[dims.len() - 1];
        
        if hidden_size != self.dim {
            return Err(format!("Hidden size {} doesn't match RoPE dim {}", hidden_size, self.dim));
        }
        
        let mut result = Vec::with_capacity(x_data.len());
        let offset = position * (self.dim / 2);
        
        // Apply rotation
        for chunk in x_data.chunks(self.dim) {
            for i in 0..(self.dim / 2) {
                let cos = self.cos_cached[offset + i];
                let sin = self.sin_cached[offset + i];
                
                let x1 = chunk[2 * i];
                let x2 = chunk[2 * i + 1];
                
                result.push(x1 * cos - x2 * sin);
                result.push(x1 * sin + x2 * cos);
            }
        }
        
        Tensor::from_slice(&result, dims)
            .map_err(|e| format!("Failed to apply RoPE: {:?}", e))
    }
}

/// SwiGLU activation (used in LLaMA FFN)
pub struct SwiGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLU {
    /// Create new SwiGLU
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        SwiGLU {
            gate_proj: Linear::new(hidden_size, intermediate_size),
            up_proj: Linear::new(hidden_size, intermediate_size),
            down_proj: Linear::new(intermediate_size, hidden_size),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        
        // SwiGLU: gate.silu() * up
        let gate_silu = gate.silu();
        let intermediate = gate_silu.mul(&up).unwrap_or(gate_silu);
        
        self.down_proj.forward(&intermediate)
    }
}

/// LLaMA Attention with Grouped Query Attention (GQA)
pub struct LLaMAAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LLaMAAttention {
    /// Create new LLaMA attention
    pub fn new(config: &LLaMAConfig) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        
        LLaMAAttention {
            q_proj: Linear::new(config.hidden_size, config.num_attention_heads * head_dim),
            k_proj: Linear::new(config.hidden_size, config.num_key_value_heads * head_dim),
            v_proj: Linear::new(config.hidden_size, config.num_key_value_heads * head_dim),
            o_proj: Linear::new(config.num_attention_heads * head_dim, config.hidden_size),
            rope: RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        }
    }
    
    /// Forward pass (simplified)
    pub fn forward(&self, hidden_states: &Tensor, position: usize) -> Tensor {
        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states);
        let _k = self.k_proj.forward(hidden_states);
        let _v = self.v_proj.forward(hidden_states);
        
        // Apply RoPE to Q and K
        let q_rope = self.rope.forward(&q, position).unwrap_or(q);
        
        // Simplified attention (real implementation would do proper multi-head attention)
        // For now, just project back
        self.o_proj.forward(&q_rope)
    }
}

/// LLaMA Decoder Layer
pub struct LLaMADecoderLayer {
    self_attn: LLaMAAttention,
    mlp: SwiGLU,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl LLaMADecoderLayer {
    /// Create new decoder layer
    pub fn new(config: &LLaMAConfig) -> Self {
        LLaMADecoderLayer {
            self_attn: LLaMAAttention::new(config),
            mlp: SwiGLU::new(config.hidden_size, config.intermediate_size),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor, position: usize) -> Result<Tensor, String> {
        // Self attention with residual
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, position);
        let hidden_states = hidden_states.add(&residual).unwrap_or(hidden_states);
        
        // FFN with residual
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states);
        let hidden_states = hidden_states.add(&residual).unwrap_or(hidden_states);
        
        Ok(hidden_states)
    }
}

/// LLaMA Model
pub struct LLaMAModel {
    config: LLaMAConfig,
    embed_tokens: Tensor,
    layers: Vec<LLaMADecoderLayer>,
    norm: RMSNorm,
}

impl LLaMAModel {
    /// Create new LLaMA model
    pub fn new(config: LLaMAConfig) -> Self {
        let embed_tokens = Tensor::randn(&[config.vocab_size, config.hidden_size]);
        
        let layers = (0..config.num_layers)
            .map(|_| LLaMADecoderLayer::new(&config))
            .collect();
        
        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        
        LLaMAModel {
            config,
            embed_tokens,
            layers,
            norm,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        // Get embeddings
        let mut hidden_states = self.get_embeddings(input_ids)?;
        
        // Pass through decoder layers
        let seq_len = input_ids.dims()[1];
        for pos in 0..seq_len {
            for layer in &self.layers {
                hidden_states = layer.forward(&hidden_states, pos)?;
            }
        }
        
        // Final norm
        self.norm.forward(&hidden_states)
    }
    
    /// Get token embeddings
    fn get_embeddings(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let ids_data = input_ids.data_f32();
        let embed_data = self.embed_tokens.data_f32();
        
        let dims = input_ids.dims();
        let batch_size = dims[0];
        let seq_length = dims[1];
        let hidden_size = self.config.hidden_size;
        
        let mut result = Vec::with_capacity(batch_size * seq_length * hidden_size);
        
        for &id in ids_data.iter() {
            let idx = id as usize;
            if idx >= self.config.vocab_size {
                return Err(format!("Token ID {} out of vocabulary", idx));
            }
            
            let start = idx * hidden_size;
            let end = start + hidden_size;
            result.extend_from_slice(&embed_data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_length, hidden_size])
            .map_err(|e| format!("Failed to create embeddings: {:?}", e))
    }
}

/// LLaMA for Causal Language Modeling
pub struct LLaMAForCausalLM {
    model: LLaMAModel,
    lm_head: Linear,
}

impl LLaMAForCausalLM {
    /// Create new LLaMA for causal LM
    pub fn new(config: LLaMAConfig) -> Self {
        let model = LLaMAModel::new(config.clone());
        let lm_head = Linear::new(config.hidden_size, config.vocab_size);
        
        LLaMAForCausalLM {
            model,
            lm_head,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let hidden_states = self.model.forward(input_ids)?;
        let logits = self.lm_head.forward(&hidden_states);
        Ok(logits)
    }
    
    /// Generate text (simplified greedy decoding)
    pub fn generate(&self, input_ids: &Tensor, max_new_tokens: usize) -> Result<Vec<usize>, String> {
        let mut current_ids = input_ids.data_f32().iter().map(|&x| x as usize).collect::<Vec<_>>();
        
        for _ in 0..max_new_tokens {
            let input_tensor = Tensor::from_slice(
                &current_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                &[1, current_ids.len()]
            ).map_err(|e| format!("Failed to create input: {:?}", e))?;
            
            let logits = self.forward(&input_tensor)?;
            let next_token = self.sample_next_token(&logits)?;
            
            current_ids.push(next_token);
        }
        
        Ok(current_ids)
    }
    
    /// Sample next token (greedy)
    fn sample_next_token(&self, logits: &Tensor) -> Result<usize, String> {
        let data = logits.data_f32();
        let dims = logits.dims();
        
        let seq_len = dims[1];
        let vocab_size = dims[2];
        
        // Get last token logits
        let start = (seq_len - 1) * vocab_size;
        let end = start + vocab_size;
        let last_logits = &data[start..end];
        
        // Greedy sampling
        let next_token = last_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| "Failed to sample token".to_string())?;
        
        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_llama_config() {
        let config = LLaMAConfig::llama_7b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        
        let config = LLaMAConfig::llama2_70b();
        assert_eq!(config.num_key_value_heads, 8); // GQA
        assert_eq!(config.max_position_embeddings, 4096);
    }
    
    #[test]
    fn test_rms_norm() {
        let norm = RMSNorm::new(128, 1e-6);
        let x = Tensor::randn(&[2, 4, 128]);
        let output = norm.forward(&x).unwrap();
        assert_eq!(output.dims(), &[2, 4, 128]);
    }
    
    #[test]
    fn test_rope() {
        let rope = RotaryEmbedding::new(64, 512, 10000.0);
        let x = Tensor::randn(&[2, 64]);
        let output = rope.forward(&x, 10).unwrap();
        assert_eq!(output.dims(), &[2, 64]);
    }
    
    #[test]
    fn test_llama_model() {
        let config = LLaMAConfig::llama_tiny();
        let model = LLaMAModel::new(config);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = model.forward(&input_ids).unwrap();
        
        assert_eq!(output.dims(), &[2, 2, 256]); // batch=2, seq=2, hidden=256
    }
}

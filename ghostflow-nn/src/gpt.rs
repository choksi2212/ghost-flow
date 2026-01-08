//! GPT (Generative Pre-trained Transformer)
//!
//! Implements GPT-style autoregressive language models
//! - Token embeddings
//! - Position embeddings
//! - Causal (masked) self-attention
//! - Transformer decoder blocks
//! - Language modeling head

use ghostflow_core::Tensor;
use crate::transformer::TransformerEncoder;
use crate::linear::Linear;
use crate::norm::LayerNorm;
use crate::activation::GELU;
use crate::Module;

/// GPT configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Context length (maximum sequence length)
    pub context_length: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Feed-forward hidden dimension
    pub ff_dim: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Use bias in linear layers
    pub bias: bool,
}

impl Default for GPTConfig {
    fn default() -> Self {
        GPTConfig {
            vocab_size: 50257,
            context_length: 1024,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            ff_dim: 3072,
            dropout: 0.1,
            bias: true,
        }
    }
}

impl GPTConfig {
    /// GPT-2 Small (117M parameters)
    pub fn gpt2_small() -> Self {
        Self::default()
    }
    
    /// GPT-2 Medium (345M parameters)
    pub fn gpt2_medium() -> Self {
        GPTConfig {
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            ff_dim: 4096,
            ..Default::default()
        }
    }
    
    /// GPT-2 Large (774M parameters)
    pub fn gpt2_large() -> Self {
        GPTConfig {
            embed_dim: 1280,
            num_layers: 36,
            num_heads: 20,
            ff_dim: 5120,
            ..Default::default()
        }
    }
    
    /// GPT-2 XL (1.5B parameters)
    pub fn gpt2_xl() -> Self {
        GPTConfig {
            embed_dim: 1600,
            num_layers: 48,
            num_heads: 25,
            ff_dim: 6400,
            ..Default::default()
        }
    }
    
    /// GPT-3 Small (125M parameters)
    pub fn gpt3_small() -> Self {
        GPTConfig {
            vocab_size: 50257,
            context_length: 2048,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            ff_dim: 3072,
            dropout: 0.0,
            bias: false,
        }
    }
    
    /// GPT-3 Medium (350M parameters)
    pub fn gpt3_medium() -> Self {
        GPTConfig {
            vocab_size: 50257,
            context_length: 2048,
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            ff_dim: 4096,
            dropout: 0.0,
            bias: false,
        }
    }
    
    /// GPT-3 Large (760M parameters)
    pub fn gpt3_large() -> Self {
        GPTConfig {
            vocab_size: 50257,
            context_length: 2048,
            embed_dim: 1280,
            num_layers: 36,
            num_heads: 20,
            ff_dim: 5120,
            dropout: 0.0,
            bias: false,
        }
    }
    
    /// GPT-3 XL (1.3B parameters)
    pub fn gpt3_xl() -> Self {
        GPTConfig {
            vocab_size: 50257,
            context_length: 2048,
            embed_dim: 1536,
            num_layers: 48,
            num_heads: 24,
            ff_dim: 6144,
            dropout: 0.0,
            bias: false,
        }
    }
    
    /// GPT-Tiny (for testing)
    pub fn gpt_tiny() -> Self {
        GPTConfig {
            vocab_size: 1000,
            context_length: 128,
            embed_dim: 128,
            num_layers: 2,
            num_heads: 2,
            ff_dim: 512,
            dropout: 0.1,
            bias: true,
        }
    }
}

/// GPT embeddings (token + position)
pub struct GPTEmbeddings {
    /// Token embeddings
    token_embeddings: Tensor,
    /// Position embeddings
    position_embeddings: Tensor,
    /// Dropout
    dropout: f32,
    /// Configuration
    config: GPTConfig,
}

impl GPTEmbeddings {
    /// Create new GPT embeddings
    pub fn new(config: GPTConfig) -> Self {
        let token_embeddings = Tensor::randn(&[config.vocab_size, config.embed_dim]);
        let position_embeddings = Tensor::randn(&[config.context_length, config.embed_dim]);
        
        GPTEmbeddings {
            token_embeddings,
            position_embeddings,
            dropout: config.dropout,
            config,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let dims = input_ids.dims();
        if dims.len() != 2 {
            return Err(format!("Expected 2D input_ids, got {}D", dims.len()));
        }
        
        let batch_size = dims[0];
        let seq_length = dims[1];
        
        if seq_length > self.config.context_length {
            return Err(format!("Sequence length {} exceeds context length {}", 
                             seq_length, self.config.context_length));
        }
        
        // Get token embeddings
        let token_embeds = self.get_token_embeddings(input_ids)?;
        
        // Get position embeddings
        let position_embeds = self.get_position_embeddings(seq_length)?;
        
        // Sum embeddings
        self.sum_embeddings(&token_embeds, &position_embeds)
    }
    
    /// Get token embeddings by lookup
    fn get_token_embeddings(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let ids_data = input_ids.data_f32();
        let embed_data = self.token_embeddings.data_f32();
        
        let dims = input_ids.dims();
        let batch_size = dims[0];
        let seq_length = dims[1];
        let embed_dim = self.config.embed_dim;
        
        let mut result = Vec::with_capacity(batch_size * seq_length * embed_dim);
        
        for &id in ids_data.iter() {
            let idx = id as usize;
            if idx >= self.config.vocab_size {
                return Err(format!("Token ID {} out of vocabulary range", idx));
            }
            
            let start = idx * embed_dim;
            let end = start + embed_dim;
            result.extend_from_slice(&embed_data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_length, embed_dim])
            .map_err(|e| format!("Failed to create token embeddings: {:?}", e))
    }
    
    /// Get position embeddings
    fn get_position_embeddings(&self, seq_length: usize) -> Result<Tensor, String> {
        let embed_data = self.position_embeddings.data_f32();
        let embed_dim = self.config.embed_dim;
        
        let result = embed_data[..seq_length * embed_dim].to_vec();
        
        Tensor::from_slice(&result, &[seq_length, embed_dim])
            .map_err(|e| format!("Failed to create position embeddings: {:?}", e))
    }
    
    /// Sum token and position embeddings
    fn sum_embeddings(&self, token: &Tensor, position: &Tensor) -> Result<Tensor, String> {
        let token_data = token.data_f32();
        let pos_data = position.data_f32();
        
        let dims = token.dims();
        let batch_size = dims[0];
        let seq_length = dims[1];
        let embed_dim = dims[2];
        
        let mut result = Vec::with_capacity(token_data.len());
        
        for b in 0..batch_size {
            for s in 0..seq_length {
                for e in 0..embed_dim {
                    let token_idx = b * seq_length * embed_dim + s * embed_dim + e;
                    let pos_idx = s * embed_dim + e;
                    result.push(token_data[token_idx] + pos_data[pos_idx]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_length, embed_dim])
            .map_err(|e| format!("Failed to sum embeddings: {:?}", e))
    }
}

/// GPT model
pub struct GPTModel {
    /// Configuration
    config: GPTConfig,
    /// Embeddings
    embeddings: GPTEmbeddings,
    /// Transformer blocks
    transformer: TransformerEncoder,
    /// Final layer norm
    ln_f: LayerNorm,
}

impl GPTModel {
    /// Create new GPT model
    pub fn new(config: GPTConfig) -> Self {
        let embeddings = GPTEmbeddings::new(config.clone());
        
        let transformer = TransformerEncoder::new(
            config.embed_dim,
            config.num_heads,
            config.ff_dim,
            config.num_layers,
            config.dropout,
        );
        
        let ln_f = LayerNorm::new(&[config.embed_dim]);
        
        GPTModel {
            config,
            embeddings,
            transformer,
            ln_f,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        // Get embeddings
        let hidden_states = self.embeddings.forward(input_ids)?;
        
        // Transformer blocks (with causal masking)
        let hidden_states = self.transformer.forward(&hidden_states);
        
        // Final layer norm
        let hidden_states = self.ln_f.forward(&hidden_states);
        
        Ok(hidden_states)
    }
}

/// GPT for Language Modeling
pub struct GPTForCausalLM {
    /// Base GPT model
    gpt: GPTModel,
    /// Language modeling head
    lm_head: Linear,
}

impl GPTForCausalLM {
    /// Create new GPT for causal language modeling
    pub fn new(config: GPTConfig) -> Self {
        let gpt = GPTModel::new(config.clone());
        let lm_head = Linear::new(config.embed_dim, config.vocab_size);
        
        GPTForCausalLM {
            gpt,
            lm_head,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let hidden_states = self.gpt.forward(input_ids)?;
        let logits = self.lm_head.forward(&hidden_states);
        Ok(logits)
    }
    
    /// Generate text autoregressively
    pub fn generate(&self, input_ids: &Tensor, max_new_tokens: usize, 
                    temperature: f32) -> Result<Vec<usize>, String> {
        let mut current_ids = input_ids.data_f32().iter().map(|&x| x as usize).collect::<Vec<_>>();
        
        for _ in 0..max_new_tokens {
            // Get logits for current sequence
            let input_tensor = Tensor::from_slice(
                &current_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                &[1, current_ids.len()]
            ).map_err(|e| format!("Failed to create input tensor: {:?}", e))?;
            
            let logits = self.forward(&input_tensor)?;
            
            // Get logits for last token
            let last_logits = self.extract_last_token_logits(&logits)?;
            
            // Apply temperature and sample
            let next_token = self.sample_token(&last_logits, temperature)?;
            
            current_ids.push(next_token);
        }
        
        Ok(current_ids)
    }
    
    /// Extract logits for last token
    fn extract_last_token_logits(&self, logits: &Tensor) -> Result<Tensor, String> {
        let data = logits.data_f32();
        let dims = logits.dims();
        
        if dims.len() != 3 {
            return Err(format!("Expected 3D logits, got {}D", dims.len()));
        }
        
        let seq_length = dims[1];
        let vocab_size = dims[2];
        
        // Extract last token: [vocab_size]
        let start = (seq_length - 1) * vocab_size;
        let end = start + vocab_size;
        let last_logits = data[start..end].to_vec();
        
        Tensor::from_slice(&last_logits, &[vocab_size])
            .map_err(|e| format!("Failed to extract last token logits: {:?}", e))
    }
    
    /// Sample next token from logits
    fn sample_token(&self, logits: &Tensor, temperature: f32) -> Result<usize, String> {
        let data = logits.data_f32();
        
        // Apply temperature
        let scaled: Vec<f32> = data.iter().map(|&x| x / temperature).collect();
        
        // Softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
        
        // Sample (greedy for now - take argmax)
        let next_token = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| "Failed to sample token".to_string())?;
        
        Ok(next_token)
    }
}

/// GPT for Text Classification
pub struct GPTForSequenceClassification {
    /// Base GPT model
    gpt: GPTModel,
    /// Classification head
    classifier: Linear,
    /// Number of labels
    num_labels: usize,
}

impl GPTForSequenceClassification {
    /// Create new GPT for classification
    pub fn new(config: GPTConfig, num_labels: usize) -> Self {
        let gpt = GPTModel::new(config.clone());
        let classifier = Linear::new(config.embed_dim, num_labels);
        
        GPTForSequenceClassification {
            gpt,
            classifier,
            num_labels,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let hidden_states = self.gpt.forward(input_ids)?;
        
        // Use last token for classification
        let last_hidden = self.extract_last_token(&hidden_states)?;
        
        let logits = self.classifier.forward(&last_hidden);
        Ok(logits)
    }
    
    /// Extract last token hidden state
    fn extract_last_token(&self, hidden_states: &Tensor) -> Result<Tensor, String> {
        let data = hidden_states.data_f32();
        let dims = hidden_states.dims();
        
        if dims.len() != 3 {
            return Err(format!("Expected 3D hidden states, got {}D", dims.len()));
        }
        
        let batch_size = dims[0];
        let seq_length = dims[1];
        let embed_dim = dims[2];
        
        let mut result = Vec::with_capacity(batch_size * embed_dim);
        
        for b in 0..batch_size {
            let start = b * seq_length * embed_dim + (seq_length - 1) * embed_dim;
            let end = start + embed_dim;
            result.extend_from_slice(&data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, embed_dim])
            .map_err(|e| format!("Failed to extract last token: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpt_config() {
        let config = GPTConfig::gpt2_small();
        assert_eq!(config.embed_dim, 768);
        assert_eq!(config.num_layers, 12);
        
        let config = GPTConfig::gpt2_xl();
        assert_eq!(config.embed_dim, 1600);
        assert_eq!(config.num_layers, 48);
        
        let config = GPTConfig::gpt3_large();
        assert_eq!(config.embed_dim, 1280);
        assert_eq!(config.context_length, 2048);
    }
    
    #[test]
    fn test_gpt_embeddings() {
        let config = GPTConfig::gpt_tiny();
        let embeddings = GPTEmbeddings::new(config);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = embeddings.forward(&input_ids).unwrap();
        
        assert_eq!(output.dims(), &[2, 2, 128]); // batch=2, seq=2, embed=128
    }
    
    #[test]
    fn test_gpt_model() {
        let config = GPTConfig::gpt_tiny();
        let gpt = GPTModel::new(config);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = gpt.forward(&input_ids).unwrap();
        
        assert_eq!(output.dims(), &[2, 2, 128]); // batch=2, seq=2, embed=128
    }
    
    #[test]
    fn test_gpt_for_causal_lm() {
        let config = GPTConfig::gpt_tiny();
        let gpt = GPTForCausalLM::new(config.clone());
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = gpt.forward(&input_ids).unwrap();
        
        assert_eq!(output.dims(), &[2, 2, 1000]); // batch=2, seq=2, vocab=1000
    }
    
    #[test]
    fn test_gpt_for_classification() {
        let config = GPTConfig::gpt_tiny();
        let gpt = GPTForSequenceClassification::new(config, 2);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = gpt.forward(&input_ids).unwrap();
        
        assert_eq!(output.dims(), &[2, 2]); // batch=2, num_labels=2
    }
}

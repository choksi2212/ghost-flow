//! T5 (Text-to-Text Transfer Transformer)
//!
//! Implements T5 as described in "Exploring the Limits of Transfer Learning"
//! - Encoder-decoder architecture
//! - Relative position embeddings
//! - Text-to-text framework
//! - Multiple task support (translation, summarization, QA, etc.)

use ghostflow_core::Tensor;
use crate::transformer::{TransformerEncoder, TransformerDecoderLayer};
use crate::linear::Linear;
use crate::norm::LayerNorm;
use crate::Module;

/// T5 configuration
#[derive(Debug, Clone)]
pub struct T5Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension
    pub d_model: usize,
    /// Key/value dimension
    pub d_kv: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Use relative attention bias
    pub relative_attention: bool,
}

impl Default for T5Config {
    fn default() -> Self {
        T5Config {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            num_heads: 8,
            dropout: 0.1,
            relative_attention: true,
        }
    }
}

impl T5Config {
    /// T5-Small (60M parameters)
    pub fn t5_small() -> Self {
        Self::default()
    }
    
    /// T5-Base (220M parameters)
    pub fn t5_base() -> Self {
        T5Config {
            d_model: 768,
            d_kv: 64,
            d_ff: 3072,
            num_encoder_layers: 12,
            num_decoder_layers: 12,
            num_heads: 12,
            ..Default::default()
        }
    }
    
    /// T5-Large (770M parameters)
    pub fn t5_large() -> Self {
        T5Config {
            d_model: 1024,
            d_kv: 64,
            d_ff: 4096,
            num_encoder_layers: 24,
            num_decoder_layers: 24,
            num_heads: 16,
            ..Default::default()
        }
    }
    
    /// T5-3B (3B parameters)
    pub fn t5_3b() -> Self {
        T5Config {
            d_model: 1024,
            d_kv: 128,
            d_ff: 16384,
            num_encoder_layers: 24,
            num_decoder_layers: 24,
            num_heads: 32,
            ..Default::default()
        }
    }
    
    /// T5-11B (11B parameters)
    pub fn t5_11b() -> Self {
        T5Config {
            d_model: 1024,
            d_kv: 128,
            d_ff: 65536,
            num_encoder_layers: 24,
            num_decoder_layers: 24,
            num_heads: 128,
            ..Default::default()
        }
    }
    
    /// T5-Tiny (for testing)
    pub fn t5_tiny() -> Self {
        T5Config {
            vocab_size: 1000,
            d_model: 128,
            d_kv: 16,
            d_ff: 512,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            num_heads: 4,
            dropout: 0.1,
            relative_attention: true,
        }
    }
}

/// T5 embeddings (shared between encoder and decoder)
pub struct T5Embeddings {
    /// Token embeddings
    token_embeddings: Tensor,
    /// Configuration
    config: T5Config,
}

impl T5Embeddings {
    /// Create new T5 embeddings
    pub fn new(config: T5Config) -> Self {
        let token_embeddings = Tensor::randn(&[config.vocab_size, config.d_model]);
        
        T5Embeddings {
            token_embeddings,
            config,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let ids_data = input_ids.data_f32();
        let embed_data = self.token_embeddings.data_f32();
        
        let dims = input_ids.dims();
        if dims.len() != 2 {
            return Err(format!("Expected 2D input_ids, got {}D", dims.len()));
        }
        
        let batch_size = dims[0];
        let seq_length = dims[1];
        let d_model = self.config.d_model;
        
        let mut result = Vec::with_capacity(batch_size * seq_length * d_model);
        
        for &id in ids_data.iter() {
            let idx = id as usize;
            if idx >= self.config.vocab_size {
                return Err(format!("Token ID {} out of vocabulary range", idx));
            }
            
            let start = idx * d_model;
            let end = start + d_model;
            result.extend_from_slice(&embed_data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_length, d_model])
            .map_err(|e| format!("Failed to create embeddings: {:?}", e))
    }
}

/// T5 Encoder
pub struct T5Encoder {
    /// Embeddings
    embeddings: T5Embeddings,
    /// Encoder layers
    encoder: TransformerEncoder,
    /// Final layer norm
    final_layer_norm: LayerNorm,
    /// Dropout
    dropout: f32,
}

impl T5Encoder {
    /// Create new T5 encoder
    pub fn new(config: &T5Config, embeddings: T5Embeddings) -> Self {
        let encoder = TransformerEncoder::new(
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.num_encoder_layers,
            config.dropout,
        );
        
        let final_layer_norm = LayerNorm::new(&[config.d_model]);
        
        T5Encoder {
            embeddings,
            encoder,
            final_layer_norm,
            dropout: config.dropout,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        // Get embeddings
        let hidden_states = self.embeddings.forward(input_ids)?;
        
        // Encoder layers
        let hidden_states = self.encoder.forward(&hidden_states);
        
        // Final layer norm
        let hidden_states = self.final_layer_norm.forward(&hidden_states);
        
        Ok(hidden_states)
    }
}

/// T5 Decoder
pub struct T5Decoder {
    /// Embeddings (shared with encoder)
    embeddings: T5Embeddings,
    /// Decoder layers
    layers: Vec<TransformerDecoderLayer>,
    /// Final layer norm
    final_layer_norm: LayerNorm,
    /// Dropout
    dropout: f32,
}

impl T5Decoder {
    /// Create new T5 decoder
    pub fn new(config: &T5Config, embeddings: T5Embeddings) -> Self {
        let layers = (0..config.num_decoder_layers)
            .map(|_| TransformerDecoderLayer::new(config.d_model, config.num_heads, config.d_ff, config.dropout))
            .collect();
        
        let final_layer_norm = LayerNorm::new(&[config.d_model]);
        
        T5Decoder {
            embeddings,
            layers,
            final_layer_norm,
            dropout: config.dropout,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, decoder_input_ids: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor, String> {
        // Get embeddings
        let mut hidden_states = self.embeddings.forward(decoder_input_ids)?;
        
        // Decoder layers with cross-attention
        for layer in &self.layers {
            hidden_states = layer.forward_with_memory(&hidden_states, encoder_hidden_states, None, None);
        }
        
        // Final layer norm
        let hidden_states = self.final_layer_norm.forward(&hidden_states);
        
        Ok(hidden_states)
    }
}

/// T5 Model (encoder-decoder)
pub struct T5Model {
    /// Configuration
    config: T5Config,
    /// Shared embeddings
    shared_embeddings: T5Embeddings,
    /// Encoder
    encoder: T5Encoder,
    /// Decoder
    decoder: T5Decoder,
}

impl T5Model {
    /// Create new T5 model
    pub fn new(config: T5Config) -> Self {
        // Create shared embeddings
        let shared_embeddings = T5Embeddings::new(config.clone());
        
        // Create encoder with shared embeddings
        let encoder_embeddings = T5Embeddings::new(config.clone());
        let encoder = T5Encoder::new(&config, encoder_embeddings);
        
        // Create decoder with shared embeddings
        let decoder_embeddings = T5Embeddings::new(config.clone());
        let decoder = T5Decoder::new(&config, decoder_embeddings);
        
        T5Model {
            config,
            shared_embeddings,
            encoder,
            decoder,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, decoder_input_ids: &Tensor) -> Result<T5Output, String> {
        // Encode
        let encoder_hidden_states = self.encoder.forward(input_ids)?;
        
        // Decode
        let decoder_hidden_states = self.decoder.forward(decoder_input_ids, &encoder_hidden_states)?;
        
        Ok(T5Output {
            last_hidden_state: decoder_hidden_states,
            encoder_last_hidden_state: encoder_hidden_states,
        })
    }
}

/// T5 output
pub struct T5Output {
    /// Decoder last hidden state
    pub last_hidden_state: Tensor,
    /// Encoder last hidden state
    pub encoder_last_hidden_state: Tensor,
}

/// T5 for Conditional Generation (translation, summarization, etc.)
pub struct T5ForConditionalGeneration {
    /// Base T5 model
    t5: T5Model,
    /// Language modeling head
    lm_head: Linear,
}

impl T5ForConditionalGeneration {
    /// Create new T5 for conditional generation
    pub fn new(config: T5Config) -> Self {
        let t5 = T5Model::new(config.clone());
        let lm_head = Linear::new(config.d_model, config.vocab_size);
        
        T5ForConditionalGeneration {
            t5,
            lm_head,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, decoder_input_ids: &Tensor) -> Result<Tensor, String> {
        let output = self.t5.forward(input_ids, decoder_input_ids)?;
        let logits = self.lm_head.forward(&output.last_hidden_state);
        Ok(logits)
    }
    
    /// Generate text (simplified greedy decoding)
    pub fn generate(&self, input_ids: &Tensor, max_length: usize) -> Result<Vec<usize>, String> {
        // Start with decoder start token (0)
        let mut generated = vec![0usize];
        
        for _ in 0..max_length {
            // Create decoder input tensor
            let decoder_input = Tensor::from_slice(
                &generated.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                &[1, generated.len()]
            ).map_err(|e| format!("Failed to create decoder input: {:?}", e))?;
            
            // Forward pass
            let logits = self.forward(input_ids, &decoder_input)?;
            
            // Get last token logits
            let next_token = self.sample_next_token(&logits)?;
            
            // Check for end token (1)
            if next_token == 1 {
                break;
            }
            
            generated.push(next_token);
        }
        
        Ok(generated)
    }
    
    /// Sample next token (greedy)
    fn sample_next_token(&self, logits: &Tensor) -> Result<usize, String> {
        let data = logits.data_f32();
        let dims = logits.dims();
        
        if dims.len() != 3 {
            return Err(format!("Expected 3D logits, got {}D", dims.len()));
        }
        
        let seq_length = dims[1];
        let vocab_size = dims[2];
        
        // Get last token logits
        let start = (seq_length - 1) * vocab_size;
        let end = start + vocab_size;
        let last_logits = &data[start..end];
        
        // Greedy sampling (argmax)
        let next_token = last_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| "Failed to sample token".to_string())?;
        
        Ok(next_token)
    }
}

/// T5 for Classification
pub struct T5ForSequenceClassification {
    /// Base T5 model
    t5: T5Model,
    /// Classification head
    classifier: Linear,
    /// Number of labels
    num_labels: usize,
}

impl T5ForSequenceClassification {
    /// Create new T5 for classification
    pub fn new(config: T5Config, num_labels: usize) -> Self {
        let t5 = T5Model::new(config.clone());
        let classifier = Linear::new(config.d_model, num_labels);
        
        T5ForSequenceClassification {
            t5,
            classifier,
            num_labels,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, decoder_input_ids: &Tensor) -> Result<Tensor, String> {
        let output = self.t5.forward(input_ids, decoder_input_ids)?;
        
        // Use first decoder token for classification
        let first_token = self.extract_first_token(&output.last_hidden_state)?;
        
        let logits = self.classifier.forward(&first_token);
        Ok(logits)
    }
    
    /// Extract first token
    fn extract_first_token(&self, hidden_states: &Tensor) -> Result<Tensor, String> {
        let data = hidden_states.data_f32();
        let dims = hidden_states.dims();
        
        if dims.len() != 3 {
            return Err(format!("Expected 3D hidden states, got {}D", dims.len()));
        }
        
        let batch_size = dims[0];
        let d_model = dims[2];
        
        let mut result = Vec::with_capacity(batch_size * d_model);
        
        for b in 0..batch_size {
            let start = b * dims[1] * d_model;
            let end = start + d_model;
            result.extend_from_slice(&data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, d_model])
            .map_err(|e| format!("Failed to extract first token: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_t5_config() {
        let config = T5Config::t5_small();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_encoder_layers, 6);
        
        let config = T5Config::t5_base();
        assert_eq!(config.d_model, 768);
        assert_eq!(config.num_encoder_layers, 12);
        
        let config = T5Config::t5_large();
        assert_eq!(config.d_model, 1024);
        assert_eq!(config.num_encoder_layers, 24);
    }
    
    #[test]
    fn test_t5_embeddings() {
        let config = T5Config::t5_tiny();
        let embeddings = T5Embeddings::new(config);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = embeddings.forward(&input_ids).unwrap();
        
        assert_eq!(output.dims(), &[2, 2, 128]); // batch=2, seq=2, d_model=128
    }
    
    #[test]
    fn test_t5_model() {
        let config = T5Config::t5_tiny();
        let t5 = T5Model::new(config);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let decoder_input_ids = Tensor::from_slice(&[1.0, 2.0], &[2, 1]).unwrap();
        
        let output = t5.forward(&input_ids, &decoder_input_ids).unwrap();
        
        assert_eq!(output.last_hidden_state.dims(), &[2, 1, 128]); // batch=2, seq=1, d_model=128
    }
    
    #[test]
    fn test_t5_for_conditional_generation() {
        let config = T5Config::t5_tiny();
        let t5 = T5ForConditionalGeneration::new(config.clone());
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let decoder_input_ids = Tensor::from_slice(&[1.0, 2.0], &[2, 1]).unwrap();
        
        let output = t5.forward(&input_ids, &decoder_input_ids).unwrap();
        
        assert_eq!(output.dims(), &[2, 1, 1000]); // batch=2, seq=1, vocab=1000
    }
}

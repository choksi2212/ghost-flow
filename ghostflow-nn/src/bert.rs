//! BERT (Bidirectional Encoder Representations from Transformers)
//!
//! Implements BERT as described in "BERT: Pre-training of Deep Bidirectional Transformers"
//! - Token embeddings
//! - Segment embeddings
//! - Position embeddings
//! - Transformer encoder layers
//! - Masked language modeling head
//! - Next sentence prediction head

use ghostflow_core::Tensor;
use crate::transformer::{TransformerEncoder, TransformerEncoderLayer};
use crate::linear::Linear;
use crate::norm::LayerNorm;
use crate::activation::GELU;
use crate::Module;

/// BERT configuration
#[derive(Debug, Clone)]
pub struct BertConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size (embedding dimension)
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Intermediate size in feed-forward network
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Number of token types (segments)
    pub type_vocab_size: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
}

impl Default for BertConfig {
    fn default() -> Self {
        BertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            dropout: 0.1,
            layer_norm_eps: 1e-12,
        }
    }
}

impl BertConfig {
    /// BERT-Base configuration
    pub fn bert_base() -> Self {
        Self::default()
    }
    
    /// BERT-Large configuration
    pub fn bert_large() -> Self {
        BertConfig {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
            ..Default::default()
        }
    }
    
    /// BERT-Tiny configuration (for testing)
    pub fn bert_tiny() -> Self {
        BertConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 2,
            intermediate_size: 512,
            max_position_embeddings: 128,
            ..Default::default()
        }
    }
}

/// BERT embeddings layer
pub struct BertEmbeddings {
    /// Token embeddings
    token_embeddings: Tensor,
    /// Position embeddings
    position_embeddings: Tensor,
    /// Token type (segment) embeddings
    token_type_embeddings: Tensor,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Configuration
    config: BertConfig,
}

impl BertEmbeddings {
    /// Create new BERT embeddings
    pub fn new(config: BertConfig) -> Self {
        // Initialize embeddings
        let token_embeddings = Tensor::randn(&[config.vocab_size, config.hidden_size]);
        let position_embeddings = Tensor::randn(&[config.max_position_embeddings, config.hidden_size]);
        let token_type_embeddings = Tensor::randn(&[config.type_vocab_size, config.hidden_size]);
        
        let layer_norm = LayerNorm::new(&[config.hidden_size]);
        
        BertEmbeddings {
            token_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            config,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor, String> {
        let dims = input_ids.dims();
        if dims.len() != 2 {
            return Err(format!("Expected 2D input_ids, got {}D", dims.len()));
        }
        
        let batch_size = dims[0];
        let seq_length = dims[1];
        
        // Get token embeddings
        let token_embeds = self.get_token_embeddings(input_ids)?;
        
        // Get position embeddings
        let position_embeds = self.get_position_embeddings(seq_length)?;
        
        // Get token type embeddings
        let token_type_embeds = if let Some(tt_ids) = token_type_ids {
            self.get_token_type_embeddings(tt_ids)?
        } else {
            // Default to all zeros (first segment)
            Tensor::zeros(&[batch_size, seq_length, self.config.hidden_size])
        };
        
        // Sum all embeddings
        let embeddings = self.sum_embeddings(&token_embeds, &position_embeds, &token_type_embeds)?;
        
        // Layer normalization
        Ok(self.layer_norm.forward(&embeddings))
    }
    
    /// Get token embeddings by lookup
    fn get_token_embeddings(&self, input_ids: &Tensor) -> Result<Tensor, String> {
        let ids_data = input_ids.data_f32();
        let embed_data = self.token_embeddings.data_f32();
        
        let dims = input_ids.dims();
        let batch_size = dims[0];
        let seq_length = dims[1];
        let hidden_size = self.config.hidden_size;
        
        let mut result = Vec::with_capacity(batch_size * seq_length * hidden_size);
        
        for &id in ids_data.iter() {
            let idx = id as usize;
            if idx >= self.config.vocab_size {
                return Err(format!("Token ID {} out of vocabulary range", idx));
            }
            
            let start = idx * hidden_size;
            let end = start + hidden_size;
            result.extend_from_slice(&embed_data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_length, hidden_size])
            .map_err(|e| format!("Failed to create token embeddings: {:?}", e))
    }
    
    /// Get position embeddings
    fn get_position_embeddings(&self, seq_length: usize) -> Result<Tensor, String> {
        let embed_data = self.position_embeddings.data_f32();
        let hidden_size = self.config.hidden_size;
        
        if seq_length > self.config.max_position_embeddings {
            return Err(format!("Sequence length {} exceeds maximum {}", 
                             seq_length, self.config.max_position_embeddings));
        }
        
        let result = embed_data[..seq_length * hidden_size].to_vec();
        
        Tensor::from_slice(&result, &[seq_length, hidden_size])
            .map_err(|e| format!("Failed to create position embeddings: {:?}", e))
    }
    
    /// Get token type embeddings
    fn get_token_type_embeddings(&self, token_type_ids: &Tensor) -> Result<Tensor, String> {
        let ids_data = token_type_ids.data_f32();
        let embed_data = self.token_type_embeddings.data_f32();
        
        let dims = token_type_ids.dims();
        let batch_size = dims[0];
        let seq_length = dims[1];
        let hidden_size = self.config.hidden_size;
        
        let mut result = Vec::with_capacity(batch_size * seq_length * hidden_size);
        
        for &id in ids_data.iter() {
            let idx = id as usize;
            if idx >= self.config.type_vocab_size {
                return Err(format!("Token type ID {} out of range", idx));
            }
            
            let start = idx * hidden_size;
            let end = start + hidden_size;
            result.extend_from_slice(&embed_data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_length, hidden_size])
            .map_err(|e| format!("Failed to create token type embeddings: {:?}", e))
    }
    
    /// Sum all embeddings
    fn sum_embeddings(&self, token: &Tensor, position: &Tensor, token_type: &Tensor) -> Result<Tensor, String> {
        let token_data = token.data_f32();
        let pos_data = position.data_f32();
        let tt_data = token_type.data_f32();
        
        let dims = token.dims();
        let batch_size = dims[0];
        let seq_length = dims[1];
        let hidden_size = dims[2];
        
        let mut result = Vec::with_capacity(token_data.len());
        
        for b in 0..batch_size {
            for s in 0..seq_length {
                for h in 0..hidden_size {
                    let token_idx = b * seq_length * hidden_size + s * hidden_size + h;
                    let pos_idx = s * hidden_size + h;
                    
                    result.push(token_data[token_idx] + pos_data[pos_idx] + tt_data[token_idx]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_length, hidden_size])
            .map_err(|e| format!("Failed to sum embeddings: {:?}", e))
    }
}

/// BERT pooler (for classification tasks)
pub struct BertPooler {
    dense: Linear,
    activation: std::marker::PhantomData<GELU>,
}

impl BertPooler {
    /// Create new BERT pooler
    pub fn new(hidden_size: usize) -> Self {
        BertPooler {
            dense: Linear::new(hidden_size, hidden_size),
            activation: std::marker::PhantomData,
        }
    }
    
    /// Forward pass - pool first token (CLS)
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, String> {
        // Extract first token: [batch, hidden_size]
        let first_token = self.extract_first_token(hidden_states)?;
        
        // Dense layer
        let pooled = self.dense.forward(&first_token);
        
        // Tanh activation (BERT uses tanh for pooler)
        self.apply_tanh(&pooled)
    }
    
    /// Extract first token from sequence
    fn extract_first_token(&self, hidden_states: &Tensor) -> Result<Tensor, String> {
        let data = hidden_states.data_f32();
        let dims = hidden_states.dims();
        
        if dims.len() != 3 {
            return Err(format!("Expected 3D hidden states, got {}D", dims.len()));
        }
        
        let batch_size = dims[0];
        let hidden_size = dims[2];
        
        let mut result = Vec::with_capacity(batch_size * hidden_size);
        
        for b in 0..batch_size {
            let start = b * dims[1] * hidden_size;
            let end = start + hidden_size;
            result.extend_from_slice(&data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, hidden_size])
            .map_err(|e| format!("Failed to extract first token: {:?}", e))
    }
    
    /// Apply tanh activation
    fn apply_tanh(&self, x: &Tensor) -> Result<Tensor, String> {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter().map(|&v| v.tanh()).collect();
        
        Tensor::from_slice(&result, x.dims())
            .map_err(|e| format!("Failed to apply tanh: {:?}", e))
    }
}

/// BERT model
pub struct BertModel {
    /// Configuration
    config: BertConfig,
    /// Embeddings layer
    embeddings: BertEmbeddings,
    /// Transformer encoder
    encoder: TransformerEncoder,
    /// Pooler (optional, for classification)
    pooler: Option<BertPooler>,
}

impl BertModel {
    /// Create new BERT model
    pub fn new(config: BertConfig, with_pooler: bool) -> Self {
        let embeddings = BertEmbeddings::new(config.clone());
        
        let encoder = TransformerEncoder::new(
            config.hidden_size,
            config.num_heads,
            config.intermediate_size,
            config.num_layers,
            config.dropout,
        );
        
        let pooler = if with_pooler {
            Some(BertPooler::new(config.hidden_size))
        } else {
            None
        };
        
        BertModel {
            config,
            embeddings,
            encoder,
            pooler,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>, 
                   attention_mask: Option<&Tensor>) -> Result<BertOutput, String> {
        // Get embeddings
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        
        // Encoder
        let sequence_output = self.encoder.forward(&embedding_output);
        
        // Pooler (if present)
        let pooled_output = if let Some(ref pooler) = self.pooler {
            Some(pooler.forward(&sequence_output)?)
        } else {
            None
        };
        
        Ok(BertOutput {
            last_hidden_state: sequence_output,
            pooler_output: pooled_output,
        })
    }
}

/// BERT output
pub struct BertOutput {
    /// Last hidden state: [batch, seq_len, hidden_size]
    pub last_hidden_state: Tensor,
    /// Pooled output (CLS token): [batch, hidden_size]
    pub pooler_output: Option<Tensor>,
}

/// BERT for Masked Language Modeling
pub struct BertForMaskedLM {
    bert: BertModel,
    mlm_head: Linear,
}

impl BertForMaskedLM {
    /// Create new BERT for MLM
    pub fn new(config: BertConfig) -> Self {
        let bert = BertModel::new(config.clone(), false);
        let mlm_head = Linear::new(config.hidden_size, config.vocab_size);
        
        BertForMaskedLM {
            bert,
            mlm_head,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor, String> {
        let output = self.bert.forward(input_ids, token_type_ids, None)?;
        Ok(self.mlm_head.forward(&output.last_hidden_state))
    }
}

/// BERT for Sequence Classification
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: Linear,
    num_labels: usize,
}

impl BertForSequenceClassification {
    /// Create new BERT for classification
    pub fn new(config: BertConfig, num_labels: usize) -> Self {
        let bert = BertModel::new(config.clone(), true);
        let classifier = Linear::new(config.hidden_size, num_labels);
        
        BertForSequenceClassification {
            bert,
            classifier,
            num_labels,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor, String> {
        let output = self.bert.forward(input_ids, token_type_ids, None)?;
        
        let pooled = output.pooler_output
            .ok_or_else(|| "Pooler output not available".to_string())?;
        
        Ok(self.classifier.forward(&pooled))
    }
}

/// BERT for Token Classification (NER, POS tagging)
pub struct BertForTokenClassification {
    bert: BertModel,
    classifier: Linear,
    num_labels: usize,
}

impl BertForTokenClassification {
    /// Create new BERT for token classification
    pub fn new(config: BertConfig, num_labels: usize) -> Self {
        let bert = BertModel::new(config.clone(), false);
        let classifier = Linear::new(config.hidden_size, num_labels);
        
        BertForTokenClassification {
            bert,
            classifier,
            num_labels,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor, String> {
        let output = self.bert.forward(input_ids, token_type_ids, None)?;
        Ok(self.classifier.forward(&output.last_hidden_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bert_config() {
        let config = BertConfig::bert_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        
        let config = BertConfig::bert_large();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_layers, 24);
    }
    
    #[test]
    fn test_bert_embeddings() {
        let config = BertConfig::bert_tiny();
        let embeddings = BertEmbeddings::new(config);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = embeddings.forward(&input_ids, None).unwrap();
        
        assert_eq!(output.dims(), &[2, 2, 128]); // batch=2, seq=2, hidden=128
    }
    
    #[test]
    fn test_bert_model() {
        let config = BertConfig::bert_tiny();
        let bert = BertModel::new(config, true);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = bert.forward(&input_ids, None, None).unwrap();
        
        assert_eq!(output.last_hidden_state.dims(), &[2, 2, 128]);
        assert!(output.pooler_output.is_some());
        assert_eq!(output.pooler_output.unwrap().dims(), &[2, 128]);
    }
    
    #[test]
    fn test_bert_for_classification() {
        let config = BertConfig::bert_tiny();
        let bert = BertForSequenceClassification::new(config, 2);
        
        let input_ids = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = bert.forward(&input_ids, None).unwrap();
        
        assert_eq!(output.dims(), &[2, 2]); // batch=2, num_labels=2
    }
}

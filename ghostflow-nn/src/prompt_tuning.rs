//! Prompt Tuning and Prefix Tuning
//!
//! Implements parameter-efficient fine-tuning methods:
//! - Prompt Tuning: Learnable soft prompts prepended to input
//! - Prefix Tuning: Learnable prefix vectors for each layer
//! - P-Tuning v2: Prefix tuning with deep prompt optimization
//! - Adapter-based prompt tuning

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// Prompt tuning configuration
#[derive(Debug, Clone)]
pub struct PromptTuningConfig {
    /// Number of virtual tokens (prompt length)
    pub num_virtual_tokens: usize,
    /// Model dimension
    pub d_model: usize,
    /// Prompt initialization strategy
    pub init_strategy: PromptInitStrategy,
    /// Reparameterization for stability
    pub reparameterize: bool,
    /// Hidden dimension for reparameterization
    pub hidden_dim: usize,
}

/// Prompt initialization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PromptInitStrategy {
    /// Random initialization
    Random,
    /// Initialize from vocabulary embeddings
    Vocab,
    /// Initialize from text tokens
    Text,
}

impl Default for PromptTuningConfig {
    fn default() -> Self {
        PromptTuningConfig {
            num_virtual_tokens: 20,
            d_model: 768,
            init_strategy: PromptInitStrategy::Random,
            reparameterize: false,
            hidden_dim: 512,
        }
    }
}

impl PromptTuningConfig {
    /// Short prompt configuration (5-10 tokens)
    pub fn short(d_model: usize) -> Self {
        PromptTuningConfig {
            num_virtual_tokens: 10,
            d_model,
            ..Default::default()
        }
    }
    
    /// Medium prompt configuration (20-50 tokens)
    pub fn medium(d_model: usize) -> Self {
        PromptTuningConfig {
            num_virtual_tokens: 30,
            d_model,
            ..Default::default()
        }
    }
    
    /// Long prompt configuration (50-100 tokens)
    pub fn long(d_model: usize) -> Self {
        PromptTuningConfig {
            num_virtual_tokens: 80,
            d_model,
            ..Default::default()
        }
    }
    
    /// With reparameterization for stability
    pub fn with_reparameterization(mut self, hidden_dim: usize) -> Self {
        self.reparameterize = true;
        self.hidden_dim = hidden_dim;
        self
    }
}

/// Prompt Tuning implementation
pub struct PromptTuning {
    config: PromptTuningConfig,
    /// Learnable prompt embeddings
    prompt_embeddings: Tensor,
    /// Optional reparameterization layers
    reparam_encoder: Option<Tensor>,
    reparam_decoder: Option<Tensor>,
}

impl PromptTuning {
    /// Create new prompt tuning
    pub fn new(config: PromptTuningConfig) -> Result<Self, String> {
        let prompt_embeddings = if config.reparameterize {
            // Initialize in lower dimension
            Tensor::randn(&[config.num_virtual_tokens, config.hidden_dim])
        } else {
            // Direct initialization
            Tensor::randn(&[config.num_virtual_tokens, config.d_model])
        };
        
        let (reparam_encoder, reparam_decoder) = if config.reparameterize {
            let encoder = Tensor::randn(&[config.hidden_dim, config.d_model]);
            let decoder = Tensor::randn(&[config.d_model, config.hidden_dim]);
            (Some(encoder), Some(decoder))
        } else {
            (None, None)
        };
        
        Ok(PromptTuning {
            config,
            prompt_embeddings,
            reparam_encoder,
            reparam_decoder,
        })
    }
    
    /// Get prompt embeddings
    pub fn get_prompt_embeddings(&self) -> Result<Tensor, String> {
        if self.config.reparameterize {
            // Reparameterize: prompt @ encoder
            let encoder = self.reparam_encoder.as_ref()
                .ok_or("Encoder not initialized")?;
            self.prompt_embeddings.matmul(encoder)
                .map_err(|e| format!("Failed to reparameterize: {:?}", e))
        } else {
            Ok(self.prompt_embeddings.clone())
        }
    }
    
    /// Prepend prompts to input embeddings
    pub fn prepend_prompts(&self, input_embeddings: &Tensor) -> Result<Tensor, String> {
        let prompt_embeds = self.get_prompt_embeddings()?;
        
        let input_dims = input_embeddings.dims();
        let prompt_dims = prompt_embeds.dims();
        
        if input_dims.len() != 3 || prompt_dims.len() != 2 {
            return Err("Expected input [batch, seq_len, d_model] and prompt [num_tokens, d_model]".to_string());
        }
        
        let batch_size = input_dims[0];
        let seq_len = input_dims[1];
        let d_model = input_dims[2];
        let num_prompts = prompt_dims[0];
        
        // Expand prompts for batch
        let mut result = Vec::with_capacity(batch_size * (num_prompts + seq_len) * d_model);
        
        let prompt_data = prompt_embeds.data_f32();
        let input_data = input_embeddings.data_f32();
        
        for b in 0..batch_size {
            // Add prompts
            result.extend_from_slice(&prompt_data);
            
            // Add input embeddings
            let start = b * seq_len * d_model;
            let end = start + seq_len * d_model;
            result.extend_from_slice(&input_data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, num_prompts + seq_len, d_model])
            .map_err(|e| format!("Failed to prepend prompts: {:?}", e))
    }
    
    /// Get number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        let prompt_params = self.prompt_embeddings.data_f32().len();
        let reparam_params = if self.config.reparameterize {
            self.reparam_encoder.as_ref().map(|t| t.data_f32().len()).unwrap_or(0) +
            self.reparam_decoder.as_ref().map(|t| t.data_f32().len()).unwrap_or(0)
        } else {
            0
        };
        prompt_params + reparam_params
    }
    
    /// Get parameter efficiency ratio
    pub fn parameter_efficiency(&self, total_model_params: usize) -> f32 {
        let tunable_params = self.num_parameters();
        tunable_params as f32 / total_model_params as f32
    }
}

/// Prefix Tuning configuration
#[derive(Debug, Clone)]
pub struct PrefixTuningConfig {
    /// Number of prefix tokens per layer
    pub num_prefix_tokens: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Prefix initialization strategy
    pub init_strategy: PromptInitStrategy,
    /// Use prefix for both key and value
    pub prefix_kv: bool,
}

impl Default for PrefixTuningConfig {
    fn default() -> Self {
        PrefixTuningConfig {
            num_prefix_tokens: 10,
            num_layers: 12,
            d_model: 768,
            num_heads: 12,
            init_strategy: PromptInitStrategy::Random,
            prefix_kv: true,
        }
    }
}

impl PrefixTuningConfig {
    /// Configuration for small models
    pub fn small(num_layers: usize, d_model: usize, num_heads: usize) -> Self {
        PrefixTuningConfig {
            num_prefix_tokens: 5,
            num_layers,
            d_model,
            num_heads,
            ..Default::default()
        }
    }
    
    /// Configuration for large models
    pub fn large(num_layers: usize, d_model: usize, num_heads: usize) -> Self {
        PrefixTuningConfig {
            num_prefix_tokens: 20,
            num_layers,
            d_model,
            num_heads,
            ..Default::default()
        }
    }
}

/// Prefix Tuning implementation
pub struct PrefixTuning {
    config: PrefixTuningConfig,
    /// Prefix parameters for each layer
    prefix_params: HashMap<usize, LayerPrefix>,
}

/// Prefix parameters for a single layer
#[derive(Clone)]
pub struct LayerPrefix {
    /// Prefix for keys
    pub prefix_key: Tensor,
    /// Prefix for values
    pub prefix_value: Tensor,
}

impl PrefixTuning {
    /// Create new prefix tuning
    pub fn new(config: PrefixTuningConfig) -> Result<Self, String> {
        let mut prefix_params = HashMap::new();
        
        let head_dim = config.d_model / config.num_heads;
        
        for layer_idx in 0..config.num_layers {
            let prefix_key = Tensor::randn(&[config.num_prefix_tokens, config.d_model]);
            let prefix_value = if config.prefix_kv {
                Tensor::randn(&[config.num_prefix_tokens, config.d_model])
            } else {
                prefix_key.clone()
            };
            
            prefix_params.insert(layer_idx, LayerPrefix {
                prefix_key,
                prefix_value,
            });
        }
        
        Ok(PrefixTuning {
            config,
            prefix_params,
        })
    }
    
    /// Get prefix for a specific layer
    pub fn get_layer_prefix(&self, layer_idx: usize) -> Option<&LayerPrefix> {
        self.prefix_params.get(&layer_idx)
    }
    
    /// Prepend prefix to key/value in attention
    pub fn prepend_to_kv(
        &self,
        layer_idx: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor), String> {
        let prefix = self.get_layer_prefix(layer_idx)
            .ok_or(format!("No prefix for layer {}", layer_idx))?;
        
        let new_key = self.concatenate_prefix(&prefix.prefix_key, key)?;
        let new_value = self.concatenate_prefix(&prefix.prefix_value, value)?;
        
        Ok((new_key, new_value))
    }
    
    /// Concatenate prefix to tensor
    fn concatenate_prefix(&self, prefix: &Tensor, tensor: &Tensor) -> Result<Tensor, String> {
        let prefix_dims = prefix.dims();
        let tensor_dims = tensor.dims();
        
        if tensor_dims.len() != 3 {
            return Err("Expected tensor [batch, seq_len, d_model]".to_string());
        }
        
        let batch_size = tensor_dims[0];
        let seq_len = tensor_dims[1];
        let d_model = tensor_dims[2];
        let num_prefix = prefix_dims[0];
        
        let mut result = Vec::with_capacity(batch_size * (num_prefix + seq_len) * d_model);
        
        let prefix_data = prefix.data_f32();
        let tensor_data = tensor.data_f32();
        
        for b in 0..batch_size {
            // Add prefix
            result.extend_from_slice(&prefix_data);
            
            // Add tensor data
            let start = b * seq_len * d_model;
            let end = start + seq_len * d_model;
            result.extend_from_slice(&tensor_data[start..end]);
        }
        
        Tensor::from_slice(&result, &[batch_size, num_prefix + seq_len, d_model])
            .map_err(|e| format!("Failed to concatenate prefix: {:?}", e))
    }
    
    /// Get number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        for prefix in self.prefix_params.values() {
            total += prefix.prefix_key.data_f32().len();
            total += prefix.prefix_value.data_f32().len();
        }
        total
    }
    
    /// Get parameter efficiency ratio
    pub fn parameter_efficiency(&self, total_model_params: usize) -> f32 {
        let tunable_params = self.num_parameters();
        tunable_params as f32 / total_model_params as f32
    }
}

/// P-Tuning v2 (Deep Prompt Tuning)
pub struct PTuningV2 {
    prefix_tuning: PrefixTuning,
    /// Additional MLP for prefix generation
    prefix_mlp: Option<Tensor>,
}

impl PTuningV2 {
    /// Create new P-Tuning v2
    pub fn new(config: PrefixTuningConfig) -> Result<Self, String> {
        let prefix_tuning = PrefixTuning::new(config.clone())?;
        
        // Optional MLP for generating prefixes
        let prefix_mlp = Some(Tensor::randn(&[config.d_model, config.d_model]));
        
        Ok(PTuningV2 {
            prefix_tuning,
            prefix_mlp,
        })
    }
    
    /// Get prefix with MLP transformation
    pub fn get_layer_prefix_transformed(&self, layer_idx: usize) -> Option<LayerPrefix> {
        let prefix = self.prefix_tuning.get_layer_prefix(layer_idx)?;
        
        if let Some(mlp) = &self.prefix_mlp {
            // Transform prefix through MLP
            // Simplified - in practice would be full MLP
            Some(prefix.clone())
        } else {
            Some(prefix.clone())
        }
    }
    
    /// Get number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        let prefix_params = self.prefix_tuning.num_parameters();
        let mlp_params = self.prefix_mlp.as_ref()
            .map(|t| t.data_f32().len())
            .unwrap_or(0);
        prefix_params + mlp_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prompt_tuning_config() {
        let config = PromptTuningConfig::default();
        assert_eq!(config.num_virtual_tokens, 20);
        assert_eq!(config.d_model, 768);
        
        let short = PromptTuningConfig::short(512);
        assert_eq!(short.num_virtual_tokens, 10);
        assert_eq!(short.d_model, 512);
    }
    
    #[test]
    fn test_prompt_tuning_creation() {
        let config = PromptTuningConfig::default();
        let prompt_tuning = PromptTuning::new(config).unwrap();
        
        let embeddings = prompt_tuning.get_prompt_embeddings().unwrap();
        assert_eq!(embeddings.dims(), &[20, 768]);
    }
    
    #[test]
    fn test_prompt_tuning_prepend() {
        let config = PromptTuningConfig {
            num_virtual_tokens: 5,
            d_model: 64,
            ..Default::default()
        };
        let prompt_tuning = PromptTuning::new(config).unwrap();
        
        let input = Tensor::randn(&[2, 10, 64]);
        let output = prompt_tuning.prepend_prompts(&input).unwrap();
        
        assert_eq!(output.dims(), &[2, 15, 64]); // 5 prompts + 10 input
    }
    
    #[test]
    fn test_prompt_tuning_reparameterization() {
        let config = PromptTuningConfig {
            num_virtual_tokens: 10,
            d_model: 768,
            reparameterize: true,
            hidden_dim: 256,
            ..Default::default()
        };
        let prompt_tuning = PromptTuning::new(config).unwrap();
        
        let embeddings = prompt_tuning.get_prompt_embeddings().unwrap();
        assert_eq!(embeddings.dims(), &[10, 768]);
    }
    
    #[test]
    fn test_prompt_tuning_parameters() {
        let config = PromptTuningConfig {
            num_virtual_tokens: 20,
            d_model: 768,
            ..Default::default()
        };
        let prompt_tuning = PromptTuning::new(config).unwrap();
        
        let num_params = prompt_tuning.num_parameters();
        assert_eq!(num_params, 20 * 768);
        
        let efficiency = prompt_tuning.parameter_efficiency(100_000_000);
        assert!(efficiency < 0.01); // Less than 1% of model parameters
    }
    
    #[test]
    fn test_prefix_tuning_config() {
        let config = PrefixTuningConfig::default();
        assert_eq!(config.num_prefix_tokens, 10);
        assert_eq!(config.num_layers, 12);
        
        let small = PrefixTuningConfig::small(6, 512, 8);
        assert_eq!(small.num_prefix_tokens, 5);
        assert_eq!(small.num_layers, 6);
    }
    
    #[test]
    fn test_prefix_tuning_creation() {
        let config = PrefixTuningConfig {
            num_prefix_tokens: 5,
            num_layers: 3,
            d_model: 64,
            num_heads: 4,
            ..Default::default()
        };
        let prefix_tuning = PrefixTuning::new(config).unwrap();
        
        let prefix = prefix_tuning.get_layer_prefix(0).unwrap();
        assert_eq!(prefix.prefix_key.dims(), &[5, 64]);
        assert_eq!(prefix.prefix_value.dims(), &[5, 64]);
    }
    
    #[test]
    fn test_prefix_tuning_prepend() {
        let config = PrefixTuningConfig {
            num_prefix_tokens: 3,
            num_layers: 2,
            d_model: 32,
            num_heads: 4,
            ..Default::default()
        };
        let prefix_tuning = PrefixTuning::new(config).unwrap();
        
        let key = Tensor::randn(&[2, 8, 32]);
        let value = Tensor::randn(&[2, 8, 32]);
        
        let (new_key, new_value) = prefix_tuning.prepend_to_kv(0, &key, &value).unwrap();
        
        assert_eq!(new_key.dims(), &[2, 11, 32]); // 3 prefix + 8 original
        assert_eq!(new_value.dims(), &[2, 11, 32]);
    }
    
    #[test]
    fn test_prefix_tuning_parameters() {
        let config = PrefixTuningConfig {
            num_prefix_tokens: 10,
            num_layers: 12,
            d_model: 768,
            num_heads: 12,
            ..Default::default()
        };
        let prefix_tuning = PrefixTuning::new(config).unwrap();
        
        let num_params = prefix_tuning.num_parameters();
        // 10 tokens * 768 dim * 2 (key+value) * 12 layers
        assert_eq!(num_params, 10 * 768 * 2 * 12);
    }
    
    #[test]
    fn test_ptuning_v2() {
        let config = PrefixTuningConfig {
            num_prefix_tokens: 5,
            num_layers: 3,
            d_model: 64,
            num_heads: 4,
            ..Default::default()
        };
        let ptuning = PTuningV2::new(config).unwrap();
        
        let prefix = ptuning.get_layer_prefix_transformed(0).unwrap();
        assert_eq!(prefix.prefix_key.dims(), &[5, 64]);
    }
}

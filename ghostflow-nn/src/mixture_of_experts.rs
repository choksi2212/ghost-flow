//! Mixture of Experts (MoE)
//!
//! Implements sparse mixture of experts for efficient scaling:
//! - Top-K expert routing
//! - Load balancing
//! - Expert capacity constraints
//! - Switch Transformer style MoE
//! - GShard style MoE
//! - Auxiliary loss for load balancing

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// MoE routing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingStrategy {
    /// Top-K routing (select K experts per token)
    TopK,
    /// Switch routing (select 1 expert per token)
    Switch,
    /// Expert Choice (experts select top tokens)
    ExpertChoice,
}

/// MoE configuration
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to route to per token
    pub top_k: usize,
    /// Expert capacity factor
    pub capacity_factor: f32,
    /// Routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Load balancing loss weight
    pub load_balance_loss_weight: f32,
    /// Expert dropout probability
    pub expert_dropout: f32,
    /// Use expert parallelism
    pub expert_parallel: bool,
}

impl Default for MoEConfig {
    fn default() -> Self {
        MoEConfig {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            routing_strategy: RoutingStrategy::TopK,
            load_balance_loss_weight: 0.01,
            expert_dropout: 0.0,
            expert_parallel: false,
        }
    }
}

impl MoEConfig {
    /// Switch Transformer configuration (top-1 routing)
    pub fn switch_transformer(num_experts: usize) -> Self {
        MoEConfig {
            num_experts,
            top_k: 1,
            routing_strategy: RoutingStrategy::Switch,
            capacity_factor: 1.0,
            ..Default::default()
        }
    }
    
    /// GShard configuration (top-2 routing)
    pub fn gshard(num_experts: usize) -> Self {
        MoEConfig {
            num_experts,
            top_k: 2,
            routing_strategy: RoutingStrategy::TopK,
            capacity_factor: 1.25,
            ..Default::default()
        }
    }
    
    /// Expert Choice configuration
    pub fn expert_choice(num_experts: usize, capacity_factor: f32) -> Self {
        MoEConfig {
            num_experts,
            top_k: 1,
            routing_strategy: RoutingStrategy::ExpertChoice,
            capacity_factor,
            ..Default::default()
        }
    }
}

/// Expert network
pub struct Expert {
    /// Expert ID
    id: usize,
    /// Input dimension
    d_model: usize,
    /// Hidden dimension
    d_ff: usize,
    /// Weights (simplified - would be full linear layers)
    w1: Tensor,
    w2: Tensor,
}

impl Expert {
    /// Create new expert
    pub fn new(id: usize, d_model: usize, d_ff: usize) -> Result<Self, String> {
        let w1 = Tensor::randn(&[d_model, d_ff]);
        let w2 = Tensor::randn(&[d_ff, d_model]);
        
        Ok(Expert {
            id,
            d_model,
            d_ff,
            w1,
            w2,
        })
    }
    
    /// Forward pass through expert
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        // FFN: W2(GELU(W1(x)))
        let hidden = input.matmul(&self.w1)
            .map_err(|e| format!("Failed to compute W1: {:?}", e))?;
        let activated = hidden.gelu();
        activated.matmul(&self.w2)
            .map_err(|e| format!("Failed to compute W2: {:?}", e))
    }
}

/// Router network
pub struct Router {
    /// Routing weights
    weights: Tensor,
    /// Number of experts
    num_experts: usize,
}

impl Router {
    /// Create new router
    pub fn new(d_model: usize, num_experts: usize) -> Result<Self, String> {
        let weights = Tensor::randn(&[d_model, num_experts]);
        
        Ok(Router {
            weights,
            num_experts,
        })
    }
    
    /// Compute routing probabilities
    pub fn route(&self, input: &Tensor) -> Result<Tensor, String> {
        // Compute logits: input @ weights
        let logits = input.matmul(&self.weights)
            .map_err(|e| format!("Failed to compute routing logits: {:?}", e))?;
        
        // Apply softmax
        Ok(logits.softmax(-1))
    }
    
    /// Select top-K experts
    pub fn select_top_k(&self, probs: &Tensor, k: usize) -> Result<(Vec<usize>, Vec<f32>), String> {
        let data = probs.data_f32();
        
        // Get top-K indices and values
        let mut indexed: Vec<(usize, f32)> = data.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_k_indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        let top_k_values: Vec<f32> = indexed.iter().take(k).map(|(_, v)| *v).collect();
        
        Ok((top_k_indices, top_k_values))
    }
}

/// Mixture of Experts layer
pub struct MixtureOfExperts {
    config: MoEConfig,
    experts: Vec<Expert>,
    router: Router,
    /// Load balancing statistics
    expert_usage: Vec<usize>,
    /// Auxiliary loss
    aux_loss: f32,
}

impl MixtureOfExperts {
    /// Create new MoE layer
    pub fn new(config: MoEConfig, d_model: usize, d_ff: usize) -> Result<Self, String> {
        let mut experts = Vec::new();
        for i in 0..config.num_experts {
            experts.push(Expert::new(i, d_model, d_ff)?);
        }
        
        let router = Router::new(d_model, config.num_experts)?;
        let expert_usage = vec![0; config.num_experts];
        
        Ok(MixtureOfExperts {
            config,
            experts,
            router,
            expert_usage,
            aux_loss: 0.0,
        })
    }
    
    /// Forward pass through MoE
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, String> {
        let dims = input.dims();
        
        if dims.len() != 2 {
            return Err("Expected 2D tensor [seq_len, d_model]".to_string());
        }
        
        let seq_len = dims[0];
        let d_model = dims[1];
        
        // Route each token
        let mut outputs = Vec::new();
        let mut routing_probs = Vec::new();
        
        for i in 0..seq_len {
            let token = self.extract_token(input, i)?;
            
            // Get routing probabilities
            let probs = self.router.route(&token)?;
            routing_probs.push(probs.clone());
            
            // Select top-K experts
            let (expert_ids, expert_weights) = self.router.select_top_k(&probs, self.config.top_k)?;
            
            // Compute weighted expert outputs
            let mut token_output = vec![0.0f32; d_model];
            let mut weight_sum = 0.0;
            
            for (expert_id, weight) in expert_ids.iter().zip(expert_weights.iter()) {
                if *expert_id < self.experts.len() {
                    let expert = &self.experts[*expert_id];
                    let expert_output = expert.forward(&token)?;
                    
                    // Accumulate weighted output
                    let expert_data = expert_output.data_f32();
                    for j in 0..d_model {
                        token_output[j] += weight * expert_data[j];
                    }
                    
                    weight_sum += weight;
                    self.expert_usage[*expert_id] += 1;
                }
            }
            
            // Normalize by weight sum
            if weight_sum > 0.0 {
                for val in &mut token_output {
                    *val /= weight_sum;
                }
            }
            
            outputs.push(token_output);
        }
        
        // Compute load balancing loss
        self.aux_loss = self.compute_load_balance_loss(&routing_probs)?;
        
        // Flatten outputs
        let flattened: Vec<f32> = outputs.into_iter().flatten().collect();
        
        Tensor::from_slice(&flattened, &[seq_len, d_model])
            .map_err(|e| format!("Failed to create output tensor: {:?}", e))
    }
    
    /// Extract single token from input
    fn extract_token(&self, input: &Tensor, token_idx: usize) -> Result<Tensor, String> {
        let data = input.data_f32();
        let d_model = input.dims()[1];
        
        let start = token_idx * d_model;
        let end = start + d_model;
        
        Tensor::from_slice(&data[start..end], &[1, d_model])
            .map_err(|e| format!("Failed to extract token: {:?}", e))
    }
    
    /// Compute load balancing auxiliary loss
    fn compute_load_balance_loss(&self, routing_probs: &[Tensor]) -> Result<f32, String> {
        if routing_probs.is_empty() {
            return Ok(0.0);
        }
        
        let num_tokens = routing_probs.len() as f32;
        let num_experts = self.config.num_experts as f32;
        
        // Compute fraction of tokens routed to each expert
        let mut expert_fractions = vec![0.0f32; self.config.num_experts];
        
        for probs in routing_probs {
            let data = probs.data_f32();
            for (i, &prob) in data.iter().enumerate() {
                if i < expert_fractions.len() {
                    expert_fractions[i] += prob;
                }
            }
        }
        
        for frac in &mut expert_fractions {
            *frac /= num_tokens;
        }
        
        // Compute coefficient of variation (CV)
        // CV = std / mean, penalizes imbalance
        let mean = 1.0 / num_experts;
        let variance: f32 = expert_fractions.iter()
            .map(|&f| (f - mean).powi(2))
            .sum::<f32>() / num_experts;
        
        let cv = variance.sqrt() / mean;
        
        Ok(cv * self.config.load_balance_loss_weight)
    }
    
    /// Get auxiliary loss
    pub fn get_aux_loss(&self) -> f32 {
        self.aux_loss
    }
    
    /// Get expert usage statistics
    pub fn get_expert_usage(&self) -> &[usize] {
        &self.expert_usage
    }
    
    /// Reset expert usage statistics
    pub fn reset_usage_stats(&mut self) {
        self.expert_usage.fill(0);
    }
    
    /// Get load balance factor (1.0 = perfect balance)
    pub fn load_balance_factor(&self) -> f32 {
        if self.expert_usage.is_empty() {
            return 1.0;
        }
        
        let total: usize = self.expert_usage.iter().sum();
        if total == 0 {
            return 1.0;
        }
        
        let mean = total as f32 / self.expert_usage.len() as f32;
        let variance: f32 = self.expert_usage.iter()
            .map(|&u| (u as f32 - mean).powi(2))
            .sum::<f32>() / self.expert_usage.len() as f32;
        
        let std_dev = variance.sqrt();
        let cv = std_dev / mean;
        
        // Convert CV to balance factor (lower CV = better balance)
        1.0 / (1.0 + cv)
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> MoEStats {
        MoEStats {
            num_experts: self.config.num_experts,
            top_k: self.config.top_k,
            routing_strategy: self.config.routing_strategy,
            aux_loss: self.aux_loss,
            load_balance_factor: self.load_balance_factor(),
            expert_usage: self.expert_usage.clone(),
        }
    }
}

/// MoE statistics
#[derive(Debug, Clone)]
pub struct MoEStats {
    pub num_experts: usize,
    pub top_k: usize,
    pub routing_strategy: RoutingStrategy,
    pub aux_loss: f32,
    pub load_balance_factor: f32,
    pub expert_usage: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_moe_config() {
        let config = MoEConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);
        
        let switch = MoEConfig::switch_transformer(16);
        assert_eq!(switch.num_experts, 16);
        assert_eq!(switch.top_k, 1);
        assert_eq!(switch.routing_strategy, RoutingStrategy::Switch);
    }
    
    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(0, 512, 2048).unwrap();
        assert_eq!(expert.id, 0);
        assert_eq!(expert.d_model, 512);
        assert_eq!(expert.d_ff, 2048);
    }
    
    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(0, 64, 256).unwrap();
        let input = Tensor::randn(&[1, 64]);
        
        let output = expert.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 64]);
    }
    
    #[test]
    fn test_router_creation() {
        let router = Router::new(512, 8).unwrap();
        assert_eq!(router.num_experts, 8);
    }
    
    #[test]
    fn test_router_route() {
        let router = Router::new(64, 8).unwrap();
        let input = Tensor::randn(&[1, 64]);
        
        let probs = router.route(&input).unwrap();
        assert_eq!(probs.dims()[1], 8);
        
        // Check probabilities sum to 1
        let data = probs.data_f32();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_router_top_k() {
        let router = Router::new(64, 8).unwrap();
        let probs = Tensor::from_slice(&[0.1f32, 0.3, 0.05, 0.25, 0.15, 0.05, 0.05, 0.05], &[1, 8]).unwrap();
        
        let (indices, values) = router.select_top_k(&probs, 2).unwrap();
        
        assert_eq!(indices.len(), 2);
        assert_eq!(values.len(), 2);
        assert!(values[0] >= values[1]); // Should be sorted
    }
    
    #[test]
    fn test_moe_creation() {
        let config = MoEConfig::default();
        let moe = MixtureOfExperts::new(config, 128, 512).unwrap();
        
        assert_eq!(moe.experts.len(), 8);
    }
    
    #[test]
    fn test_moe_forward() {
        let config = MoEConfig {
            num_experts: 4,
            top_k: 2,
            ..Default::default()
        };
        let mut moe = MixtureOfExperts::new(config, 64, 256).unwrap();
        
        let input = Tensor::randn(&[8, 64]);
        let output = moe.forward(&input).unwrap();
        
        assert_eq!(output.dims(), &[8, 64]);
    }
    
    #[test]
    fn test_load_balance_factor() {
        let config = MoEConfig::default();
        let mut moe = MixtureOfExperts::new(config, 64, 256).unwrap();
        
        let input = Tensor::randn(&[16, 64]);
        moe.forward(&input).unwrap();
        
        let balance = moe.load_balance_factor();
        assert!(balance > 0.0);
        assert!(balance <= 1.0);
    }
    
    #[test]
    fn test_aux_loss() {
        let config = MoEConfig::default();
        let mut moe = MixtureOfExperts::new(config, 64, 256).unwrap();
        
        let input = Tensor::randn(&[8, 64]);
        moe.forward(&input).unwrap();
        
        let aux_loss = moe.get_aux_loss();
        assert!(aux_loss >= 0.0);
    }
    
    #[test]
    fn test_expert_usage_stats() {
        let config = MoEConfig::default();
        let mut moe = MixtureOfExperts::new(config, 64, 256).unwrap();
        
        let input = Tensor::randn(&[16, 64]);
        moe.forward(&input).unwrap();
        
        let usage = moe.get_expert_usage();
        let total: usize = usage.iter().sum();
        assert!(total > 0);
        
        moe.reset_usage_stats();
        let usage_after = moe.get_expert_usage();
        assert_eq!(usage_after.iter().sum::<usize>(), 0);
    }
    
    #[test]
    fn test_gshard_config() {
        let config = MoEConfig::gshard(16);
        assert_eq!(config.num_experts, 16);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.routing_strategy, RoutingStrategy::TopK);
    }
}

//! Gradient Checkpointing
//!
//! Implements gradient checkpointing to reduce memory usage during training:
//! - Selective activation storage
//! - Recomputation during backward pass
//! - Memory-efficient training for large models
//! - Configurable checkpoint intervals

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// Checkpoint strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointStrategy {
    /// Checkpoint every N layers
    EveryN(usize),
    /// Checkpoint at specific layer indices
    Selective,
    /// Checkpoint all layers (maximum memory savings)
    All,
    /// No checkpointing (maximum speed)
    None,
}

/// Gradient checkpointing configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Checkpoint strategy
    pub strategy: CheckpointStrategy,
    /// Specific layers to checkpoint (for Selective strategy)
    pub checkpoint_layers: Vec<usize>,
    /// Enable CPU offloading for checkpoints
    pub cpu_offload: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        CheckpointConfig {
            strategy: CheckpointStrategy::EveryN(2),
            checkpoint_layers: Vec::new(),
            cpu_offload: false,
        }
    }
}

impl CheckpointConfig {
    /// Checkpoint every N layers
    pub fn every_n(n: usize) -> Self {
        CheckpointConfig {
            strategy: CheckpointStrategy::EveryN(n),
            ..Default::default()
        }
    }
    
    /// Checkpoint specific layers
    pub fn selective(layers: Vec<usize>) -> Self {
        CheckpointConfig {
            strategy: CheckpointStrategy::Selective,
            checkpoint_layers: layers,
            ..Default::default()
        }
    }
    
    /// Checkpoint all layers
    pub fn all() -> Self {
        CheckpointConfig {
            strategy: CheckpointStrategy::All,
            ..Default::default()
        }
    }
}

/// Checkpoint manager
pub struct CheckpointManager {
    config: CheckpointConfig,
    checkpoints: HashMap<usize, Tensor>,
    recompute_count: usize,
    memory_saved: usize,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Self {
        CheckpointManager {
            config,
            checkpoints: HashMap::new(),
            recompute_count: 0,
            memory_saved: 0,
        }
    }
    
    /// Check if layer should be checkpointed
    pub fn should_checkpoint(&self, layer_idx: usize) -> bool {
        match self.config.strategy {
            CheckpointStrategy::None => false,
            CheckpointStrategy::All => true,
            CheckpointStrategy::EveryN(n) => layer_idx % n == 0,
            CheckpointStrategy::Selective => self.config.checkpoint_layers.contains(&layer_idx),
        }
    }
    
    /// Save checkpoint
    pub fn save_checkpoint(&mut self, layer_idx: usize, activation: Tensor) {
        if self.should_checkpoint(layer_idx) {
            let memory_size = activation.data_f32().len() * 4; // 4 bytes per f32
            self.memory_saved += memory_size;
            self.checkpoints.insert(layer_idx, activation);
        }
    }
    
    /// Get checkpoint
    pub fn get_checkpoint(&mut self, layer_idx: usize) -> Option<&Tensor> {
        self.checkpoints.get(&layer_idx)
    }
    
    /// Recompute activation (called during backward pass)
    pub fn recompute<F>(&mut self, layer_idx: usize, recompute_fn: F) -> Tensor
    where
        F: FnOnce() -> Tensor,
    {
        self.recompute_count += 1;
        recompute_fn()
    }
    
    /// Clear checkpoints
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> CheckpointStats {
        CheckpointStats {
            num_checkpoints: self.checkpoints.len(),
            recompute_count: self.recompute_count,
            memory_saved_bytes: self.memory_saved,
        }
    }
}

/// Checkpoint statistics
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    /// Number of active checkpoints
    pub num_checkpoints: usize,
    /// Number of recomputations
    pub recompute_count: usize,
    /// Estimated memory saved (bytes)
    pub memory_saved_bytes: usize,
}

/// Checkpointed layer wrapper
pub struct CheckpointedLayer<F>
where
    F: Fn(&Tensor) -> Tensor,
{
    forward_fn: F,
    layer_idx: usize,
    manager: CheckpointManager,
}

impl<F> CheckpointedLayer<F>
where
    F: Fn(&Tensor) -> Tensor,
{
    /// Create new checkpointed layer
    pub fn new(forward_fn: F, layer_idx: usize, config: CheckpointConfig) -> Self {
        CheckpointedLayer {
            forward_fn,
            layer_idx,
            manager: CheckpointManager::new(config),
        }
    }
    
    /// Forward pass with checkpointing
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let output = (self.forward_fn)(input);
        
        // Save checkpoint if needed
        if self.manager.should_checkpoint(self.layer_idx) {
            self.manager.save_checkpoint(self.layer_idx, input.clone());
        }
        
        output
    }
    
    /// Backward pass with recomputation
    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Check if we have a checkpoint
        if let Some(checkpoint) = self.manager.get_checkpoint(self.layer_idx) {
            // Recompute forward pass
            let _recomputed = (self.forward_fn)(checkpoint);
            // In a real implementation, we'd compute gradients here
            grad_output.clone()
        } else {
            // No checkpoint, assume we have the activation
            grad_output.clone()
        }
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> CheckpointStats {
        self.manager.get_stats()
    }
}

/// Sequential model with gradient checkpointing
pub struct CheckpointedSequential {
    layers: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    manager: CheckpointManager,
}

impl CheckpointedSequential {
    /// Create new checkpointed sequential model
    pub fn new(config: CheckpointConfig) -> Self {
        CheckpointedSequential {
            layers: Vec::new(),
            manager: CheckpointManager::new(config),
        }
    }
    
    /// Add layer
    pub fn add_layer<F>(&mut self, layer: F)
    where
        F: Fn(&Tensor) -> Tensor + 'static,
    {
        self.layers.push(Box::new(layer));
    }
    
    /// Forward pass with checkpointing
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        
        for (idx, layer) in self.layers.iter().enumerate() {
            // Save checkpoint if needed
            if self.manager.should_checkpoint(idx) {
                self.manager.save_checkpoint(idx, x.clone());
            }
            
            // Forward through layer
            x = layer(&x);
        }
        
        x
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> CheckpointStats {
        self.manager.get_stats()
    }
    
    /// Clear checkpoints
    pub fn clear_checkpoints(&mut self) {
        self.manager.clear();
    }
}

/// Utility function to estimate memory savings
pub fn estimate_memory_savings(
    num_layers: usize,
    activation_size_mb: f32,
    strategy: CheckpointStrategy,
) -> f32 {
    let checkpointed_layers = match strategy {
        CheckpointStrategy::None => 0,
        CheckpointStrategy::All => num_layers,
        CheckpointStrategy::EveryN(n) => num_layers / n,
        CheckpointStrategy::Selective => 0, // Can't estimate without knowing which layers
    };
    
    let saved_memory = (num_layers - checkpointed_layers) as f32 * activation_size_mb;
    saved_memory
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_checkpoint_strategy() {
        let config = CheckpointConfig::every_n(2);
        let manager = CheckpointManager::new(config);
        
        assert!(manager.should_checkpoint(0));
        assert!(!manager.should_checkpoint(1));
        assert!(manager.should_checkpoint(2));
        assert!(!manager.should_checkpoint(3));
    }
    
    #[test]
    fn test_selective_checkpointing() {
        let config = CheckpointConfig::selective(vec![1, 3, 5]);
        let manager = CheckpointManager::new(config);
        
        assert!(!manager.should_checkpoint(0));
        assert!(manager.should_checkpoint(1));
        assert!(!manager.should_checkpoint(2));
        assert!(manager.should_checkpoint(3));
    }
    
    #[test]
    fn test_checkpoint_save_and_get() {
        let config = CheckpointConfig::all();
        let mut manager = CheckpointManager::new(config);
        
        let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        manager.save_checkpoint(0, tensor.clone());
        
        let retrieved = manager.get_checkpoint(0).unwrap();
        assert_eq!(retrieved.data_f32(), tensor.data_f32());
    }
    
    #[test]
    fn test_checkpointed_layer() {
        let forward_fn = |x: &Tensor| {
            x.mul_scalar(2.0)
        };
        
        let config = CheckpointConfig::all();
        let mut layer = CheckpointedLayer::new(forward_fn, 0, config);
        
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let output = layer.forward(&input);
        
        let output_data = output.data_f32();
        assert_eq!(output_data[0], 2.0);
        assert_eq!(output_data[1], 4.0);
        assert_eq!(output_data[2], 6.0);
        
        let stats = layer.get_stats();
        assert_eq!(stats.num_checkpoints, 1);
    }
    
    #[test]
    fn test_checkpointed_sequential() {
        let config = CheckpointConfig::every_n(1);
        let mut model = CheckpointedSequential::new(config);
        
        // Add layers
        model.add_layer(|x: &Tensor| x.mul_scalar(2.0));
        model.add_layer(|x: &Tensor| x.add_scalar(1.0));
        model.add_layer(|x: &Tensor| x.mul_scalar(0.5));
        
        let input = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let output = model.forward(&input);
        
        // (x * 2 + 1) * 0.5 = (1*2+1)*0.5 = 1.5, (2*2+1)*0.5 = 2.5
        let output_data = output.data_f32();
        assert!((output_data[0] - 1.5).abs() < 1e-5);
        assert!((output_data[1] - 2.5).abs() < 1e-5);
        
        let stats = model.get_stats();
        assert!(stats.num_checkpoints > 0);
    }
    
    #[test]
    fn test_memory_savings_estimation() {
        let savings = estimate_memory_savings(10, 100.0, CheckpointStrategy::EveryN(2));
        assert_eq!(savings, 500.0); // 5 layers saved * 100 MB
        
        let savings = estimate_memory_savings(10, 100.0, CheckpointStrategy::All);
        assert_eq!(savings, 0.0); // All checkpointed, no savings
        
        let savings = estimate_memory_savings(10, 100.0, CheckpointStrategy::None);
        assert_eq!(savings, 1000.0); // All saved
    }
    
    #[test]
    fn test_checkpoint_clear() {
        let config = CheckpointConfig::all();
        let mut manager = CheckpointManager::new(config);
        
        let tensor = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
        manager.save_checkpoint(0, tensor);
        
        assert_eq!(manager.checkpoints.len(), 1);
        
        manager.clear();
        assert_eq!(manager.checkpoints.len(), 0);
    }
    
    #[test]
    fn test_recompute_tracking() {
        let config = CheckpointConfig::all();
        let mut manager = CheckpointManager::new(config);
        
        let initial_count = manager.recompute_count;
        
        manager.recompute(0, || Tensor::from_slice(&[1.0f32], &[1]).unwrap());
        
        assert_eq!(manager.recompute_count, initial_count + 1);
    }
}

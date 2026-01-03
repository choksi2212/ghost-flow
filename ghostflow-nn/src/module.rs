//! Base Module trait for neural network layers

use ghostflow_core::Tensor;

/// Base trait for all neural network modules
pub trait Module: Send + Sync {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Tensor;
    
    /// Get all trainable parameters
    fn parameters(&self) -> Vec<Tensor>;
    
    /// Number of trainable parameters
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
    
    /// Set module to training mode
    fn train(&mut self);
    
    /// Set module to evaluation mode
    fn eval(&mut self);
    
    /// Check if in training mode
    fn is_training(&self) -> bool;
}

/// Container for sequential layers
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
            training: true,
        }
    }

    /// Add a layer to the sequential model
    pub fn add_layer<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter()
            .flat_map(|l| l.parameters())
            .collect()
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

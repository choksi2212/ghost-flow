//! Base Optimizer trait

use ghostflow_core::Tensor;

/// Base trait for all optimizers
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self);
    
    /// Zero all gradients
    fn zero_grad(&mut self);
    
    /// Get current learning rate
    fn get_lr(&self) -> f32;
    
    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
    
    /// Get all parameters
    fn parameters(&self) -> &[Tensor];
}

/// Parameter group for different learning rates
pub struct ParamGroup {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub weight_decay: f32,
}

impl ParamGroup {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        ParamGroup {
            params,
            lr,
            weight_decay: 0.0,
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

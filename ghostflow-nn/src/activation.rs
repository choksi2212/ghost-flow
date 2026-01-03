//! Activation function modules

use ghostflow_core::Tensor;
use crate::module::Module;

/// ReLU activation module
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self { ReLU }
}

impl Default for ReLU {
    fn default() -> Self { Self::new() }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Leaky ReLU activation module
pub struct LeakyReLU {
    alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self {
        LeakyReLU { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self { Self::new(0.01) }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.leaky_relu(self.alpha)
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// GELU activation module
pub struct GELU;

impl GELU {
    pub fn new() -> Self { GELU }
}

impl Default for GELU {
    fn default() -> Self { Self::new() }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.gelu()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Sigmoid activation module
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self { Sigmoid }
}

impl Default for Sigmoid {
    fn default() -> Self { Self::new() }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Tanh activation module
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self { Tanh }
}

impl Default for Tanh {
    fn default() -> Self { Self::new() }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// SiLU/Swish activation module
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self { SiLU }
}

impl Default for SiLU {
    fn default() -> Self { Self::new() }
}

impl Module for SiLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.silu()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Softmax activation module
pub struct Softmax {
    dim: i32,
}

impl Softmax {
    pub fn new(dim: i32) -> Self {
        Softmax { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self { Self::new(-1) }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.softmax(self.dim)
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

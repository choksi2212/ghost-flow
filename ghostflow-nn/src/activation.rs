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

/// Swish activation module (parameterized version of SiLU)
pub struct Swish {
    beta: f32,
}

impl Swish {
    pub fn new(beta: f32) -> Self {
        Swish { beta }
    }
}

impl Default for Swish {
    fn default() -> Self { Self::new(1.0) }
}

impl Module for Swish {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&x| x / (1.0 + (-self.beta * x).exp()))
            .collect();
        Tensor::from_slice(&result, input.dims()).unwrap()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Mish activation module
/// f(x) = x * tanh(softplus(x))
pub struct Mish;

impl Mish {
    pub fn new() -> Self { Mish }
}

impl Default for Mish {
    fn default() -> Self { Self::new() }
}

impl Module for Mish {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&x| {
                let softplus = (1.0 + x.exp()).ln();
                x * softplus.tanh()
            })
            .collect();
        Tensor::from_slice(&result, input.dims()).unwrap()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// ELU (Exponential Linear Unit) activation module
pub struct ELU {
    alpha: f32,
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        ELU { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self { Self::new(1.0) }
}

impl Module for ELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&x| {
                if x > 0.0 {
                    x
                } else {
                    self.alpha * (x.exp() - 1.0)
                }
            })
            .collect();
        Tensor::from_slice(&result, input.dims()).unwrap()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// SELU (Scaled Exponential Linear Unit) activation module
pub struct SELU {
    alpha: f32,
    scale: f32,
}

impl SELU {
    pub fn new() -> Self {
        // Standard SELU parameters
        SELU {
            alpha: 1.6732632423543772848170429916717,
            scale: 1.0507009873554804934193349852946,
        }
    }
    
    pub fn with_params(alpha: f32, scale: f32) -> Self {
        SELU { alpha, scale }
    }
}

impl Default for SELU {
    fn default() -> Self { Self::new() }
}

impl Module for SELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&x| {
                if x > 0.0 {
                    self.scale * x
                } else {
                    self.scale * self.alpha * (x.exp() - 1.0)
                }
            })
            .collect();
        Tensor::from_slice(&result, input.dims()).unwrap()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Softplus activation module
/// f(x) = ln(1 + exp(x))
pub struct Softplus {
    beta: f32,
    threshold: f32,
}

impl Softplus {
    pub fn new(beta: f32, threshold: f32) -> Self {
        Softplus { beta, threshold }
    }
}

impl Default for Softplus {
    fn default() -> Self { Self::new(1.0, 20.0) }
}

impl Module for Softplus {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&x| {
                let beta_x = self.beta * x;
                if beta_x > self.threshold {
                    // For large values, use linear approximation to avoid overflow
                    x
                } else {
                    (1.0 + beta_x.exp()).ln() / self.beta
                }
            })
            .collect();
        Tensor::from_slice(&result, input.dims()).unwrap()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

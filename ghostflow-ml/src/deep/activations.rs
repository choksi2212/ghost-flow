//! Activation Functions - ReLU, Sigmoid, Tanh, GELU, Swish, etc.

use ghostflow_core::Tensor;

/// Activation function trait
pub trait ActivationFn: Send + Sync {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor;
}

/// ReLU activation
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self { ReLU }
}

impl Default for ReLU {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| if v > 0.0 { g } else { 0.0 })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Leaky ReLU activation
pub struct LeakyReLU {
    pub negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        LeakyReLU { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self { Self::new(0.01) }
}

impl ActivationFn for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| if v > 0.0 { v } else { self.negative_slope * v })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| if v > 0.0 { g } else { self.negative_slope * g })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// PReLU (Parametric ReLU)
pub struct PReLU {
    pub alpha: Vec<f32>,
    pub num_parameters: usize,
}

impl PReLU {
    pub fn new(num_parameters: usize) -> Self {
        PReLU {
            alpha: vec![0.25; num_parameters],
            num_parameters,
        }
    }
}

impl ActivationFn for PReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter().enumerate()
            .map(|(i, &v)| {
                let alpha = self.alpha[i % self.num_parameters];
                if v > 0.0 { v } else { alpha * v }
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter()).enumerate()
            .map(|(i, (&v, &g))| {
                let alpha = self.alpha[i % self.num_parameters];
                if v > 0.0 { g } else { alpha * g }
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// ELU (Exponential Linear Unit)
pub struct ELU {
    pub alpha: f32,
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        ELU { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self { Self::new(1.0) }
}

impl ActivationFn for ELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| if v > 0.0 { v } else { self.alpha * (v.exp() - 1.0) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                if v > 0.0 { g } else { g * self.alpha * v.exp() }
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// SELU (Scaled ELU)
pub struct SELU;

impl SELU {
    pub fn new() -> Self { SELU }
    
    const ALPHA: f32 = 1.6732632423543772;
    const SCALE: f32 = 1.0507009873554805;
}

impl Default for SELU {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for SELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| {
                Self::SCALE * if v > 0.0 { v } else { Self::ALPHA * (v.exp() - 1.0) }
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                Self::SCALE * g * if v > 0.0 { 1.0 } else { Self::ALPHA * v.exp() }
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Sigmoid activation
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self { Sigmoid }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }
}

impl Default for Sigmoid {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for Sigmoid {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter().map(|&v| Self::sigmoid(v)).collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                let s = Self::sigmoid(v);
                g * s * (1.0 - s)
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Tanh activation
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self { Tanh }
}

impl Default for Tanh {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for Tanh {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter().map(|&v| v.tanh()).collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                let t = v.tanh();
                g * (1.0 - t * t)
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Softmax activation
pub struct Softmax {
    pub dim: i32,
}

impl Softmax {
    pub fn new(dim: i32) -> Self {
        Softmax { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self { Self::new(-1) }
}

impl ActivationFn for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let dims = x.dims();
        
        if dims.len() == 1 {
            let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = data.iter().map(|&v| (v - max_val).exp()).sum();
            let result: Vec<f32> = data.iter().map(|&v| (v - max_val).exp() / exp_sum).collect();
            return Tensor::from_slice(&result, dims).unwrap();
        }

        // For 2D: apply softmax along last dimension
        let batch_size = dims[0];
        let n_classes = dims[1];
        let mut result = vec![0.0f32; data.len()];

        for b in 0..batch_size {
            let start = b * n_classes;
            let end = start + n_classes;
            let row = &data[start..end];
            
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
            
            for (i, &v) in row.iter().enumerate() {
                result[start + i] = (v - max_val).exp() / exp_sum;
            }
        }

        Tensor::from_slice(&result, dims).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        // Softmax backward is typically combined with cross-entropy
        // Here we provide the Jacobian-based gradient
        let softmax_output = self.forward(x);
        let s = softmax_output.data_f32();
        let grad = grad_output.data_f32();
        let dims = x.dims();

        if dims.len() == 1 {
            let n = s.len();
            let mut result = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    let jacobian = if i == j {
                        s[i] * (1.0 - s[i])
                    } else {
                        -s[i] * s[j]
                    };
                    result[i] += jacobian * grad[j];
                }
            }
            return Tensor::from_slice(&result, dims).unwrap();
        }

        // For batched input
        let batch_size = dims[0];
        let n_classes = dims[1];
        let mut result = vec![0.0f32; s.len()];

        for b in 0..batch_size {
            let start = b * n_classes;
            for i in 0..n_classes {
                for j in 0..n_classes {
                    let jacobian = if i == j {
                        s[start + i] * (1.0 - s[start + i])
                    } else {
                        -s[start + i] * s[start + j]
                    };
                    result[start + i] += jacobian * grad[start + j];
                }
            }
        }

        Tensor::from_slice(&result, dims).unwrap()
    }
}

/// GELU (Gaussian Error Linear Unit)
pub struct GELU {
    pub approximate: bool,
}

impl GELU {
    pub fn new() -> Self {
        GELU { approximate: false }
    }

    pub fn approximate(mut self, approx: bool) -> Self {
        self.approximate = approx;
        self
    }

    fn erf(x: f32) -> f32 {
        // Approximation of error function
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }
}

impl Default for GELU {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for GELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = if self.approximate {
            // Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
            data.iter()
                .map(|&v| {
                    0.5 * v * (1.0 + (sqrt_2_pi * (v + 0.044715 * v.powi(3))).tanh())
                })
                .collect()
        } else {
            // Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
            let sqrt_2 = std::f32::consts::SQRT_2;
            data.iter()
                .map(|&v| 0.5 * v * (1.0 + Self::erf(v / sqrt_2)))
                .collect()
        };
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let sqrt_2 = std::f32::consts::SQRT_2;
        let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();

        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                let cdf = 0.5 * (1.0 + Self::erf(v / sqrt_2));
                let pdf = (-v * v / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
                g * (cdf + v * pdf)
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Swish activation (SiLU)
pub struct Swish;

impl Swish {
    pub fn new() -> Self { Swish }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }
}

impl Default for Swish {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for Swish {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| v * Self::sigmoid(v))
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                let s = Self::sigmoid(v);
                g * (s + v * s * (1.0 - s))
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Mish activation
pub struct Mish;

impl Mish {
    pub fn new() -> Self { Mish }

    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            0.0
        } else {
            (1.0 + x.exp()).ln()
        }
    }
}

impl Default for Mish {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for Mish {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| v * Self::softplus(v).tanh())
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                let sp = Self::softplus(v);
                let tanh_sp = sp.tanh();
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                let sech_sq = 1.0 - tanh_sp * tanh_sp;
                g * (tanh_sp + v * sech_sq * sigmoid)
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Hardswish activation
pub struct Hardswish;

impl Hardswish {
    pub fn new() -> Self { Hardswish }
}

impl Default for Hardswish {
    fn default() -> Self { Self::new() }
}

impl ActivationFn for Hardswish {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| {
                if v <= -3.0 {
                    0.0
                } else if v >= 3.0 {
                    v
                } else {
                    v * (v + 3.0) / 6.0
                }
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn backward(&self, x: &Tensor, grad_output: &Tensor) -> Tensor {
        let data = x.data_f32();
        let grad = grad_output.data_f32();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(&v, &g)| {
                if v <= -3.0 {
                    0.0
                } else if v >= 3.0 {
                    g
                } else {
                    g * (2.0 * v + 3.0) / 6.0
                }
            })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let relu = ReLU::new();
        let y = relu.forward(&x);
        let y_data = y.data_f32();
        assert_eq!(y_data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let x = Tensor::from_slice(&[0.0], &[1]).unwrap();
        let sigmoid = Sigmoid::new();
        let y = sigmoid.forward(&x);
        assert!((y.data_f32()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gelu() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0], &[3]).unwrap();
        let gelu = GELU::new();
        let y = gelu.forward(&x);
        assert!(y.data_f32()[0].abs() < 1e-6); // GELU(0) = 0
    }
}

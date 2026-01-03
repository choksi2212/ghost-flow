//! Linear (fully connected) layer

use ghostflow_core::Tensor;
use crate::module::Module;
use crate::init;

/// Linear transformation: y = xW^T + b
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, true)
    }

    /// Create a linear layer with optional bias
    pub fn with_bias(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Kaiming initialization for weights
        let weight = init::kaiming_uniform(&[out_features, in_features], in_features);
        
        let bias = if bias {
            // Uniform initialization for bias
            let bound = 1.0 / (in_features as f32).sqrt();
            Some(init::uniform(&[out_features], -bound, bound))
        } else {
            None
        };

        Linear {
            weight,
            bias,
            in_features,
            out_features,
            training: true,
        }
    }

    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [*, in_features]
        // weight: [out_features, in_features]
        // output: [*, out_features]
        
        let weight_t = self.weight.t().unwrap();
        let mut output = input.matmul(&weight_t).unwrap();
        
        if let Some(ref bias) = self.bias {
            output = output.add(bias).unwrap();
        }
        
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(10, 5);
        let input = Tensor::randn(&[2, 10]);
        let output = linear.forward(&input);
        
        assert_eq!(output.dims(), &[2, 5]);
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::with_bias(10, 5, false);
        let input = Tensor::randn(&[2, 10]);
        let output = linear.forward(&input);
        
        assert_eq!(output.dims(), &[2, 5]);
    }

    #[test]
    fn test_linear_parameters() {
        let linear = Linear::new(10, 5);
        let params = linear.parameters();
        
        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].numel(), 50); // 10 * 5
        assert_eq!(params[1].numel(), 5);
    }
}

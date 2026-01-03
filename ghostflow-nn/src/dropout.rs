//! Dropout regularization

use ghostflow_core::Tensor;
use crate::module::Module;
use rand::Rng;

/// Dropout layer
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");
        Dropout { p, training: true }
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let data = input.data_f32();
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);

        let output: Vec<f32> = data.iter()
            .map(|&x| {
                if rng.gen::<f32>() < self.p {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect();

        Tensor::from_slice(&output, input.dims()).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
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

/// Dropout2d - drops entire channels
pub struct Dropout2d {
    p: f32,
    training: bool,
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0);
        Dropout2d { p, training: true }
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let dims = input.dims();
        let batch = dims[0];
        let channels = dims[1];
        let spatial: usize = dims[2..].iter().product();

        let data = input.data_f32();
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);

        let mut output = data.clone();

        for b in 0..batch {
            for c in 0..channels {
                if rng.gen::<f32>() < self.p {
                    // Drop entire channel
                    let start = (b * channels + c) * spatial;
                    for i in 0..spatial {
                        output[start + i] = 0.0;
                    }
                } else {
                    // Scale
                    let start = (b * channels + c) * spatial;
                    for i in 0..spatial {
                        output[start + i] *= scale;
                    }
                }
            }
        }

        Tensor::from_slice(&output, dims).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_eval() {
        let mut dropout = Dropout::new(0.5);
        dropout.eval();
        
        let input = Tensor::ones(&[10, 10]);
        let output = dropout.forward(&input);
        
        // In eval mode, output should equal input
        assert_eq!(output.data_f32(), input.data_f32());
    }

    #[test]
    fn test_dropout_train() {
        let dropout = Dropout::new(0.5);
        let input = Tensor::ones(&[100, 100]);
        let output = dropout.forward(&input);
        
        // Some values should be zero
        let zeros = output.data_f32().iter().filter(|&&x| x == 0.0).count();
        assert!(zeros > 0);
    }
}

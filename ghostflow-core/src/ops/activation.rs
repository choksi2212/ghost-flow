//! Activation functions

use crate::tensor::Tensor;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Tensor {
    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        #[cfg(feature = "simd")]
        {
            use crate::ops::simd::relu_simd;
            let data = self.data_f32();
            let result = relu_simd(&data);
            Tensor::from_slice(&result, self.dims()).unwrap()
        }
        
        #[cfg(not(feature = "simd"))]
        {
            let data: Vec<f32> = self.data_f32()
                .iter()
                .map(|&x| x.max(0.0))
                .collect();
            Tensor::from_slice(&data, self.dims()).unwrap()
        }
    }

    /// Leaky ReLU: max(alpha * x, x)
    pub fn leaky_relu(&self, alpha: f32) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * x })
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// ELU activation
    pub fn elu(&self, alpha: f32) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// SELU activation (self-normalizing)
    pub fn selu(&self) -> Tensor {
        const ALPHA: f32 = 1.673_263_2;
        const SCALE: f32 = 1.050_701;
        
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| {
                SCALE * if x > 0.0 { x } else { ALPHA * (x.exp() - 1.0) }
            })
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        #[cfg(feature = "simd")]
        {
            use crate::ops::simd::sigmoid_simd;
            let data = self.data_f32();
            let result = sigmoid_simd(&data);
            Tensor::from_slice(&result, self.dims()).unwrap()
        }
        
        #[cfg(not(feature = "simd"))]
        {
            let data: Vec<f32> = self.data_f32()
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            Tensor::from_slice(&data, self.dims()).unwrap()
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| x.tanh())
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// GELU activation (Gaussian Error Linear Unit)
    pub fn gelu(&self) -> Tensor {
        #[cfg(feature = "simd")]
        {
            use crate::ops::simd::gelu_simd;
            let data = self.data_f32();
            let result = gelu_simd(&data);
            Tensor::from_slice(&result, self.dims()).unwrap()
        }
        
        #[cfg(not(feature = "simd"))]
        {
            const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
            const COEFF: f32 = 0.044715;
            
            let data: Vec<f32> = self.data_f32()
                .iter()
                .map(|&x| {
                    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x.powi(3));
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect();
            Tensor::from_slice(&data, self.dims()).unwrap()
        }
    }

    /// SiLU / Swish activation: x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Mish activation: x * tanh(softplus(x))
    pub fn mish(&self) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| {
                let softplus = (1.0 + x.exp()).ln();
                x * softplus.tanh()
            })
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Softplus: log(1 + exp(x))
    pub fn softplus(&self) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| {
                // Numerically stable version
                if x > 20.0 {
                    x
                } else if x < -20.0 {
                    x.exp()
                } else {
                    (1.0 + x.exp()).ln()
                }
            })
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Softsign: x / (1 + |x|)
    pub fn softsign(&self) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| x / (1.0 + x.abs()))
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Softmax along last dimension
    pub fn softmax(&self, dim: i32) -> Tensor {
        let dims = self.dims();
        let ndim = dims.len();
        let dim = if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize };
        
        let data = self.data_f32();
        let dim_size = dims[dim];
        
        // Compute stride for the softmax dimension
        let inner_size: usize = dims[dim + 1..].iter().product();
        let _outer_size: usize = dims[..dim].iter().product();
        
        let mut result = vec![0.0f32; data.len()];
        
        // Parallelize over outer dimension for better performance
        result.par_chunks_mut(dim_size * inner_size)
            .enumerate()
            .for_each(|(outer, outer_chunk)| {
                for inner in 0..inner_size {
                    // Find max for numerical stability
                    let mut max_val = f32::NEG_INFINITY;
                    for d in 0..dim_size {
                        let idx = d * inner_size + inner;
                        let val = data[outer * dim_size * inner_size + idx];
                        max_val = max_val.max(val);
                    }
                    
                    // Compute exp and sum
                    let mut sum = 0.0f32;
                    for d in 0..dim_size {
                        let idx = d * inner_size + inner;
                        let data_idx = outer * dim_size * inner_size + idx;
                        let exp_val = (data[data_idx] - max_val).exp();
                        outer_chunk[idx] = exp_val;
                        sum += exp_val;
                    }
                    
                    // Normalize
                    for d in 0..dim_size {
                        let idx = d * inner_size + inner;
                        outer_chunk[idx] /= sum;
                    }
                }
            });
        
        Tensor::from_slice(&result, dims).unwrap()
    }

    /// Log softmax (numerically stable)
    pub fn log_softmax(&self, dim: i32) -> Tensor {
        let softmax = self.softmax(dim);
        softmax.log()
    }

    /// Hardtanh: clamp(x, min, max)
    pub fn hardtanh(&self, min_val: f32, max_val: f32) -> Tensor {
        self.clamp(min_val, max_val)
    }

    /// Hard sigmoid: clamp((x + 3) / 6, 0, 1)
    pub fn hardsigmoid(&self) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| ((x + 3.0) / 6.0).clamp(0.0, 1.0))
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Hard swish: x * hardsigmoid(x)
    pub fn hardswish(&self) -> Tensor {
        let data: Vec<f32> = self.data_f32()
            .iter()
            .map(|&x| x * ((x + 3.0) / 6.0).clamp(0.0, 1.0))
            .collect();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let r = t.relu();
        assert_eq!(r.data_f32(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::from_slice(&[0.0f32], &[1]).unwrap();
        let s = t.sigmoid();
        assert!((s.data_f32()[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let s = t.softmax(-1);
        let sum: f32 = s.data_f32().iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_slice(&[0.0f32, 1.0, -1.0], &[3]).unwrap();
        let g = t.gelu();
        // GELU(0) ≈ 0
        assert!(g.data_f32()[0].abs() < 0.001);
    }
}

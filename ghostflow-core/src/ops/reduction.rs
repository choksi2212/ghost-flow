//! Reduction operations (sum, mean, max, min, etc.)

use crate::tensor::Tensor;
use crate::error::{GhostError, Result};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Tensor {
    /// Sum all elements
    pub fn sum(&self) -> Tensor {
        let data = self.data_f32();
        let sum: f32 = data.iter().sum();
        Tensor::from_slice(&[sum], &[]).unwrap()
    }

    /// Sum along dimension
    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        self.reduce_dim(dim, keepdim, |slice| slice.iter().sum())
    }

    /// Mean of all elements
    pub fn mean(&self) -> Tensor {
        let data = self.data_f32();
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        Tensor::from_slice(&[mean], &[]).unwrap()
    }

    /// Mean along dimension
    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        let dim_size = self.dims()[dim] as f32;
        self.reduce_dim(dim, keepdim, |slice| {
            slice.iter().sum::<f32>() / dim_size
        })
    }

    /// Maximum element
    pub fn max(&self) -> Tensor {
        let data = self.data_f32();
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Tensor::from_slice(&[max], &[]).unwrap()
    }

    /// Maximum along dimension
    pub fn max_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        self.reduce_dim(dim, keepdim, |slice| {
            slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        })
    }

    /// Minimum element
    pub fn min(&self) -> Tensor {
        let data = self.data_f32();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        Tensor::from_slice(&[min], &[]).unwrap()
    }

    /// Minimum along dimension
    pub fn min_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        self.reduce_dim(dim, keepdim, |slice| {
            slice.iter().cloned().fold(f32::INFINITY, f32::min)
        })
    }

    /// Product of all elements
    pub fn prod(&self) -> Tensor {
        let data = self.data_f32();
        let prod: f32 = data.iter().product();
        Tensor::from_slice(&[prod], &[]).unwrap()
    }

    /// Variance of all elements
    pub fn var(&self, unbiased: bool) -> Tensor {
        let data = self.data_f32();
        let n = data.len() as f32;
        let mean: f32 = data.iter().sum::<f32>() / n;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>();
        let divisor = if unbiased { n - 1.0 } else { n };
        Tensor::from_slice(&[var / divisor], &[]).unwrap()
    }

    /// Standard deviation
    pub fn std(&self, unbiased: bool) -> Tensor {
        let var = self.var(unbiased);
        var.sqrt()
    }

    /// Argmax - index of maximum element
    pub fn argmax(&self) -> Tensor {
        let data = self.data_f32();
        let (idx, _) = data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Tensor::from_slice(&[idx as f32], &[]).unwrap()
    }

    /// Argmax along dimension
    pub fn argmax_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        self.reduce_dim_with_index(dim, keepdim, |slice| {
            slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as f32)
                .unwrap_or(0.0)
        })
    }

    /// Argmin - index of minimum element
    pub fn argmin(&self) -> Tensor {
        let data = self.data_f32();
        let (idx, _) = data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Tensor::from_slice(&[idx as f32], &[]).unwrap()
    }

    /// Generic reduction along dimension
    fn reduce_dim<F>(&self, dim: usize, keepdim: bool, reducer: F) -> Result<Tensor>
    where
        F: Fn(&[f32]) -> f32 + Sync,
    {
        if dim >= self.ndim() {
            return Err(GhostError::DimOutOfBounds {
                dim,
                ndim: self.ndim(),
            });
        }

        let dims = self.dims();
        let dim_size = dims[dim];
        
        // Compute output shape
        let mut out_shape: Vec<usize> = dims.iter()
            .enumerate()
            .filter(|&(i, _)| i != dim || keepdim)
            .map(|(i, &d)| if i == dim { 1 } else { d })
            .collect();
        
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let data = self.data_f32();
        let out_numel: usize = out_shape.iter().product();
        
        // Compute strides for iteration
        let stride: usize = dims[dim + 1..].iter().product();
        let outer_stride = dim_size * stride;
        let _outer_size: usize = dims[..dim].iter().product();

        let result: Vec<f32> = (0..out_numel)
            .into_iter()
            .map(|out_idx| {
                let outer = out_idx / stride;
                let inner = out_idx % stride;
                
                let slice: Vec<f32> = (0..dim_size)
                    .map(|d| {
                        let idx = outer * outer_stride + d * stride + inner;
                        data[idx]
                    })
                    .collect();
                
                reducer(&slice)
            })
            .collect();

        Tensor::from_slice(&result, &out_shape)
    }

    /// Reduction with index output
    fn reduce_dim_with_index<F>(&self, dim: usize, keepdim: bool, reducer: F) -> Result<Tensor>
    where
        F: Fn(&[f32]) -> f32 + Sync,
    {
        // Same as reduce_dim but returns indices
        self.reduce_dim(dim, keepdim, reducer)
    }

    /// L2 norm
    pub fn norm(&self) -> Tensor {
        let data = self.data_f32();
        let sum_sq: f32 = data.iter().map(|&x| x * x).sum();
        Tensor::from_slice(&[sum_sq.sqrt()], &[]).unwrap()
    }

    /// L1 norm
    pub fn norm_l1(&self) -> Tensor {
        let data = self.data_f32();
        let sum: f32 = data.iter().map(|&x| x.abs()).sum();
        Tensor::from_slice(&[sum], &[]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let sum = t.sum();
        assert_eq!(sum.data_f32()[0], 10.0);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let mean = t.mean();
        assert_eq!(mean.data_f32()[0], 2.5);
    }

    #[test]
    fn test_max_min() {
        let t = Tensor::from_slice(&[1.0f32, 5.0, 2.0, 4.0], &[4]).unwrap();
        assert_eq!(t.max().data_f32()[0], 5.0);
        assert_eq!(t.min().data_f32()[0], 1.0);
    }

    #[test]
    fn test_var_std() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let var = t.var(false);
        assert!((var.data_f32()[0] - 2.0).abs() < 0.001);
    }
}

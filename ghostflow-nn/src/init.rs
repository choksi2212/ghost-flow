//! Weight initialization strategies

use ghostflow_core::Tensor;
use rand_distr::{Distribution, Normal, Uniform};

/// Initialize tensor with Xavier/Glorot uniform
pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Tensor {
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-bound, bound);
    
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    
    Tensor::from_slice(&data, shape).unwrap()
}

/// Initialize tensor with Xavier/Glorot normal
pub fn xavier_normal(shape: &[usize], fan_in: usize, fan_out: usize) -> Tensor {
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, std).unwrap();
    
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    
    Tensor::from_slice(&data, shape).unwrap()
}

/// Initialize tensor with Kaiming/He uniform (for ReLU)
pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Tensor {
    let bound = (6.0 / fan_in as f32).sqrt();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-bound, bound);
    
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    
    Tensor::from_slice(&data, shape).unwrap()
}

/// Initialize tensor with Kaiming/He normal (for ReLU)
pub fn kaiming_normal(shape: &[usize], fan_in: usize) -> Tensor {
    let std = (2.0 / fan_in as f32).sqrt();
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, std).unwrap();
    
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    
    Tensor::from_slice(&data, shape).unwrap()
}

/// Initialize tensor with uniform distribution
pub fn uniform(shape: &[usize], low: f32, high: f32) -> Tensor {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    
    Tensor::from_slice(&data, shape).unwrap()
}

/// Initialize tensor with normal distribution
pub fn normal(shape: &[usize], mean: f32, std: f32) -> Tensor {
    let mut rng = rand::thread_rng();
    let dist = Normal::new(mean, std).unwrap();
    
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    
    Tensor::from_slice(&data, shape).unwrap()
}

/// Initialize tensor with constant value
pub fn constant(shape: &[usize], value: f32) -> Tensor {
    Tensor::full(shape, value)
}

/// Initialize tensor with zeros
pub fn zeros(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape)
}

/// Initialize tensor with ones
pub fn ones(shape: &[usize]) -> Tensor {
    Tensor::ones(shape)
}

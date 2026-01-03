//! Pooling layers

use ghostflow_core::Tensor;
use crate::module::Module;

/// Max Pooling 2D
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0)
    }

    pub fn with_params(kernel_size: usize, stride: usize, padding: usize) -> Self {
        MaxPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (padding, padding),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let batch = dims[0];
        let channels = dims[1];
        let in_h = dims[2];
        let in_w = dims[3];
        
        let out_h = (in_h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        
        let data = input.data_f32();
        let mut output = vec![f32::NEG_INFINITY; batch * channels * out_h * out_w];
        
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = (oh * self.stride.0 + kh) as i32 - self.padding.0 as i32;
                                let iw = (ow * self.stride.1 + kw) as i32 - self.padding.1 as i32;
                                
                                if ih >= 0 && (ih as usize) < in_h && iw >= 0 && (iw as usize) < in_w {
                                    let idx = b * channels * in_h * in_w 
                                        + c * in_h * in_w 
                                        + (ih as usize) * in_w 
                                        + iw as usize;
                                    max_val = max_val.max(data[idx]);
                                }
                            }
                        }
                        
                        let out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, &[batch, channels, out_h, out_w]).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Average Pooling 2D
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    count_include_pad: bool,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0)
    }

    pub fn with_params(kernel_size: usize, stride: usize, padding: usize) -> Self {
        AvgPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (padding, padding),
            count_include_pad: true,
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let batch = dims[0];
        let channels = dims[1];
        let in_h = dims[2];
        let in_w = dims[3];
        
        let out_h = (in_h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        
        let data = input.data_f32();
        let mut output = vec![0.0f32; batch * channels * out_h * out_w];
        
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0;
                        
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = (oh * self.stride.0 + kh) as i32 - self.padding.0 as i32;
                                let iw = (ow * self.stride.1 + kw) as i32 - self.padding.1 as i32;
                                
                                if ih >= 0 && (ih as usize) < in_h && iw >= 0 && (iw as usize) < in_w {
                                    let idx = b * channels * in_h * in_w 
                                        + c * in_h * in_w 
                                        + (ih as usize) * in_w 
                                        + iw as usize;
                                    sum += data[idx];
                                    count += 1;
                                } else if self.count_include_pad {
                                    count += 1;
                                }
                            }
                        }
                        
                        let out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, &[batch, channels, out_h, out_w]).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Global Average Pooling 2D - reduces spatial dimensions to 1x1
pub struct GlobalAvgPool2d;

impl GlobalAvgPool2d {
    pub fn new() -> Self {
        GlobalAvgPool2d
    }
}

impl Default for GlobalAvgPool2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GlobalAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let batch = dims[0];
        let channels = dims[1];
        let spatial_size = dims[2] * dims[3];
        
        let data = input.data_f32();
        let mut output = vec![0.0f32; batch * channels];
        
        for b in 0..batch {
            for c in 0..channels {
                let start = b * channels * spatial_size + c * spatial_size;
                let sum: f32 = data[start..start + spatial_size].iter().sum();
                output[b * channels + c] = sum / spatial_size as f32;
            }
        }
        
        Tensor::from_slice(&output, &[batch, channels, 1, 1]).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Global Max Pooling 2D
pub struct GlobalMaxPool2d;

impl GlobalMaxPool2d {
    pub fn new() -> Self {
        GlobalMaxPool2d
    }
}

impl Default for GlobalMaxPool2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GlobalMaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let batch = dims[0];
        let channels = dims[1];
        let spatial_size = dims[2] * dims[3];
        
        let data = input.data_f32();
        let mut output = vec![f32::NEG_INFINITY; batch * channels];
        
        for b in 0..batch {
            for c in 0..channels {
                let start = b * channels * spatial_size + c * spatial_size;
                let max_val = data[start..start + spatial_size]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                output[b * channels + c] = max_val;
            }
        }
        
        Tensor::from_slice(&output, &[batch, channels, 1, 1]).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

/// Adaptive Average Pooling 2D - pools to target output size
pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        AdaptiveAvgPool2d { output_size }
    }

    pub fn square(size: usize) -> Self {
        Self::new((size, size))
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.dims();
        let batch = dims[0];
        let channels = dims[1];
        let in_h = dims[2];
        let in_w = dims[3];
        let (out_h, out_w) = self.output_size;
        
        let data = input.data_f32();
        let mut output = vec![0.0f32; batch * channels * out_h * out_w];
        
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Compute input region for this output
                        let ih_start = (oh * in_h) / out_h;
                        let ih_end = ((oh + 1) * in_h) / out_h;
                        let iw_start = (ow * in_w) / out_w;
                        let iw_end = ((ow + 1) * in_w) / out_w;
                        
                        let mut sum = 0.0f32;
                        let mut count = 0;
                        
                        for ih in ih_start..ih_end {
                            for iw in iw_start..iw_end {
                                let idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                                sum += data[idx];
                                count += 1;
                            }
                        }
                        
                        let out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, &[batch, channels, out_h, out_w]).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool2d() {
        let pool = MaxPool2d::new(2);
        let input = Tensor::randn(&[1, 3, 8, 8]);
        let output = pool.forward(&input);
        
        assert_eq!(output.dims(), &[1, 3, 4, 4]);
    }

    #[test]
    fn test_avg_pool2d() {
        let pool = AvgPool2d::new(2);
        let input = Tensor::randn(&[1, 3, 8, 8]);
        let output = pool.forward(&input);
        
        assert_eq!(output.dims(), &[1, 3, 4, 4]);
    }

    #[test]
    fn test_global_avg_pool() {
        let pool = GlobalAvgPool2d::new();
        let input = Tensor::randn(&[2, 64, 7, 7]);
        let output = pool.forward(&input);
        
        assert_eq!(output.dims(), &[2, 64, 1, 1]);
    }

    #[test]
    fn test_adaptive_avg_pool() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Tensor::randn(&[2, 64, 7, 7]);
        let output = pool.forward(&input);
        
        assert_eq!(output.dims(), &[2, 64, 1, 1]);
    }
}

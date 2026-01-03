//! Convolutional layers

use ghostflow_core::Tensor;
use crate::module::Module;
use crate::init;

/// 1D Convolution layer
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let weight = init::kaiming_uniform(
            &[out_channels, in_channels, kernel_size],
            fan_in,
        );
        
        let bound = 1.0 / (fan_in as f32).sqrt();
        let bias = Some(init::uniform(&[out_channels], -bound, bound));

        Conv1d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_channels, length]
        // weight: [out_channels, in_channels, kernel_size]
        // output: [batch, out_channels, out_length]
        
        let dims = input.dims();
        let batch = dims[0];
        let in_len = dims[2];
        
        let out_len = (in_len + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        // Simple implementation (not optimized)
        let input_data = input.data_f32();
        let weight_data = self.weight.data_f32();
        
        let mut output = vec![0.0f32; batch * self.out_channels * out_len];
        
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for ol in 0..out_len {
                    let mut sum = 0.0f32;
                    
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k;
                            let il = il as i32 - self.padding as i32;
                            
                            if il >= 0 && (il as usize) < in_len {
                                let input_idx = b * self.in_channels * in_len 
                                    + ic * in_len + il as usize;
                                let weight_idx = oc * self.in_channels * self.kernel_size 
                                    + ic * self.kernel_size + k;
                                sum += input_data[input_idx] * weight_data[weight_idx];
                            }
                        }
                    }
                    
                    let out_idx = b * self.out_channels * out_len + oc * out_len + ol;
                    output[out_idx] = sum;
                }
            }
        }
        
        let mut result = Tensor::from_slice(&output, &[batch, self.out_channels, out_len]).unwrap();
        
        if let Some(ref bias) = self.bias {
            // Add bias (broadcast over batch and length)
            let bias_data = bias.data_f32();
            let mut result_data = result.data_f32();
            
            for b in 0..batch {
                for oc in 0..self.out_channels {
                    for ol in 0..out_len {
                        let idx = b * self.out_channels * out_len + oc * out_len + ol;
                        result_data[idx] += bias_data[oc];
                    }
                }
            }
            
            result = Tensor::from_slice(&result_data, &[batch, self.out_channels, out_len]).unwrap();
        }
        
        result
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// 2D Convolution layer
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    training: bool,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self::with_params(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (stride, stride),
            (padding, padding),
        )
    }

    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let weight = init::kaiming_uniform(
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
            fan_in,
        );
        
        let bound = 1.0 / (fan_in as f32).sqrt();
        let bias = Some(init::uniform(&[out_channels], -bound, bound));

        Conv2d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Use optimized convolution from ghostflow-core
        #[cfg(feature = "optimized-conv")]
        {
            use ghostflow_core::ops::conv::conv2d_optimized;
            let bias = self.bias.as_ref();
            return conv2d_optimized(input, &self.weight, bias, self.stride, self.padding).unwrap();
        }
        
        // Fallback to direct implementation
        #[cfg(not(feature = "optimized-conv"))]
        {
            self.forward_direct(input)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

#[allow(dead_code)]
impl Conv2d {
    /// Direct convolution implementation (fallback)
    fn forward_direct(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_channels, height, width]
        // weight: [out_channels, in_channels, kH, kW]
        // output: [batch, out_channels, out_height, out_width]
        
        let dims = input.dims();
        let batch = dims[0];
        let in_h = dims[2];
        let in_w = dims[3];
        
        let out_h = (in_h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        
        let input_data = input.data_f32();
        let weight_data = self.weight.data_f32();
        
        let mut output = vec![0.0f32; batch * self.out_channels * out_h * out_w];
        
        // Naive convolution (im2col would be faster)
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    let ih = (oh * self.stride.0 + kh) as i32 - self.padding.0 as i32;
                                    let iw = (ow * self.stride.1 + kw) as i32 - self.padding.1 as i32;
                                    
                                    if ih >= 0 && (ih as usize) < in_h && iw >= 0 && (iw as usize) < in_w {
                                        let input_idx = b * self.in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + (ih as usize) * in_w
                                            + iw as usize;
                                        let weight_idx = oc * self.in_channels * self.kernel_size.0 * self.kernel_size.1
                                            + ic * self.kernel_size.0 * self.kernel_size.1
                                            + kh * self.kernel_size.1
                                            + kw;
                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                        }
                        
                        let out_idx = b * self.out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
        
        let mut result = Tensor::from_slice(&output, &[batch, self.out_channels, out_h, out_w]).unwrap();
        
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_f32();
            let mut result_data = result.data_f32();
            
            for b in 0..batch {
                for oc in 0..self.out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let idx = b * self.out_channels * out_h * out_w
                                + oc * out_h * out_w
                                + oh * out_w
                                + ow;
                            result_data[idx] += bias_data[oc];
                        }
                    }
                }
            }
            
            result = Tensor::from_slice(&result_data, &[batch, self.out_channels, out_h, out_w]).unwrap();
        }
        
        result
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2d::new(3, 16, 3, 1, 1);
        let input = Tensor::randn(&[2, 3, 32, 32]);
        let output = conv.forward(&input);
        
        assert_eq!(output.dims(), &[2, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_stride() {
        let conv = Conv2d::new(3, 16, 3, 2, 1);
        let input = Tensor::randn(&[2, 3, 32, 32]);
        let output = conv.forward(&input);
        
        assert_eq!(output.dims(), &[2, 16, 16, 16]);
    }
}

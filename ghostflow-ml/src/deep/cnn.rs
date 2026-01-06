//! Convolutional Neural Network Layers

use ghostflow_core::Tensor;
use rand::prelude::*;

/// 2D Convolution Layer
pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub use_bias: bool,
    weights: Vec<f32>,
    bias: Vec<f32>,
    grad_weights: Vec<f32>,
    grad_bias: Vec<f32>,
    input_cache: Vec<f32>,
    input_shape: Vec<usize>,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        let mut rng = thread_rng();
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let scale = (2.0 / fan_in as f32).sqrt();
        
        let weight_size = out_channels * in_channels * kernel_size.0 * kernel_size.1;
        let weights: Vec<f32> = (0..weight_size)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();

        Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            use_bias: true,
            weights,
            bias: vec![0.0; out_channels],
            grad_weights: vec![0.0; weight_size],
            grad_bias: vec![0.0; out_channels],
            input_cache: Vec::new(),
            input_shape: Vec::new(),
        }
    }

    pub fn stride(mut self, s: (usize, usize)) -> Self {
        self.stride = s;
        self
    }

    pub fn padding(mut self, p: (usize, usize)) -> Self {
        self.padding = p;
        self
    }

    fn output_size(&self, input_size: usize, kernel: usize, stride: usize, padding: usize) -> usize {
        (input_size + 2 * padding - kernel) / stride + 1
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        self.input_shape = input.dims().to_vec();
        self.input_cache = input_data.clone();

        let batch_size = self.input_shape[0];
        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];

        let out_h = self.output_size(in_h, self.kernel_size.0, self.stride.0, self.padding.0);
        let out_w = self.output_size(in_w, self.kernel_size.1, self.stride.1, self.padding.1);

        let mut output = vec![0.0f32; batch_size * self.out_channels * out_h * out_w];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = if self.use_bias { self.bias[oc] } else { 0.0 };

                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    let ih = oh * self.stride.0 + kh;
                                    let iw = ow * self.stride.1 + kw;

                                    let ih_pad = ih as i32 - self.padding.0 as i32;
                                    let iw_pad = iw as i32 - self.padding.1 as i32;

                                    if ih_pad >= 0 && ih_pad < in_h as i32 && 
                                       iw_pad >= 0 && iw_pad < in_w as i32 {
                                        let input_idx = b * self.in_channels * in_h * in_w +
                                                       ic * in_h * in_w +
                                                       ih_pad as usize * in_w +
                                                       iw_pad as usize;
                                        let weight_idx = oc * self.in_channels * self.kernel_size.0 * self.kernel_size.1 +
                                                        ic * self.kernel_size.0 * self.kernel_size.1 +
                                                        kh * self.kernel_size.1 + kw;
                                        sum += input_data[input_idx] * self.weights[weight_idx];
                                    }
                                }
                            }
                        }

                        let output_idx = b * self.out_channels * out_h * out_w +
                                        oc * out_h * out_w +
                                        oh * out_w + ow;
                        output[output_idx] = sum;
                    }
                }
            }
        }

        Tensor::from_slice(&output, &[batch_size, self.out_channels, out_h, out_w]).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let grad_shape = grad_output.dims();

        let batch_size = grad_shape[0];
        let out_h = grad_shape[2];
        let out_w = grad_shape[3];
        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];

        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
        let mut grad_input = vec![0.0f32; self.input_cache.len()];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grad_idx = b * self.out_channels * out_h * out_w +
                                      oc * out_h * out_w +
                                      oh * out_w + ow;
                        let grad = grad_data[grad_idx];

                        self.grad_bias[oc] += grad;

                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    let ih = oh * self.stride.0 + kh;
                                    let iw = ow * self.stride.1 + kw;

                                    let ih_pad = ih as i32 - self.padding.0 as i32;
                                    let iw_pad = iw as i32 - self.padding.1 as i32;

                                    if ih_pad >= 0 && ih_pad < in_h as i32 && 
                                       iw_pad >= 0 && iw_pad < in_w as i32 {
                                        let input_idx = b * self.in_channels * in_h * in_w +
                                                       ic * in_h * in_w +
                                                       ih_pad as usize * in_w +
                                                       iw_pad as usize;
                                        let weight_idx = oc * self.in_channels * self.kernel_size.0 * self.kernel_size.1 +
                                                        ic * self.kernel_size.0 * self.kernel_size.1 +
                                                        kh * self.kernel_size.1 + kw;

                                        self.grad_weights[weight_idx] += self.input_cache[input_idx] * grad;
                                        grad_input[input_idx] += self.weights[weight_idx] * grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&grad_input, &self.input_shape).unwrap()
    }

    pub fn update(&mut self, lr: f32) {
        for (w, g) in self.weights.iter_mut().zip(self.grad_weights.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.bias.iter_mut().zip(self.grad_bias.iter()) {
            *b -= lr * g;
        }
    }
}

/// 2D Max Pooling Layer
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    max_indices: Vec<usize>,
    input_shape: Vec<usize>,
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        MaxPool2d {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
            max_indices: Vec::new(),
            input_shape: Vec::new(),
        }
    }

    pub fn stride(mut self, s: (usize, usize)) -> Self {
        self.stride = s;
        self
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        self.input_shape = input.dims().to_vec();

        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];

        let out_h = (in_h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = vec![0.0f32; batch_size * channels * out_h * out_w];
        self.max_indices = vec![0usize; batch_size * channels * out_h * out_w];

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * self.stride.0 + kh;
                                let iw = ow * self.stride.1 + kw;

                                if ih < in_h && iw < in_w {
                                    let input_idx = b * channels * in_h * in_w +
                                                   c * in_h * in_w +
                                                   ih * in_w + iw;
                                    if input_data[input_idx] > max_val {
                                        max_val = input_data[input_idx];
                                        max_idx = input_idx;
                                    }
                                }
                            }
                        }

                        let output_idx = b * channels * out_h * out_w +
                                        c * out_h * out_w +
                                        oh * out_w + ow;
                        output[output_idx] = max_val;
                        self.max_indices[output_idx] = max_idx;
                    }
                }
            }
        }

        Tensor::from_slice(&output, &[batch_size, channels, out_h, out_w]).unwrap()
    }

    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let mut grad_input = vec![0.0f32; self.input_shape.iter().product()];

        for (i, &grad) in grad_data.iter().enumerate() {
            grad_input[self.max_indices[i]] += grad;
        }

        Tensor::from_slice(&grad_input, &self.input_shape).unwrap()
    }
}

/// 2D Average Pooling Layer
pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    input_shape: Vec<usize>,
}

impl AvgPool2d {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        AvgPool2d {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
            input_shape: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        self.input_shape = input.dims().to_vec();

        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];

        let out_h = (in_h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let pool_size = (self.kernel_size.0 * self.kernel_size.1) as f32;
        let mut output = vec![0.0f32; batch_size * channels * out_h * out_w];

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;

                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * self.stride.0 + kh;
                                let iw = ow * self.stride.1 + kw;

                                if ih < in_h && iw < in_w {
                                    let input_idx = b * channels * in_h * in_w +
                                                   c * in_h * in_w +
                                                   ih * in_w + iw;
                                    sum += input_data[input_idx];
                                }
                            }
                        }

                        let output_idx = b * channels * out_h * out_w +
                                        c * out_h * out_w +
                                        oh * out_w + ow;
                        output[output_idx] = sum / pool_size;
                    }
                }
            }
        }

        Tensor::from_slice(&output, &[batch_size, channels, out_h, out_w]).unwrap()
    }

    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let grad_shape = grad_output.dims();

        let batch_size = grad_shape[0];
        let channels = grad_shape[1];
        let out_h = grad_shape[2];
        let out_w = grad_shape[3];
        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];

        let pool_size = (self.kernel_size.0 * self.kernel_size.1) as f32;
        let mut grad_input = vec![0.0f32; self.input_shape.iter().product()];

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grad_idx = b * channels * out_h * out_w +
                                      c * out_h * out_w +
                                      oh * out_w + ow;
                        let grad = grad_data[grad_idx] / pool_size;

                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * self.stride.0 + kh;
                                let iw = ow * self.stride.1 + kw;

                                if ih < in_h && iw < in_w {
                                    let input_idx = b * channels * in_h * in_w +
                                                   c * in_h * in_w +
                                                   ih * in_w + iw;
                                    grad_input[input_idx] += grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&grad_input, &self.input_shape).unwrap()
    }
}

/// Global Average Pooling 2D
pub struct GlobalAvgPool2d {
    input_shape: Vec<usize>,
}

impl GlobalAvgPool2d {
    pub fn new() -> Self {
        GlobalAvgPool2d { input_shape: Vec::new() }
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        self.input_shape = input.dims().to_vec();

        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let spatial_size = self.input_shape[2] * self.input_shape[3];

        let mut output = vec![0.0f32; batch_size * channels];

        for b in 0..batch_size {
            for c in 0..channels {
                let start = b * channels * spatial_size + c * spatial_size;
                let sum: f32 = input_data[start..start + spatial_size].iter().sum();
                output[b * channels + c] = sum / spatial_size as f32;
            }
        }

        Tensor::from_slice(&output, &[batch_size, channels]).unwrap()
    }

    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let spatial_size = self.input_shape[2] * self.input_shape[3];

        let mut grad_input = vec![0.0f32; self.input_shape.iter().product()];

        for b in 0..batch_size {
            for c in 0..channels {
                let grad = grad_data[b * channels + c] / spatial_size as f32;
                let start = b * channels * spatial_size + c * spatial_size;
                for i in 0..spatial_size {
                    grad_input[start + i] = grad;
                }
            }
        }

        Tensor::from_slice(&grad_input, &self.input_shape).unwrap()
    }
}

impl Default for GlobalAvgPool2d {
    fn default() -> Self { Self::new() }
}

/// 1D Convolution Layer
pub struct Conv1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    grad_weights: Vec<f32>,
    grad_bias: Vec<f32>,
    input_cache: Vec<f32>,
    input_shape: Vec<usize>,
}

impl Conv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut rng = thread_rng();
        let fan_in = in_channels * kernel_size;
        let scale = (2.0 / fan_in as f32).sqrt();
        
        let weight_size = out_channels * in_channels * kernel_size;
        let weights: Vec<f32> = (0..weight_size)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();

        Conv1d {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            weights,
            bias: vec![0.0; out_channels],
            grad_weights: vec![0.0; weight_size],
            grad_bias: vec![0.0; out_channels],
            input_cache: Vec::new(),
            input_shape: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        self.input_shape = input.dims().to_vec();
        self.input_cache = input_data.clone();

        let batch_size = self.input_shape[0];
        let in_len = self.input_shape[2];
        let out_len = (in_len + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = vec![0.0f32; batch_size * self.out_channels * out_len];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_len {
                    let mut sum = self.bias[oc];

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k;
                            let il_pad = il as i32 - self.padding as i32;

                            if il_pad >= 0 && il_pad < in_len as i32 {
                                let input_idx = b * self.in_channels * in_len +
                                               ic * in_len + il_pad as usize;
                                let weight_idx = oc * self.in_channels * self.kernel_size +
                                                ic * self.kernel_size + k;
                                sum += input_data[input_idx] * self.weights[weight_idx];
                            }
                        }
                    }

                    let output_idx = b * self.out_channels * out_len + oc * out_len + ol;
                    output[output_idx] = sum;
                }
            }
        }

        Tensor::from_slice(&output, &[batch_size, self.out_channels, out_len]).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let grad_shape = grad_output.dims();

        let batch_size = grad_shape[0];
        let out_len = grad_shape[2];
        let in_len = self.input_shape[2];

        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
        let mut grad_input = vec![0.0f32; self.input_cache.len()];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_len {
                    let grad_idx = b * self.out_channels * out_len + oc * out_len + ol;
                    let grad = grad_data[grad_idx];

                    self.grad_bias[oc] += grad;

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k;
                            let il_pad = il as i32 - self.padding as i32;

                            if il_pad >= 0 && il_pad < in_len as i32 {
                                let input_idx = b * self.in_channels * in_len +
                                               ic * in_len + il_pad as usize;
                                let weight_idx = oc * self.in_channels * self.kernel_size +
                                                ic * self.kernel_size + k;

                                self.grad_weights[weight_idx] += self.input_cache[input_idx] * grad;
                                grad_input[input_idx] += self.weights[weight_idx] * grad;
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&grad_input, &self.input_shape).unwrap()
    }

    pub fn update(&mut self, lr: f32) {
        for (w, g) in self.weights.iter_mut().zip(self.grad_weights.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.bias.iter_mut().zip(self.grad_bias.iter()) {
            *b -= lr * g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d() {
        let x = Tensor::from_slice(&vec![1.0f32; 2 * 3 * 4 * 4], &[2, 3, 4, 4]).unwrap();
        let mut conv = Conv2d::new(3, 6, (3, 3));
        let out = conv.forward(&x, true);
        assert_eq!(out.dims(), &[2, 6, 2, 2]);
    }

    #[test]
    fn test_maxpool2d() {
        let x = Tensor::from_slice(&vec![1.0f32; 2 * 3 * 4 * 4], &[2, 3, 4, 4]).unwrap();
        let mut pool = MaxPool2d::new((2, 2));
        let out = pool.forward(&x, true);
        assert_eq!(out.dims(), &[2, 3, 2, 2]);
    }
}



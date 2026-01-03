//! Neural Network Layers - Dense, Dropout, BatchNorm, etc.

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Layer trait for neural network components
pub trait Layer: Send + Sync {
    fn forward(&mut self, input: &Tensor, training: bool) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Vec<f32>>;
    fn gradients(&self) -> Vec<&Vec<f32>>;
    fn update_parameters(&mut self, lr: f32);
}

/// Dense (Fully Connected) Layer
pub struct Dense {
    pub in_features: usize,
    pub out_features: usize,
    pub use_bias: bool,
    weights: Vec<f32>,
    bias: Vec<f32>,
    grad_weights: Vec<f32>,
    grad_bias: Vec<f32>,
    input_cache: Vec<f32>,
}

impl Dense {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (2.0 / in_features as f32).sqrt();
        
        let weights: Vec<f32> = (0..in_features * out_features)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let bias = vec![0.0; out_features];

        Dense {
            in_features,
            out_features,
            use_bias: true,
            weights,
            bias,
            grad_weights: vec![0.0; in_features * out_features],
            grad_bias: vec![0.0; out_features],
            input_cache: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        let batch_size = input.dims()[0];
        
        self.input_cache = input_data.clone();

        let mut output = vec![0.0f32; batch_size * self.out_features];

        for b in 0..batch_size {
            for j in 0..self.out_features {
                let mut sum = if self.use_bias { self.bias[j] } else { 0.0 };
                for i in 0..self.in_features {
                    sum += input_data[b * self.in_features + i] * self.weights[i * self.out_features + j];
                }
                output[b * self.out_features + j] = sum;
            }
        }

        Tensor::from_slice(&output, &[batch_size, self.out_features]).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let batch_size = grad_output.dims()[0];

        // Reset gradients
        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);

        let mut grad_input = vec![0.0f32; batch_size * self.in_features];

        for b in 0..batch_size {
            for j in 0..self.out_features {
                let grad = grad_data[b * self.out_features + j];
                
                // Gradient w.r.t. bias
                self.grad_bias[j] += grad;

                for i in 0..self.in_features {
                    // Gradient w.r.t. weights
                    self.grad_weights[i * self.out_features + j] += 
                        self.input_cache[b * self.in_features + i] * grad;
                    
                    // Gradient w.r.t. input
                    grad_input[b * self.in_features + i] += 
                        self.weights[i * self.out_features + j] * grad;
                }
            }
        }

        // Average gradients over batch
        let batch_f = batch_size as f32;
        for g in &mut self.grad_weights { *g /= batch_f; }
        for g in &mut self.grad_bias { *g /= batch_f; }

        Tensor::from_slice(&grad_input, &[batch_size, self.in_features]).unwrap()
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

/// Dropout Layer
pub struct Dropout {
    pub rate: f32,
    mask: Vec<f32>,
}

impl Dropout {
    pub fn new(rate: f32) -> Self {
        Dropout {
            rate: rate.clamp(0.0, 1.0),
            mask: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor, training: bool) -> Tensor {
        let input_data = input.data_f32();
        
        if !training || self.rate == 0.0 {
            return input.clone();
        }

        let mut rng = thread_rng();
        let scale = 1.0 / (1.0 - self.rate);
        
        self.mask = input_data.iter()
            .map(|_| if rng.gen::<f32>() > self.rate { scale } else { 0.0 })
            .collect();

        let output: Vec<f32> = input_data.iter()
            .zip(self.mask.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        Tensor::from_slice(&output, input.dims()).unwrap()
    }

    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        
        let grad_input: Vec<f32> = grad_data.iter()
            .zip(self.mask.iter())
            .map(|(&g, &m)| g * m)
            .collect();

        Tensor::from_slice(&grad_input, grad_output.dims()).unwrap()
    }
}

/// Batch Normalization Layer
pub struct BatchNorm {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    grad_gamma: Vec<f32>,
    grad_beta: Vec<f32>,
    // Cache for backward
    x_norm: Vec<f32>,
    std_inv: Vec<f32>,
    batch_mean: Vec<f32>,
}

impl BatchNorm {
    pub fn new(num_features: usize) -> Self {
        BatchNorm {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            grad_gamma: vec![0.0; num_features],
            grad_beta: vec![0.0; num_features],
            x_norm: Vec::new(),
            std_inv: Vec::new(),
            batch_mean: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor, training: bool) -> Tensor {
        let input_data = input.data_f32();
        let batch_size = input.dims()[0];

        let (mean, var) = if training {
            // Compute batch statistics
            let mean: Vec<f32> = (0..self.num_features)
                .map(|j| {
                    (0..batch_size).map(|i| input_data[i * self.num_features + j]).sum::<f32>() 
                        / batch_size as f32
                })
                .collect();

            let var: Vec<f32> = (0..self.num_features)
                .map(|j| {
                    (0..batch_size)
                        .map(|i| (input_data[i * self.num_features + j] - mean[j]).powi(2))
                        .sum::<f32>() / batch_size as f32
                })
                .collect();

            // Update running statistics
            for j in 0..self.num_features {
                self.running_mean[j] = (1.0 - self.momentum) * self.running_mean[j] + self.momentum * mean[j];
                self.running_var[j] = (1.0 - self.momentum) * self.running_var[j] + self.momentum * var[j];
            }

            self.batch_mean = mean.clone();
            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        // Normalize
        self.std_inv = var.iter().map(|&v| 1.0 / (v + self.eps).sqrt()).collect();
        
        let mut output = vec![0.0f32; batch_size * self.num_features];
        self.x_norm = vec![0.0f32; batch_size * self.num_features];

        for i in 0..batch_size {
            for j in 0..self.num_features {
                let idx = i * self.num_features + j;
                self.x_norm[idx] = (input_data[idx] - mean[j]) * self.std_inv[j];
                output[idx] = self.gamma[j] * self.x_norm[idx] + self.beta[j];
            }
        }

        Tensor::from_slice(&output, &[batch_size, self.num_features]).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let batch_size = grad_output.dims()[0];
        let n = batch_size as f32;

        // Gradients w.r.t. gamma and beta
        self.grad_gamma.fill(0.0);
        self.grad_beta.fill(0.0);

        for i in 0..batch_size {
            for j in 0..self.num_features {
                let idx = i * self.num_features + j;
                self.grad_gamma[j] += grad_data[idx] * self.x_norm[idx];
                self.grad_beta[j] += grad_data[idx];
            }
        }

        // Gradient w.r.t. input
        let mut grad_input = vec![0.0f32; batch_size * self.num_features];

        for j in 0..self.num_features {
            let mut dx_norm_sum = 0.0f32;
            let mut dx_norm_x_sum = 0.0f32;

            for i in 0..batch_size {
                let idx = i * self.num_features + j;
                let dx_norm = grad_data[idx] * self.gamma[j];
                dx_norm_sum += dx_norm;
                dx_norm_x_sum += dx_norm * self.x_norm[idx];
            }

            for i in 0..batch_size {
                let idx = i * self.num_features + j;
                let dx_norm = grad_data[idx] * self.gamma[j];
                grad_input[idx] = self.std_inv[j] * (dx_norm - dx_norm_sum / n - self.x_norm[idx] * dx_norm_x_sum / n);
            }
        }

        Tensor::from_slice(&grad_input, &[batch_size, self.num_features]).unwrap()
    }

    pub fn update(&mut self, lr: f32) {
        for (g, grad) in self.gamma.iter_mut().zip(self.grad_gamma.iter()) {
            *g -= lr * grad;
        }
        for (b, grad) in self.beta.iter_mut().zip(self.grad_beta.iter()) {
            *b -= lr * grad;
        }
    }
}

/// Layer Normalization
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    grad_gamma: Vec<f32>,
    grad_beta: Vec<f32>,
    x_norm: Vec<f32>,
    std_inv: Vec<f32>,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let size: usize = normalized_shape.iter().product();
        LayerNorm {
            normalized_shape,
            eps: 1e-5,
            gamma: vec![1.0; size],
            beta: vec![0.0; size],
            grad_gamma: vec![0.0; size],
            grad_beta: vec![0.0; size],
            x_norm: Vec::new(),
            std_inv: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        let dims = input.dims();
        let batch_size = dims[0];
        let feature_size: usize = self.normalized_shape.iter().product();

        let mut output = vec![0.0f32; input_data.len()];
        self.x_norm = vec![0.0f32; input_data.len()];
        self.std_inv = vec![0.0f32; batch_size];

        for b in 0..batch_size {
            let start = b * feature_size;
            let end = start + feature_size;
            let slice = &input_data[start..end];

            let mean: f32 = slice.iter().sum::<f32>() / feature_size as f32;
            let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / feature_size as f32;
            let std_inv = 1.0 / (var + self.eps).sqrt();
            self.std_inv[b] = std_inv;

            for i in 0..feature_size {
                let idx = start + i;
                self.x_norm[idx] = (input_data[idx] - mean) * std_inv;
                output[idx] = self.gamma[i] * self.x_norm[idx] + self.beta[i];
            }
        }

        Tensor::from_slice(&output, dims).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        let dims = grad_output.dims();
        let batch_size = dims[0];
        let feature_size: usize = self.normalized_shape.iter().product();
        let n = feature_size as f32;

        self.grad_gamma.fill(0.0);
        self.grad_beta.fill(0.0);

        let mut grad_input = vec![0.0f32; grad_data.len()];

        for b in 0..batch_size {
            let start = b * feature_size;

            // Accumulate gradients for gamma and beta
            for i in 0..feature_size {
                let idx = start + i;
                self.grad_gamma[i] += grad_data[idx] * self.x_norm[idx];
                self.grad_beta[i] += grad_data[idx];
            }

            // Compute gradient w.r.t. input
            let mut dx_norm_sum = 0.0f32;
            let mut dx_norm_x_sum = 0.0f32;

            for i in 0..feature_size {
                let idx = start + i;
                let dx_norm = grad_data[idx] * self.gamma[i];
                dx_norm_sum += dx_norm;
                dx_norm_x_sum += dx_norm * self.x_norm[idx];
            }

            for i in 0..feature_size {
                let idx = start + i;
                let dx_norm = grad_data[idx] * self.gamma[i];
                grad_input[idx] = self.std_inv[b] * (dx_norm - dx_norm_sum / n - self.x_norm[idx] * dx_norm_x_sum / n);
            }
        }

        Tensor::from_slice(&grad_input, dims).unwrap()
    }

    pub fn update(&mut self, lr: f32) {
        for (g, grad) in self.gamma.iter_mut().zip(self.grad_gamma.iter()) {
            *g -= lr * grad;
        }
        for (b, grad) in self.beta.iter_mut().zip(self.grad_beta.iter()) {
            *b -= lr * grad;
        }
    }
}

/// Embedding Layer
pub struct Embedding {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
    weights: Vec<f32>,
    grad_weights: Vec<f32>,
    input_cache: Vec<usize>,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut rng = thread_rng();
        let weights: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| rng.gen::<f32>() - 0.5)
            .collect();

        Embedding {
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            weights,
            grad_weights: vec![0.0; num_embeddings * embedding_dim],
            input_cache: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        let input_shape = input.dims();
        
        self.input_cache = input_data.iter().map(|&x| x as usize).collect();

        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        let output: Vec<f32> = self.input_cache.iter()
            .flat_map(|&idx| {
                let start = idx * self.embedding_dim;
                self.weights[start..start + self.embedding_dim].to_vec()
            })
            .collect();

        Tensor::from_slice(&output, &output_shape).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        
        self.grad_weights.fill(0.0);

        for (i, &idx) in self.input_cache.iter().enumerate() {
            if Some(idx) == self.padding_idx {
                continue;
            }
            let start = idx * self.embedding_dim;
            let grad_start = i * self.embedding_dim;
            for j in 0..self.embedding_dim {
                self.grad_weights[start + j] += grad_data[grad_start + j];
            }
        }

        // Return zero gradient for input (indices don't have gradients)
        Tensor::from_slice(&vec![0.0f32; self.input_cache.len()], &[self.input_cache.len()]).unwrap()
    }

    pub fn update(&mut self, lr: f32) {
        for (w, g) in self.weights.iter_mut().zip(self.grad_weights.iter()) {
            *w -= lr * g;
        }
    }
}

/// Flatten Layer
pub struct Flatten {
    input_shape: Vec<usize>,
}

impl Flatten {
    pub fn new() -> Self {
        Flatten { input_shape: Vec::new() }
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        self.input_shape = input.dims().to_vec();
        
        let batch_size = self.input_shape[0];
        let flat_size: usize = self.input_shape[1..].iter().product();

        Tensor::from_slice(&input_data, &[batch_size, flat_size]).unwrap()
    }

    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        let grad_data = grad_output.data_f32();
        Tensor::from_slice(&grad_data, &self.input_shape).unwrap()
    }
}

impl Default for Flatten {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut dense = Dense::new(2, 3);
        let out = dense.forward(&x, true);
        assert_eq!(out.dims(), &[2, 3]);
    }

    #[test]
    fn test_dropout() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut dropout = Dropout::new(0.5);
        let out = dropout.forward(&x, true);
        assert_eq!(out.dims(), &[2, 2]);
    }

    #[test]
    fn test_batch_norm() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let mut bn = BatchNorm::new(2);
        let out = bn.forward(&x, true);
        assert_eq!(out.dims(), &[3, 2]);
    }
}


/// 3D Convolutional Layer
pub struct Conv3d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub use_bias: bool,
    weights: Vec<f32>,
    bias: Vec<f32>,
    grad_weights: Vec<f32>,
    grad_bias: Vec<f32>,
    input_cache: Vec<f32>,
    input_shape: Vec<usize>,
}

impl Conv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        let mut rng = thread_rng();
        let (kd, kh, kw) = kernel_size;
        let fan_in = in_channels * kd * kh * kw;
        let scale = (2.0 / fan_in as f32).sqrt();

        let weight_size = out_channels * in_channels * kd * kh * kw;
        let weights: Vec<f32> = (0..weight_size)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let bias = vec![0.0; out_channels];

        Conv3d {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            use_bias: true,
            weights,
            bias,
            grad_weights: vec![0.0; weight_size],
            grad_bias: vec![0.0; out_channels],
            input_cache: Vec::new(),
            input_shape: Vec::new(),
        }
    }

    pub fn stride(mut self, s: (usize, usize, usize)) -> Self {
        self.stride = s;
        self
    }

    pub fn padding(mut self, p: (usize, usize, usize)) -> Self {
        self.padding = p;
        self
    }

    pub fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        let input_data = input.data_f32();
        let dims = input.dims();
        
        // Input shape: [batch, in_channels, depth, height, width]
        let batch = dims[0];
        let in_d = dims[2];
        let in_h = dims[3];
        let in_w = dims[4];

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;

        // Calculate output dimensions
        let out_d = (in_d + 2 * pd - kd) / sd + 1;
        let out_h = (in_h + 2 * ph - kh) / sh + 1;
        let out_w = (in_w + 2 * pw - kw) / sw + 1;

        let output_size = batch * self.out_channels * out_d * out_h * out_w;
        let mut output = vec![0.0f32; output_size];

        // Cache input for backward pass
        self.input_cache = input_data.to_vec();
        self.input_shape = dims.to_vec();

        // Perform 3D convolution
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = if self.use_bias { self.bias[oc] } else { 0.0 };

                            for ic in 0..self.in_channels {
                                for kd_i in 0..kd {
                                    for kh_i in 0..kh {
                                        for kw_i in 0..kw {
                                            let id = od * sd + kd_i;
                                            let ih = oh * sh + kh_i;
                                            let iw = ow * sw + kw_i;

                                            if id >= pd && ih >= ph && iw >= pw {
                                                let id = id - pd;
                                                let ih = ih - ph;
                                                let iw = iw - pw;

                                                if id < in_d && ih < in_h && iw < in_w {
                                                    let input_idx = ((b * self.in_channels + ic) * in_d + id) * in_h * in_w
                                                        + ih * in_w + iw;
                                                    let weight_idx = ((oc * self.in_channels + ic) * kd + kd_i) * kh * kw
                                                        + kh_i * kw + kw_i;
                                                    sum += input_data[input_idx] * self.weights[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            let output_idx = ((b * self.out_channels + oc) * out_d + od) * out_h * out_w
                                + oh * out_w + ow;
                            output[output_idx] = sum;
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&output, &[batch, self.out_channels, out_d, out_h, out_w]).unwrap()
    }

    pub fn parameters(&self) -> Vec<&Vec<f32>> {
        if self.use_bias {
            vec![&self.weights, &self.bias]
        } else {
            vec![&self.weights]
        }
    }

    pub fn gradients(&self) -> Vec<&Vec<f32>> {
        if self.use_bias {
            vec![&self.grad_weights, &self.grad_bias]
        } else {
            vec![&self.grad_weights]
        }
    }
}

/// 2D Batch Normalization Layer
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    grad_gamma: Vec<f32>,
    grad_beta: Vec<f32>,
    // Cache for backward pass
    input_cache: Vec<f32>,
    normalized_cache: Vec<f32>,
    std_cache: Vec<f32>,
    batch_size: usize,
    spatial_size: usize,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        BatchNorm2d {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            grad_gamma: vec![0.0; num_features],
            grad_beta: vec![0.0; num_features],
            input_cache: Vec::new(),
            normalized_cache: Vec::new(),
            std_cache: Vec::new(),
            batch_size: 0,
            spatial_size: 0,
        }
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    pub fn forward(&mut self, input: &Tensor, training: bool) -> Tensor {
        let input_data = input.data_f32();
        let dims = input.dims();
        
        // Input shape: [batch, channels, height, width]
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let spatial_size = height * width;

        assert_eq!(channels, self.num_features, "Input channels must match num_features");

        let mut output = vec![0.0f32; input_data.len()];

        if training {
            // Compute batch statistics
            let mut batch_mean = vec![0.0f32; channels];
            let mut batch_var = vec![0.0f32; channels];

            // Calculate mean
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch {
                    for s in 0..spatial_size {
                        let idx = (b * channels + c) * spatial_size + s;
                        sum += input_data[idx];
                    }
                }
                batch_mean[c] = sum / (batch * spatial_size) as f32;
            }

            // Calculate variance
            for c in 0..channels {
                let mut sum_sq = 0.0f32;
                for b in 0..batch {
                    for s in 0..spatial_size {
                        let idx = (b * channels + c) * spatial_size + s;
                        let diff = input_data[idx] - batch_mean[c];
                        sum_sq += diff * diff;
                    }
                }
                batch_var[c] = sum_sq / (batch * spatial_size) as f32;
            }

            // Update running statistics
            for c in 0..channels {
                self.running_mean[c] = (1.0 - self.momentum) * self.running_mean[c] 
                    + self.momentum * batch_mean[c];
                self.running_var[c] = (1.0 - self.momentum) * self.running_var[c] 
                    + self.momentum * batch_var[c];
            }

            // Normalize and scale
            let mut normalized = vec![0.0f32; input_data.len()];
            let mut std = vec![0.0f32; channels];

            for c in 0..channels {
                std[c] = (batch_var[c] + self.eps).sqrt();
            }

            for c in 0..channels {
                for b in 0..batch {
                    for s in 0..spatial_size {
                        let idx = (b * channels + c) * spatial_size + s;
                        normalized[idx] = (input_data[idx] - batch_mean[c]) / std[c];
                        
                        if self.affine {
                            output[idx] = self.gamma[c] * normalized[idx] + self.beta[c];
                        } else {
                            output[idx] = normalized[idx];
                        }
                    }
                }
            }

            // Cache for backward
            self.input_cache = input_data.to_vec();
            self.normalized_cache = normalized;
            self.std_cache = std;
            self.batch_size = batch;
            self.spatial_size = spatial_size;

        } else {
            // Use running statistics for inference
            for c in 0..channels {
                let std = (self.running_var[c] + self.eps).sqrt();
                for b in 0..batch {
                    for s in 0..spatial_size {
                        let idx = (b * channels + c) * spatial_size + s;
                        let normalized = (input_data[idx] - self.running_mean[c]) / std;
                        
                        if self.affine {
                            output[idx] = self.gamma[c] * normalized + self.beta[c];
                        } else {
                            output[idx] = normalized;
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&output, dims).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let grad_output_data = grad_output.data_f32();
        let dims = grad_output.dims();
        let channels = dims[1];
        let batch = self.batch_size;
        let spatial_size = self.spatial_size;
        let n = (batch * spatial_size) as f32;

        let mut grad_input = vec![0.0f32; grad_output_data.len()];

        // Reset gradients
        self.grad_gamma.fill(0.0);
        self.grad_beta.fill(0.0);

        // Compute gradients for gamma and beta
        if self.affine {
            for c in 0..channels {
                for b in 0..batch {
                    for s in 0..spatial_size {
                        let idx = (b * channels + c) * spatial_size + s;
                        self.grad_gamma[c] += grad_output_data[idx] * self.normalized_cache[idx];
                        self.grad_beta[c] += grad_output_data[idx];
                    }
                }
            }
        }

        // Compute gradient w.r.t. input
        for c in 0..channels {
            let std = self.std_cache[c];
            
            // Compute intermediate gradients
            let mut grad_var = 0.0f32;
            let mut grad_mean = 0.0f32;

            for b in 0..batch {
                for s in 0..spatial_size {
                    let idx = (b * channels + c) * spatial_size + s;
                    let grad_out = if self.affine {
                        grad_output_data[idx] * self.gamma[c]
                    } else {
                        grad_output_data[idx]
                    };

                    grad_var += grad_out * self.normalized_cache[idx] * (-0.5) / std;
                }
            }

            for b in 0..batch {
                for s in 0..spatial_size {
                    let idx = (b * channels + c) * spatial_size + s;
                    let grad_out = if self.affine {
                        grad_output_data[idx] * self.gamma[c]
                    } else {
                        grad_output_data[idx]
                    };

                    grad_mean += grad_out * (-1.0 / std) 
                        + grad_var * (-2.0 * self.normalized_cache[idx] * std / n);
                }
            }

            // Compute gradient w.r.t. input
            for b in 0..batch {
                for s in 0..spatial_size {
                    let idx = (b * channels + c) * spatial_size + s;
                    let grad_out = if self.affine {
                        grad_output_data[idx] * self.gamma[c]
                    } else {
                        grad_output_data[idx]
                    };

                    grad_input[idx] = grad_out / std 
                        + grad_var * 2.0 * self.normalized_cache[idx] * std / n
                        + grad_mean / n;
                }
            }
        }

        Tensor::from_slice(&grad_input, dims).unwrap()
    }

    pub fn parameters(&self) -> Vec<&Vec<f32>> {
        if self.affine {
            vec![&self.gamma, &self.beta]
        } else {
            vec![]
        }
    }

    pub fn gradients(&self) -> Vec<&Vec<f32>> {
        if self.affine {
            vec![&self.grad_gamma, &self.grad_beta]
        } else {
            vec![]
        }
    }

    pub fn update_parameters(&mut self, lr: f32) {
        if self.affine {
            for i in 0..self.num_features {
                self.gamma[i] -= lr * self.grad_gamma[i];
                self.beta[i] -= lr * self.grad_beta[i];
            }
        }
    }
}

/// Layer Normalization

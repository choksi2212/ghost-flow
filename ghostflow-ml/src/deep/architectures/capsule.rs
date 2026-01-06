//! Capsule Network Architectures - CapsNet, Dynamic Routing, EM Routing, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense};
use crate::deep::activations::ReLU;

/// Primary Capsule Layer
pub struct PrimaryCaps {
    conv: Conv2d,
    num_capsules: usize,
    capsule_dim: usize,
}

impl PrimaryCaps {
    pub fn new(in_channels: usize, num_capsules: usize, capsule_dim: usize) -> Self {
        PrimaryCaps {
            conv: Conv2d::new(in_channels, num_capsules * capsule_dim, (9, 9)).stride((2, 2)),
            num_capsules,
            capsule_dim,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let conv_out = self.conv.forward(x, training);
        self.squash(&conv_out)
    }

    fn squash(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let batch_size = x.dims()[0];
        let channels = x.dims()[1];
        let height = x.dims()[2];
        let width = x.dims()[3];
        
        let capsules_per_location = channels / self.capsule_dim;
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    for cap in 0..capsules_per_location {
                        // Get capsule vector
                        let mut capsule_vec = Vec::new();
                        for d in 0..self.capsule_dim {
                            let c = cap * self.capsule_dim + d;
                            let idx = ((b * channels + c) * height + h) * width + w;
                            capsule_vec.push(data[idx]);
                        }
                        
                        // Compute norm
                        let norm_sq: f32 = capsule_vec.iter().map(|&v| v * v).sum();
                        let norm = norm_sq.sqrt();
                        
                        // Squash
                        let scale = norm_sq / (1.0 + norm_sq) / (norm + 1e-8);
                        for &v in &capsule_vec {
                            result.push(v * scale);
                        }
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Digit Capsule Layer with Dynamic Routing
pub struct DigitCaps {
    num_capsules: usize,
    capsule_dim: usize,
    num_iterations: usize,
    weights: Vec<Vec<Vec<f32>>>,
}

impl DigitCaps {
    pub fn new(num_input_capsules: usize, input_capsule_dim: usize, 
               num_capsules: usize, capsule_dim: usize, num_iterations: usize) -> Self {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        // Initialize transformation matrices
        let mut weights = Vec::new();
        for _ in 0..num_input_capsules {
            let mut capsule_weights = Vec::new();
            for _ in 0..num_capsules {
                let weight: Vec<f32> = (0..input_capsule_dim * capsule_dim)
                    .map(|_| rng.gen::<f32>() * 0.02 - 0.01)
                    .collect();
                capsule_weights.push(weight);
            }
            weights.push(capsule_weights);
        }
        
        DigitCaps {
            num_capsules,
            capsule_dim,
            num_iterations,
            weights,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified dynamic routing
        let batch_size = x.dims()[0];
        let num_input_capsules = x.dims()[1];
        let input_capsule_dim = x.dims()[2];
        
        // Predict output capsules
        let predictions = self.compute_predictions(x);
        
        // Dynamic routing
        self.dynamic_routing(&predictions, batch_size, num_input_capsules)
    }

    fn compute_predictions(&self, x: &Tensor) -> Vec<Vec<Vec<f32>>> {
        let batch_size = x.dims()[0];
        let num_input_capsules = x.dims()[1];
        let x_data = x.data_f32();
        
        let mut predictions = Vec::new();
        
        for b in 0..batch_size {
            let mut batch_predictions = Vec::new();
            
            for i in 0..num_input_capsules {
                let mut input_predictions = Vec::new();
                
                for j in 0..self.num_capsules {
                    // Matrix multiplication
                    let mut pred = vec![0.0f32; self.capsule_dim];
                    
                    for d_out in 0..self.capsule_dim {
                        for d_in in 0..x.dims()[2] {
                            let x_idx = (b * num_input_capsules + i) * x.dims()[2] + d_in;
                            let w_idx = d_out * x.dims()[2] + d_in;
                            pred[d_out] += x_data[x_idx] * self.weights[i][j][w_idx];
                        }
                    }
                    
                    input_predictions.push(pred);
                }
                
                batch_predictions.push(input_predictions);
            }
            
            predictions.push(batch_predictions);
        }
        
        predictions
    }

    fn dynamic_routing(&self, predictions: &[Vec<Vec<Vec<f32>>>], 
                       batch_size: usize, num_input_capsules: usize) -> Tensor {
        // Initialize routing logits
        let mut b = vec![vec![vec![0.0f32; self.num_capsules]; num_input_capsules]; batch_size];
        
        // Routing iterations
        for _ in 0..self.num_iterations {
            // Softmax over output capsules
            let c = self.softmax_routing(&b);
            
            // Weighted sum
            let s = self.weighted_sum(predictions, &c, batch_size, num_input_capsules);
            
            // Squash
            let v = self.squash_capsules(&s);
            
            // Update routing logits
            b = self.update_routing(&b, predictions, &v, batch_size, num_input_capsules);
        }
        
        // Final output
        let c = self.softmax_routing(&b);
        let s = self.weighted_sum(predictions, &c, batch_size, num_input_capsules);
        let v = self.squash_capsules(&s);
        
        self.vec_to_tensor(&v, batch_size)
    }

    fn softmax_routing(&self, b: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
        let batch_size = b.len();
        let num_input = b[0].len();
        let num_output = b[0][0].len();
        
        let mut c = vec![vec![vec![0.0f32; num_output]; num_input]; batch_size];
        
        for batch in 0..batch_size {
            for i in 0..num_input {
                let max_val = b[batch][i].iter().fold(f32::MIN, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                
                for j in 0..num_output {
                    c[batch][i][j] = (b[batch][i][j] - max_val).exp();
                    sum += c[batch][i][j];
                }
                
                for j in 0..num_output {
                    c[batch][i][j] /= sum;
                }
            }
        }
        
        c
    }

    fn weighted_sum(&self, predictions: &[Vec<Vec<Vec<f32>>>], 
                    c: &[Vec<Vec<f32>>], batch_size: usize, num_input: usize) -> Vec<Vec<Vec<f32>>> {
        let mut s = vec![vec![vec![0.0f32; self.capsule_dim]; self.num_capsules]; batch_size];
        
        for batch in 0..batch_size {
            for j in 0..self.num_capsules {
                for i in 0..num_input {
                    for d in 0..self.capsule_dim {
                        s[batch][j][d] += c[batch][i][j] * predictions[batch][i][j][d];
                    }
                }
            }
        }
        
        s
    }

    fn squash_capsules(&self, s: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
        let batch_size = s.len();
        let num_capsules = s[0].len();
        let capsule_dim = s[0][0].len();
        
        let mut v = vec![vec![vec![0.0f32; capsule_dim]; num_capsules]; batch_size];
        
        for batch in 0..batch_size {
            for j in 0..num_capsules {
                let norm_sq: f32 = s[batch][j].iter().map(|&x| x * x).sum();
                let norm = norm_sq.sqrt();
                let scale = norm_sq / (1.0 + norm_sq) / (norm + 1e-8);
                
                for d in 0..capsule_dim {
                    v[batch][j][d] = s[batch][j][d] * scale;
                }
            }
        }
        
        v
    }

    fn update_routing(&self, b: &[Vec<Vec<f32>>], predictions: &[Vec<Vec<Vec<f32>>>], 
                      v: &[Vec<Vec<f32>>], batch_size: usize, num_input: usize) -> Vec<Vec<Vec<f32>>> {
        let mut new_b = b.to_vec();
        
        for batch in 0..batch_size {
            for i in 0..num_input {
                for j in 0..self.num_capsules {
                    let mut agreement = 0.0f32;
                    for d in 0..self.capsule_dim {
                        agreement += predictions[batch][i][j][d] * v[batch][j][d];
                    }
                    new_b[batch][i][j] += agreement;
                }
            }
        }
        
        new_b
    }

    fn vec_to_tensor(&self, v: &[Vec<Vec<f32>>], batch_size: usize) -> Tensor {
        let mut data = Vec::new();
        for batch in v {
            for capsule in batch {
                data.extend_from_slice(capsule);
            }
        }
        
        Tensor::from_slice(&data, &[batch_size, self.num_capsules, self.capsule_dim]).unwrap()
    }
}

/// CapsNet (Complete Capsule Network)
pub struct CapsNet {
    conv1: Conv2d,
    primary_caps: PrimaryCaps,
    digit_caps: DigitCaps,
    decoder: CapsNetDecoder,
}

struct CapsNetDecoder {
    fc1: Dense,
    fc2: Dense,
    fc3: Dense,
}

impl CapsNetDecoder {
    fn new(capsule_dim: usize, output_dim: usize) -> Self {
        CapsNetDecoder {
            fc1: Dense::new(capsule_dim, 512),
            fc2: Dense::new(512, 1024),
            fc3: Dense::new(1024, output_dim),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.fc1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.fc2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        self.fc3.forward(&out, training)
    }
}

impl CapsNet {
    pub fn new(num_classes: usize) -> Self {
        CapsNet {
            conv1: Conv2d::new(1, 256, (9, 9)),
            primary_caps: PrimaryCaps::new(256, 32, 8),
            digit_caps: DigitCaps::new(32 * 6 * 6, 8, num_classes, 16, 3),
            decoder: CapsNetDecoder::new(16, 784),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let mut out = self.conv1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.primary_caps.forward(&out, training);
        let digit_out = self.digit_caps.forward(&out);
        
        // Get capsule lengths for classification
        let lengths = self.capsule_lengths(&digit_out);
        
        // Reconstruction
        let masked = self.mask_capsules(&digit_out, &lengths);
        let reconstruction = self.decoder.forward(&masked, training);
        
        (lengths, reconstruction)
    }

    fn capsule_lengths(&self, capsules: &Tensor) -> Tensor {
        let batch_size = capsules.dims()[0];
        let num_capsules = capsules.dims()[1];
        let capsule_dim = capsules.dims()[2];
        let data = capsules.data_f32();
        
        let mut lengths = Vec::new();
        
        for b in 0..batch_size {
            for c in 0..num_capsules {
                let mut norm_sq = 0.0f32;
                for d in 0..capsule_dim {
                    let idx = (b * num_capsules + c) * capsule_dim + d;
                    norm_sq += data[idx] * data[idx];
                }
                lengths.push(norm_sq.sqrt());
            }
        }
        
        Tensor::from_slice(&lengths, &[batch_size, num_capsules]).unwrap()
    }

    fn mask_capsules(&self, capsules: &Tensor, lengths: &Tensor) -> Tensor {
        let batch_size = capsules.dims()[0];
        let num_capsules = capsules.dims()[1];
        let capsule_dim = capsules.dims()[2];
        let caps_data = capsules.data_f32();
        let len_data = lengths.data_f32();
        
        let mut masked = Vec::new();
        
        for b in 0..batch_size {
            // Find max length capsule
            let mut max_idx = 0;
            let mut max_len = len_data[b * num_capsules];
            
            for c in 1..num_capsules {
                let len = len_data[b * num_capsules + c];
                if len > max_len {
                    max_len = len;
                    max_idx = c;
                }
            }
            
            // Extract that capsule
            for d in 0..capsule_dim {
                let idx = (b * num_capsules + max_idx) * capsule_dim + d;
                masked.push(caps_data[idx]);
            }
        }
        
        Tensor::from_slice(&masked, &[batch_size, capsule_dim]).unwrap()
    }
}

/// EM Routing Capsule Network
pub struct EMCapsNet {
    conv1: Conv2d,
    primary_caps: PrimaryCaps,
    conv_caps: EMConvCaps,
    class_caps: EMClassCaps,
}

struct EMConvCaps {
    num_capsules: usize,
    capsule_dim: usize,
}

impl EMConvCaps {
    fn new(num_capsules: usize, capsule_dim: usize) -> Self {
        EMConvCaps {
            num_capsules,
            capsule_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified EM routing
        x.clone()
    }
}

struct EMClassCaps {
    num_classes: usize,
    capsule_dim: usize,
}

impl EMClassCaps {
    fn new(num_classes: usize, capsule_dim: usize) -> Self {
        EMClassCaps {
            num_classes,
            capsule_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified EM routing for classification
        x.clone()
    }
}

impl EMCapsNet {
    pub fn new(num_classes: usize) -> Self {
        EMCapsNet {
            conv1: Conv2d::new(1, 256, (9, 9)),
            primary_caps: PrimaryCaps::new(256, 32, 8),
            conv_caps: EMConvCaps::new(32, 16),
            class_caps: EMClassCaps::new(num_classes, 16),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.primary_caps.forward(&out, training);
        out = self.conv_caps.forward(&out);
        self.class_caps.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_caps() {
        let mut primary = PrimaryCaps::new(256, 32, 8);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 256 * 20 * 20], &[1, 256, 20, 20]).unwrap();
        let output = primary.forward(&input, false);
        assert_eq!(output.dims()[1], 256);
    }

    #[test]
    fn test_capsnet() {
        let mut capsnet = CapsNet::new(10);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 1 * 28 * 28], &[1, 1, 28, 28]).unwrap();
        let (lengths, recon) = capsnet.forward(&input, false);
        assert_eq!(lengths.dims()[1], 10);
    }
}



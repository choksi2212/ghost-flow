//! Feed-forward Neural Networks and Variants - MLP, Deep FFN, Highway Networks, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::Dense;
use crate::deep::activations::{ReLU, Sigmoid, Tanh};

/// Multi-Layer Perceptron (MLP)
pub struct MLP {
    layers: Vec<Dense>,
    num_layers: usize,
}

impl MLP {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Dense::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        
        MLP {
            num_layers: layers.len(),
            layers,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            out = layer.forward(&out, training);
            
            // Apply activation to all layers except the last
            if i < self.num_layers - 1 {
                out = ReLU::new().forward(&out);
            }
        }
        
        out
    }
}

/// Deep Feed-Forward Network with Dropout
pub struct DeepFFN {
    layers: Vec<Dense>,
    dropout_rate: f32,
}

impl DeepFFN {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize, dropout_rate: f32) -> Self {
        let mut layers = Vec::new();
        
        // Input to first hidden
        layers.push(Dense::new(input_dim, hidden_dims[0]));
        
        // Hidden to hidden
        for i in 0..hidden_dims.len() - 1 {
            layers.push(Dense::new(hidden_dims[i], hidden_dims[i + 1]));
        }
        
        // Last hidden to output
        layers.push(Dense::new(hidden_dims[hidden_dims.len() - 1], output_dim));
        
        DeepFFN {
            layers,
            dropout_rate,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            out = layer.forward(&out, training);
            
            if i < self.layers.len() - 1 {
                out = ReLU::new().forward(&out);
                
                if training {
                    out = self.dropout(&out);
                }
            }
        }
        
        out
    }

    fn dropout(&self, x: &Tensor) -> Tensor {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| {
                if rng.gen::<f32>() < self.dropout_rate {
                    0.0
                } else {
                    val / (1.0 - self.dropout_rate)
                }
            })
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Highway Network
pub struct HighwayNetwork {
    layers: Vec<HighwayLayer>,
}

struct HighwayLayer {
    transform: Dense,
    gate: Dense,
}

impl HighwayLayer {
    fn new(size: usize) -> Self {
        HighwayLayer {
            transform: Dense::new(size, size),
            gate: Dense::new(size, size),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut h = self.transform.forward(x, training);
        h = ReLU::new().forward(&h);
        
        let mut t = self.gate.forward(x, training);
        t = Sigmoid::new().forward(&t);
        
        // Highway connection: y = h * t + x * (1 - t)
        self.highway_connection(&h, x, &t)
    }

    fn highway_connection(&self, h: &Tensor, x: &Tensor, t: &Tensor) -> Tensor {
        let h_data = h.data_f32();
        let x_data = x.data_f32();
        let t_data = t.data_f32();
        
        let result: Vec<f32> = (0..h_data.len())
            .map(|i| h_data[i] * t_data[i] + x_data[i] * (1.0 - t_data[i]))
            .collect();
        
        Tensor::from_slice(&result, h.dims()).unwrap()
    }
}

impl HighwayNetwork {
    pub fn new(size: usize, num_layers: usize) -> Self {
        HighwayNetwork {
            layers: (0..num_layers).map(|_| HighwayLayer::new(size)).collect(),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for layer in &mut self.layers {
            out = layer.forward(&out, training);
        }
        
        out
    }
}

/// Residual Feed-Forward Network
pub struct ResidualFFN {
    blocks: Vec<ResidualBlock>,
    final_layer: Dense,
}

struct ResidualBlock {
    fc1: Dense,
    fc2: Dense,
}

impl ResidualBlock {
    fn new(size: usize) -> Self {
        ResidualBlock {
            fc1: Dense::new(size, size),
            fc2: Dense::new(size, size),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        
        let mut out = self.fc1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.fc2.forward(&out, training);
        
        // Add residual
        self.add_tensors(&out, &identity)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl ResidualFFN {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_blocks: usize) -> Self {
        ResidualFFN {
            blocks: (0..num_blocks).map(|_| ResidualBlock::new(hidden_dim)).collect(),
            final_layer: Dense::new(hidden_dim, output_dim),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for block in &mut self.blocks {
            out = block.forward(&out, training);
            out = ReLU::new().forward(&out);
        }
        
        self.final_layer.forward(&out, training)
    }
}

/// Mixture of Experts (MoE)
pub struct MixtureOfExperts {
    experts: Vec<Expert>,
    gating_network: Dense,
    num_experts: usize,
}

struct Expert {
    layers: Vec<Dense>,
}

impl Expert {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Expert {
            layers: vec![
                Dense::new(input_dim, hidden_dim),
                Dense::new(hidden_dim, output_dim),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.layers[0].forward(x, training);
        out = ReLU::new().forward(&out);
        self.layers[1].forward(&out, training)
    }
}

impl MixtureOfExperts {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_experts: usize) -> Self {
        MixtureOfExperts {
            experts: (0..num_experts)
                .map(|_| Expert::new(input_dim, hidden_dim, output_dim))
                .collect(),
            gating_network: Dense::new(input_dim, num_experts),
            num_experts,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // Compute gating weights
        let gates = self.gating_network.forward(x, training);
        let gates_softmax = self.softmax(&gates);
        
        // Get expert outputs
        let mut expert_outputs = Vec::new();
        for expert in &mut self.experts {
            expert_outputs.push(expert.forward(x, training));
        }
        
        // Weighted combination
        self.combine_experts(&expert_outputs, &gates_softmax)
    }

    fn softmax(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let batch_size = x.dims()[0];
        let num_experts = x.dims()[1];
        
        let mut result = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            let offset = b * num_experts;
            
            let mut max_val = data[offset];
            for i in 1..num_experts {
                max_val = max_val.max(data[offset + i]);
            }
            
            let mut sum = 0.0f32;
            for i in 0..num_experts {
                let exp_val = (data[offset + i] - max_val).exp();
                result[offset + i] = exp_val;
                sum += exp_val;
            }
            
            for i in 0..num_experts {
                result[offset + i] /= sum;
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn combine_experts(&self, expert_outputs: &[Tensor], gates: &Tensor) -> Tensor {
        let batch_size = expert_outputs[0].dims()[0];
        let output_dim = expert_outputs[0].dims()[1];
        let gates_data = gates.data_f32();
        
        let mut result = vec![0.0f32; batch_size * output_dim];
        
        for b in 0..batch_size {
            for e in 0..self.num_experts {
                let gate_weight = gates_data[b * self.num_experts + e];
                let expert_data = expert_outputs[e].data_f32();
                
                for d in 0..output_dim {
                    result[b * output_dim + d] += gate_weight * expert_data[b * output_dim + d];
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, output_dim]).unwrap()
    }
}

/// Kolmogorov-Arnold Network (KAN)
pub struct KAN {
    layers: Vec<KANLayer>,
}

struct KANLayer {
    basis_functions: Vec<Dense>,
    combination_weights: Dense,
}

impl KANLayer {
    fn new(input_dim: usize, output_dim: usize, num_basis: usize) -> Self {
        KANLayer {
            basis_functions: (0..num_basis)
                .map(|_| Dense::new(input_dim, output_dim))
                .collect(),
            combination_weights: Dense::new(num_basis, output_dim),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut basis_outputs = Vec::new();
        
        for basis in &mut self.basis_functions {
            let out = basis.forward(x, training);
            basis_outputs.push(out);
        }
        
        // Combine basis function outputs
        self.combine_basis(&basis_outputs)
    }

    fn combine_basis(&self, outputs: &[Tensor]) -> Tensor {
        if outputs.is_empty() {
            return Tensor::from_slice(&[0.0f32], &[1, 1]).unwrap();
        }
        
        let mut result = outputs[0].data_f32().to_vec();
        
        for output in &outputs[1..] {
            let data = output.data_f32();
            for (i, &val) in data.iter().enumerate() {
                result[i] += val;
            }
        }
        
        Tensor::from_slice(&result, outputs[0].dims()).unwrap()
    }
}

impl KAN {
    pub fn new(layer_sizes: Vec<usize>, num_basis: usize) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(KANLayer::new(layer_sizes[i], layer_sizes[i + 1], num_basis));
        }
        
        KAN { layers }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for layer in &mut self.layers {
            out = layer.forward(&out, training);
        }
        
        out
    }
}

/// Radial Basis Function Network (RBF)
pub struct RBFNetwork {
    centers: Vec<Vec<f32>>,
    widths: Vec<f32>,
    output_layer: Dense,
    num_centers: usize,
}

impl RBFNetwork {
    pub fn new(input_dim: usize, num_centers: usize, output_dim: usize) -> Self {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let mut centers = Vec::new();
        for _ in 0..num_centers {
            let center: Vec<f32> = (0..input_dim)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .collect();
            centers.push(center);
        }
        
        RBFNetwork {
            centers,
            widths: vec![1.0f32; num_centers],
            output_layer: Dense::new(num_centers, output_dim),
            num_centers,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let rbf_activations = self.compute_rbf_activations(x);
        self.output_layer.forward(&rbf_activations, training)
    }

    fn compute_rbf_activations(&self, x: &Tensor) -> Tensor {
        let batch_size = x.dims()[0];
        let input_dim = x.dims()[1];
        let x_data = x.data_f32();
        
        let mut activations = Vec::new();
        
        for b in 0..batch_size {
            for c in 0..self.num_centers {
                let mut dist_sq = 0.0f32;
                
                for d in 0..input_dim {
                    let diff = x_data[b * input_dim + d] - self.centers[c][d];
                    dist_sq += diff * diff;
                }
                
                let activation = (-dist_sq / (2.0 * self.widths[c].powi(2))).exp();
                activations.push(activation);
            }
        }
        
        Tensor::from_slice(&activations, &[batch_size, self.num_centers]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp() {
        let mut mlp = MLP::new(vec![784, 256, 128, 10]);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 784], &[2, 784]).unwrap();
        let output = mlp.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }

    #[test]
    fn test_highway_network() {
        let mut highway = HighwayNetwork::new(256, 5);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 256], &[2, 256]).unwrap();
        let output = highway.forward(&input, false);
        assert_eq!(output.dims()[1], 256);
    }

    #[test]
    fn test_mixture_of_experts() {
        let mut moe = MixtureOfExperts::new(100, 64, 10, 4);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 100], &[2, 100]).unwrap();
        let output = moe.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }
}



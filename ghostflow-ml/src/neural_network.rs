//! Neural Network models - Perceptron, MLP

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Perceptron classifier
pub struct Perceptron {
    pub max_iter: usize,
    pub eta0: f32,
    pub tol: f32,
    pub shuffle: bool,
    pub penalty: Option<PerceptronPenalty>,
    pub alpha: f32,
    coef_: Option<Vec<f32>>,
    intercept_: Option<f32>,
    n_iter_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum PerceptronPenalty {
    L1,
    L2,
    ElasticNet,
}

impl Perceptron {
    pub fn new() -> Self {
        Perceptron {
            max_iter: 1000,
            eta0: 1.0,
            tol: 1e-3,
            shuffle: true,
            penalty: None,
            alpha: 0.0001,
            coef_: None,
            intercept_: None,
            n_iter_: 0,
        }
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    pub fn eta0(mut self, eta: f32) -> Self {
        self.eta0 = eta;
        self
    }

    pub fn penalty(mut self, p: PerceptronPenalty) -> Self {
        self.penalty = Some(p);
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Convert labels to {-1, 1}
        let y_binary: Vec<f32> = y_data.iter()
            .map(|&y| if y > 0.5 { 1.0 } else { -1.0 })
            .collect();

        let mut weights = vec![0.0f32; n_features];
        let mut bias = 0.0f32;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for iter in 0..self.max_iter {
            if self.shuffle {
                indices.shuffle(&mut thread_rng());
            }

            let mut n_errors = 0;

            for &i in &indices {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                
                // Compute prediction
                let mut pred = bias;
                for j in 0..n_features {
                    pred += weights[j] * xi[j];
                }
                let pred_sign = if pred >= 0.0 { 1.0 } else { -1.0 };

                // Update if misclassified
                if pred_sign != y_binary[i] {
                    n_errors += 1;
                    
                    for j in 0..n_features {
                        weights[j] += self.eta0 * y_binary[i] * xi[j];
                    }
                    bias += self.eta0 * y_binary[i];

                    // Apply penalty
                    if let Some(penalty) = self.penalty {
                        match penalty {
                            PerceptronPenalty::L2 => {
                                for w in &mut weights {
                                    *w *= 1.0 - self.alpha * self.eta0;
                                }
                            }
                            PerceptronPenalty::L1 => {
                                for w in &mut weights {
                                    let sign = w.signum();
                                    *w = (*w - self.alpha * self.eta0 * sign).max(0.0) * sign;
                                }
                            }
                            PerceptronPenalty::ElasticNet => {
                                for w in &mut weights {
                                    *w *= 1.0 - 0.5 * self.alpha * self.eta0;
                                    let sign = w.signum();
                                    *w = (*w - 0.5 * self.alpha * self.eta0 * sign).max(0.0) * sign;
                                }
                            }
                        }
                    }
                }
            }

            self.n_iter_ = iter + 1;

            if n_errors == 0 {
                break;
            }
        }

        self.coef_ = Some(weights);
        self.intercept_ = Some(bias);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let weights = self.coef_.as_ref().expect("Model not fitted");
        let bias = self.intercept_.unwrap_or(0.0);

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = bias;
                for j in 0..n_features {
                    pred += weights[j] * xi[j];
                }
                if pred >= 0.0 { 1.0 } else { 0.0 }
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn score(&self, x: &Tensor, y: &Tensor) -> f32 {
        let predictions = self.predict(x);
        let pred_data = predictions.data_f32();
        let y_data = y.data_f32();

        let correct: usize = pred_data.iter()
            .zip(y_data.iter())
            .filter(|(&p, &y)| (p - y).abs() < 0.5)
            .count();

        correct as f32 / y_data.len() as f32
    }
}

impl Default for Perceptron {
    fn default() -> Self {
        Self::new()
    }
}


/// Multi-Layer Perceptron Classifier
pub struct MLPClassifier {
    pub hidden_layer_sizes: Vec<usize>,
    pub activation: Activation,
    pub solver: MLPSolver,
    pub alpha: f32,
    pub learning_rate_init: f32,
    pub max_iter: usize,
    pub tol: f32,
    pub batch_size: usize,
    pub momentum: f32,
    weights_: Vec<Vec<f32>>,
    biases_: Vec<Vec<f32>>,
    n_classes_: usize,
    n_features_: usize,
    n_iter_: usize,
    loss_: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Identity,
}

#[derive(Clone, Copy, Debug)]
pub enum MLPSolver {
    SGD,
    Adam,
}

impl MLPClassifier {
    pub fn new(hidden_layer_sizes: Vec<usize>) -> Self {
        MLPClassifier {
            hidden_layer_sizes,
            activation: Activation::ReLU,
            solver: MLPSolver::Adam,
            alpha: 0.0001,
            learning_rate_init: 0.001,
            max_iter: 200,
            tol: 1e-4,
            batch_size: 32,
            momentum: 0.9,
            weights_: Vec::new(),
            biases_: Vec::new(),
            n_classes_: 0,
            n_features_: 0,
            n_iter_: 0,
            loss_: 0.0,
        }
    }

    pub fn activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    pub fn solver(mut self, solver: MLPSolver) -> Self {
        self.solver = solver;
        self
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate_init = lr;
        self
    }

    fn activate(&self, x: f32) -> f32 {
        match self.activation {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Identity => x,
        }
    }

    fn activate_derivative(&self, x: f32) -> f32 {
        match self.activation {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Identity => 1.0,
        }
    }

    fn softmax(x: &[f32]) -> Vec<f32> {
        let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_x: Vec<f32> = x.iter().map(|&xi| (xi - max_x).exp()).collect();
        let sum: f32 = exp_x.iter().sum();
        exp_x.iter().map(|&e| e / sum).collect()
    }

    fn forward(&self, x: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut activations = vec![x.to_vec()];
        let mut pre_activations = Vec::new();

        let mut current = x.to_vec();

        for (layer_idx, (weights, biases)) in self.weights_.iter().zip(self.biases_.iter()).enumerate() {
            let n_in = current.len();
            let n_out = biases.len();

            let mut z = vec![0.0f32; n_out];
            for j in 0..n_out {
                z[j] = biases[j];
                for i in 0..n_in {
                    z[j] += weights[i * n_out + j] * current[i];
                }
            }

            pre_activations.push(z.clone());

            // Apply activation (softmax for last layer)
            let a = if layer_idx == self.weights_.len() - 1 {
                Self::softmax(&z)
            } else {
                z.iter().map(|&zi| self.activate(zi)).collect()
            };

            activations.push(a.clone());
            current = a;
        }

        (activations, pre_activations)
    }

    fn initialize_weights(&mut self, n_features: usize, n_classes: usize) {
        let mut rng = thread_rng();
        
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.hidden_layer_sizes);
        layer_sizes.push(n_classes);

        self.weights_.clear();
        self.biases_.clear();

        for i in 0..layer_sizes.len() - 1 {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];

            // Xavier initialization
            let scale = (2.0 / (n_in + n_out) as f32).sqrt();
            let weights: Vec<f32> = (0..n_in * n_out)
                .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            let biases = vec![0.0f32; n_out];

            self.weights_.push(weights);
            self.biases_.push(biases);
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        // One-hot encode labels
        let mut y_onehot = vec![vec![0.0f32; self.n_classes_]; n_samples];
        for i in 0..n_samples {
            let class = y_data[i] as usize;
            if class < self.n_classes_ {
                y_onehot[i][class] = 1.0;
            }
        }

        // Initialize weights
        self.initialize_weights(n_features, self.n_classes_);

        // Adam optimizer state
        let mut m_weights: Vec<Vec<f32>> = self.weights_.iter().map(|w| vec![0.0f32; w.len()]).collect();
        let mut v_weights: Vec<Vec<f32>> = self.weights_.iter().map(|w| vec![0.0f32; w.len()]).collect();
        let mut m_biases: Vec<Vec<f32>> = self.biases_.iter().map(|b| vec![0.0f32; b.len()]).collect();
        let mut v_biases: Vec<Vec<f32>> = self.biases_.iter().map(|b| vec![0.0f32; b.len()]).collect();

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut prev_loss = f32::INFINITY;

        for iter in 0..self.max_iter {
            indices.shuffle(&mut thread_rng());

            let mut total_loss = 0.0f32;

            // Mini-batch training
            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_samples);
                let batch_size = batch_end - batch_start;

                // Accumulate gradients
                let mut grad_weights: Vec<Vec<f32>> = self.weights_.iter().map(|w| vec![0.0f32; w.len()]).collect();
                let mut grad_biases: Vec<Vec<f32>> = self.biases_.iter().map(|b| vec![0.0f32; b.len()]).collect();

                for &idx in &indices[batch_start..batch_end] {
                    let xi = &x_data[idx * n_features..(idx + 1) * n_features];
                    let yi = &y_onehot[idx];

                    // Forward pass
                    let (activations, pre_activations) = self.forward(xi);

                    // Compute loss (cross-entropy)
                    let output = activations.last().unwrap();
                    for c in 0..self.n_classes_ {
                        if yi[c] > 0.5 {
                            total_loss -= output[c].max(1e-10).ln();
                        }
                    }

                    // Backward pass
                    let n_layers = self.weights_.len();
                    let mut delta: Vec<f32> = output.iter().zip(yi.iter())
                        .map(|(&o, &y)| o - y)
                        .collect();

                    for layer in (0..n_layers).rev() {
                        let a_prev = &activations[layer];
                        let n_in = a_prev.len();
                        let n_out = delta.len();

                        // Gradient for weights and biases
                        for i in 0..n_in {
                            for j in 0..n_out {
                                grad_weights[layer][i * n_out + j] += a_prev[i] * delta[j];
                            }
                        }
                        for j in 0..n_out {
                            grad_biases[layer][j] += delta[j];
                        }

                        // Propagate delta to previous layer
                        if layer > 0 {
                            let mut new_delta = vec![0.0f32; n_in];
                            for i in 0..n_in {
                                for j in 0..n_out {
                                    new_delta[i] += self.weights_[layer][i * n_out + j] * delta[j];
                                }
                                new_delta[i] *= self.activate_derivative(pre_activations[layer - 1][i]);
                            }
                            delta = new_delta;
                        }
                    }
                }

                // Update weights using Adam
                let t = (iter * (n_samples / self.batch_size) + batch_start / self.batch_size + 1) as f32;

                for layer in 0..self.weights_.len() {
                    for i in 0..self.weights_[layer].len() {
                        let g = grad_weights[layer][i] / batch_size as f32 + self.alpha * self.weights_[layer][i];
                        
                        m_weights[layer][i] = beta1 * m_weights[layer][i] + (1.0 - beta1) * g;
                        v_weights[layer][i] = beta2 * v_weights[layer][i] + (1.0 - beta2) * g * g;
                        
                        let m_hat = m_weights[layer][i] / (1.0 - beta1.powf(t));
                        let v_hat = v_weights[layer][i] / (1.0 - beta2.powf(t));
                        
                        self.weights_[layer][i] -= self.learning_rate_init * m_hat / (v_hat.sqrt() + eps);
                    }

                    for i in 0..self.biases_[layer].len() {
                        let g = grad_biases[layer][i] / batch_size as f32;
                        
                        m_biases[layer][i] = beta1 * m_biases[layer][i] + (1.0 - beta1) * g;
                        v_biases[layer][i] = beta2 * v_biases[layer][i] + (1.0 - beta2) * g * g;
                        
                        let m_hat = m_biases[layer][i] / (1.0 - beta1.powf(t));
                        let v_hat = v_biases[layer][i] / (1.0 - beta2.powf(t));
                        
                        self.biases_[layer][i] -= self.learning_rate_init * m_hat / (v_hat.sqrt() + eps);
                    }
                }
            }

            self.loss_ = total_loss / n_samples as f32;
            self.n_iter_ = iter + 1;

            // Check convergence
            if (prev_loss - self.loss_).abs() < self.tol {
                break;
            }
            prev_loss = self.loss_;
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let start = i * self.n_classes_;
                let probs = &proba_data[start..start + self.n_classes_];
                probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(c, _)| c as f32)
                    .unwrap_or(0.0)
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut probs: Vec<f32> = Vec::with_capacity(n_samples * self.n_classes_);

        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            let (activations, _) = self.forward(xi);
            probs.extend(activations.last().unwrap());
        }

        Tensor::from_slice(&probs, &[n_samples, self.n_classes_]).unwrap()
    }

    pub fn score(&self, x: &Tensor, y: &Tensor) -> f32 {
        let predictions = self.predict(x);
        let pred_data = predictions.data_f32();
        let y_data = y.data_f32();

        let correct: usize = pred_data.iter()
            .zip(y_data.iter())
            .filter(|(&p, &y)| (p - y).abs() < 0.5)
            .count();

        correct as f32 / y_data.len() as f32
    }
}


/// Multi-Layer Perceptron Regressor
pub struct MLPRegressor {
    pub hidden_layer_sizes: Vec<usize>,
    pub activation: Activation,
    pub solver: MLPSolver,
    pub alpha: f32,
    pub learning_rate_init: f32,
    pub max_iter: usize,
    pub tol: f32,
    pub batch_size: usize,
    weights_: Vec<Vec<f32>>,
    biases_: Vec<Vec<f32>>,
    n_features_: usize,
    n_outputs_: usize,
    n_iter_: usize,
    loss_: f32,
}

impl MLPRegressor {
    pub fn new(hidden_layer_sizes: Vec<usize>) -> Self {
        MLPRegressor {
            hidden_layer_sizes,
            activation: Activation::ReLU,
            solver: MLPSolver::Adam,
            alpha: 0.0001,
            learning_rate_init: 0.001,
            max_iter: 200,
            tol: 1e-4,
            batch_size: 32,
            weights_: Vec::new(),
            biases_: Vec::new(),
            n_features_: 0,
            n_outputs_: 1,
            n_iter_: 0,
            loss_: 0.0,
        }
    }

    pub fn activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    fn activate(&self, x: f32) -> f32 {
        match self.activation {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Identity => x,
        }
    }

    fn activate_derivative(&self, x: f32) -> f32 {
        match self.activation {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Identity => 1.0,
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut activations = vec![x.to_vec()];
        let mut pre_activations = Vec::new();
        let mut current = x.to_vec();

        for (layer_idx, (weights, biases)) in self.weights_.iter().zip(self.biases_.iter()).enumerate() {
            let n_in = current.len();
            let n_out = biases.len();

            let mut z = vec![0.0f32; n_out];
            for j in 0..n_out {
                z[j] = biases[j];
                for i in 0..n_in {
                    z[j] += weights[i * n_out + j] * current[i];
                }
            }

            pre_activations.push(z.clone());

            // Identity activation for output layer
            let a = if layer_idx == self.weights_.len() - 1 {
                z.clone()
            } else {
                z.iter().map(|&zi| self.activate(zi)).collect()
            };

            activations.push(a.clone());
            current = a;
        }

        (activations, pre_activations)
    }

    fn initialize_weights(&mut self, n_features: usize, n_outputs: usize) {
        let mut rng = thread_rng();
        
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.hidden_layer_sizes);
        layer_sizes.push(n_outputs);

        self.weights_.clear();
        self.biases_.clear();

        for i in 0..layer_sizes.len() - 1 {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];

            let scale = (2.0 / (n_in + n_out) as f32).sqrt();
            let weights: Vec<f32> = (0..n_in * n_out)
                .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            let biases = vec![0.0f32; n_out];

            self.weights_.push(weights);
            self.biases_.push(biases);
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;
        self.n_outputs_ = 1;

        self.initialize_weights(n_features, self.n_outputs_);

        // Adam optimizer state
        let mut m_weights: Vec<Vec<f32>> = self.weights_.iter().map(|w| vec![0.0f32; w.len()]).collect();
        let mut v_weights: Vec<Vec<f32>> = self.weights_.iter().map(|w| vec![0.0f32; w.len()]).collect();
        let mut m_biases: Vec<Vec<f32>> = self.biases_.iter().map(|b| vec![0.0f32; b.len()]).collect();
        let mut v_biases: Vec<Vec<f32>> = self.biases_.iter().map(|b| vec![0.0f32; b.len()]).collect();

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut prev_loss = f32::INFINITY;

        for iter in 0..self.max_iter {
            indices.shuffle(&mut thread_rng());
            let mut total_loss = 0.0f32;

            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_samples);
                let batch_size = batch_end - batch_start;

                let mut grad_weights: Vec<Vec<f32>> = self.weights_.iter().map(|w| vec![0.0f32; w.len()]).collect();
                let mut grad_biases: Vec<Vec<f32>> = self.biases_.iter().map(|b| vec![0.0f32; b.len()]).collect();

                for &idx in &indices[batch_start..batch_end] {
                    let xi = &x_data[idx * n_features..(idx + 1) * n_features];
                    let yi = y_data[idx];

                    let (activations, pre_activations) = self.forward(xi);
                    let output = activations.last().unwrap()[0];

                    // MSE loss
                    let error = output - yi;
                    total_loss += error * error;

                    // Backward pass
                    let n_layers = self.weights_.len();
                    let mut delta = vec![error];

                    for layer in (0..n_layers).rev() {
                        let a_prev = &activations[layer];
                        let n_in = a_prev.len();
                        let n_out = delta.len();

                        for i in 0..n_in {
                            for j in 0..n_out {
                                grad_weights[layer][i * n_out + j] += a_prev[i] * delta[j];
                            }
                        }
                        for j in 0..n_out {
                            grad_biases[layer][j] += delta[j];
                        }

                        if layer > 0 {
                            let mut new_delta = vec![0.0f32; n_in];
                            for i in 0..n_in {
                                for j in 0..n_out {
                                    new_delta[i] += self.weights_[layer][i * n_out + j] * delta[j];
                                }
                                new_delta[i] *= self.activate_derivative(pre_activations[layer - 1][i]);
                            }
                            delta = new_delta;
                        }
                    }
                }

                // Adam update
                let t = (iter * (n_samples / self.batch_size) + batch_start / self.batch_size + 1) as f32;

                for layer in 0..self.weights_.len() {
                    for i in 0..self.weights_[layer].len() {
                        let g = grad_weights[layer][i] / batch_size as f32 + self.alpha * self.weights_[layer][i];
                        m_weights[layer][i] = beta1 * m_weights[layer][i] + (1.0 - beta1) * g;
                        v_weights[layer][i] = beta2 * v_weights[layer][i] + (1.0 - beta2) * g * g;
                        let m_hat = m_weights[layer][i] / (1.0 - beta1.powf(t));
                        let v_hat = v_weights[layer][i] / (1.0 - beta2.powf(t));
                        self.weights_[layer][i] -= self.learning_rate_init * m_hat / (v_hat.sqrt() + eps);
                    }

                    for i in 0..self.biases_[layer].len() {
                        let g = grad_biases[layer][i] / batch_size as f32;
                        m_biases[layer][i] = beta1 * m_biases[layer][i] + (1.0 - beta1) * g;
                        v_biases[layer][i] = beta2 * v_biases[layer][i] + (1.0 - beta2) * g * g;
                        let m_hat = m_biases[layer][i] / (1.0 - beta1.powf(t));
                        let v_hat = v_biases[layer][i] / (1.0 - beta2.powf(t));
                        self.biases_[layer][i] -= self.learning_rate_init * m_hat / (v_hat.sqrt() + eps);
                    }
                }
            }

            self.loss_ = total_loss / n_samples as f32;
            self.n_iter_ = iter + 1;

            if (prev_loss - self.loss_).abs() < self.tol {
                break;
            }
            prev_loss = self.loss_;
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let (activations, _) = self.forward(xi);
                activations.last().unwrap()[0]
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn score(&self, x: &Tensor, y: &Tensor) -> f32 {
        let predictions = self.predict(x);
        let pred_data = predictions.data_f32();
        let y_data = y.data_f32();

        let y_mean: f32 = y_data.iter().sum::<f32>() / y_data.len() as f32;
        let ss_res: f32 = pred_data.iter().zip(y_data.iter()).map(|(&p, &y)| (y - p).powi(2)).sum();
        let ss_tot: f32 = y_data.iter().map(|&y| (y - y_mean).powi(2)).sum();

        1.0 - ss_res / ss_tot.max(1e-10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0], &[4]).unwrap();
        
        let mut p = Perceptron::new().max_iter(100);
        p.fit(&x, &y);
        
        let predictions = p.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }

    #[test]
    fn test_mlp_classifier() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[4]).unwrap();
        
        let mut mlp = MLPClassifier::new(vec![4]).max_iter(100);
        mlp.fit(&x, &y);
        
        let predictions = mlp.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }

    #[test]
    fn test_mlp_regressor() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap();
        let y = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
        
        let mut mlp = MLPRegressor::new(vec![10]).max_iter(500);
        mlp.fit(&x, &y);
        
        let predictions = mlp.predict(&x);
        assert_eq!(predictions.dims(), &[5]);
    }
}



//! Conditional Random Fields (CRF)
//!
//! Discriminative probabilistic models for structured prediction,
//! particularly useful for sequence labeling tasks.

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// Linear-chain Conditional Random Field
/// 
/// Used for sequence labeling tasks like:
/// - Named Entity Recognition (NER)
/// - Part-of-Speech (POS) tagging
/// - Chunking
/// - Segmentation
pub struct LinearChainCRF {
    pub n_labels: usize,
    pub n_features: usize,
    pub max_iter: usize,
    pub learning_rate: f32,
    pub l2_penalty: f32,
    pub tol: f32,
    
    // Model parameters
    weights: Vec<f32>,              // Feature weights (n_features * n_labels)
    transitions: Vec<Vec<f32>>,     // Transition scores (n_labels, n_labels)
    converged: bool,
}

impl LinearChainCRF {
    pub fn new(n_labels: usize, n_features: usize) -> Self {
        Self {
            n_labels,
            n_features,
            max_iter: 100,
            learning_rate: 0.01,
            l2_penalty: 0.1,
            tol: 1e-3,
            weights: vec![0.0; n_features * n_labels],
            transitions: vec![vec![0.0; n_labels]; n_labels],
            converged: false,
        }
    }

    pub fn max_iter(mut self, iter: usize) -> Self {
        self.max_iter = iter;
        self
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn l2_penalty(mut self, penalty: f32) -> Self {
        self.l2_penalty = penalty;
        self
    }

    /// Fit the CRF using stochastic gradient descent
    pub fn fit(&mut self, sequences: &[Tensor], labels: &[Tensor]) {
        assert_eq!(sequences.len(), labels.len(), "Number of sequences and labels must match");

        let mut prev_loss = f32::INFINITY;

        for iteration in 0..self.max_iter {
            let mut total_loss = 0.0;
            let mut n_samples = 0;

            // Process each sequence
            for (seq_idx, (sequence, label_seq)) in sequences.iter().zip(labels.iter()).enumerate() {
                let seq_data = sequence.data_f32();
                let label_data = label_seq.data_f32();
                let seq_len = sequence.dims()[0];

                // Forward-backward to compute marginals
                let (alpha, beta, z) = self.forward_backward(&seq_data, seq_len);

                // Compute gradients
                let (weight_grad, trans_grad) = self.compute_gradients(
                    &seq_data,
                    &label_data,
                    &alpha,
                    &beta,
                    z,
                    seq_len,
                );

                // Update parameters
                self.update_parameters(&weight_grad, &trans_grad);

                // Compute loss for this sequence
                let loss = self.compute_loss(&seq_data, &label_data, seq_len);
                total_loss += loss;
                n_samples += 1;
            }

            let avg_loss = total_loss / n_samples as f32;

            // Check convergence
            if (prev_loss - avg_loss).abs() < self.tol {
                self.converged = true;
                println!("CRF converged at iteration {}", iteration);
                break;
            }

            prev_loss = avg_loss;

            if iteration % 10 == 0 {
                println!("Iteration {}: Loss = {:.4}", iteration, avg_loss);
            }
        }
    }

    /// Forward-backward algorithm for CRF
    fn forward_backward(&self, seq_data: &[f32], seq_len: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f32) {
        // Forward pass
        let mut alpha = vec![vec![f32::NEG_INFINITY; self.n_labels]; seq_len];
        
        // Initialize first position
        for j in 0..self.n_labels {
            alpha[0][j] = self.emission_score(&seq_data, 0, j);
        }

        // Forward recursion
        for t in 1..seq_len {
            for j in 0..self.n_labels {
                let emission = self.emission_score(&seq_data, t, j);
                let mut max_score = f32::NEG_INFINITY;
                
                for i in 0..self.n_labels {
                    let score = alpha[t - 1][i] + self.transitions[i][j] + emission;
                    max_score = max_score.max(score);
                }
                
                // Log-sum-exp for numerical stability
                let mut sum = 0.0;
                for i in 0..self.n_labels {
                    let score = alpha[t - 1][i] + self.transitions[i][j] + emission;
                    sum += (score - max_score).exp();
                }
                alpha[t][j] = max_score + sum.ln();
            }
        }

        // Compute partition function Z
        let mut max_alpha = f32::NEG_INFINITY;
        for j in 0..self.n_labels {
            max_alpha = max_alpha.max(alpha[seq_len - 1][j]);
        }
        
        let mut z_sum = 0.0;
        for j in 0..self.n_labels {
            z_sum += (alpha[seq_len - 1][j] - max_alpha).exp();
        }
        let z = max_alpha + z_sum.ln();

        // Backward pass
        let mut beta = vec![vec![f32::NEG_INFINITY; self.n_labels]; seq_len];
        
        // Initialize last position
        for j in 0..self.n_labels {
            beta[seq_len - 1][j] = 0.0;
        }

        // Backward recursion
        for t in (0..seq_len - 1).rev() {
            for i in 0..self.n_labels {
                let mut max_score = f32::NEG_INFINITY;
                
                for j in 0..self.n_labels {
                    let emission = self.emission_score(&seq_data, t + 1, j);
                    let score = self.transitions[i][j] + emission + beta[t + 1][j];
                    max_score = max_score.max(score);
                }
                
                // Log-sum-exp
                let mut sum = 0.0;
                for j in 0..self.n_labels {
                    let emission = self.emission_score(&seq_data, t + 1, j);
                    let score = self.transitions[i][j] + emission + beta[t + 1][j];
                    sum += (score - max_score).exp();
                }
                beta[t][i] = max_score + sum.ln();
            }
        }

        (alpha, beta, z)
    }

    /// Compute emission score for a position and label
    fn emission_score(&self, seq_data: &[f32], position: usize, label: usize) -> f32 {
        let features = &seq_data[position * self.n_features..(position + 1) * self.n_features];
        let mut score = 0.0;
        
        for (feat_idx, &feat_val) in features.iter().enumerate() {
            let weight_idx = feat_idx * self.n_labels + label;
            score += self.weights[weight_idx] * feat_val;
        }
        
        score
    }

    /// Compute gradients using forward-backward marginals
    fn compute_gradients(
        &self,
        seq_data: &[f32],
        label_data: &[f32],
        alpha: &[Vec<f32>],
        beta: &[Vec<f32>],
        z: f32,
        seq_len: usize,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut weight_grad = vec![0.0; self.n_features * self.n_labels];
        let mut trans_grad = vec![vec![0.0; self.n_labels]; self.n_labels];

        // Compute expected feature counts (model)
        for t in 0..seq_len {
            let features = &seq_data[t * self.n_features..(t + 1) * self.n_features];
            
            for j in 0..self.n_labels {
                // Marginal probability of label j at position t
                let marginal = (alpha[t][j] + beta[t][j] - z).exp();
                
                // Update weight gradients (expected - observed)
                for (feat_idx, &feat_val) in features.iter().enumerate() {
                    let weight_idx = feat_idx * self.n_labels + j;
                    weight_grad[weight_idx] -= marginal * feat_val;
                }
            }
        }

        // Compute expected transition counts (model)
        for t in 0..seq_len - 1 {
            for i in 0..self.n_labels {
                for j in 0..self.n_labels {
                    let emission = self.emission_score(&seq_data, t + 1, j);
                    let marginal = (alpha[t][i] + self.transitions[i][j] + emission + beta[t + 1][j] - z).exp();
                    trans_grad[i][j] -= marginal;
                }
            }
        }

        // Add observed counts
        for t in 0..seq_len {
            let label = label_data[t] as usize;
            let features = &seq_data[t * self.n_features..(t + 1) * self.n_features];
            
            for (feat_idx, &feat_val) in features.iter().enumerate() {
                let weight_idx = feat_idx * self.n_labels + label;
                weight_grad[weight_idx] += feat_val;
            }
        }

        for t in 0..seq_len - 1 {
            let prev_label = label_data[t] as usize;
            let curr_label = label_data[t + 1] as usize;
            trans_grad[prev_label][curr_label] += 1.0;
        }

        // Add L2 regularization
        for i in 0..weight_grad.len() {
            weight_grad[i] -= self.l2_penalty * self.weights[i];
        }

        for i in 0..self.n_labels {
            for j in 0..self.n_labels {
                trans_grad[i][j] -= self.l2_penalty * self.transitions[i][j];
            }
        }

        (weight_grad, trans_grad)
    }

    /// Update parameters using gradients
    fn update_parameters(&mut self, weight_grad: &[f32], trans_grad: &[Vec<f32>]) {
        // Update weights
        for i in 0..self.weights.len() {
            self.weights[i] += self.learning_rate * weight_grad[i];
        }

        // Update transitions
        for i in 0..self.n_labels {
            for j in 0..self.n_labels {
                self.transitions[i][j] += self.learning_rate * trans_grad[i][j];
            }
        }
    }

    /// Compute negative log-likelihood loss
    fn compute_loss(&self, seq_data: &[f32], label_data: &[f32], seq_len: usize) -> f32 {
        // Compute score of true sequence
        let mut true_score = 0.0;
        
        for t in 0..seq_len {
            let label = label_data[t] as usize;
            true_score += self.emission_score(&seq_data, t, label);
        }
        
        for t in 0..seq_len - 1 {
            let prev_label = label_data[t] as usize;
            let curr_label = label_data[t + 1] as usize;
            true_score += self.transitions[prev_label][curr_label];
        }

        // Compute partition function
        let (_, _, z) = self.forward_backward(&seq_data, seq_len);

        // Negative log-likelihood
        let nll = z - true_score;

        // Add L2 regularization term
        let mut reg_term = 0.0;
        for &w in &self.weights {
            reg_term += w * w;
        }
        for i in 0..self.n_labels {
            for j in 0..self.n_labels {
                reg_term += self.transitions[i][j] * self.transitions[i][j];
            }
        }
        reg_term *= 0.5 * self.l2_penalty;

        nll + reg_term
    }

    /// Predict label sequence using Viterbi algorithm
    pub fn predict(&self, sequence: &Tensor) -> Tensor {
        let seq_data = sequence.data_f32();
        let seq_len = sequence.dims()[0];

        let mut delta = vec![vec![f32::NEG_INFINITY; self.n_labels]; seq_len];
        let mut psi = vec![vec![0; self.n_labels]; seq_len];

        // Initialize
        for j in 0..self.n_labels {
            delta[0][j] = self.emission_score(&seq_data, 0, j);
        }

        // Viterbi recursion
        for t in 1..seq_len {
            for j in 0..self.n_labels {
                let emission = self.emission_score(&seq_data, t, j);
                let mut max_score = f32::NEG_INFINITY;
                let mut max_idx = 0;

                for i in 0..self.n_labels {
                    let score = delta[t - 1][i] + self.transitions[i][j] + emission;
                    if score > max_score {
                        max_score = score;
                        max_idx = i;
                    }
                }

                delta[t][j] = max_score;
                psi[t][j] = max_idx;
            }
        }

        // Backtrack
        let mut path = vec![0; seq_len];
        path[seq_len - 1] = delta[seq_len - 1]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        for t in (0..seq_len - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }

        let path_f32: Vec<f32> = path.iter().map(|&x| x as f32).collect();
        Tensor::from_slice(&path_f32, &[seq_len]).unwrap()
    }

    /// Predict marginal probabilities for each position
    pub fn predict_marginals(&self, sequence: &Tensor) -> Tensor {
        let seq_data = sequence.data_f32();
        let seq_len = sequence.dims()[0];

        let (alpha, beta, z) = self.forward_backward(&seq_data, seq_len);

        let mut marginals = Vec::with_capacity(seq_len * self.n_labels);

        for t in 0..seq_len {
            for j in 0..self.n_labels {
                let marginal = (alpha[t][j] + beta[t][j] - z).exp();
                marginals.push(marginal);
            }
        }

        Tensor::from_slice(&marginals, &[seq_len, self.n_labels]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_chain_crf() {
        // Simple sequence labeling task
        let seq1 = Tensor::from_slice(
            &[
                1.0f32, 0.0, 0.0,  // Position 0
                0.0, 1.0, 0.0,     // Position 1
                0.0, 0.0, 1.0,     // Position 2
            ],
            &[3, 3],
        ).unwrap();

        let labels1 = Tensor::from_slice(&[0.0f32, 1.0, 2.0], &[3]).unwrap();

        let sequences = vec![seq1.clone()];
        let labels = vec![labels1];

        let mut crf = LinearChainCRF::new(3, 3)
            .max_iter(50)
            .learning_rate(0.1)
            .l2_penalty(0.01);

        crf.fit(&sequences, &labels);

        let predictions = crf.predict(&seq1);
        assert_eq!(predictions.dims(), &[3]);

        let marginals = crf.predict_marginals(&seq1);
        assert_eq!(marginals.dims(), &[3, 3]);
    }
}

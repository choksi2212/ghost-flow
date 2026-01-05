//! Hidden Markov Models (HMM)
//!
//! Statistical models for sequential data where the system is assumed to be
//! a Markov process with hidden states.

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Hidden Markov Model with Gaussian emissions
pub struct GaussianHMM {
    pub n_components: usize,  // Number of hidden states
    pub n_features: usize,    // Dimensionality of observations
    pub covariance_type: HMMCovarianceType,
    pub max_iter: usize,
    pub tol: f32,
    pub n_init: usize,
    
    // Model parameters
    start_prob: Vec<f32>,           // Initial state probabilities (n_components,)
    trans_prob: Vec<Vec<f32>>,      // Transition probabilities (n_components, n_components)
    means: Vec<Vec<f32>>,           // Emission means (n_components, n_features)
    covariances: Vec<Vec<f32>>,     // Emission covariances
    converged: bool,
}

#[derive(Clone, Copy)]
pub enum HMMCovarianceType {
    Diag,      // Diagonal covariance
    Full,      // Full covariance
    Spherical, // Single variance
}

impl GaussianHMM {
    pub fn new(n_components: usize, n_features: usize) -> Self {
        Self {
            n_components,
            n_features,
            covariance_type: HMMCovarianceType::Diag,
            max_iter: 100,
            tol: 1e-2,
            n_init: 1,
            start_prob: vec![1.0 / n_components as f32; n_components],
            trans_prob: vec![vec![1.0 / n_components as f32; n_components]; n_components],
            means: Vec::new(),
            covariances: Vec::new(),
            converged: false,
        }
    }

    pub fn covariance_type(mut self, cov_type: HMMCovarianceType) -> Self {
        self.covariance_type = cov_type;
        self
    }

    pub fn max_iter(mut self, iter: usize) -> Self {
        self.max_iter = iter;
        self
    }

    /// Fit the HMM using Baum-Welch algorithm (EM for HMMs)
    pub fn fit(&mut self, sequences: &[Tensor]) {
        if sequences.is_empty() {
            return;
        }

        let mut best_log_likelihood = f32::NEG_INFINITY;
        let mut best_start_prob = Vec::new();
        let mut best_trans_prob = Vec::new();
        let mut best_means = Vec::new();
        let mut best_covariances = Vec::new();

        for _ in 0..self.n_init {
            // Initialize parameters
            self.initialize_parameters(sequences);

            let mut prev_log_likelihood = f32::NEG_INFINITY;

            // Baum-Welch algorithm
            for _ in 0..self.max_iter {
                // E-step: Forward-backward algorithm
                let (log_likelihood, gamma, xi) = self.e_step(sequences);

                // M-step: Update parameters
                self.m_step(sequences, &gamma, &xi);

                // Check convergence
                if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                    self.converged = true;
                    break;
                }

                prev_log_likelihood = log_likelihood;
            }

            // Keep best result
            let final_log_likelihood = self.compute_log_likelihood(sequences);
            if final_log_likelihood > best_log_likelihood {
                best_log_likelihood = final_log_likelihood;
                best_start_prob = self.start_prob.clone();
                best_trans_prob = self.trans_prob.clone();
                best_means = self.means.clone();
                best_covariances = self.covariances.clone();
            }
        }

        self.start_prob = best_start_prob;
        self.trans_prob = best_trans_prob;
        self.means = best_means;
        self.covariances = best_covariances;
    }

    fn initialize_parameters(&mut self, sequences: &[Tensor]) {
        let mut rng = thread_rng();

        // Initialize start probabilities uniformly
        self.start_prob = vec![1.0 / self.n_components as f32; self.n_components];

        // Initialize transition probabilities uniformly
        self.trans_prob = vec![vec![1.0 / self.n_components as f32; self.n_components]; self.n_components];

        // Initialize means using k-means++ on all observations
        let mut all_obs = Vec::new();
        for seq in sequences {
            let seq_data = seq.data_f32();
            let seq_len = seq.dims()[0];
            for t in 0..seq_len {
                all_obs.push(seq_data[t * self.n_features..(t + 1) * self.n_features].to_vec());
            }
        }

        self.means = Vec::with_capacity(self.n_components);
        
        // First mean: random observation
        let first_idx = rng.gen_range(0..all_obs.len());
        self.means.push(all_obs[first_idx].clone());

        // Remaining means: k-means++ strategy
        for _ in 1..self.n_components {
            let mut distances = vec![f32::MAX; all_obs.len()];
            
            for (i, obs) in all_obs.iter().enumerate() {
                let min_dist = self.means.iter()
                    .map(|mean| {
                        obs.iter().zip(mean.iter())
                            .map(|(x, m)| (x - m).powi(2))
                            .sum::<f32>()
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                distances[i] = min_dist;
            }

            let total_dist: f32 = distances.iter().sum();
            let mut cumsum = 0.0;
            let rand_val = rng.gen::<f32>() * total_dist;
            
            let mut selected_idx = 0;
            for (i, &dist) in distances.iter().enumerate() {
                cumsum += dist;
                if cumsum >= rand_val {
                    selected_idx = i;
                    break;
                }
            }

            self.means.push(all_obs[selected_idx].clone());
        }

        // Initialize covariances
        self.covariances = match self.covariance_type {
            HMMCovarianceType::Diag | HMMCovarianceType::Full => {
                (0..self.n_components)
                    .map(|_| vec![1.0; self.n_features])
                    .collect()
            }
            HMMCovarianceType::Spherical => {
                (0..self.n_components)
                    .map(|_| vec![1.0])
                    .collect()
            }
        };
    }

    /// E-step: Forward-backward algorithm
    fn e_step(&self, sequences: &[Tensor]) -> (f32, Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<Vec<f32>>>>) {
        let mut total_log_likelihood = 0.0;
        let mut all_gamma = Vec::new();
        let mut all_xi = Vec::new();

        for seq in sequences {
            let seq_data = seq.data_f32();
            let seq_len = seq.dims()[0];

            // Forward algorithm
            let (alpha, log_likelihood) = self.forward(&seq_data, seq_len);
            total_log_likelihood += log_likelihood;

            // Backward algorithm
            let beta = self.backward(&seq_data, seq_len);

            // Calculate gamma (state probabilities)
            let gamma = self.calculate_gamma(&alpha, &beta, seq_len);

            // Calculate xi (transition probabilities)
            let xi = self.calculate_xi(&alpha, &beta, &seq_data, seq_len);

            all_gamma.push(gamma);
            all_xi.push(xi);
        }

        (total_log_likelihood, all_gamma, all_xi)
    }

    /// Forward algorithm
    fn forward(&self, seq_data: &[f32], seq_len: usize) -> (Vec<Vec<f32>>, f32) {
        let mut alpha = vec![vec![0.0; self.n_components]; seq_len];
        let mut scaling = vec![0.0; seq_len];

        // Initialize
        for i in 0..self.n_components {
            let obs = &seq_data[0..self.n_features];
            alpha[0][i] = self.start_prob[i] * self.emission_prob(obs, i);
            scaling[0] += alpha[0][i];
        }

        // Scale
        if scaling[0] > 0.0 {
            for i in 0..self.n_components {
                alpha[0][i] /= scaling[0];
            }
        }

        // Recursion
        for t in 1..seq_len {
            for j in 0..self.n_components {
                let mut sum = 0.0;
                for i in 0..self.n_components {
                    sum += alpha[t - 1][i] * self.trans_prob[i][j];
                }
                let obs = &seq_data[t * self.n_features..(t + 1) * self.n_features];
                alpha[t][j] = sum * self.emission_prob(obs, j);
                scaling[t] += alpha[t][j];
            }

            // Scale
            if scaling[t] > 0.0 {
                for j in 0..self.n_components {
                    alpha[t][j] /= scaling[t];
                }
            }
        }

        // Calculate log likelihood
        let log_likelihood: f32 = scaling.iter().map(|&s| s.max(1e-10).ln()).sum();

        (alpha, log_likelihood)
    }

    /// Backward algorithm
    fn backward(&self, seq_data: &[f32], seq_len: usize) -> Vec<Vec<f32>> {
        let mut beta = vec![vec![0.0; self.n_components]; seq_len];

        // Initialize
        for i in 0..self.n_components {
            beta[seq_len - 1][i] = 1.0;
        }

        // Recursion
        for t in (0..seq_len - 1).rev() {
            for i in 0..self.n_components {
                let mut sum = 0.0;
                for j in 0..self.n_components {
                    let obs = &seq_data[(t + 1) * self.n_features..(t + 2) * self.n_features];
                    sum += self.trans_prob[i][j] * self.emission_prob(obs, j) * beta[t + 1][j];
                }
                beta[t][i] = sum;
            }

            // Normalize
            let total: f32 = beta[t].iter().sum();
            if total > 0.0 {
                for i in 0..self.n_components {
                    beta[t][i] /= total;
                }
            }
        }

        beta
    }

    /// Calculate gamma (state probabilities)
    fn calculate_gamma(&self, alpha: &[Vec<f32>], beta: &[Vec<f32>], seq_len: usize) -> Vec<Vec<f32>> {
        let mut gamma = vec![vec![0.0; self.n_components]; seq_len];

        for t in 0..seq_len {
            let mut total = 0.0;
            for i in 0..self.n_components {
                gamma[t][i] = alpha[t][i] * beta[t][i];
                total += gamma[t][i];
            }

            // Normalize
            if total > 0.0 {
                for i in 0..self.n_components {
                    gamma[t][i] /= total;
                }
            }
        }

        gamma
    }

    /// Calculate xi (transition probabilities)
    fn calculate_xi(&self, alpha: &[Vec<f32>], beta: &[Vec<f32>], seq_data: &[f32], seq_len: usize) -> Vec<Vec<Vec<f32>>> {
        let mut xi = vec![vec![vec![0.0; self.n_components]; self.n_components]; seq_len - 1];

        for t in 0..seq_len - 1 {
            let mut total = 0.0;
            for i in 0..self.n_components {
                for j in 0..self.n_components {
                    let obs = &seq_data[(t + 1) * self.n_features..(t + 2) * self.n_features];
                    xi[t][i][j] = alpha[t][i] * self.trans_prob[i][j] * 
                                  self.emission_prob(obs, j) * beta[t + 1][j];
                    total += xi[t][i][j];
                }
            }

            // Normalize
            if total > 0.0 {
                for i in 0..self.n_components {
                    for j in 0..self.n_components {
                        xi[t][i][j] /= total;
                    }
                }
            }
        }

        xi
    }

    /// M-step: Update parameters
    fn m_step(&mut self, sequences: &[Tensor], all_gamma: &[Vec<Vec<f32>>], all_xi: &[Vec<Vec<Vec<f32>>>]) {
        // Update start probabilities
        for i in 0..self.n_components {
            self.start_prob[i] = all_gamma.iter().map(|gamma| gamma[0][i]).sum::<f32>() / sequences.len() as f32;
        }

        // Update transition probabilities
        for i in 0..self.n_components {
            let mut denom = 0.0;
            for j in 0..self.n_components {
                let mut numer = 0.0;
                for xi in all_xi {
                    for t in 0..xi.len() {
                        numer += xi[t][i][j];
                    }
                }
                
                for gamma in all_gamma {
                    for t in 0..gamma.len() - 1 {
                        denom += gamma[t][i];
                    }
                }
                
                self.trans_prob[i][j] = if denom > 0.0 { numer / denom } else { 1.0 / self.n_components as f32 };
            }
        }

        // Update emission parameters
        for i in 0..self.n_components {
            let mut weighted_sum = vec![0.0; self.n_features];
            let mut weight_total = 0.0;

            for (seq_idx, seq) in sequences.iter().enumerate() {
                let seq_data = seq.data_f32();
                let seq_len = seq.dims()[0];
                let gamma = &all_gamma[seq_idx];

                for t in 0..seq_len {
                    let obs = &seq_data[t * self.n_features..(t + 1) * self.n_features];
                    for j in 0..self.n_features {
                        weighted_sum[j] += gamma[t][i] * obs[j];
                    }
                    weight_total += gamma[t][i];
                }
            }

            // Update mean
            for j in 0..self.n_features {
                self.means[i][j] = if weight_total > 0.0 { weighted_sum[j] / weight_total } else { 0.0 };
            }

            // Update covariance
            let mut weighted_var = vec![0.0; self.n_features];
            for (seq_idx, seq) in sequences.iter().enumerate() {
                let seq_data = seq.data_f32();
                let seq_len = seq.dims()[0];
                let gamma = &all_gamma[seq_idx];

                for t in 0..seq_len {
                    let obs = &seq_data[t * self.n_features..(t + 1) * self.n_features];
                    for j in 0..self.n_features {
                        let diff = obs[j] - self.means[i][j];
                        weighted_var[j] += gamma[t][i] * diff * diff;
                    }
                }
            }

            match self.covariance_type {
                HMMCovarianceType::Diag | HMMCovarianceType::Full => {
                    for j in 0..self.n_features {
                        self.covariances[i][j] = if weight_total > 0.0 { 
                            (weighted_var[j] / weight_total).max(1e-6)
                        } else { 
                            1.0 
                        };
                    }
                }
                HMMCovarianceType::Spherical => {
                    let avg_var = weighted_var.iter().sum::<f32>() / self.n_features as f32;
                    self.covariances[i][0] = if weight_total > 0.0 { 
                        (avg_var / weight_total).max(1e-6)
                    } else { 
                        1.0 
                    };
                }
            }
        }
    }

    /// Calculate emission probability
    fn emission_prob(&self, obs: &[f32], state: usize) -> f32 {
        let mean = &self.means[state];
        let cov = &self.covariances[state];

        match self.covariance_type {
            HMMCovarianceType::Diag | HMMCovarianceType::Full => {
                let mut exponent = 0.0;
                let mut det = 1.0;
                
                for i in 0..self.n_features {
                    let diff = obs[i] - mean[i];
                    exponent += diff * diff / cov[i];
                    det *= cov[i];
                }

                let norm = 1.0 / ((2.0 * std::f32::consts::PI).powf(self.n_features as f32 / 2.0) * det.sqrt());
                (norm * (-0.5 * exponent).exp()).max(1e-10)
            }
            HMMCovarianceType::Spherical => {
                let variance = cov[0];
                let mut exponent = 0.0;
                
                for i in 0..self.n_features {
                    let diff = obs[i] - mean[i];
                    exponent += diff * diff;
                }

                let norm = 1.0 / ((2.0 * std::f32::consts::PI * variance).powf(self.n_features as f32 / 2.0));
                (norm * (-exponent / (2.0 * variance)).exp()).max(1e-10)
            }
        }
    }

    /// Compute log likelihood
    fn compute_log_likelihood(&self, sequences: &[Tensor]) -> f32 {
        let mut total_log_likelihood = 0.0;

        for seq in sequences {
            let seq_data = seq.data_f32();
            let seq_len = seq.dims()[0];
            let (_, log_likelihood) = self.forward(&seq_data, seq_len);
            total_log_likelihood += log_likelihood;
        }

        total_log_likelihood
    }

    /// Predict hidden state sequence using Viterbi algorithm
    pub fn predict(&self, sequence: &Tensor) -> Tensor {
        let seq_data = sequence.data_f32();
        let seq_len = sequence.dims()[0];

        let mut delta = vec![vec![0.0; self.n_components]; seq_len];
        let mut psi = vec![vec![0; self.n_components]; seq_len];

        // Initialize
        for i in 0..self.n_components {
            let obs = &seq_data[0..self.n_features];
            delta[0][i] = self.start_prob[i].ln() + self.emission_prob(obs, i).ln();
        }

        // Recursion
        for t in 1..seq_len {
            for j in 0..self.n_components {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0;

                for i in 0..self.n_components {
                    let val = delta[t - 1][i] + self.trans_prob[i][j].ln();
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }

                let obs = &seq_data[t * self.n_features..(t + 1) * self.n_features];
                delta[t][j] = max_val + self.emission_prob(obs, j).ln();
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_hmm() {
        // Create simple sequence
        let seq1 = Tensor::from_slice(
            &[0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1],
            &[4, 2],
        ).unwrap();

        let sequences = vec![seq1];

        let mut hmm = GaussianHMM::new(2, 2)
            .covariance_type(HMMCovarianceType::Diag)
            .max_iter(20);

        hmm.fit(&sequences);

        let test_seq = Tensor::from_slice(&[0.0, 0.0, 5.0, 5.0], &[2, 2]).unwrap();
        let states = hmm.predict(&test_seq);

        assert_eq!(states.dims(), &[2]);
    }
}

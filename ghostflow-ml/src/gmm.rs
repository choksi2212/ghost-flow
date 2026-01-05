//! Gaussian Mixture Models (GMM)
//!
//! Probabilistic model for representing normally distributed subpopulations
//! within an overall population.

use ghostflow_core::Tensor;
use rand::prelude::*;
use std::f32::consts::PI;

/// Gaussian Mixture Model
pub struct GaussianMixture {
    pub n_components: usize,
    pub covariance_type: CovarianceType,
    pub max_iter: usize,
    pub tol: f32,
    pub reg_covar: f32,
    pub n_init: usize,
    
    // Learned parameters
    weights: Vec<f32>,          // Mixture weights (n_components,)
    means: Vec<Vec<f32>>,       // Component means (n_components, n_features)
    covariances: Vec<Vec<f32>>, // Component covariances
    converged: bool,
}

#[derive(Clone, Copy)]
pub enum CovarianceType {
    Full,      // Each component has its own general covariance matrix
    Tied,      // All components share the same general covariance matrix
    Diag,      // Each component has its own diagonal covariance matrix
    Spherical, // Each component has its own single variance
}

impl GaussianMixture {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
            n_init: 1,
            weights: Vec::new(),
            means: Vec::new(),
            covariances: Vec::new(),
            converged: false,
        }
    }

    pub fn covariance_type(mut self, cov_type: CovarianceType) -> Self {
        self.covariance_type = cov_type;
        self
    }

    pub fn max_iter(mut self, iter: usize) -> Self {
        self.max_iter = iter;
        self
    }

    pub fn tol(mut self, tolerance: f32) -> Self {
        self.tol = tolerance;
        self
    }

    /// Fit the Gaussian Mixture Model using EM algorithm
    pub fn fit(&mut self, x: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        let mut best_log_likelihood = f32::NEG_INFINITY;
        let mut best_weights = Vec::new();
        let mut best_means = Vec::new();
        let mut best_covariances = Vec::new();

        // Try multiple initializations
        for _ in 0..self.n_init {
            // Initialize parameters
            self.initialize_parameters(&x_data, n_samples, n_features);

            let mut prev_log_likelihood = f32::NEG_INFINITY;

            // EM algorithm
            for _iteration in 0..self.max_iter {
                // E-step: Calculate responsibilities
                let responsibilities = self.e_step(&x_data, n_samples, n_features);

                // M-step: Update parameters
                self.m_step(&x_data, &responsibilities, n_samples, n_features);

                // Calculate log likelihood
                let log_likelihood = self.compute_log_likelihood(&x_data, n_samples, n_features);

                // Check convergence
                if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                    self.converged = true;
                    break;
                }

                prev_log_likelihood = log_likelihood;
            }

            // Keep best result
            let final_log_likelihood = self.compute_log_likelihood(&x_data, n_samples, n_features);
            if final_log_likelihood > best_log_likelihood {
                best_log_likelihood = final_log_likelihood;
                best_weights = self.weights.clone();
                best_means = self.means.clone();
                best_covariances = self.covariances.clone();
            }
        }

        // Set best parameters
        self.weights = best_weights;
        self.means = best_means;
        self.covariances = best_covariances;
    }

    /// Initialize parameters using k-means++
    fn initialize_parameters(&mut self, x_data: &[f32], n_samples: usize, n_features: usize) {
        let mut rng = thread_rng();

        // Initialize weights uniformly
        self.weights = vec![1.0 / self.n_components as f32; self.n_components];

        // Initialize means using k-means++ strategy
        self.means = Vec::with_capacity(self.n_components);
        
        // First center: random sample
        let first_idx = rng.gen_range(0..n_samples);
        self.means.push(x_data[first_idx * n_features..(first_idx + 1) * n_features].to_vec());

        // Remaining centers: weighted by distance
        for _ in 1..self.n_components {
            let mut distances = vec![f32::MAX; n_samples];
            
            for i in 0..n_samples {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                let min_dist = self.means.iter()
                    .map(|mean| {
                        sample.iter().zip(mean.iter())
                            .map(|(x, m)| (x - m).powi(2))
                            .sum::<f32>()
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                distances[i] = min_dist;
            }

            // Sample proportional to squared distance
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

            self.means.push(x_data[selected_idx * n_features..(selected_idx + 1) * n_features].to_vec());
        }

        // Initialize covariances
        self.covariances = match self.covariance_type {
            CovarianceType::Full => {
                (0..self.n_components)
                    .map(|_| {
                        let mut cov = vec![0.0; n_features * n_features];
                        for i in 0..n_features {
                            cov[i * n_features + i] = 1.0;
                        }
                        cov
                    })
                    .collect()
            }
            CovarianceType::Diag => {
                (0..self.n_components)
                    .map(|_| vec![1.0; n_features])
                    .collect()
            }
            CovarianceType::Spherical => {
                (0..self.n_components)
                    .map(|_| vec![1.0])
                    .collect()
            }
            CovarianceType::Tied => {
                let mut cov = vec![0.0; n_features * n_features];
                for i in 0..n_features {
                    cov[i * n_features + i] = 1.0;
                }
                vec![cov]
            }
        };
    }

    /// E-step: Calculate responsibilities
    fn e_step(&self, x_data: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut responsibilities = vec![vec![0.0; self.n_components]; n_samples];

        for i in 0..n_samples {
            let sample = &x_data[i * n_features..(i + 1) * n_features];
            let mut total = 0.0;

            for k in 0..self.n_components {
                let prob = self.weights[k] * self.gaussian_pdf(sample, k, n_features);
                responsibilities[i][k] = prob;
                total += prob;
            }

            // Normalize
            if total > 0.0 {
                for k in 0..self.n_components {
                    responsibilities[i][k] /= total;
                }
            }
        }

        responsibilities
    }

    /// M-step: Update parameters
    fn m_step(&mut self, x_data: &[f32], responsibilities: &[Vec<f32>], n_samples: usize, n_features: usize) {
        // Update weights
        for k in 0..self.n_components {
            let n_k: f32 = responsibilities.iter().map(|r| r[k]).sum();
            self.weights[k] = n_k / n_samples as f32;

            // Update means
            let mut new_mean = vec![0.0; n_features];
            for i in 0..n_samples {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                for j in 0..n_features {
                    new_mean[j] += responsibilities[i][k] * sample[j];
                }
            }
            for j in 0..n_features {
                new_mean[j] /= n_k;
            }
            self.means[k] = new_mean;

            // Update covariances
            match self.covariance_type {
                CovarianceType::Diag => {
                    let mut new_cov = vec![0.0; n_features];
                    for i in 0..n_samples {
                        let sample = &x_data[i * n_features..(i + 1) * n_features];
                        for j in 0..n_features {
                            let diff = sample[j] - self.means[k][j];
                            new_cov[j] += responsibilities[i][k] * diff * diff;
                        }
                    }
                    for j in 0..n_features {
                        new_cov[j] = (new_cov[j] / n_k) + self.reg_covar;
                    }
                    self.covariances[k] = new_cov;
                }
                CovarianceType::Spherical => {
                    let mut variance = 0.0;
                    for i in 0..n_samples {
                        let sample = &x_data[i * n_features..(i + 1) * n_features];
                        for j in 0..n_features {
                            let diff = sample[j] - self.means[k][j];
                            variance += responsibilities[i][k] * diff * diff;
                        }
                    }
                    variance = (variance / (n_k * n_features as f32)) + self.reg_covar;
                    self.covariances[k] = vec![variance];
                }
                _ => {
                    // Full and Tied covariance (simplified implementation)
                    let mut new_cov = vec![0.0; n_features];
                    for i in 0..n_samples {
                        let sample = &x_data[i * n_features..(i + 1) * n_features];
                        for j in 0..n_features {
                            let diff = sample[j] - self.means[k][j];
                            new_cov[j] += responsibilities[i][k] * diff * diff;
                        }
                    }
                    for j in 0..n_features {
                        new_cov[j] = (new_cov[j] / n_k) + self.reg_covar;
                    }
                    self.covariances[k] = new_cov;
                }
            }
        }
    }

    /// Calculate Gaussian PDF
    fn gaussian_pdf(&self, sample: &[f32], component: usize, n_features: usize) -> f32 {
        let mean = &self.means[component];
        let cov = &self.covariances[component];

        match self.covariance_type {
            CovarianceType::Diag | CovarianceType::Full => {
                let mut exponent = 0.0;
                let mut det = 1.0;
                
                for i in 0..n_features {
                    let diff = sample[i] - mean[i];
                    exponent += diff * diff / cov[i];
                    det *= cov[i];
                }

                let norm = 1.0 / ((2.0 * PI).powf(n_features as f32 / 2.0) * det.sqrt());
                norm * (-0.5 * exponent).exp()
            }
            CovarianceType::Spherical => {
                let variance = cov[0];
                let mut exponent = 0.0;
                
                for i in 0..n_features {
                    let diff = sample[i] - mean[i];
                    exponent += diff * diff;
                }

                let norm = 1.0 / ((2.0 * PI * variance).powf(n_features as f32 / 2.0));
                norm * (-exponent / (2.0 * variance)).exp()
            }
            CovarianceType::Tied => {
                // Simplified: treat as diagonal
                let mut exponent = 0.0;
                let mut det = 1.0;
                
                for i in 0..n_features {
                    let diff = sample[i] - mean[i];
                    let var = if component == 0 { cov[i] } else { self.covariances[0][i] };
                    exponent += diff * diff / var;
                    det *= var;
                }

                let norm = 1.0 / ((2.0 * PI).powf(n_features as f32 / 2.0) * det.sqrt());
                norm * (-0.5 * exponent).exp()
            }
        }
    }

    /// Compute log likelihood
    fn compute_log_likelihood(&self, x_data: &[f32], n_samples: usize, n_features: usize) -> f32 {
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let sample = &x_data[i * n_features..(i + 1) * n_features];
            let mut prob = 0.0;

            for k in 0..self.n_components {
                prob += self.weights[k] * self.gaussian_pdf(sample, k, n_features);
            }

            log_likelihood += prob.max(1e-10).ln();
        }

        log_likelihood
    }

    /// Predict cluster labels
    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        let labels: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                let mut max_prob = 0.0;
                let mut best_component = 0;

                for k in 0..self.n_components {
                    let prob = self.weights[k] * self.gaussian_pdf(sample, k, n_features);
                    if prob > max_prob {
                        max_prob = prob;
                        best_component = k;
                    }
                }

                best_component as f32
            })
            .collect();

        Tensor::from_slice(&labels, &[n_samples]).unwrap()
    }

    /// Predict probabilities for each component
    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        let mut probabilities = Vec::with_capacity(n_samples * self.n_components);

        for i in 0..n_samples {
            let sample = &x_data[i * n_features..(i + 1) * n_features];
            let mut total = 0.0;
            let mut probs = vec![0.0; self.n_components];

            for k in 0..self.n_components {
                probs[k] = self.weights[k] * self.gaussian_pdf(sample, k, n_features);
                total += probs[k];
            }

            // Normalize
            for k in 0..self.n_components {
                probabilities.push(probs[k] / total);
            }
        }

        Tensor::from_slice(&probabilities, &[n_samples, self.n_components]).unwrap()
    }

    /// Sample from the fitted model
    pub fn sample(&self, n_samples: usize) -> Tensor {
        let mut rng = thread_rng();
        let n_features = self.means[0].len();
        let mut samples = Vec::with_capacity(n_samples * n_features);

        for _ in 0..n_samples {
            // Choose component
            let rand_val = rng.gen::<f32>();
            let mut cumsum = 0.0;
            let mut component = 0;
            
            for (k, &weight) in self.weights.iter().enumerate() {
                cumsum += weight;
                if cumsum >= rand_val {
                    component = k;
                    break;
                }
            }

            // Sample from Gaussian
            let mean = &self.means[component];
            let cov = &self.covariances[component];

            for j in 0..n_features {
                let std = match self.covariance_type {
                    CovarianceType::Spherical => cov[0].sqrt(),
                    _ => cov[j].sqrt(),
                };
                let sample = mean[j] + rng.gen::<f32>() * std;
                samples.push(sample);
            }
        }

        Tensor::from_slice(&samples, &[n_samples, n_features]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mixture() {
        // Create simple 2-cluster data
        let x = Tensor::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                5.0, 5.0,
                5.1, 5.1,
            ],
            &[4, 2],
        ).unwrap();

        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Diag)
            .max_iter(50);

        gmm.fit(&x);
        let labels = gmm.predict(&x);

        assert_eq!(labels.dims(), &[4]);
    }

    #[test]
    fn test_gmm_predict_proba() {
        let x = Tensor::from_slice(
            &[0.0, 0.0, 1.0, 1.0],
            &[2, 2],
        ).unwrap();

        let mut gmm = GaussianMixture::new(2);
        gmm.fit(&x);
        let proba = gmm.predict_proba(&x);

        assert_eq!(proba.dims(), &[2, 2]);
    }
}

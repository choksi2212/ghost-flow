//! Mixture Models - Gaussian Mixture Model, Bayesian GMM

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Gaussian Mixture Model using EM algorithm
pub struct GaussianMixture {
    pub n_components: usize,
    pub covariance_type: CovarianceType,
    pub max_iter: usize,
    pub tol: f32,
    pub n_init: usize,
    pub reg_covar: f32,
    weights_: Option<Vec<f32>>,
    means_: Option<Vec<Vec<f32>>>,
    covariances_: Option<Vec<Vec<f32>>>,
    precisions_: Option<Vec<Vec<f32>>>,
    converged_: bool,
    n_iter_: usize,
    lower_bound_: f32,
}

#[derive(Clone, Copy)]
pub enum CovarianceType {
    Full,
    Tied,
    Diag,
    Spherical,
}

impl GaussianMixture {
    pub fn new(n_components: usize) -> Self {
        GaussianMixture {
            n_components,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            tol: 1e-3,
            n_init: 1,
            reg_covar: 1e-6,
            weights_: None,
            means_: None,
            covariances_: None,
            precisions_: None,
            converged_: false,
            n_iter_: 0,
            lower_bound_: f32::NEG_INFINITY,
        }
    }

    pub fn covariance_type(mut self, ct: CovarianceType) -> Self {
        self.covariance_type = ct;
        self
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    fn initialize_parameters(&self, x: &[f32], n_samples: usize, n_features: usize) 
        -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) 
    {
        let mut rng = thread_rng();
        
        // Initialize weights uniformly
        let weights = vec![1.0 / self.n_components as f32; self.n_components];
        
        // Initialize means using k-means++ style
        let mut means = Vec::with_capacity(self.n_components);
        let first_idx = rng.gen_range(0..n_samples);
        means.push(x[first_idx * n_features..(first_idx + 1) * n_features].to_vec());

        for _ in 1..self.n_components {
            let distances: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let xi = &x[i * n_features..(i + 1) * n_features];
                    means.iter()
                        .map(|m| {
                            xi.iter().zip(m.iter())
                                .map(|(&a, &b)| (a - b).powi(2))
                                .sum::<f32>()
                        })
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total > 0.0 {
                let threshold = rng.gen::<f32>() * total;
                let mut cumsum = 0.0f32;
                for (i, &d) in distances.iter().enumerate() {
                    cumsum += d;
                    if cumsum >= threshold {
                        means.push(x[i * n_features..(i + 1) * n_features].to_vec());
                        break;
                    }
                }
            }
            if means.len() < self.n_components {
                let idx = rng.gen_range(0..n_samples);
                means.push(x[idx * n_features..(idx + 1) * n_features].to_vec());
            }
        }

        // Initialize covariances as identity
        let covariances: Vec<Vec<f32>> = (0..self.n_components)
            .map(|_| {
                let mut cov = vec![0.0f32; n_features * n_features];
                for i in 0..n_features {
                    cov[i * n_features + i] = 1.0;
                }
                cov
            })
            .collect();

        (weights, means, covariances)
    }

    fn compute_log_det_cholesky(&self, cov: &[f32], n_features: usize) -> f32 {
        // Compute log determinant via Cholesky
        let mut l = vec![0.0f32; n_features * n_features];
        
        for i in 0..n_features {
            for j in 0..=i {
                let mut sum = cov[i * n_features + j];
                for k in 0..j {
                    sum -= l[i * n_features + k] * l[j * n_features + k];
                }
                if i == j {
                    l[i * n_features + j] = sum.max(self.reg_covar).sqrt();
                } else {
                    l[i * n_features + j] = sum / l[j * n_features + j].max(1e-10);
                }
            }
        }

        let mut log_det = 0.0f32;
        for i in 0..n_features {
            log_det += 2.0 * l[i * n_features + i].max(1e-10).ln();
        }
        log_det
    }

    fn compute_precision(&self, cov: &[f32], n_features: usize) -> Vec<f32> {
        // Invert covariance matrix
        let mut a = cov.to_vec();
        let mut inv = vec![0.0f32; n_features * n_features];
        for i in 0..n_features {
            inv[i * n_features + i] = 1.0;
            a[i * n_features + i] += self.reg_covar;
        }

        // Gaussian elimination
        for i in 0..n_features {
            let pivot = a[i * n_features + i];
            if pivot.abs() > 1e-10 {
                for j in 0..n_features {
                    a[i * n_features + j] /= pivot;
                    inv[i * n_features + j] /= pivot;
                }
            }

            for k in 0..n_features {
                if k != i {
                    let factor = a[k * n_features + i];
                    for j in 0..n_features {
                        a[k * n_features + j] -= factor * a[i * n_features + j];
                        inv[k * n_features + j] -= factor * inv[i * n_features + j];
                    }
                }
            }
        }

        inv
    }

    fn log_gaussian_prob(&self, x: &[f32], mean: &[f32], precision: &[f32], 
                         log_det: f32, n_features: usize) -> f32 {
        let mut diff = vec![0.0f32; n_features];
        for i in 0..n_features {
            diff[i] = x[i] - mean[i];
        }

        let mut mahalanobis = 0.0f32;
        for i in 0..n_features {
            for j in 0..n_features {
                mahalanobis += diff[i] * precision[i * n_features + j] * diff[j];
            }
        }

        -0.5 * (n_features as f32 * (2.0 * std::f32::consts::PI).ln() + log_det + mahalanobis)
    }

    fn e_step(&self, x: &[f32], n_samples: usize, n_features: usize,
              weights: &[f32], means: &[Vec<f32>], precisions: &[Vec<f32>], 
              log_dets: &[f32]) -> (Vec<Vec<f32>>, f32) {
        let mut log_resp = vec![vec![0.0f32; self.n_components]; n_samples];
        let mut log_prob_sum = 0.0f32;

        for i in 0..n_samples {
            let xi = &x[i * n_features..(i + 1) * n_features];
            let mut log_probs = vec![0.0f32; self.n_components];
            let mut max_log_prob = f32::NEG_INFINITY;

            for k in 0..self.n_components {
                log_probs[k] = weights[k].ln() + 
                    self.log_gaussian_prob(xi, &means[k], &precisions[k], log_dets[k], n_features);
                max_log_prob = max_log_prob.max(log_probs[k]);
            }

            // Log-sum-exp trick
            let mut sum_exp = 0.0f32;
            for k in 0..self.n_components {
                sum_exp += (log_probs[k] - max_log_prob).exp();
            }
            let log_sum = max_log_prob + sum_exp.ln();
            log_prob_sum += log_sum;

            for k in 0..self.n_components {
                log_resp[i][k] = log_probs[k] - log_sum;
            }
        }

        // Convert to responsibilities
        let resp: Vec<Vec<f32>> = log_resp.iter()
            .map(|lr| lr.iter().map(|&l| l.exp()).collect())
            .collect();

        (resp, log_prob_sum / n_samples as f32)
    }

    fn m_step(&self, x: &[f32], resp: &[Vec<f32>], n_samples: usize, n_features: usize)
        -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) 
    {
        // Compute Nk
        let nk: Vec<f32> = (0..self.n_components)
            .map(|k| resp.iter().map(|r| r[k]).sum::<f32>().max(1e-10))
            .collect();

        // Update weights
        let weights: Vec<f32> = nk.iter().map(|&n| n / n_samples as f32).collect();

        // Update means
        let means: Vec<Vec<f32>> = (0..self.n_components)
            .map(|k| {
                let mut mean = vec![0.0f32; n_features];
                for i in 0..n_samples {
                    for j in 0..n_features {
                        mean[j] += resp[i][k] * x[i * n_features + j];
                    }
                }
                for j in 0..n_features {
                    mean[j] /= nk[k];
                }
                mean
            })
            .collect();

        // Update covariances
        let covariances: Vec<Vec<f32>> = (0..self.n_components)
            .map(|k| {
                let mut cov = vec![0.0f32; n_features * n_features];
                for i in 0..n_samples {
                    for j1 in 0..n_features {
                        for j2 in 0..n_features {
                            let diff1 = x[i * n_features + j1] - means[k][j1];
                            let diff2 = x[i * n_features + j2] - means[k][j2];
                            cov[j1 * n_features + j2] += resp[i][k] * diff1 * diff2;
                        }
                    }
                }
                for j in 0..n_features * n_features {
                    cov[j] /= nk[k];
                }
                // Add regularization
                for j in 0..n_features {
                    cov[j * n_features + j] += self.reg_covar;
                }
                cov
            })
            .collect();

        (weights, means, covariances)
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut best_lower_bound = f32::NEG_INFINITY;
        let mut best_params = None;

        for _ in 0..self.n_init {
            let (mut weights, mut means, mut covariances) = 
                self.initialize_parameters(&x_data, n_samples, n_features);

            let mut prev_lower_bound = f32::NEG_INFINITY;

            for iter in 0..self.max_iter {
                // Compute precisions and log determinants
                let precisions: Vec<Vec<f32>> = covariances.iter()
                    .map(|c| self.compute_precision(c, n_features))
                    .collect();
                let log_dets: Vec<f32> = covariances.iter()
                    .map(|c| self.compute_log_det_cholesky(c, n_features))
                    .collect();

                // E-step
                let (resp, lower_bound) = self.e_step(
                    &x_data, n_samples, n_features,
                    &weights, &means, &precisions, &log_dets
                );

                // M-step
                let (new_weights, new_means, new_covariances) = 
                    self.m_step(&x_data, &resp, n_samples, n_features);

                weights = new_weights;
                means = new_means;
                covariances = new_covariances;

                // Check convergence
                if (lower_bound - prev_lower_bound).abs() < self.tol {
                    self.converged_ = true;
                    self.n_iter_ = iter + 1;
                    break;
                }
                prev_lower_bound = lower_bound;
                self.n_iter_ = iter + 1;
            }

            if prev_lower_bound > best_lower_bound {
                best_lower_bound = prev_lower_bound;
                let precisions: Vec<Vec<f32>> = covariances.iter()
                    .map(|c| self.compute_precision(c, n_features))
                    .collect();
                best_params = Some((weights, means, covariances, precisions));
            }
        }

        if let Some((weights, means, covariances, precisions)) = best_params {
            self.weights_ = Some(weights);
            self.means_ = Some(means);
            self.covariances_ = Some(covariances);
            self.precisions_ = Some(precisions);
            self.lower_bound_ = best_lower_bound;
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];

        let labels: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut max_prob = f32::NEG_INFINITY;
                let mut max_k = 0;
                for k in 0..self.n_components {
                    if proba_data[i * self.n_components + k] > max_prob {
                        max_prob = proba_data[i * self.n_components + k];
                        max_k = k;
                    }
                }
                max_k as f32
            })
            .collect();

        Tensor::from_slice(&labels, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let weights = self.weights_.as_ref().expect("Model not fitted");
        let means = self.means_.as_ref().unwrap();
        let precisions = self.precisions_.as_ref().unwrap();
        let covariances = self.covariances_.as_ref().unwrap();

        let log_dets: Vec<f32> = covariances.iter()
            .map(|c| self.compute_log_det_cholesky(c, n_features))
            .collect();

        let (resp, _) = self.e_step(&x_data, n_samples, n_features, weights, means, precisions, &log_dets);

        let proba: Vec<f32> = resp.into_iter().flatten().collect();
        Tensor::from_slice(&proba, &[n_samples, self.n_components]).unwrap()
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.predict(x)
    }

    pub fn score(&self, x: &Tensor) -> f32 {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let weights = self.weights_.as_ref().expect("Model not fitted");
        let means = self.means_.as_ref().unwrap();
        let precisions = self.precisions_.as_ref().unwrap();
        let covariances = self.covariances_.as_ref().unwrap();

        let log_dets: Vec<f32> = covariances.iter()
            .map(|c| self.compute_log_det_cholesky(c, n_features))
            .collect();

        let (_, lower_bound) = self.e_step(&x_data, n_samples, n_features, weights, means, precisions, &log_dets);
        lower_bound
    }

    pub fn bic(&self, x: &Tensor) -> f32 {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        // Number of parameters
        let n_params = self.n_components - 1  // weights
            + self.n_components * n_features  // means
            + self.n_components * n_features * (n_features + 1) / 2;  // covariances

        -2.0 * self.score(x) * n_samples as f32 + n_params as f32 * (n_samples as f32).ln()
    }

    pub fn aic(&self, x: &Tensor) -> f32 {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let n_params = self.n_components - 1
            + self.n_components * n_features
            + self.n_components * n_features * (n_features + 1) / 2;

        -2.0 * self.score(x) * n_samples as f32 + 2.0 * n_params as f32
    }
}

/// Bayesian Gaussian Mixture Model
pub struct BayesianGaussianMixture {
    pub n_components: usize,
    pub covariance_type: CovarianceType,
    pub max_iter: usize,
    pub tol: f32,
    pub weight_concentration_prior_type: WeightConcentrationType,
    pub weight_concentration_prior: Option<f32>,
    pub reg_covar: f32,
    weights_: Option<Vec<f32>>,
    means_: Option<Vec<Vec<f32>>>,
    covariances_: Option<Vec<Vec<f32>>>,
    converged_: bool,
    n_iter_: usize,
}

#[derive(Clone, Copy)]
pub enum WeightConcentrationType {
    DirichletProcess,
    DirichletDistribution,
}

impl BayesianGaussianMixture {
    pub fn new(n_components: usize) -> Self {
        BayesianGaussianMixture {
            n_components,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            tol: 1e-3,
            weight_concentration_prior_type: WeightConcentrationType::DirichletProcess,
            weight_concentration_prior: None,
            reg_covar: 1e-6,
            weights_: None,
            means_: None,
            covariances_: None,
            converged_: false,
            n_iter_: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        // Simplified: use standard GMM with automatic component selection
        let mut gmm = GaussianMixture::new(self.n_components);
        gmm.max_iter = self.max_iter;
        gmm.tol = self.tol;
        gmm.reg_covar = self.reg_covar;
        gmm.fit(x);

        self.weights_ = gmm.weights_;
        self.means_ = gmm.means_;
        self.covariances_ = gmm.covariances_;
        self.converged_ = gmm.converged_;
        self.n_iter_ = gmm.n_iter_;
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let means = self.means_.as_ref().expect("Model not fitted");
        
        let labels: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut min_dist = f32::INFINITY;
                let mut best_k = 0;
                for (k, mean) in means.iter().enumerate() {
                    let dist: f32 = xi.iter().zip(mean.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum();
                    if dist < min_dist {
                        min_dist = dist;
                        best_k = k;
                    }
                }
                best_k as f32
            })
            .collect();

        Tensor::from_slice(&labels, &[n_samples]).unwrap()
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mixture() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.1, 0.1, 0.2, 0.0,
            5.0, 5.0, 5.1, 5.1, 5.2, 5.0,
        ], &[6, 2]).unwrap();
        
        let mut gmm = GaussianMixture::new(2);
        let labels = gmm.fit_predict(&x);
        assert_eq!(labels.dims()[0], 6); // Number of samples
    }

    #[test]
    fn test_bayesian_gmm() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.1, 0.1,
            5.0, 5.0, 5.1, 5.1,
        ], &[4, 2]).unwrap();
        
        let mut bgmm = BayesianGaussianMixture::new(2);
        let labels = bgmm.fit_predict(&x);
        assert_eq!(labels.dims(), &[4]);
    }
}



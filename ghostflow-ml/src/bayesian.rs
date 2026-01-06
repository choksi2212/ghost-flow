//! Bayesian Linear Models - Bayesian Ridge, ARD Regression

use ghostflow_core::Tensor;

/// Bayesian Ridge Regression with automatic relevance determination
pub struct BayesianRidge {
    pub n_iter: usize,
    pub tol: f32,
    pub alpha_1: f32,
    pub alpha_2: f32,
    pub lambda_1: f32,
    pub lambda_2: f32,
    pub fit_intercept: bool,
    coef_: Option<Vec<f32>>,
    intercept_: Option<f32>,
    alpha_: Option<f32>,
    lambda_: Option<f32>,
    sigma_: Option<Vec<f32>>,
    n_iter_: usize,
}

impl BayesianRidge {
    pub fn new() -> Self {
        BayesianRidge {
            n_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            fit_intercept: true,
            coef_: None,
            intercept_: None,
            alpha_: None,
            lambda_: None,
            sigma_: None,
            n_iter_: 0,
        }
    }

    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean: Vec<f32> = (0..n_features)
                .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
                .collect();
            let y_mean = y_data.iter().sum::<f32>() / n_samples as f32;

            let x_centered: Vec<f32> = (0..n_samples)
                .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - x_mean[j]).collect::<Vec<_>>())
                .collect();
            let y_centered: Vec<f32> = y_data.iter().map(|&y| y - y_mean).collect();

            (x_centered, y_centered, x_mean, y_mean)
        } else {
            (x_data.clone(), y_data.clone(), vec![0.0; n_features], 0.0)
        };

        // Initialize hyperparameters
        let mut alpha = 1.0f32;  // Noise precision
        let mut lambda = 1.0f32; // Weight precision

        // Compute X^T X
        let mut xtx = vec![0.0f32; n_features * n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    xtx[i * n_features + j] += x_centered[k * n_features + i] * x_centered[k * n_features + j];
                }
            }
        }

        // Compute X^T y
        let mut xty = vec![0.0f32; n_features];
        for i in 0..n_features {
            for k in 0..n_samples {
                xty[i] += x_centered[k * n_features + i] * y_centered[k];
            }
        }

        let mut coef = vec![0.0f32; n_features];
        let mut sigma = vec![0.0f32; n_features * n_features];

        for iter in 0..self.n_iter {
            let alpha_old = alpha;
            let lambda_old = lambda;

            // Compute posterior covariance: Sigma = (alpha * X^T X + lambda * I)^-1
            let mut a = xtx.clone();
            for i in 0..n_features {
                a[i * n_features + i] += lambda / alpha;
            }

            // Cholesky decomposition and solve
            sigma = self.invert_matrix(&a, n_features);

            // Compute posterior mean: m = alpha * Sigma * X^T y
            for i in 0..n_features {
                coef[i] = 0.0;
                for j in 0..n_features {
                    coef[i] += sigma[i * n_features + j] * xty[j];
                }
            }

            // Update alpha (noise precision)
            let mut residual_sum = 0.0f32;
            for i in 0..n_samples {
                let mut pred = 0.0f32;
                for j in 0..n_features {
                    pred += x_centered[i * n_features + j] * coef[j];
                }
                residual_sum += (y_centered[i] - pred).powi(2);
            }

            // Compute effective number of parameters
            let mut gamma = 0.0f32;
            for i in 0..n_features {
                let mut eigenvalue_approx = 0.0f32;
                for j in 0..n_features {
                    eigenvalue_approx += xtx[i * n_features + j] * sigma[j * n_features + i];
                }
                gamma += alpha * eigenvalue_approx;
            }
            gamma = gamma.clamp(0.0, n_features as f32);

            // Update hyperparameters
            alpha = (n_samples as f32 - gamma + 2.0 * self.alpha_1) / 
                    (residual_sum + 2.0 * self.alpha_2);

            let coef_sq_sum: f32 = coef.iter().map(|&c| c * c).sum();
            lambda = (gamma + 2.0 * self.lambda_1) / (coef_sq_sum + 2.0 * self.lambda_2);

            self.n_iter_ = iter + 1;

            // Check convergence
            if (alpha - alpha_old).abs() < self.tol && (lambda - lambda_old).abs() < self.tol {
                break;
            }
        }

        // Compute intercept
        let intercept = if self.fit_intercept {
            let mut int = y_mean;
            for j in 0..n_features {
                int -= coef[j] * x_mean[j];
            }
            int
        } else {
            0.0
        };

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        self.alpha_ = Some(alpha);
        self.lambda_ = Some(lambda);
        self.sigma_ = Some(sigma);
    }

    fn invert_matrix(&self, a: &[f32], n: usize) -> Vec<f32> {
        let mut l = vec![0.0f32; n * n];
        
        // Cholesky decomposition
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f32;
                if i == j {
                    for k in 0..j {
                        sum += l[j * n + k] * l[j * n + k];
                    }
                    l[j * n + j] = (a[j * n + j] - sum).max(1e-10).sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j].max(1e-10);
                }
            }
        }

        // Invert L
        let mut l_inv = vec![0.0f32; n * n];
        for i in 0..n {
            l_inv[i * n + i] = 1.0 / l[i * n + i].max(1e-10);
            for j in (i + 1)..n {
                let mut sum = 0.0f32;
                for k in i..j {
                    sum += l[j * n + k] * l_inv[k * n + i];
                }
                l_inv[j * n + i] = -sum / l[j * n + j].max(1e-10);
            }
        }

        // A^-1 = L_inv^T * L_inv
        let mut inv = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    inv[i * n + j] += l_inv[k * n + i] * l_inv[k * n + j];
                }
            }
        }

        inv
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.unwrap_or(0.0);

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_with_std(&self, x: &Tensor) -> (Tensor, Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.unwrap_or(0.0);
        let sigma = self.sigma_.as_ref().expect("Model not fitted");
        let alpha = self.alpha_.unwrap_or(1.0);

        let mut predictions = Vec::with_capacity(n_samples);
        let mut stds = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            
            let mut pred = intercept;
            for j in 0..n_features {
                pred += coef[j] * xi[j];
            }
            predictions.push(pred);

            // Predictive variance: 1/alpha + x^T Sigma x
            let mut var = 1.0 / alpha;
            for j in 0..n_features {
                for k in 0..n_features {
                    var += xi[j] * sigma[j * n_features + k] * xi[k];
                }
            }
            stds.push(var.sqrt());
        }

        (
            Tensor::from_slice(&predictions, &[n_samples]).unwrap(),
            Tensor::from_slice(&stds, &[n_samples]).unwrap(),
        )
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

impl Default for BayesianRidge {
    fn default() -> Self {
        Self::new()
    }
}


/// Automatic Relevance Determination Regression
pub struct ARDRegression {
    pub n_iter: usize,
    pub tol: f32,
    pub alpha_1: f32,
    pub alpha_2: f32,
    pub lambda_1: f32,
    pub lambda_2: f32,
    pub threshold_lambda: f32,
    pub fit_intercept: bool,
    coef_: Option<Vec<f32>>,
    intercept_: Option<f32>,
    alpha_: Option<f32>,
    lambda_: Option<Vec<f32>>,
    sigma_: Option<Vec<f32>>,
    n_iter_: usize,
}

impl ARDRegression {
    pub fn new() -> Self {
        ARDRegression {
            n_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            threshold_lambda: 1e4,
            fit_intercept: true,
            coef_: None,
            intercept_: None,
            alpha_: None,
            lambda_: None,
            sigma_: None,
            n_iter_: 0,
        }
    }

    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Center data
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean: Vec<f32> = (0..n_features)
                .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
                .collect();
            let y_mean = y_data.iter().sum::<f32>() / n_samples as f32;

            let x_centered: Vec<f32> = (0..n_samples)
                .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - x_mean[j]).collect::<Vec<_>>())
                .collect();
            let y_centered: Vec<f32> = y_data.iter().map(|&y| y - y_mean).collect();

            (x_centered, y_centered, x_mean, y_mean)
        } else {
            (x_data.clone(), y_data.clone(), vec![0.0; n_features], 0.0)
        };

        // Initialize
        let mut alpha = 1.0f32;
        let mut lambda = vec![1.0f32; n_features];  // Per-feature precision

        // Compute X^T X and X^T y
        let mut xtx = vec![0.0f32; n_features * n_features];
        let mut xty = vec![0.0f32; n_features];
        
        for i in 0..n_features {
            for k in 0..n_samples {
                xty[i] += x_centered[k * n_features + i] * y_centered[k];
            }
            for j in 0..n_features {
                for k in 0..n_samples {
                    xtx[i * n_features + j] += x_centered[k * n_features + i] * x_centered[k * n_features + j];
                }
            }
        }

        let mut coef = vec![0.0f32; n_features];
        let mut sigma = vec![0.0f32; n_features * n_features];

        for iter in 0..self.n_iter {
            let alpha_old = alpha;
            let lambda_old = lambda.clone();

            // Compute posterior covariance: Sigma = (alpha * X^T X + diag(lambda))^-1
            let mut a = xtx.clone();
            for i in 0..n_features {
                a[i * n_features + i] += lambda[i] / alpha;
            }

            sigma = self.invert_matrix(&a, n_features);

            // Compute posterior mean
            for i in 0..n_features {
                coef[i] = 0.0;
                for j in 0..n_features {
                    coef[i] += sigma[i * n_features + j] * xty[j];
                }
            }

            // Update alpha
            let mut residual_sum = 0.0f32;
            for i in 0..n_samples {
                let mut pred = 0.0f32;
                for j in 0..n_features {
                    pred += x_centered[i * n_features + j] * coef[j];
                }
                residual_sum += (y_centered[i] - pred).powi(2);
            }

            let mut gamma = 0.0f32;
            for i in 0..n_features {
                gamma += 1.0 - lambda[i] * sigma[i * n_features + i];
            }
            gamma = gamma.clamp(0.0, n_features as f32);

            alpha = (n_samples as f32 - gamma + 2.0 * self.alpha_1) / 
                    (residual_sum + 2.0 * self.alpha_2);

            // Update per-feature lambda (ARD)
            for i in 0..n_features {
                let gamma_i = 1.0 - lambda[i] * sigma[i * n_features + i];
                lambda[i] = (gamma_i + 2.0 * self.lambda_1) / 
                           (coef[i] * coef[i] + 2.0 * self.lambda_2);
                
                // Threshold very large lambda (feature pruning)
                if lambda[i] > self.threshold_lambda {
                    lambda[i] = self.threshold_lambda;
                }
            }

            self.n_iter_ = iter + 1;

            // Check convergence
            let alpha_diff = (alpha - alpha_old).abs();
            let lambda_diff: f32 = lambda.iter().zip(lambda_old.iter())
                .map(|(&l, &lo)| (l - lo).abs())
                .sum::<f32>() / n_features as f32;

            if alpha_diff < self.tol && lambda_diff < self.tol {
                break;
            }
        }

        // Compute intercept
        let intercept = if self.fit_intercept {
            let mut int = y_mean;
            for j in 0..n_features {
                int -= coef[j] * x_mean[j];
            }
            int
        } else {
            0.0
        };

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        self.alpha_ = Some(alpha);
        self.lambda_ = Some(lambda);
        self.sigma_ = Some(sigma);
    }

    fn invert_matrix(&self, a: &[f32], n: usize) -> Vec<f32> {
        let mut l = vec![0.0f32; n * n];
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f32;
                if i == j {
                    for k in 0..j {
                        sum += l[j * n + k] * l[j * n + k];
                    }
                    l[j * n + j] = (a[j * n + j] - sum).max(1e-10).sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j].max(1e-10);
                }
            }
        }

        let mut l_inv = vec![0.0f32; n * n];
        for i in 0..n {
            l_inv[i * n + i] = 1.0 / l[i * n + i].max(1e-10);
            for j in (i + 1)..n {
                let mut sum = 0.0f32;
                for k in i..j {
                    sum += l[j * n + k] * l_inv[k * n + i];
                }
                l_inv[j * n + i] = -sum / l[j * n + j].max(1e-10);
            }
        }

        let mut inv = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    inv[i * n + j] += l_inv[k * n + i] * l_inv[k * n + j];
                }
            }
        }

        inv
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.unwrap_or(0.0);

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                pred
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

impl Default for ARDRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_ridge() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap();
        let y = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();

        let mut br = BayesianRidge::new().n_iter(100);
        br.fit(&x, &y);

        let score = br.score(&x, &y);
        assert!(score > 0.9);
    }

    #[test]
    fn test_ard_regression() {
        let x = Tensor::from_slice(&[1.0f32, 0.0,
            2.0, 0.0,
            3.0, 0.0,
            4.0, 0.0,
        ], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[4]).unwrap();

        let mut ard = ARDRegression::new().n_iter(100);
        ard.fit(&x, &y);

        let predictions = ard.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }
}



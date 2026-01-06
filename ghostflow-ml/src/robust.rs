//! Robust Regression - Huber, RANSAC, Theil-Sen, Quantile Regression

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Huber Regressor - robust to outliers
pub struct HuberRegressor {
    pub epsilon: f32,
    pub max_iter: usize,
    pub alpha: f32,
    pub fit_intercept: bool,
    pub tol: f32,
    coef_: Option<Vec<f32>>,
    intercept_: f32,
    scale_: f32,
    n_iter_: usize,
}

impl HuberRegressor {
    pub fn new() -> Self {
        HuberRegressor {
            epsilon: 1.35,
            max_iter: 100,
            alpha: 0.0001,
            fit_intercept: true,
            tol: 1e-5,
            coef_: None,
            intercept_: 0.0,
            scale_: 1.0,
            n_iter_: 0,
        }
    }

    pub fn epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
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

            let x_c: Vec<f32> = (0..n_samples)
                .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - x_mean[j]).collect::<Vec<_>>())
                .collect();
            let y_c: Vec<f32> = y_data.iter().map(|&y| y - y_mean).collect();
            (x_c, y_c, x_mean, y_mean)
        } else {
            (x_data.clone(), y_data.clone(), vec![0.0; n_features], 0.0)
        };

        let mut coef = vec![0.0f32; n_features];

        // Initial scale estimate using MAD
        let mut residuals: Vec<f32> = (0..n_samples)
            .map(|i| y_centered[i])
            .collect();
        residuals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        let mut scale = residuals[n_samples / 2].abs() / 0.6745;
        scale = scale.max(1e-10);

        // IRLS iterations
        for iter in 0..self.max_iter {
            let coef_old = coef.clone();

            // Compute weights
            let mut weights = vec![0.0f32; n_samples];
            for i in 0..n_samples {
                let mut pred = 0.0f32;
                for j in 0..n_features {
                    pred += coef[j] * x_centered[i * n_features + j];
                }
                let residual = (y_centered[i] - pred) / scale;
                weights[i] = if residual.abs() <= self.epsilon {
                    1.0
                } else {
                    self.epsilon / residual.abs()
                };
            }

            // Weighted least squares
            let mut xtx = vec![0.0f32; n_features * n_features];
            let mut xty = vec![0.0f32; n_features];

            for i in 0..n_features {
                for k in 0..n_samples {
                    xty[i] += weights[k] * x_centered[k * n_features + i] * y_centered[k];
                }
                for j in 0..n_features {
                    for k in 0..n_samples {
                        xtx[i * n_features + j] += weights[k] * 
                            x_centered[k * n_features + i] * x_centered[k * n_features + j];
                    }
                }
                xtx[i * n_features + i] += self.alpha;
            }

            coef = solve_linear_system(&xtx, &xty, n_features);

            // Update scale
            let mut new_residuals: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let mut pred = 0.0f32;
                    for j in 0..n_features {
                        pred += coef[j] * x_centered[i * n_features + j];
                    }
                    (y_centered[i] - pred).abs()
                })
                .collect();
            new_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            scale = new_residuals[n_samples / 2] / 0.6745;
            scale = scale.max(1e-10);

            self.n_iter_ = iter + 1;

            let max_change = coef.iter().zip(coef_old.iter())
                .map(|(&new, &old)| (new - old).abs())
                .fold(0.0f32, f32::max);
            if max_change < self.tol {
                break;
            }
        }

        self.intercept_ = if self.fit_intercept {
            y_mean - coef.iter().zip(x_mean.iter()).map(|(&c, &m)| c * m).sum::<f32>()
        } else {
            0.0
        };
        self.scale_ = scale;
        self.coef_ = Some(coef);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let coef = self.coef_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = self.intercept_;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for HuberRegressor {
    fn default() -> Self { Self::new() }
}

/// RANSAC Regressor
pub struct RANSACRegressor {
    pub min_samples: Option<usize>,
    pub residual_threshold: Option<f32>,
    pub max_trials: usize,
    coef_: Option<Vec<f32>>,
    intercept_: f32,
    inlier_mask_: Option<Vec<bool>>,
}

impl RANSACRegressor {
    pub fn new() -> Self {
        RANSACRegressor {
            min_samples: None,
            residual_threshold: None,
            max_trials: 100,
            coef_: None,
            intercept_: 0.0,
            inlier_mask_: None,
        }
    }

    pub fn max_trials(mut self, n: usize) -> Self {
        self.max_trials = n;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let min_samples = self.min_samples.unwrap_or(n_features + 1);
        
        // Estimate threshold using MAD
        let y_mean = y_data.iter().sum::<f32>() / n_samples as f32;
        let mut deviations: Vec<f32> = y_data.iter().map(|&y| (y - y_mean).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = self.residual_threshold.unwrap_or(deviations[n_samples / 2] * 2.0);

        let mut rng = thread_rng();
        let mut best_coef = vec![0.0f32; n_features];
        let mut best_intercept = 0.0f32;
        let mut best_inliers = vec![false; n_samples];
        let mut best_n_inliers = 0;

        for _ in 0..self.max_trials {
            let indices: Vec<usize> = (0..n_samples).choose_multiple(&mut rng, min_samples);

            // Fit on sample
            let x_sample: Vec<f32> = indices.iter()
                .flat_map(|&i| (0..n_features).map(|j| x_data[i * n_features + j]).collect::<Vec<_>>())
                .collect();
            let y_sample: Vec<f32> = indices.iter().map(|&i| y_data[i]).collect();

            let (coef, intercept) = fit_ols(&x_sample, &y_sample, min_samples, n_features);

            // Count inliers
            let mut inliers = vec![false; n_samples];
            let mut n_inliers = 0;
            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                if (y_data[i] - pred).abs() <= threshold {
                    inliers[i] = true;
                    n_inliers += 1;
                }
            }

            if n_inliers > best_n_inliers {
                // Refit on inliers
                let inlier_indices: Vec<usize> = inliers.iter()
                    .enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
                
                if inlier_indices.len() >= min_samples {
                    let x_inliers: Vec<f32> = inlier_indices.iter()
                        .flat_map(|&i| (0..n_features).map(|j| x_data[i * n_features + j]).collect::<Vec<_>>())
                        .collect();
                    let y_inliers: Vec<f32> = inlier_indices.iter().map(|&i| y_data[i]).collect();

                    let (c, int) = fit_ols(&x_inliers, &y_inliers, inlier_indices.len(), n_features);
                    best_coef = c;
                    best_intercept = int;
                    best_inliers = inliers;
                    best_n_inliers = n_inliers;
                }
            }
        }

        self.coef_ = Some(best_coef);
        self.intercept_ = best_intercept;
        self.inlier_mask_ = Some(best_inliers);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let coef = self.coef_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = self.intercept_;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn inlier_mask(&self) -> Option<&Vec<bool>> {
        self.inlier_mask_.as_ref()
    }
}

impl Default for RANSACRegressor {
    fn default() -> Self { Self::new() }
}

/// Theil-Sen Regressor - median-based estimator
pub struct TheilSenRegressor {
    pub fit_intercept: bool,
    pub max_subpopulation: usize,
    coef_: Option<Vec<f32>>,
    intercept_: f32,
}

impl TheilSenRegressor {
    pub fn new() -> Self {
        TheilSenRegressor {
            fit_intercept: true,
            max_subpopulation: 10000,
            coef_: None,
            intercept_: 0.0,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut coef = vec![0.0f32; n_features];
        
        // Compute median of pairwise slopes for each feature
        for j in 0..n_features {
            let mut slopes = Vec::new();
            let max_pairs = self.max_subpopulation.min(n_samples * (n_samples - 1) / 2);
            let mut count = 0;
            
            'outer: for i1 in 0..n_samples {
                for i2 in (i1 + 1)..n_samples {
                    let dx = x_data[i2 * n_features + j] - x_data[i1 * n_features + j];
                    if dx.abs() > 1e-10 {
                        let dy = y_data[i2] - y_data[i1];
                        slopes.push(dy / dx);
                        count += 1;
                        if count >= max_pairs {
                            break 'outer;
                        }
                    }
                }
            }
            
            if !slopes.is_empty() {
                slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
                coef[j] = slopes[slopes.len() / 2];
            }
        }

        // Compute intercept as median of residuals
        self.intercept_ = if self.fit_intercept {
            let mut intercepts: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let mut pred = 0.0f32;
                    for j in 0..n_features {
                        pred += coef[j] * x_data[i * n_features + j];
                    }
                    y_data[i] - pred
                })
                .collect();
            intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            intercepts[n_samples / 2]
        } else {
            0.0
        };

        self.coef_ = Some(coef);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let coef = self.coef_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = self.intercept_;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for TheilSenRegressor {
    fn default() -> Self { Self::new() }
}

/// Quantile Regressor
pub struct QuantileRegressor {
    pub quantile: f32,
    pub alpha: f32,
    pub max_iter: usize,
    pub tol: f32,
    pub fit_intercept: bool,
    coef_: Option<Vec<f32>>,
    intercept_: f32,
}

impl QuantileRegressor {
    pub fn new(quantile: f32) -> Self {
        QuantileRegressor {
            quantile,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            coef_: None,
            intercept_: 0.0,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut coef = vec![0.0f32; n_features];
        let mut intercept = 0.0f32;
        let lr = 0.01f32;

        // Subgradient descent for quantile loss
        for _ in 0..self.max_iter {
            let mut grad_coef = vec![0.0f32; n_features];
            let mut grad_intercept = 0.0f32;

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                let residual = y_data[i] - pred;
                
                // Quantile loss subgradient
                let weight = if residual >= 0.0 { self.quantile } else { self.quantile - 1.0 };
                
                grad_intercept -= weight;
                for j in 0..n_features {
                    grad_coef[j] -= weight * x_data[i * n_features + j];
                }
            }

            // Update with L2 regularization
            for j in 0..n_features {
                grad_coef[j] = grad_coef[j] / n_samples as f32 + self.alpha * coef[j];
                coef[j] -= lr * grad_coef[j];
            }
            if self.fit_intercept {
                intercept -= lr * grad_intercept / n_samples as f32;
            }
        }

        self.coef_ = Some(coef);
        self.intercept_ = intercept;
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let coef = self.coef_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = self.intercept_;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

/// Passive Aggressive Regressor
pub struct PassiveAggressiveRegressor {
    pub c: f32,
    pub max_iter: usize,
    pub tol: f32,
    pub epsilon: f32,
    pub fit_intercept: bool,
    coef_: Option<Vec<f32>>,
    intercept_: f32,
}

impl PassiveAggressiveRegressor {
    pub fn new() -> Self {
        PassiveAggressiveRegressor {
            c: 1.0,
            max_iter: 1000,
            tol: 1e-3,
            epsilon: 0.1,
            fit_intercept: true,
            coef_: None,
            intercept_: 0.0,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut coef = vec![0.0f32; n_features];
        let mut intercept = 0.0f32;

        for _ in 0..self.max_iter {
            let mut max_update = 0.0f32;

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                
                let loss = (y_data[i] - pred).abs() - self.epsilon;
                if loss > 0.0 {
                    let mut x_norm_sq = 1.0f32; // for intercept
                    for j in 0..n_features {
                        x_norm_sq += x_data[i * n_features + j].powi(2);
                    }
                    
                    let sign = if y_data[i] > pred { 1.0 } else { -1.0 };
                    let tau = (loss / x_norm_sq).min(self.c);
                    
                    for j in 0..n_features {
                        let update = tau * sign * x_data[i * n_features + j];
                        coef[j] += update;
                        max_update = max_update.max(update.abs());
                    }
                    if self.fit_intercept {
                        intercept += tau * sign;
                    }
                }
            }

            if max_update < self.tol {
                break;
            }
        }

        self.coef_ = Some(coef);
        self.intercept_ = intercept;
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let coef = self.coef_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = self.intercept_;
                for j in 0..n_features {
                    pred += coef[j] * x_data[i * n_features + j];
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for PassiveAggressiveRegressor {
    fn default() -> Self { Self::new() }
}

// Helper functions
fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n * n];
    let mut a_copy = a.to_vec();
    
    for i in 0..n {
        a_copy[i * n + i] += 1e-10;
    }

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            if i == j {
                for k in 0..j {
                    sum += l[j * n + k] * l[j * n + k];
                }
                l[j * n + j] = (a_copy[j * n + j] - sum).max(1e-10).sqrt();
            } else {
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = (a_copy[i * n + j] - sum) / l[j * n + j].max(1e-10);
            }
        }
    }

    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i].max(1e-10);
    }

    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i].max(1e-10);
    }
    x
}

fn fit_ols(x: &[f32], y: &[f32], n_samples: usize, n_features: usize) -> (Vec<f32>, f32) {
    let x_mean: Vec<f32> = (0..n_features)
        .map(|j| (0..n_samples).map(|i| x[i * n_features + j]).sum::<f32>() / n_samples as f32)
        .collect();
    let y_mean = y.iter().sum::<f32>() / n_samples as f32;

    let mut xtx = vec![0.0f32; n_features * n_features];
    let mut xty = vec![0.0f32; n_features];

    for i in 0..n_features {
        for k in 0..n_samples {
            let xki = x[k * n_features + i] - x_mean[i];
            xty[i] += xki * (y[k] - y_mean);
        }
        for j in 0..n_features {
            for k in 0..n_samples {
                xtx[i * n_features + j] += 
                    (x[k * n_features + i] - x_mean[i]) * (x[k * n_features + j] - x_mean[j]);
            }
        }
        xtx[i * n_features + i] += 1e-10;
    }

    let coef = solve_linear_system(&xtx, &xty, n_features);
    let intercept = y_mean - coef.iter().zip(x_mean.iter()).map(|(&c, &m)| c * m).sum::<f32>();
    (coef, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huber() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let mut model = HuberRegressor::new();
        model.fit(&x, &y);
        let pred = model.predict(&x);
        assert_eq!(pred.dims(), &[3]);
    }

    #[test]
    fn test_ransac() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let mut model = RANSACRegressor::new();
        model.fit(&x, &y);
        let pred = model.predict(&x);
        assert_eq!(pred.dims(), &[3]);
    }
}



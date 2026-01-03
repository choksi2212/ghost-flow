//! Linear Models - Real implementations with gradient descent and closed-form solutions

use ghostflow_core::Tensor;

/// Linear Regression using Ordinary Least Squares (closed-form) or Gradient Descent
pub struct LinearRegression {
    /// Coefficients (weights)
    pub coef_: Option<Vec<f32>>,
    /// Intercept (bias)
    pub intercept_: Option<f32>,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Solver to use
    pub solver: LinearSolver,
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Maximum iterations for gradient descent
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum LinearSolver {
    /// Normal equation: (X^T X)^-1 X^T y
    ClosedForm,
    /// Gradient descent
    GradientDescent,
    /// Stochastic gradient descent
    SGD,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            coef_: None,
            intercept_: None,
            fit_intercept: true,
            solver: LinearSolver::ClosedForm,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    pub fn fit_intercept(mut self, fit: bool) -> Self {
        self.fit_intercept = fit;
        self
    }

    pub fn solver(mut self, solver: LinearSolver) -> Self {
        self.solver = solver;
        self
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Fit the linear regression model
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        match self.solver {
            LinearSolver::ClosedForm => self.fit_closed_form(x, y),
            LinearSolver::GradientDescent => self.fit_gradient_descent(x, y),
            LinearSolver::SGD => self.fit_sgd(x, y),
        }
    }

    /// Closed-form solution using normal equation
    fn fit_closed_form(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Add bias column if fitting intercept
        let n_cols = if self.fit_intercept { n_features + 1 } else { n_features };
        
        let mut x_aug = Vec::with_capacity(n_samples * n_cols);
        for i in 0..n_samples {
            if self.fit_intercept {
                x_aug.push(1.0); // Bias term
            }
            for j in 0..n_features {
                x_aug.push(x_data[i * n_features + j]);
            }
        }

        // Compute X^T X
        let mut xtx = vec![0.0f32; n_cols * n_cols];
        for i in 0..n_cols {
            for j in 0..n_cols {
                let mut sum = 0.0f32;
                for k in 0..n_samples {
                    sum += x_aug[k * n_cols + i] * x_aug[k * n_cols + j];
                }
                xtx[i * n_cols + j] = sum;
            }
        }

        // Compute X^T y
        let mut xty = vec![0.0f32; n_cols];
        for i in 0..n_cols {
            let mut sum = 0.0f32;
            for k in 0..n_samples {
                sum += x_aug[k * n_cols + i] * y_data[k];
            }
            xty[i] = sum;
        }

        // Solve (X^T X) w = X^T y using Cholesky decomposition
        let weights = self.solve_linear_system(&xtx, &xty, n_cols);

        if self.fit_intercept {
            self.intercept_ = Some(weights[0]);
            self.coef_ = Some(weights[1..].to_vec());
        } else {
            self.intercept_ = Some(0.0);
            self.coef_ = Some(weights);
        }
    }

    /// Solve linear system Ax = b using Cholesky decomposition
    fn solve_linear_system(&self, a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        // Add small regularization for numerical stability
        let mut a_reg = a.to_vec();
        for i in 0..n {
            a_reg[i * n + i] += 1e-8;
        }

        // Cholesky decomposition: A = L L^T
        let mut l = vec![0.0f32; n * n];
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f32;
                
                if i == j {
                    for k in 0..j {
                        sum += l[j * n + k] * l[j * n + k];
                    }
                    let val = a_reg[j * n + j] - sum;
                    l[j * n + j] = if val > 0.0 { val.sqrt() } else { 1e-10 };
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (a_reg[i * n + j] - sum) / l[j * n + j].max(1e-10);
                }
            }
        }

        // Forward substitution: L y = b
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..i {
                sum += l[i * n + j] * y[j];
            }
            y[i] = (b[i] - sum) / l[i * n + i].max(1e-10);
        }

        // Backward substitution: L^T x = y
        let mut x = vec![0.0f32; n];
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            for j in (i + 1)..n {
                sum += l[j * n + i] * x[j];
            }
            x[i] = (y[i] - sum) / l[i * n + i].max(1e-10);
        }

        x
    }

    /// Fit using batch gradient descent
    fn fit_gradient_descent(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Initialize weights
        let mut weights = vec![0.0f32; n_features];
        let mut bias = 0.0f32;

        for _iter in 0..self.max_iter {
            // Compute predictions
            let mut predictions = vec![0.0f32; n_samples];
            for i in 0..n_samples {
                let mut pred = bias;
                for j in 0..n_features {
                    pred += weights[j] * x_data[i * n_features + j];
                }
                predictions[i] = pred;
            }

            // Compute gradients
            let mut grad_w = vec![0.0f32; n_features];
            let mut grad_b = 0.0f32;

            for i in 0..n_samples {
                let error = predictions[i] - y_data[i];
                grad_b += error;
                for j in 0..n_features {
                    grad_w[j] += error * x_data[i * n_features + j];
                }
            }

            // Average gradients
            let scale = 1.0 / n_samples as f32;
            grad_b *= scale;
            for g in &mut grad_w {
                *g *= scale;
            }

            // Update weights
            if self.fit_intercept {
                bias -= self.learning_rate * grad_b;
            }
            for j in 0..n_features {
                weights[j] -= self.learning_rate * grad_w[j];
            }

            // Check convergence
            let grad_norm: f32 = grad_w.iter().map(|&g| g * g).sum::<f32>() + grad_b * grad_b;
            if grad_norm.sqrt() < self.tol {
                break;
            }
        }

        self.coef_ = Some(weights);
        self.intercept_ = Some(bias);
    }

    /// Fit using stochastic gradient descent
    fn fit_sgd(&mut self, x: &Tensor, y: &Tensor) {
        use rand::seq::SliceRandom;
        
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut weights = vec![0.0f32; n_features];
        let mut bias = 0.0f32;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for _epoch in 0..self.max_iter {
            indices.shuffle(&mut rand::thread_rng());

            for &i in &indices {
                // Compute prediction for single sample
                let mut pred = bias;
                for j in 0..n_features {
                    pred += weights[j] * x_data[i * n_features + j];
                }

                let error = pred - y_data[i];

                // Update weights
                if self.fit_intercept {
                    bias -= self.learning_rate * error;
                }
                for j in 0..n_features {
                    weights[j] -= self.learning_rate * error * x_data[i * n_features + j];
                }
            }
        }

        self.coef_ = Some(weights);
        self.intercept_ = Some(bias);
    }

    /// Predict target values
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

    /// Compute R² score
    pub fn score(&self, x: &Tensor, y: &Tensor) -> f32 {
        let predictions = self.predict(x);
        let pred_data = predictions.data_f32();
        let y_data = y.data_f32();

        let y_mean: f32 = y_data.iter().sum::<f32>() / y_data.len() as f32;

        let ss_res: f32 = pred_data.iter()
            .zip(y_data.iter())
            .map(|(&p, &y)| (y - p).powi(2))
            .sum();

        let ss_tot: f32 = y_data.iter()
            .map(|&y| (y - y_mean).powi(2))
            .sum();

        1.0 - ss_res / ss_tot.max(1e-10)
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}


/// Ridge Regression (L2 regularization)
pub struct Ridge {
    pub coef_: Option<Vec<f32>>,
    pub intercept_: Option<f32>,
    pub alpha: f32,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f32,
}

impl Ridge {
    pub fn new(alpha: f32) -> Self {
        Ridge {
            coef_: None,
            intercept_: None,
            alpha,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean: Vec<f32> = (0..n_features)
                .map(|j| {
                    (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32
                })
                .collect();
            
            let y_mean = y_data.iter().sum::<f32>() / n_samples as f32;

            let x_centered: Vec<f32> = (0..n_samples)
                .flat_map(|i| {
                    (0..n_features).map(|j| x_data[i * n_features + j] - x_mean[j]).collect::<Vec<_>>()
                })
                .collect();

            let y_centered: Vec<f32> = y_data.iter().map(|&y| y - y_mean).collect();

            (x_centered, y_centered, x_mean, y_mean)
        } else {
            (x_data.to_vec(), y_data.to_vec(), vec![0.0; n_features], 0.0)
        };

        // Compute X^T X + alpha * I
        let mut xtx = vec![0.0f32; n_features * n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0f32;
                for k in 0..n_samples {
                    sum += x_centered[k * n_features + i] * x_centered[k * n_features + j];
                }
                xtx[i * n_features + j] = sum;
                if i == j {
                    xtx[i * n_features + j] += self.alpha;
                }
            }
        }

        // Compute X^T y
        let mut xty = vec![0.0f32; n_features];
        for i in 0..n_features {
            let mut sum = 0.0f32;
            for k in 0..n_samples {
                sum += x_centered[k * n_features + i] * y_centered[k];
            }
            xty[i] = sum;
        }

        // Solve using Cholesky
        let weights = self.solve_cholesky(&xtx, &xty, n_features);

        self.coef_ = Some(weights.clone());

        // Compute intercept
        if self.fit_intercept {
            let mut intercept = y_mean;
            for j in 0..n_features {
                intercept -= weights[j] * x_mean[j];
            }
            self.intercept_ = Some(intercept);
        } else {
            self.intercept_ = Some(0.0);
        }
    }

    fn solve_cholesky(&self, a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
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

        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..i {
                sum += l[i * n + j] * y[j];
            }
            y[i] = (b[i] - sum) / l[i * n + i].max(1e-10);
        }

        let mut x = vec![0.0f32; n];
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            for j in (i + 1)..n {
                sum += l[j * n + i] * x[j];
            }
            x[i] = (y[i] - sum) / l[i * n + i].max(1e-10);
        }

        x
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


/// Lasso Regression (L1 regularization) using Coordinate Descent
pub struct Lasso {
    pub coef_: Option<Vec<f32>>,
    pub intercept_: Option<f32>,
    pub alpha: f32,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f32,
}

impl Lasso {
    pub fn new(alpha: f32) -> Self {
        Lasso {
            coef_: None,
            intercept_: None,
            alpha,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Soft thresholding operator
    fn soft_threshold(x: f32, lambda: f32) -> f32 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Center data
        let x_mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();
        
        let y_mean = if self.fit_intercept {
            y_data.iter().sum::<f32>() / n_samples as f32
        } else {
            0.0
        };

        let x_centered: Vec<f32> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - x_mean[j]).collect::<Vec<_>>())
            .collect();

        let y_centered: Vec<f32> = y_data.iter().map(|&y| y - y_mean).collect();

        // Precompute X^T X diagonal and X^T y
        let mut x_sq_sum = vec![0.0f32; n_features];
        for j in 0..n_features {
            for i in 0..n_samples {
                x_sq_sum[j] += x_centered[i * n_features + j].powi(2);
            }
        }

        // Initialize weights
        let mut weights = vec![0.0f32; n_features];
        let mut residuals = y_centered.clone();

        // Coordinate descent
        for _iter in 0..self.max_iter {
            let weights_old = weights.clone();

            for j in 0..n_features {
                // Add back contribution of current weight
                for i in 0..n_samples {
                    residuals[i] += weights[j] * x_centered[i * n_features + j];
                }

                // Compute correlation
                let mut rho = 0.0f32;
                for i in 0..n_samples {
                    rho += x_centered[i * n_features + j] * residuals[i];
                }

                // Update weight with soft thresholding
                let lambda = self.alpha * n_samples as f32;
                weights[j] = Self::soft_threshold(rho, lambda) / x_sq_sum[j].max(1e-10);

                // Update residuals
                for i in 0..n_samples {
                    residuals[i] -= weights[j] * x_centered[i * n_features + j];
                }
            }

            // Check convergence
            let max_change: f32 = weights.iter()
                .zip(weights_old.iter())
                .map(|(&w, &w_old)| (w - w_old).abs())
                .fold(0.0f32, f32::max);

            if max_change < self.tol {
                break;
            }
        }

        self.coef_ = Some(weights.clone());

        // Compute intercept
        if self.fit_intercept {
            let mut intercept = y_mean;
            for j in 0..n_features {
                intercept -= weights[j] * x_mean[j];
            }
            self.intercept_ = Some(intercept);
        } else {
            self.intercept_ = Some(0.0);
        }
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


/// Elastic Net (L1 + L2 regularization) using Coordinate Descent
pub struct ElasticNet {
    pub coef_: Option<Vec<f32>>,
    pub intercept_: Option<f32>,
    pub alpha: f32,
    pub l1_ratio: f32,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f32,
}

impl ElasticNet {
    pub fn new(alpha: f32, l1_ratio: f32) -> Self {
        ElasticNet {
            coef_: None,
            intercept_: None,
            alpha,
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    fn soft_threshold(x: f32, lambda: f32) -> f32 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let l1_reg = self.alpha * self.l1_ratio * n_samples as f32;
        let l2_reg = self.alpha * (1.0 - self.l1_ratio) * n_samples as f32;

        // Center data
        let x_mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();
        
        let y_mean = if self.fit_intercept {
            y_data.iter().sum::<f32>() / n_samples as f32
        } else {
            0.0
        };

        let x_centered: Vec<f32> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - x_mean[j]).collect::<Vec<_>>())
            .collect();

        let y_centered: Vec<f32> = y_data.iter().map(|&y| y - y_mean).collect();

        // Precompute X^T X diagonal
        let mut x_sq_sum = vec![0.0f32; n_features];
        for j in 0..n_features {
            for i in 0..n_samples {
                x_sq_sum[j] += x_centered[i * n_features + j].powi(2);
            }
        }

        let mut weights = vec![0.0f32; n_features];
        let mut residuals = y_centered.clone();

        for _iter in 0..self.max_iter {
            let weights_old = weights.clone();

            for j in 0..n_features {
                for i in 0..n_samples {
                    residuals[i] += weights[j] * x_centered[i * n_features + j];
                }

                let mut rho = 0.0f32;
                for i in 0..n_samples {
                    rho += x_centered[i * n_features + j] * residuals[i];
                }

                // Elastic net update
                let denom = x_sq_sum[j] + l2_reg;
                weights[j] = Self::soft_threshold(rho, l1_reg) / denom.max(1e-10);

                for i in 0..n_samples {
                    residuals[i] -= weights[j] * x_centered[i * n_features + j];
                }
            }

            let max_change: f32 = weights.iter()
                .zip(weights_old.iter())
                .map(|(&w, &w_old)| (w - w_old).abs())
                .fold(0.0f32, f32::max);

            if max_change < self.tol {
                break;
            }
        }

        self.coef_ = Some(weights.clone());

        if self.fit_intercept {
            let mut intercept = y_mean;
            for j in 0..n_features {
                intercept -= weights[j] * x_mean[j];
            }
            self.intercept_ = Some(intercept);
        } else {
            self.intercept_ = Some(0.0);
        }
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


/// Logistic Regression for binary and multiclass classification
pub struct LogisticRegression {
    pub coef_: Option<Vec<Vec<f32>>>,
    pub intercept_: Option<Vec<f32>>,
    pub penalty: Penalty,
    pub c: f32,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f32,
    pub multi_class: MultiClass,
    n_classes: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Penalty {
    None,
    L1,
    L2,
    ElasticNet(f32),
}

#[derive(Clone, Copy, Debug)]
pub enum MultiClass {
    /// One-vs-Rest for multiclass
    OvR,
    /// Multinomial (softmax)
    Multinomial,
}

impl LogisticRegression {
    pub fn new() -> Self {
        LogisticRegression {
            coef_: None,
            intercept_: None,
            penalty: Penalty::L2,
            c: 1.0,
            fit_intercept: true,
            max_iter: 100,
            tol: 1e-4,
            multi_class: MultiClass::OvR,
            n_classes: 0,
        }
    }

    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    pub fn c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    pub fn multi_class(mut self, mc: MultiClass) -> Self {
        self.multi_class = mc;
        self
    }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let y_data = y.data_f32();
        self.n_classes = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        if self.n_classes == 2 {
            self.fit_binary(x, y);
        } else {
            match self.multi_class {
                MultiClass::OvR => self.fit_ovr(x, y),
                MultiClass::Multinomial => self.fit_multinomial(x, y),
            }
        }
    }

    fn fit_binary(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut weights = vec![0.0f32; n_features];
        let mut bias = 0.0f32;

        let reg = 1.0 / self.c;

        // L-BFGS or gradient descent
        for _iter in 0..self.max_iter {
            let mut grad_w = vec![0.0f32; n_features];
            let mut grad_b = 0.0f32;

            for i in 0..n_samples {
                let mut z = bias;
                for j in 0..n_features {
                    z += weights[j] * x_data[i * n_features + j];
                }
                let p = Self::sigmoid(z);
                let error = p - y_data[i];

                grad_b += error;
                for j in 0..n_features {
                    grad_w[j] += error * x_data[i * n_features + j];
                }
            }

            // Add regularization gradient
            match self.penalty {
                Penalty::L2 => {
                    for j in 0..n_features {
                        grad_w[j] += reg * weights[j];
                    }
                }
                Penalty::L1 => {
                    for j in 0..n_features {
                        grad_w[j] += reg * weights[j].signum();
                    }
                }
                _ => {}
            }

            // Scale gradients
            let scale = 1.0 / n_samples as f32;
            grad_b *= scale;
            for g in &mut grad_w {
                *g *= scale;
            }

            // Update with learning rate
            let lr = 0.1;
            if self.fit_intercept {
                bias -= lr * grad_b;
            }
            for j in 0..n_features {
                weights[j] -= lr * grad_w[j];
            }

            // Check convergence
            let grad_norm: f32 = grad_w.iter().map(|&g| g * g).sum::<f32>().sqrt();
            if grad_norm < self.tol {
                break;
            }
        }

        self.coef_ = Some(vec![weights]);
        self.intercept_ = Some(vec![bias]);
    }

    fn fit_ovr(&mut self, x: &Tensor, y: &Tensor) {
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let _n_features = x.dims()[1];

        let mut all_coef = Vec::with_capacity(self.n_classes);
        let mut all_intercept = Vec::with_capacity(self.n_classes);

        for c in 0..self.n_classes {
            // Create binary labels
            let y_binary: Vec<f32> = y_data.iter()
                .map(|&yi| if yi as usize == c { 1.0 } else { 0.0 })
                .collect();
            let y_tensor = Tensor::from_slice(&y_binary, &[n_samples]).unwrap();

            let mut binary_lr = LogisticRegression::new()
                .penalty(self.penalty)
                .c(self.c)
                .max_iter(self.max_iter);
            binary_lr.n_classes = 2;
            binary_lr.fit_binary(x, &y_tensor);

            all_coef.push(binary_lr.coef_.unwrap().into_iter().next().unwrap());
            all_intercept.push(binary_lr.intercept_.unwrap().into_iter().next().unwrap());
        }

        self.coef_ = Some(all_coef);
        self.intercept_ = Some(all_intercept);
    }

    fn fit_multinomial(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut weights = vec![vec![0.0f32; n_features]; self.n_classes];
        let mut biases = vec![0.0f32; self.n_classes];

        let reg = 1.0 / self.c;
        let lr = 0.1;

        for _iter in 0..self.max_iter {
            // Compute softmax probabilities
            let mut probs = vec![vec![0.0f32; self.n_classes]; n_samples];
            
            for i in 0..n_samples {
                let mut max_z = f32::NEG_INFINITY;
                let mut z = vec![0.0f32; self.n_classes];
                
                for c in 0..self.n_classes {
                    z[c] = biases[c];
                    for j in 0..n_features {
                        z[c] += weights[c][j] * x_data[i * n_features + j];
                    }
                    max_z = max_z.max(z[c]);
                }

                let mut sum_exp = 0.0f32;
                for c in 0..self.n_classes {
                    probs[i][c] = (z[c] - max_z).exp();
                    sum_exp += probs[i][c];
                }
                for c in 0..self.n_classes {
                    probs[i][c] /= sum_exp;
                }
            }

            // Compute gradients
            let mut grad_w = vec![vec![0.0f32; n_features]; self.n_classes];
            let mut grad_b = vec![0.0f32; self.n_classes];

            for i in 0..n_samples {
                let true_class = y_data[i] as usize;
                for c in 0..self.n_classes {
                    let target = if c == true_class { 1.0 } else { 0.0 };
                    let error = probs[i][c] - target;
                    
                    grad_b[c] += error;
                    for j in 0..n_features {
                        grad_w[c][j] += error * x_data[i * n_features + j];
                    }
                }
            }

            // Add regularization and scale
            let scale = 1.0 / n_samples as f32;
            for c in 0..self.n_classes {
                grad_b[c] *= scale;
                for j in 0..n_features {
                    grad_w[c][j] = grad_w[c][j] * scale + reg * weights[c][j];
                }
            }

            // Update
            for c in 0..self.n_classes {
                if self.fit_intercept {
                    biases[c] -= lr * grad_b[c];
                }
                for j in 0..n_features {
                    weights[c][j] -= lr * grad_w[c][j];
                }
            }
        }

        self.coef_ = Some(weights);
        self.intercept_ = Some(biases);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let start = i * self.n_classes;
                let sample_probs = &proba_data[start..start + self.n_classes];
                sample_probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as f32)
                    .unwrap_or(0.0)
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.as_ref().expect("Model not fitted");

        let mut probs = Vec::with_capacity(n_samples * self.n_classes);

        for i in 0..n_samples {
            if self.n_classes == 2 {
                let mut z = intercept[0];
                for j in 0..n_features {
                    z += coef[0][j] * x_data[i * n_features + j];
                }
                let p = Self::sigmoid(z);
                probs.push(1.0 - p);
                probs.push(p);
            } else {
                let mut z = vec![0.0f32; self.n_classes];
                let mut max_z = f32::NEG_INFINITY;

                for c in 0..self.n_classes {
                    z[c] = intercept[c];
                    for j in 0..n_features {
                        z[c] += coef[c][j] * x_data[i * n_features + j];
                    }
                    max_z = max_z.max(z[c]);
                }

                let mut sum_exp = 0.0f32;
                for c in 0..self.n_classes {
                    z[c] = (z[c] - max_z).exp();
                    sum_exp += z[c];
                }

                for c in 0..self.n_classes {
                    probs.push(z[c] / sum_exp);
                }
            }
        }

        Tensor::from_slice(&probs, &[n_samples, self.n_classes]).unwrap()
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

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap();
        let y = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();

        let mut lr = LinearRegression::new();
        lr.fit(&x, &y);

        let predictions = lr.predict(&x);
        let score = lr.score(&x, &y);

        assert!(score > 0.99, "R² should be close to 1.0");
    }

    #[test]
    fn test_ridge() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap();
        let y = Tensor::from_slice(&[2.1, 3.9, 6.2, 7.8, 10.1], &[5]).unwrap();

        let mut ridge = Ridge::new(1.0);
        ridge.fit(&x, &y);

        let score = ridge.score(&x, &y);
        assert!(score > 0.95);
    }

    #[test]
    fn test_logistic_regression() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0], &[4]).unwrap();

        let mut lr = LogisticRegression::new().max_iter(1000);
        lr.fit(&x, &y);

        let predictions = lr.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }
}

//! Gaussian Process models for regression and classification

use ghostflow_core::Tensor;

/// Gaussian Process Regressor
pub struct GaussianProcessRegressor {
    pub kernel: GPKernel,
    pub alpha: f32,  // Noise level
    pub normalize_y: bool,
    pub n_restarts_optimizer: usize,
    x_train_: Option<Vec<f32>>,
    y_train_: Option<Vec<f32>>,
    #[allow(dead_code)]
    k_inv_: Option<Vec<f32>>,  // Inverse of kernel matrix
    alpha_: Option<Vec<f32>>,  // K^-1 * y
    y_mean_: f32,
    y_std_: f32,
    n_samples_: usize,
    n_features_: usize,
}

#[derive(Clone, Debug)]
pub enum GPKernel {
    RBF { length_scale: f32 },
    Matern { length_scale: f32, nu: f32 },
    RationalQuadratic { length_scale: f32, alpha: f32 },
    DotProduct { sigma_0: f32 },
    WhiteKernel { noise_level: f32 },
    Sum(Box<GPKernel>, Box<GPKernel>),
    Product(Box<GPKernel>, Box<GPKernel>),
}

impl GPKernel {
    pub fn rbf(length_scale: f32) -> Self {
        GPKernel::RBF { length_scale }
    }

    pub fn matern(length_scale: f32, nu: f32) -> Self {
        GPKernel::Matern { length_scale, nu }
    }

    fn compute(&self, x1: &[f32], x2: &[f32]) -> f32 {
        match self {
            GPKernel::RBF { length_scale } => {
                let dist_sq: f32 = x1.iter().zip(x2.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                (-dist_sq / (2.0 * length_scale * length_scale)).exp()
            }
            GPKernel::Matern { length_scale, nu } => {
                let dist: f32 = x1.iter().zip(x2.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let d = dist / length_scale;
                
                if *nu == 0.5 {
                    (-d).exp()
                } else if *nu == 1.5 {
                    let sqrt3_d = 3.0f32.sqrt() * d;
                    (1.0 + sqrt3_d) * (-sqrt3_d).exp()
                } else if *nu == 2.5 {
                    let sqrt5_d = 5.0f32.sqrt() * d;
                    (1.0 + sqrt5_d + sqrt5_d * sqrt5_d / 3.0) * (-sqrt5_d).exp()
                } else {
                    (-d).exp()  // Fallback to RBF-like
                }
            }
            GPKernel::RationalQuadratic { length_scale, alpha } => {
                let dist_sq: f32 = x1.iter().zip(x2.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                (1.0 + dist_sq / (2.0 * alpha * length_scale * length_scale)).powf(-*alpha)
            }
            GPKernel::DotProduct { sigma_0 } => {
                let dot: f32 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
                sigma_0 * sigma_0 + dot
            }
            GPKernel::WhiteKernel { noise_level } => {
                let same = x1.iter().zip(x2.iter()).all(|(&a, &b)| (a - b).abs() < 1e-10);
                if same { *noise_level } else { 0.0 }
            }
            GPKernel::Sum(k1, k2) => k1.compute(x1, x2) + k2.compute(x1, x2),
            GPKernel::Product(k1, k2) => k1.compute(x1, x2) * k2.compute(x1, x2),
        }
    }
}

impl GaussianProcessRegressor {
    pub fn new(kernel: GPKernel) -> Self {
        GaussianProcessRegressor {
            kernel,
            alpha: 1e-10,
            normalize_y: false,
            n_restarts_optimizer: 0,
            x_train_: None,
            y_train_: None,
            k_inv_: None,
            alpha_: None,
            y_mean_: 0.0,
            y_std_: 1.0,
            n_samples_: 0,
            n_features_: 0,
        }
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn normalize_y(mut self, normalize: bool) -> Self {
        self.normalize_y = normalize;
        self
    }

    fn compute_kernel_matrix(&self, x1: &[f32], n1: usize, x2: &[f32], n2: usize, n_features: usize) -> Vec<f32> {
        let mut k = vec![0.0f32; n1 * n2];
        
        for i in 0..n1 {
            for j in 0..n2 {
                let xi = &x1[i * n_features..(i + 1) * n_features];
                let xj = &x2[j * n_features..(j + 1) * n_features];
                k[i * n2 + j] = self.kernel.compute(xi, xj);
            }
        }
        
        k
    }

    fn cholesky_solve(&self, l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        // Forward substitution: L * y = b
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i * n + j] * y[j];
            }
            y[i] = sum / l[i * n + i].max(1e-10);
        }

        // Backward substitution: L^T * x = y
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

    fn cholesky_decomposition(&self, a: &[f32], n: usize) -> Vec<f32> {
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
        
        l
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let mut y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_samples_ = n_samples;
        self.n_features_ = n_features;

        // Normalize y if requested
        if self.normalize_y {
            self.y_mean_ = y_data.iter().sum::<f32>() / n_samples as f32;
            let variance: f32 = y_data.iter().map(|&y| (y - self.y_mean_).powi(2)).sum::<f32>() / n_samples as f32;
            self.y_std_ = variance.sqrt().max(1e-10);
            y_data = y_data.iter().map(|&y| (y - self.y_mean_) / self.y_std_).collect();
        } else {
            self.y_mean_ = 0.0;
            self.y_std_ = 1.0;
        }

        // Compute kernel matrix K
        let mut k = self.compute_kernel_matrix(&x_data, n_samples, &x_data, n_samples, n_features);
        
        // Add noise to diagonal: K + alpha * I
        for i in 0..n_samples {
            k[i * n_samples + i] += self.alpha;
        }

        // Cholesky decomposition
        let l = self.cholesky_decomposition(&k, n_samples);
        
        // Compute alpha = K^-1 * y using Cholesky
        let alpha = self.cholesky_solve(&l, &y_data, n_samples);

        self.x_train_ = Some(x_data);
        self.y_train_ = Some(y_data);
        self.alpha_ = Some(alpha);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let x_train = self.x_train_.as_ref().expect("Model not fitted");
        let alpha = self.alpha_.as_ref().expect("Model not fitted");
        let n_train = self.n_samples_;

        // Compute K_star = K(X_test, X_train)
        let k_star = self.compute_kernel_matrix(&x_data, n_samples, x_train, n_train, n_features);

        // Predict: y = K_star * alpha
        let mut predictions = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            for j in 0..n_train {
                predictions[i] += k_star[i * n_train + j] * alpha[j];
            }
            // Denormalize
            predictions[i] = predictions[i] * self.y_std_ + self.y_mean_;
        }

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_with_std(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mean = self.predict(x);
        
        // For simplicity, return constant std (full implementation would compute posterior variance)
        let n_samples = x.dims()[0];
        let std = vec![self.alpha.sqrt(); n_samples];
        
        (mean, Tensor::from_slice(&std, &[n_samples]).unwrap())
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


/// Gaussian Process Classifier using Laplace approximation
pub struct GaussianProcessClassifier {
    pub kernel: GPKernel,
    pub max_iter_predict: usize,
    x_train_: Option<Vec<f32>>,
    y_train_: Option<Vec<f32>>,
    f_cached_: Option<Vec<f32>>,
    n_samples_: usize,
    n_features_: usize,
    n_classes_: usize,
}

impl GaussianProcessClassifier {
    pub fn new(kernel: GPKernel) -> Self {
        GaussianProcessClassifier {
            kernel,
            max_iter_predict: 100,
            x_train_: None,
            y_train_: None,
            f_cached_: None,
            n_samples_: 0,
            n_features_: 0,
            n_classes_: 0,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }

    fn compute_kernel_matrix(&self, x1: &[f32], n1: usize, x2: &[f32], n2: usize, n_features: usize) -> Vec<f32> {
        let mut k = vec![0.0f32; n1 * n2];
        
        for i in 0..n1 {
            for j in 0..n2 {
                let xi = &x1[i * n_features..(i + 1) * n_features];
                let xj = &x2[j * n_features..(j + 1) * n_features];
                k[i * n2 + j] = self.kernel.compute(xi, xj);
            }
        }
        
        k
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_samples_ = n_samples;
        self.n_features_ = n_features;
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        // Convert labels to {-1, 1} for binary classification
        let y_binary: Vec<f32> = y_data.iter()
            .map(|&y| if y > 0.5 { 1.0 } else { -1.0 })
            .collect();

        // Compute kernel matrix
        let k = self.compute_kernel_matrix(&x_data, n_samples, &x_data, n_samples, n_features);

        // Laplace approximation: find mode of posterior using Newton's method
        let mut f = vec![0.0f32; n_samples];
        
        for _ in 0..self.max_iter_predict {
            // Compute pi = sigmoid(f)
            let pi: Vec<f32> = f.iter().map(|&fi| Self::sigmoid(fi)).collect();
            
            // W = diag(pi * (1 - pi))
            let _w: Vec<f32> = pi.iter().map(|&p| p * (1.0 - p)).collect();
            
            // Gradient: grad = y_binary * (1 - pi) - (1 - y_binary) * pi (simplified for {-1,1})
            // For binary: grad = (y + 1)/2 - pi
            let grad: Vec<f32> = y_binary.iter().zip(pi.iter())
                .map(|(&y, &p)| (y + 1.0) / 2.0 - p)
                .collect();
            
            // Newton update (simplified)
            let step_size = 0.1;
            for i in 0..n_samples {
                let mut k_grad = 0.0f32;
                for j in 0..n_samples {
                    k_grad += k[i * n_samples + j] * grad[j];
                }
                f[i] += step_size * k_grad;
            }
        }

        self.x_train_ = Some(x_data);
        self.y_train_ = Some(y_data);
        self.f_cached_ = Some(f);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                if self.n_classes_ == 2 {
                    if proba_data[i * 2 + 1] > 0.5 { 1.0 } else { 0.0 }
                } else {
                    let start = i * self.n_classes_;
                    let probs = &proba_data[start..start + self.n_classes_];
                    probs.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(c, _)| c as f32)
                        .unwrap_or(0.0)
                }
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let x_train = self.x_train_.as_ref().expect("Model not fitted");
        let f_cached = self.f_cached_.as_ref().expect("Model not fitted");
        let n_train = self.n_samples_;

        // Compute K_star
        let k_star = self.compute_kernel_matrix(&x_data, n_samples, x_train, n_train, n_features);

        // Predict latent function values
        let mut f_star = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            for j in 0..n_train {
                f_star[i] += k_star[i * n_train + j] * f_cached[j] / n_train as f32;
            }
        }

        // Convert to probabilities
        let mut probs = Vec::with_capacity(n_samples * self.n_classes_);
        for i in 0..n_samples {
            let p1 = Self::sigmoid(f_star[i]);
            if self.n_classes_ == 2 {
                probs.push(1.0 - p1);
                probs.push(p1);
            } else {
                // Multi-class: use softmax approximation
                for c in 0..self.n_classes_ {
                    probs.push(if c == 1 { p1 } else { (1.0 - p1) / (self.n_classes_ - 1) as f32 });
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gp_regressor() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0,
        ], &[5, 1]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 1.0, 4.0, 9.0, 16.0], &[5]).unwrap();
        
        let mut gpr = GaussianProcessRegressor::new(GPKernel::rbf(1.0)).alpha(0.1);
        gpr.fit(&x, &y);
        
        let predictions = gpr.predict(&x);
        assert_eq!(predictions.dims(), &[5]);
    }

    #[test]
    fn test_gp_classifier() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0], &[4]).unwrap();
        
        let mut gpc = GaussianProcessClassifier::new(GPKernel::rbf(1.0));
        gpc.fit(&x, &y);
        
        let predictions = gpc.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }
}



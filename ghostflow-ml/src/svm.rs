//! Support Vector Machine implementations

use ghostflow_core::Tensor;
use rayon::prelude::*;

/// Support Vector Classifier using SMO (Sequential Minimal Optimization)
pub struct SVC {
    pub c: f32,
    pub kernel: Kernel,
    pub gamma: f32,
    pub degree: i32,
    pub coef0: f32,
    pub tol: f32,
    pub max_iter: usize,
    support_vectors_: Option<Vec<Vec<f32>>>,
    support_: Option<Vec<usize>>,
    dual_coef_: Option<Vec<f32>>,
    intercept_: Option<f32>,
    y_train_: Option<Vec<f32>>,
    n_features_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    Linear,
    Poly,
    RBF,
    Sigmoid,
}

impl SVC {
    pub fn new() -> Self {
        SVC {
            c: 1.0,
            kernel: Kernel::RBF,
            gamma: 1.0,
            degree: 3,
            coef0: 0.0,
            tol: 1e-3,
            max_iter: 1000,
            support_vectors_: None,
            support_: None,
            dual_coef_: None,
            intercept_: None,
            y_train_: None,
            n_features_: 0,
        }
    }

    pub fn c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }


    fn kernel_function(&self, xi: &[f32], xj: &[f32]) -> f32 {
        match self.kernel {
            Kernel::Linear => {
                xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum()
            }
            Kernel::Poly => {
                let dot: f32 = xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum();
                (self.gamma * dot + self.coef0).powi(self.degree)
            }
            Kernel::RBF => {
                let dist_sq: f32 = xi.iter().zip(xj.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
                (-self.gamma * dist_sq).exp()
            }
            Kernel::Sigmoid => {
                let dot: f32 = xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum();
                (self.gamma * dot + self.coef0).tanh()
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        self.n_features_ = n_features;
        
        // Convert labels to -1/+1
        let y_binary: Vec<f32> = y_data.iter()
            .map(|&label| if label > 0.5 { 1.0 } else { -1.0 })
            .collect();
        
        // Initialize alphas and bias
        let mut alpha = vec![0.0f32; n_samples];
        let mut b = 0.0f32;
        
        // Precompute kernel matrix
        let mut kernel_matrix = vec![vec![0.0f32; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in i..n_samples {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let xj = &x_data[j * n_features..(j + 1) * n_features];
                let k = self.kernel_function(xi, xj);
                kernel_matrix[i][j] = k;
                kernel_matrix[j][i] = k;
            }
        }
        
        // Simplified SMO
        let eps = 1e-5;
        for _iter in 0..self.max_iter {
            let mut num_changed = 0;
            
            for i in 0..n_samples {
                // Compute error for i
                let mut f_i = -b;
                for j in 0..n_samples {
                    f_i += alpha[j] * y_binary[j] * kernel_matrix[i][j];
                }
                let e_i = f_i - y_binary[i];
                
                // Check KKT conditions
                if (y_binary[i] * e_i < -self.tol && alpha[i] < self.c) ||
                   (y_binary[i] * e_i > self.tol && alpha[i] > 0.0) {
                    
                    // Select j randomly (simplified)
                    let j = (i + 1) % n_samples;
                    
                    // Compute error for j
                    let mut f_j = -b;
                    for k in 0..n_samples {
                        f_j += alpha[k] * y_binary[k] * kernel_matrix[j][k];
                    }
                    let e_j = f_j - y_binary[j];
                    
                    let alpha_i_old = alpha[i];
                    let alpha_j_old = alpha[j];
                    
                    // Compute bounds
                    let (l, h) = if y_binary[i] != y_binary[j] {
                        ((alpha[j] - alpha[i]).max(0.0), self.c.min(self.c + alpha[j] - alpha[i]))
                    } else {
                        ((alpha[i] + alpha[j] - self.c).max(0.0), self.c.min(alpha[i] + alpha[j]))
                    };
                    
                    if l >= h { continue; }
                    
                    let eta = 2.0 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j];
                    if eta >= 0.0 { continue; }
                    
                    // Update alpha_j
                    alpha[j] = alpha_j_old - y_binary[j] * (e_i - e_j) / eta;
                    alpha[j] = alpha[j].clamp(l, h);
                    
                    if (alpha[j] - alpha_j_old).abs() < eps { continue; }
                    
                    // Update alpha_i
                    alpha[i] = alpha_i_old + y_binary[i] * y_binary[j] * (alpha_j_old - alpha[j]);
                    
                    // Update bias
                    let b1 = b - e_i - y_binary[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i][i]
                                     - y_binary[j] * (alpha[j] - alpha_j_old) * kernel_matrix[i][j];
                    let b2 = b - e_j - y_binary[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i][j]
                                     - y_binary[j] * (alpha[j] - alpha_j_old) * kernel_matrix[j][j];
                    
                    b = if alpha[i] > 0.0 && alpha[i] < self.c {
                        b1
                    } else if alpha[j] > 0.0 && alpha[j] < self.c {
                        b2
                    } else {
                        (b1 + b2) / 2.0
                    };
                    
                    num_changed += 1;
                }
            }
            
            if num_changed == 0 { break; }
        }
        
        // Extract support vectors
        let support_indices: Vec<usize> = alpha.iter()
            .enumerate()
            .filter(|(_, &a)| a > eps)
            .map(|(i, _)| i)
            .collect();
        
        let support_vectors: Vec<Vec<f32>> = support_indices.iter()
            .map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
            .collect();
        
        let dual_coef: Vec<f32> = support_indices.iter()
            .map(|&i| alpha[i] * y_binary[i])
            .collect();
        
        self.support_vectors_ = Some(support_vectors);
        self.support_ = Some(support_indices.clone());
        self.dual_coef_ = Some(dual_coef);
        self.intercept_ = Some(b);
        self.y_train_ = Some(y_binary);
    }


    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let support_vectors = self.support_vectors_.as_ref().expect("Model not fitted");
        let dual_coef = self.dual_coef_.as_ref().expect("Model not fitted");
        let b = self.intercept_.unwrap_or(0.0);
        
        let predictions: Vec<f32> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut decision = -b;
                
                for (sv, &coef) in support_vectors.iter().zip(dual_coef.iter()) {
                    decision += coef * self.kernel_function(xi, sv);
                }
                
                if decision >= 0.0 { 1.0 } else { 0.0 }
            })
            .collect();
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn decision_function(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let support_vectors = self.support_vectors_.as_ref().expect("Model not fitted");
        let dual_coef = self.dual_coef_.as_ref().expect("Model not fitted");
        let b = self.intercept_.unwrap_or(0.0);
        
        let decisions: Vec<f32> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut decision = -b;
                
                for (sv, &coef) in support_vectors.iter().zip(dual_coef.iter()) {
                    decision += coef * self.kernel_function(xi, sv);
                }
                
                decision
            })
            .collect();
        
        Tensor::from_slice(&decisions, &[n_samples]).unwrap()
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

impl Default for SVC {
    fn default() -> Self {
        Self::new()
    }
}

/// Support Vector Regressor
pub struct SVR {
    pub c: f32,
    pub kernel: Kernel,
    pub gamma: f32,
    pub epsilon: f32,
    pub max_iter: usize,
    support_vectors_: Option<Vec<Vec<f32>>>,
    dual_coef_: Option<Vec<f32>>,
    intercept_: Option<f32>,
    n_features_: usize,
}

impl SVR {
    pub fn new() -> Self {
        SVR {
            c: 1.0,
            kernel: Kernel::RBF,
            gamma: 1.0,
            epsilon: 0.1,
            max_iter: 1000,
            support_vectors_: None,
            dual_coef_: None,
            intercept_: None,
            n_features_: 0,
        }
    }

    pub fn c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    fn kernel_function(&self, xi: &[f32], xj: &[f32]) -> f32 {
        match self.kernel {
            Kernel::Linear => {
                xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum()
            }
            Kernel::RBF => {
                let dist_sq: f32 = xi.iter().zip(xj.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
                (-self.gamma * dist_sq).exp()
            }
            _ => {
                xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum()
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        self.n_features_ = n_features;
        
        // Simplified SVR using gradient descent on dual
        let mut alpha = vec![0.0f32; n_samples];
        let mut alpha_star = vec![0.0f32; n_samples];
        
        let lr = 0.01;
        
        for _iter in 0..self.max_iter {
            for i in 0..n_samples {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                
                // Compute prediction
                let mut pred = 0.0f32;
                for j in 0..n_samples {
                    let xj = &x_data[j * n_features..(j + 1) * n_features];
                    pred += (alpha[j] - alpha_star[j]) * self.kernel_function(xi, xj);
                }
                
                let error = pred - y_data[i];
                
                // Update alphas
                if error > self.epsilon {
                    alpha_star[i] = (alpha_star[i] + lr).min(self.c);
                } else if error < -self.epsilon {
                    alpha[i] = (alpha[i] + lr).min(self.c);
                }
            }
        }
        
        // Extract support vectors
        let eps = 1e-5;
        let mut support_vectors = Vec::new();
        let mut dual_coef = Vec::new();
        
        for i in 0..n_samples {
            let coef = alpha[i] - alpha_star[i];
            if coef.abs() > eps {
                support_vectors.push(x_data[i * n_features..(i + 1) * n_features].to_vec());
                dual_coef.push(coef);
            }
        }
        
        // Compute intercept
        let mut b = 0.0f32;
        let mut count = 0;
        for i in 0..n_samples {
            if alpha[i] > eps && alpha[i] < self.c - eps {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = 0.0f32;
                for (sv, &coef) in support_vectors.iter().zip(dual_coef.iter()) {
                    pred += coef * self.kernel_function(xi, sv);
                }
                b += y_data[i] - pred - self.epsilon;
                count += 1;
            }
        }
        if count > 0 {
            b /= count as f32;
        }
        
        self.support_vectors_ = Some(support_vectors);
        self.dual_coef_ = Some(dual_coef);
        self.intercept_ = Some(b);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let support_vectors = self.support_vectors_.as_ref().expect("Model not fitted");
        let dual_coef = self.dual_coef_.as_ref().expect("Model not fitted");
        let b = self.intercept_.unwrap_or(0.0);
        
        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = b;
                
                for (sv, &coef) in support_vectors.iter().zip(dual_coef.iter()) {
                    pred += coef * self.kernel_function(xi, sv);
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

impl Default for SVR {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svc() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0], &[4]).unwrap();
        
        let mut svc = SVC::new().kernel(Kernel::RBF).c(1.0);
        svc.fit(&x, &y);
        
        let predictions = svc.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }

    #[test]
    fn test_svr() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap();
        let y = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
        
        let mut svr = SVR::new().kernel(Kernel::Linear);
        svr.fit(&x, &y);
        
        let predictions = svr.predict(&x);
        assert_eq!(predictions.dims(), &[5]);
    }
}



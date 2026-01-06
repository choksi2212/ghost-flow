//! Linear and Quadratic Discriminant Analysis

use ghostflow_core::Tensor;

/// Linear Discriminant Analysis
pub struct LinearDiscriminantAnalysis {
    pub n_components: Option<usize>,
    pub solver: LDASolver,
    pub shrinkage: Option<f32>,
    pub priors: Option<Vec<f32>>,
    means_: Option<Vec<Vec<f32>>>,
    covariance_: Option<Vec<f32>>,  // Pooled covariance (flattened)
    #[allow(dead_code)]
    scalings_: Option<Vec<Vec<f32>>>,
    classes_: Option<Vec<usize>>,
    priors_: Option<Vec<f32>>,
    n_features_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum LDASolver {
    SVD,
    Eigen,
}

impl LinearDiscriminantAnalysis {
    pub fn new() -> Self {
        LinearDiscriminantAnalysis {
            n_components: None,
            solver: LDASolver::SVD,
            shrinkage: None,
            priors: None,
            means_: None,
            covariance_: None,
            scalings_: None,
            classes_: None,
            priors_: None,
            n_features_: 0,
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = Some(n);
        self
    }

    pub fn shrinkage(mut self, s: f32) -> Self {
        self.shrinkage = Some(s.clamp(0.0, 1.0));
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Get unique classes
        let mut classes: Vec<usize> = y_data.iter().map(|&v| v as usize).collect();
        classes.sort();
        classes.dedup();
        let n_classes = classes.len();

        // Compute class priors and means
        let mut class_counts = vec![0usize; n_classes];
        let mut means = vec![vec![0.0f32; n_features]; n_classes];

        for i in 0..n_samples {
            let class_idx = classes.iter().position(|&c| c == y_data[i] as usize).unwrap();
            class_counts[class_idx] += 1;
            for j in 0..n_features {
                means[class_idx][j] += x_data[i * n_features + j];
            }
        }

        for c in 0..n_classes {
            if class_counts[c] > 0 {
                for j in 0..n_features {
                    means[c][j] /= class_counts[c] as f32;
                }
            }
        }

        // Compute priors
        let priors: Vec<f32> = if let Some(ref p) = self.priors {
            p.clone()
        } else {
            class_counts.iter().map(|&c| c as f32 / n_samples as f32).collect()
        };

        // Compute pooled within-class covariance
        let mut cov = vec![0.0f32; n_features * n_features];

        for i in 0..n_samples {
            let class_idx = classes.iter().position(|&c| c == y_data[i] as usize).unwrap();
            for j in 0..n_features {
                for k in 0..n_features {
                    let diff_j = x_data[i * n_features + j] - means[class_idx][j];
                    let diff_k = x_data[i * n_features + k] - means[class_idx][k];
                    cov[j * n_features + k] += diff_j * diff_k;
                }
            }
        }

        // Normalize covariance
        let denom = (n_samples - n_classes).max(1) as f32;
        for c in &mut cov {
            *c /= denom;
        }

        // Apply shrinkage if specified
        if let Some(shrinkage) = self.shrinkage {
            let trace: f32 = (0..n_features).map(|i| cov[i * n_features + i]).sum();
            let mu = trace / n_features as f32;
            
            for i in 0..n_features {
                for j in 0..n_features {
                    if i == j {
                        cov[i * n_features + j] = (1.0 - shrinkage) * cov[i * n_features + j] + shrinkage * mu;
                    } else {
                        cov[i * n_features + j] *= 1.0 - shrinkage;
                    }
                }
            }
        }

        // Compute LDA scalings using eigendecomposition of Sw^-1 * Sb
        // For simplicity, we'll use the means directly for classification
        
        self.means_ = Some(means);
        self.covariance_ = Some(cov);
        self.classes_ = Some(classes);
        self.priors_ = Some(priors);
    }

    fn solve_linear_system(&self, cov: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        // Solve Ax = b using Cholesky decomposition
        let mut a = cov.to_vec();
        
        // Add regularization
        for i in 0..n {
            a[i * n + i] += 1e-6;
        }

        // Cholesky decomposition
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

        // Forward substitution
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..i {
                sum += l[i * n + j] * y[j];
            }
            y[i] = (b[i] - sum) / l[i * n + i].max(1e-10);
        }

        // Backward substitution
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

        let means = self.means_.as_ref().unwrap();
        let cov = self.covariance_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        let priors = self.priors_.as_ref().unwrap();
        let n_classes = classes.len();

        // Precompute Sigma^-1 * mu for each class
        let cov_inv_means: Vec<Vec<f32>> = means.iter()
            .map(|mean| self.solve_linear_system(cov, mean, n_features))
            .collect();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                
                let mut best_class = 0;
                let mut best_score = f32::NEG_INFINITY;

                for c in 0..n_classes {
                    // Linear discriminant function
                    let mut score = priors[c].ln();
                    
                    // x^T * Sigma^-1 * mu_c
                    for j in 0..n_features {
                        score += sample[j] * cov_inv_means[c][j];
                    }
                    
                    // -0.5 * mu_c^T * Sigma^-1 * mu_c
                    let mut quad = 0.0f32;
                    for j in 0..n_features {
                        quad += means[c][j] * cov_inv_means[c][j];
                    }
                    score -= 0.5 * quad;

                    if score > best_score {
                        best_score = score;
                        best_class = c;
                    }
                }

                classes[best_class] as f32
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        // Project data onto LDA components
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let means = self.means_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        let n_classes = classes.len();
        let n_components = self.n_components.unwrap_or(n_classes - 1).min(n_classes - 1);

        // Compute global mean
        let priors = self.priors_.as_ref().unwrap();
        let mut global_mean = vec![0.0f32; n_features];
        for c in 0..n_classes {
            for j in 0..n_features {
                global_mean[j] += priors[c] * means[c][j];
            }
        }

        // Use class means as projection directions (simplified)
        let mut result = vec![0.0f32; n_samples * n_components];
        
        for i in 0..n_samples {
            for c in 0..n_components {
                let mut proj = 0.0f32;
                for j in 0..n_features {
                    proj += (x_data[i * n_features + j] - global_mean[j]) * 
                            (means[c][j] - global_mean[j]);
                }
                result[i * n_components + c] = proj;
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_components]).unwrap()
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

impl Default for LinearDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}


/// Quadratic Discriminant Analysis
pub struct QuadraticDiscriminantAnalysis {
    pub reg_param: f32,
    pub priors: Option<Vec<f32>>,
    means_: Option<Vec<Vec<f32>>>,
    covariances_: Option<Vec<Vec<f32>>>,  // Per-class covariances
    classes_: Option<Vec<usize>>,
    priors_: Option<Vec<f32>>,
    n_features_: usize,
}

impl QuadraticDiscriminantAnalysis {
    pub fn new() -> Self {
        QuadraticDiscriminantAnalysis {
            reg_param: 0.0,
            priors: None,
            means_: None,
            covariances_: None,
            classes_: None,
            priors_: None,
            n_features_: 0,
        }
    }

    pub fn reg_param(mut self, r: f32) -> Self {
        self.reg_param = r;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Get unique classes
        let mut classes: Vec<usize> = y_data.iter().map(|&v| v as usize).collect();
        classes.sort();
        classes.dedup();
        let n_classes = classes.len();

        // Compute class counts and means
        let mut class_counts = vec![0usize; n_classes];
        let mut means = vec![vec![0.0f32; n_features]; n_classes];

        for i in 0..n_samples {
            let class_idx = classes.iter().position(|&c| c == y_data[i] as usize).unwrap();
            class_counts[class_idx] += 1;
            for j in 0..n_features {
                means[class_idx][j] += x_data[i * n_features + j];
            }
        }

        for c in 0..n_classes {
            if class_counts[c] > 0 {
                for j in 0..n_features {
                    means[c][j] /= class_counts[c] as f32;
                }
            }
        }

        // Compute priors
        let priors: Vec<f32> = if let Some(ref p) = self.priors {
            p.clone()
        } else {
            class_counts.iter().map(|&c| c as f32 / n_samples as f32).collect()
        };

        // Compute per-class covariances
        let mut covariances = vec![vec![0.0f32; n_features * n_features]; n_classes];

        for i in 0..n_samples {
            let class_idx = classes.iter().position(|&c| c == y_data[i] as usize).unwrap();
            for j in 0..n_features {
                for k in 0..n_features {
                    let diff_j = x_data[i * n_features + j] - means[class_idx][j];
                    let diff_k = x_data[i * n_features + k] - means[class_idx][k];
                    covariances[class_idx][j * n_features + k] += diff_j * diff_k;
                }
            }
        }

        // Normalize and regularize
        for c in 0..n_classes {
            let denom = (class_counts[c] - 1).max(1) as f32;
            for idx in 0..n_features * n_features {
                covariances[c][idx] /= denom;
            }

            // Add regularization
            if self.reg_param > 0.0 {
                for i in 0..n_features {
                    covariances[c][i * n_features + i] += self.reg_param;
                }
            }
        }

        self.means_ = Some(means);
        self.covariances_ = Some(covariances);
        self.classes_ = Some(classes);
        self.priors_ = Some(priors);
    }

    fn log_det_and_inv(&self, cov: &[f32], n: usize) -> (f32, Vec<f32>) {
        // Compute log determinant and inverse using Cholesky
        let mut a = cov.to_vec();
        for i in 0..n {
            a[i * n + i] += 1e-6;
        }

        // Cholesky decomposition
        let mut l = vec![0.0f32; n * n];
        let mut log_det = 0.0f32;

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f32;
                if i == j {
                    for k in 0..j {
                        sum += l[j * n + k] * l[j * n + k];
                    }
                    let val = (a[j * n + j] - sum).max(1e-10).sqrt();
                    l[j * n + j] = val;
                    log_det += 2.0 * val.ln();
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j].max(1e-10);
                }
            }
        }

        // Compute inverse
        let mut inv = vec![0.0f32; n * n];
        for col in 0..n {
            let mut e = vec![0.0f32; n];
            e[col] = 1.0;

            // Forward substitution
            let mut y = vec![0.0f32; n];
            for i in 0..n {
                let mut sum = 0.0f32;
                for j in 0..i {
                    sum += l[i * n + j] * y[j];
                }
                y[i] = (e[i] - sum) / l[i * n + i].max(1e-10);
            }

            // Backward substitution
            let mut x = vec![0.0f32; n];
            for i in (0..n).rev() {
                let mut sum = 0.0f32;
                for j in (i + 1)..n {
                    sum += l[j * n + i] * x[j];
                }
                x[i] = (y[i] - sum) / l[i * n + i].max(1e-10);
            }

            for i in 0..n {
                inv[i * n + col] = x[i];
            }
        }

        (log_det, inv)
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let means = self.means_.as_ref().unwrap();
        let covariances = self.covariances_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        let priors = self.priors_.as_ref().unwrap();
        let n_classes = classes.len();

        // Precompute log determinants and inverses
        let cov_info: Vec<(f32, Vec<f32>)> = covariances.iter()
            .map(|cov| self.log_det_and_inv(cov, n_features))
            .collect();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                
                let mut best_class = 0;
                let mut best_score = f32::NEG_INFINITY;

                for c in 0..n_classes {
                    let (log_det, ref cov_inv) = cov_info[c];
                    
                    // Quadratic discriminant function
                    let mut score = priors[c].ln() - 0.5 * log_det;
                    
                    // -0.5 * (x - mu)^T * Sigma^-1 * (x - mu)
                    let mut quad = 0.0f32;
                    for j in 0..n_features {
                        for k in 0..n_features {
                            let diff_j = sample[j] - means[c][j];
                            let diff_k = sample[k] - means[c][k];
                            quad += diff_j * cov_inv[j * n_features + k] * diff_k;
                        }
                    }
                    score -= 0.5 * quad;

                    if score > best_score {
                        best_score = score;
                        best_class = c;
                    }
                }

                classes[best_class] as f32
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

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lda() {
        let x = Tensor::from_slice(&[1.0f32, 2.0,
            1.5, 1.8,
            5.0, 8.0,
            6.0, 9.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut lda = LinearDiscriminantAnalysis::new();
        lda.fit(&x, &y);
        
        let score = lda.score(&x, &y);
        assert!(score >= 0.5);
    }

    #[test]
    fn test_qda() {
        let x = Tensor::from_slice(&[1.0f32, 2.0,
            1.5, 1.8,
            5.0, 8.0,
            6.0, 9.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut qda = QuadraticDiscriminantAnalysis::new().reg_param(0.1);
        qda.fit(&x, &y);
        
        let score = qda.score(&x, &y);
        assert!(score >= 0.5);
    }
}



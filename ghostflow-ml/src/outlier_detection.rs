//! Outlier Detection - Local Outlier Factor, One-Class SVM, Elliptic Envelope

use ghostflow_core::Tensor;

/// Local Outlier Factor for anomaly detection
pub struct LocalOutlierFactor {
    pub n_neighbors: usize,
    pub contamination: f32,
    pub metric: LOFMetric,
    x_train_: Option<Vec<f32>>,
    lrd_: Option<Vec<f32>>,
    lof_scores_: Option<Vec<f32>>,
    threshold_: f32,
    n_samples_: usize,
    n_features_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum LOFMetric {
    Euclidean,
    Manhattan,
    Minkowski(f32),
}

impl LocalOutlierFactor {
    pub fn new(n_neighbors: usize) -> Self {
        LocalOutlierFactor {
            n_neighbors,
            contamination: 0.1,
            metric: LOFMetric::Euclidean,
            x_train_: None,
            lrd_: None,
            lof_scores_: None,
            threshold_: 0.0,
            n_samples_: 0,
            n_features_: 0,
        }
    }

    pub fn contamination(mut self, c: f32) -> Self {
        self.contamination = c.clamp(0.0, 0.5);
        self
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            LOFMetric::Euclidean => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f32>().sqrt()
            }
            LOFMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).abs()).sum()
            }
            LOFMetric::Minkowski(p) => {
                a.iter().zip(b.iter())
                    .map(|(&ai, &bi)| (ai - bi).abs().powf(p))
                    .sum::<f32>()
                    .powf(1.0 / p)
            }
        }
    }

    fn k_neighbors(&self, x: &[f32], point_idx: usize, n_samples: usize, n_features: usize) -> Vec<(usize, f32)> {
        let point = &x[point_idx * n_features..(point_idx + 1) * n_features];
        
        let mut distances: Vec<(usize, f32)> = (0..n_samples)
            .filter(|&i| i != point_idx)
            .map(|i| {
                let other = &x[i * n_features..(i + 1) * n_features];
                (i, self.distance(point, other))
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(self.n_neighbors);
        distances
    }

    fn k_distance(&self, neighbors: &[(usize, f32)]) -> f32 {
        neighbors.last().map(|(_, d)| *d).unwrap_or(0.0)
    }

    fn reachability_distance(&self, x: &[f32], point_a: usize, point_b: usize, k_dist_b: f32, n_features: usize) -> f32 {
        let a = &x[point_a * n_features..(point_a + 1) * n_features];
        let b = &x[point_b * n_features..(point_b + 1) * n_features];
        let dist = self.distance(a, b);
        dist.max(k_dist_b)
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_samples_ = n_samples;
        self.n_features_ = n_features;

        // Compute k-neighbors and k-distances for all points
        let all_neighbors: Vec<Vec<(usize, f32)>> = (0..n_samples)
            .map(|i| self.k_neighbors(&x_data, i, n_samples, n_features))
            .collect();

        let k_distances: Vec<f32> = all_neighbors.iter()
            .map(|neighbors| self.k_distance(neighbors))
            .collect();

        // Compute Local Reachability Density (LRD)
        let lrd: Vec<f32> = (0..n_samples)
            .map(|i| {
                let neighbors = &all_neighbors[i];
                let sum_reach_dist: f32 = neighbors.iter()
                    .map(|&(j, _)| self.reachability_distance(&x_data, i, j, k_distances[j], n_features))
                    .sum();
                
                if sum_reach_dist > 0.0 {
                    neighbors.len() as f32 / sum_reach_dist
                } else {
                    f32::INFINITY
                }
            })
            .collect();

        // Compute LOF scores
        let lof_scores: Vec<f32> = (0..n_samples)
            .map(|i| {
                let neighbors = &all_neighbors[i];
                let sum_lrd_ratio: f32 = neighbors.iter()
                    .map(|&(j, _)| lrd[j] / lrd[i].max(1e-10))
                    .sum();
                
                sum_lrd_ratio / neighbors.len() as f32
            })
            .collect();

        // Compute threshold based on contamination
        let mut sorted_scores = lof_scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());  // Descending
        let threshold_idx = (self.contamination * n_samples as f32) as usize;
        self.threshold_ = sorted_scores[threshold_idx.min(n_samples - 1)];

        self.x_train_ = Some(x_data);
        self.lrd_ = Some(lrd);
        self.lof_scores_ = Some(lof_scores);
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        
        let lof_scores = self.lof_scores_.as_ref().unwrap();
        let predictions: Vec<f32> = lof_scores.iter()
            .map(|&score| if score > self.threshold_ { -1.0 } else { 1.0 })  // -1 = outlier, 1 = inlier
            .collect();

        Tensor::from_slice(&predictions, &[predictions.len()]).unwrap()
    }

    pub fn decision_function(&self, x: &Tensor) -> Tensor {
        // For new data, compute LOF scores
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let x_train = self.x_train_.as_ref().expect("Model not fitted");
        let lrd_train = self.lrd_.as_ref().expect("Model not fitted");
        let n_train = self.n_samples_;

        let scores: Vec<f32> = (0..n_samples)
            .map(|i| {
                let point = &x_data[i * n_features..(i + 1) * n_features];
                
                // Find k-nearest neighbors in training data
                let mut distances: Vec<(usize, f32)> = (0..n_train)
                    .map(|j| {
                        let train_point = &x_train[j * n_features..(j + 1) * n_features];
                        (j, self.distance(point, train_point))
                    })
                    .collect();

                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let neighbors: Vec<(usize, f32)> = distances.into_iter().take(self.n_neighbors).collect();

                // Compute LRD for this point
                let k_dist = neighbors.last().map(|(_, d)| *d).unwrap_or(0.0);
                let sum_reach_dist: f32 = neighbors.iter()
                    .map(|&(_j, d)| d.max(k_dist))
                    .sum();
                
                let lrd_point = if sum_reach_dist > 0.0 {
                    neighbors.len() as f32 / sum_reach_dist
                } else {
                    f32::INFINITY
                };

                // Compute LOF
                let sum_lrd_ratio: f32 = neighbors.iter()
                    .map(|&(j, _)| lrd_train[j] / lrd_point.max(1e-10))
                    .sum();
                
                -(sum_lrd_ratio / neighbors.len() as f32)  // Negative so higher = more normal
            })
            .collect();

        Tensor::from_slice(&scores, &[n_samples]).unwrap()
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let scores = self.decision_function(x);
        let score_data = scores.data_f32();
        let n_samples = x.dims()[0];

        let predictions: Vec<f32> = score_data.iter()
            .map(|&s| if -s > self.threshold_ { -1.0 } else { 1.0 })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

/// One-Class SVM for novelty detection
pub struct OneClassSVM {
    pub kernel: OCSVMKernel,
    pub nu: f32,
    pub gamma: f32,
    pub max_iter: usize,
    support_vectors_: Option<Vec<Vec<f32>>>,
    dual_coef_: Option<Vec<f32>>,
    intercept_: Option<f32>,
    n_features_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum OCSVMKernel {
    RBF,
    Linear,
    Poly { degree: i32, coef0: f32 },
    Sigmoid { coef0: f32 },
}

impl OneClassSVM {
    pub fn new() -> Self {
        OneClassSVM {
            kernel: OCSVMKernel::RBF,
            nu: 0.5,
            gamma: 1.0,
            max_iter: 1000,
            support_vectors_: None,
            dual_coef_: None,
            intercept_: None,
            n_features_: 0,
        }
    }

    pub fn nu(mut self, nu: f32) -> Self {
        self.nu = nu.clamp(0.0, 1.0);
        self
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn kernel(mut self, kernel: OCSVMKernel) -> Self {
        self.kernel = kernel;
        self
    }

    fn kernel_function(&self, xi: &[f32], xj: &[f32]) -> f32 {
        match self.kernel {
            OCSVMKernel::RBF => {
                let dist_sq: f32 = xi.iter().zip(xj.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
                (-self.gamma * dist_sq).exp()
            }
            OCSVMKernel::Linear => {
                xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum()
            }
            OCSVMKernel::Poly { degree, coef0 } => {
                let dot: f32 = xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum();
                (self.gamma * dot + coef0).powi(degree)
            }
            OCSVMKernel::Sigmoid { coef0 } => {
                let dot: f32 = xi.iter().zip(xj.iter()).map(|(&a, &b)| a * b).sum();
                (self.gamma * dot + coef0).tanh()
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Simplified One-Class SVM using SMO-like approach
        let mut alpha = vec![0.0f32; n_samples];
        let upper_bound = 1.0 / (self.nu * n_samples as f32);

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

        // Initialize alphas uniformly
        let init_alpha = 1.0 / n_samples as f32;
        for a in &mut alpha {
            *a = init_alpha.min(upper_bound);
        }

        // Simple coordinate descent
        for _ in 0..self.max_iter {
            let mut changed = false;

            for i in 0..n_samples {
                // Compute gradient
                let mut grad = 0.0f32;
                for j in 0..n_samples {
                    grad += alpha[j] * kernel_matrix[i][j];
                }

                // Update alpha
                let old_alpha = alpha[i];
                alpha[i] = (alpha[i] - 0.01 * grad).clamp(0.0, upper_bound);

                if (alpha[i] - old_alpha).abs() > 1e-5 {
                    changed = true;
                }
            }

            // Normalize to sum to 1
            let sum: f32 = alpha.iter().sum();
            if sum > 0.0 {
                for a in &mut alpha {
                    *a /= sum;
                }
            }

            if !changed {
                break;
            }
        }

        // Extract support vectors
        let eps = 1e-5;
        let mut support_vectors = Vec::new();
        let mut dual_coef = Vec::new();

        for i in 0..n_samples {
            if alpha[i] > eps {
                support_vectors.push(x_data[i * n_features..(i + 1) * n_features].to_vec());
                dual_coef.push(alpha[i]);
            }
        }

        // Compute intercept (rho)
        let mut rho = 0.0f32;
        let mut count = 0;
        for i in 0..n_samples {
            if alpha[i] > eps && alpha[i] < upper_bound - eps {
                let mut sum = 0.0f32;
                for j in 0..n_samples {
                    sum += alpha[j] * kernel_matrix[i][j];
                }
                rho += sum;
                count += 1;
            }
        }
        if count > 0 {
            rho /= count as f32;
        }

        self.support_vectors_ = Some(support_vectors);
        self.dual_coef_ = Some(dual_coef);
        self.intercept_ = Some(rho);
    }

    pub fn decision_function(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let support_vectors = self.support_vectors_.as_ref().expect("Model not fitted");
        let dual_coef = self.dual_coef_.as_ref().expect("Model not fitted");
        let rho = self.intercept_.unwrap_or(0.0);

        let scores: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut score = -rho;
                
                for (sv, &coef) in support_vectors.iter().zip(dual_coef.iter()) {
                    score += coef * self.kernel_function(xi, sv);
                }
                
                score
            })
            .collect();

        Tensor::from_slice(&scores, &[n_samples]).unwrap()
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let scores = self.decision_function(x);
        let score_data = scores.data_f32();
        let n_samples = x.dims()[0];

        let predictions: Vec<f32> = score_data.iter()
            .map(|&s| if s >= 0.0 { 1.0 } else { -1.0 })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.predict(x)
    }
}

impl Default for OneClassSVM {
    fn default() -> Self {
        Self::new()
    }
}


/// Elliptic Envelope for Gaussian-distributed data
pub struct EllipticEnvelope {
    pub contamination: f32,
    pub support_fraction: Option<f32>,
    location_: Option<Vec<f32>>,
    covariance_: Option<Vec<f32>>,
    precision_: Option<Vec<f32>>,
    threshold_: f32,
    n_features_: usize,
}

impl EllipticEnvelope {
    pub fn new() -> Self {
        EllipticEnvelope {
            contamination: 0.1,
            support_fraction: None,
            location_: None,
            covariance_: None,
            precision_: None,
            threshold_: 0.0,
            n_features_: 0,
        }
    }

    pub fn contamination(mut self, c: f32) -> Self {
        self.contamination = c.clamp(0.0, 0.5);
        self
    }

    fn compute_mean(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let mut mean = vec![0.0f32; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += x[i * n_features + j];
            }
        }
        for m in &mut mean {
            *m /= n_samples as f32;
        }
        mean
    }

    fn compute_covariance(&self, x: &[f32], mean: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let mut cov = vec![0.0f32; n_features * n_features];
        
        for i in 0..n_samples {
            for j in 0..n_features {
                for k in 0..n_features {
                    let diff_j = x[i * n_features + j] - mean[j];
                    let diff_k = x[i * n_features + k] - mean[k];
                    cov[j * n_features + k] += diff_j * diff_k;
                }
            }
        }

        let denom = (n_samples - 1).max(1) as f32;
        for c in &mut cov {
            *c /= denom;
        }

        cov
    }

    fn invert_matrix(&self, matrix: &[f32], n: usize) -> Vec<f32> {
        // Cholesky-based inversion
        let mut a = matrix.to_vec();
        
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

        // Precision = L_inv^T * L_inv
        let mut precision = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    precision[i * n + j] += l_inv[k * n + i] * l_inv[k * n + j];
                }
            }
        }

        precision
    }

    fn mahalanobis_distance(&self, x: &[f32], mean: &[f32], precision: &[f32], n_features: usize) -> f32 {
        let diff: Vec<f32> = x.iter().zip(mean.iter()).map(|(&xi, &mi)| xi - mi).collect();
        
        let mut dist = 0.0f32;
        for i in 0..n_features {
            for j in 0..n_features {
                dist += diff[i] * precision[i * n_features + j] * diff[j];
            }
        }
        
        dist.sqrt()
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Compute robust estimates using all data (simplified - full MCD would be better)
        let mean = self.compute_mean(&x_data, n_samples, n_features);
        let cov = self.compute_covariance(&x_data, &mean, n_samples, n_features);
        let precision = self.invert_matrix(&cov, n_features);

        // Compute Mahalanobis distances
        let distances: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                self.mahalanobis_distance(xi, &mean, &precision, n_features)
            })
            .collect();

        // Set threshold based on contamination
        let mut sorted_distances = distances.clone();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = ((1.0 - self.contamination) * n_samples as f32) as usize;
        self.threshold_ = sorted_distances[threshold_idx.min(n_samples - 1)];

        self.location_ = Some(mean);
        self.covariance_ = Some(cov);
        self.precision_ = Some(precision);
    }

    pub fn decision_function(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean = self.location_.as_ref().expect("Model not fitted");
        let precision = self.precision_.as_ref().expect("Model not fitted");

        let scores: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                -self.mahalanobis_distance(xi, mean, precision, n_features)
            })
            .collect();

        Tensor::from_slice(&scores, &[n_samples]).unwrap()
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let scores = self.decision_function(x);
        let score_data = scores.data_f32();
        let n_samples = x.dims()[0];

        let predictions: Vec<f32> = score_data.iter()
            .map(|&s| if -s <= self.threshold_ { 1.0 } else { -1.0 })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.predict(x)
    }

    pub fn mahalanobis(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean = self.location_.as_ref().expect("Model not fitted");
        let precision = self.precision_.as_ref().expect("Model not fitted");

        let distances: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                self.mahalanobis_distance(xi, mean, precision, n_features)
            })
            .collect();

        Tensor::from_slice(&distances, &[n_samples]).unwrap()
    }
}

impl Default for EllipticEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lof() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            10.0, 10.0,  // Outlier
        ], &[4, 2]).unwrap();

        let mut lof = LocalOutlierFactor::new(2).contamination(0.25);
        let predictions = lof.fit_predict(&x);
        
        assert_eq!(predictions.dims(), &[4]);
    }

    #[test]
    fn test_one_class_svm() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            0.1, 0.2,
        ], &[4, 2]).unwrap();

        let mut ocsvm = OneClassSVM::new().nu(0.1);
        ocsvm.fit(&x);
        
        let predictions = ocsvm.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }

    #[test]
    fn test_elliptic_envelope() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            10.0, 10.0,  // Outlier
        ], &[4, 2]).unwrap();

        let mut ee = EllipticEnvelope::new().contamination(0.25);
        let predictions = ee.fit_predict(&x);
        
        assert_eq!(predictions.dims(), &[4]);
    }
}

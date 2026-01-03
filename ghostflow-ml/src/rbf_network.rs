//! RBF Network - Radial Basis Function Neural Network

use ghostflow_core::Tensor;
use rand::prelude::*;

/// RBF Network for classification and regression
pub struct RBFNetwork {
    pub n_centers: usize,
    pub gamma: f32,
    pub max_iter: usize,
    pub tol: f32,
    centers_: Option<Vec<Vec<f32>>>,
    weights_: Option<Vec<f32>>,
    bias_: f32,
    n_features_: usize,
}

impl RBFNetwork {
    pub fn new(n_centers: usize) -> Self {
        RBFNetwork {
            n_centers,
            gamma: 1.0,
            max_iter: 100,
            tol: 1e-4,
            centers_: None,
            weights_: None,
            bias_: 0.0,
            n_features_: 0,
        }
    }

    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    fn rbf_kernel(&self, x: &[f32], center: &[f32]) -> f32 {
        let sq_dist: f32 = x.iter().zip(center.iter())
            .map(|(&a, &b)| (a - b).powi(2)).sum();
        (-self.gamma * sq_dist).exp()
    }

    fn kmeans_init(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut rng = thread_rng();
        let n_centers = self.n_centers.min(n_samples);

        // K-means++ initialization
        let mut centers = Vec::with_capacity(n_centers);
        
        // First center: random
        let first_idx = rng.gen_range(0..n_samples);
        centers.push(x[first_idx * n_features..(first_idx + 1) * n_features].to_vec());

        // Remaining centers
        for _ in 1..n_centers {
            let distances: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let xi = &x[i * n_features..(i + 1) * n_features];
                    centers.iter()
                        .map(|c| {
                            xi.iter().zip(c.iter())
                                .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>()
                        })
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total < 1e-10 {
                let idx = rng.gen_range(0..n_samples);
                centers.push(x[idx * n_features..(idx + 1) * n_features].to_vec());
                continue;
            }

            // Sample proportional to distance squared
            let threshold = rng.gen::<f32>() * total;
            let mut cumsum = 0.0f32;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    centers.push(x[i * n_features..(i + 1) * n_features].to_vec());
                    break;
                }
            }
        }

        // Run a few k-means iterations
        for _ in 0..10 {
            // Assign points to nearest center
            let mut assignments = vec![0usize; n_samples];
            for i in 0..n_samples {
                let xi = &x[i * n_features..(i + 1) * n_features];
                let mut min_dist = f32::MAX;
                for (j, center) in centers.iter().enumerate() {
                    let dist: f32 = xi.iter().zip(center.iter())
                        .map(|(&a, &b)| (a - b).powi(2)).sum();
                    if dist < min_dist {
                        min_dist = dist;
                        assignments[i] = j;
                    }
                }
            }

            // Update centers
            let mut new_centers = vec![vec![0.0f32; n_features]; n_centers];
            let mut counts = vec![0usize; n_centers];

            for i in 0..n_samples {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..n_features {
                    new_centers[c][j] += x[i * n_features + j];
                }
            }

            for c in 0..n_centers {
                if counts[c] > 0 {
                    for j in 0..n_features {
                        new_centers[c][j] /= counts[c] as f32;
                    }
                } else {
                    // Keep old center if no points assigned
                    new_centers[c] = centers[c].clone();
                }
            }

            centers = new_centers;
        }

        centers
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Initialize centers using k-means
        let centers = self.kmeans_init(&x_data, n_samples, n_features);
        let n_centers = centers.len();

        // Compute RBF activations
        let mut phi = vec![0.0f32; n_samples * (n_centers + 1)]; // +1 for bias
        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            for (j, center) in centers.iter().enumerate() {
                phi[i * (n_centers + 1) + j] = self.rbf_kernel(xi, center);
            }
            phi[i * (n_centers + 1) + n_centers] = 1.0; // Bias term
        }

        // Solve least squares: (Phi^T Phi + lambda I) w = Phi^T y
        let lambda = 1e-6f32;
        let m = n_centers + 1;

        let mut ata = vec![0.0f32; m * m];
        let mut aty = vec![0.0f32; m];

        for i in 0..m {
            for k in 0..n_samples {
                aty[i] += phi[k * m + i] * y_data[k];
            }
            for j in 0..m {
                for k in 0..n_samples {
                    ata[i * m + j] += phi[k * m + i] * phi[k * m + j];
                }
            }
            ata[i * m + i] += lambda;
        }

        // Solve linear system
        let weights = solve_linear_system(&ata, &aty, m);

        self.centers_ = Some(centers);
        self.weights_ = Some(weights[..n_centers].to_vec());
        self.bias_ = weights[n_centers];
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let centers = self.centers_.as_ref().expect("Model not fitted");
        let weights = self.weights_.as_ref().unwrap();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = self.bias_;
                for (j, center) in centers.iter().enumerate() {
                    pred += weights[j] * self.rbf_kernel(xi, center);
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let predictions = self.predict(x);
        let pred_data = predictions.data_f32();
        
        // Apply sigmoid for probability
        let proba: Vec<f32> = pred_data.iter()
            .map(|&p| 1.0 / (1.0 + (-p).exp()))
            .collect();

        Tensor::from_slice(&proba, &[proba.len()]).unwrap()
    }
}

/// RBF Classifier
pub struct RBFClassifier {
    pub n_centers: usize,
    pub gamma: f32,
    networks_: Vec<RBFNetwork>,
    classes_: Vec<i32>,
}

impl RBFClassifier {
    pub fn new(n_centers: usize) -> Self {
        RBFClassifier {
            n_centers,
            gamma: 1.0,
            networks_: Vec::new(),
            classes_: Vec::new(),
        }
    }

    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let y_data = y.data_f32();

        // Find unique classes
        let mut classes: Vec<i32> = y_data.iter().map(|&v| v as i32).collect();
        classes.sort();
        classes.dedup();

        if classes.len() == 2 {
            // Binary classification
            let mut network = RBFNetwork::new(self.n_centers);
            network.gamma = self.gamma;
            
            // Convert labels to 0/1
            let y_binary: Vec<f32> = y_data.iter()
                .map(|&v| if v as i32 == classes[1] { 1.0 } else { 0.0 })
                .collect();
            let y_tensor = Tensor::from_slice(&y_binary, &[y_binary.len()]).unwrap();
            
            network.fit(x, &y_tensor);
            self.networks_ = vec![network];
        } else {
            // One-vs-rest for multiclass
            for &class in &classes {
                let mut network = RBFNetwork::new(self.n_centers);
                network.gamma = self.gamma;
                
                let y_binary: Vec<f32> = y_data.iter()
                    .map(|&v| if v as i32 == class { 1.0 } else { 0.0 })
                    .collect();
                let y_tensor = Tensor::from_slice(&y_binary, &[y_binary.len()]).unwrap();
                
                network.fit(x, &y_tensor);
                self.networks_.push(network);
            }
        }

        self.classes_ = classes;
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];

        if self.networks_.len() == 1 {
            // Binary classification
            let proba = self.networks_[0].predict_proba(x);
            let proba_data = proba.data_f32();
            
            let predictions: Vec<f32> = proba_data.iter()
                .map(|&p| if p >= 0.5 { self.classes_[1] as f32 } else { self.classes_[0] as f32 })
                .collect();
            
            Tensor::from_slice(&predictions, &[n_samples]).unwrap()
        } else {
            // Multiclass: pick class with highest score
            let scores: Vec<Vec<f32>> = self.networks_.iter()
                .map(|net| net.predict(x).data_f32().clone())
                .collect();

            let predictions: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let mut max_score = f32::NEG_INFINITY;
                    let mut max_class = self.classes_[0];
                    for (j, class) in self.classes_.iter().enumerate() {
                        if scores[j][i] > max_score {
                            max_score = scores[j][i];
                            max_class = *class;
                        }
                    }
                    max_class as f32
                })
                .collect();

            Tensor::from_slice(&predictions, &[n_samples]).unwrap()
        }
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let n_classes = self.classes_.len();

        if self.networks_.len() == 1 {
            // Binary classification
            let proba = self.networks_[0].predict_proba(x);
            let proba_data = proba.data_f32();
            
            let result: Vec<f32> = proba_data.iter()
                .flat_map(|&p| vec![1.0 - p, p])
                .collect();
            
            Tensor::from_slice(&result, &[n_samples, 2]).unwrap()
        } else {
            // Multiclass: softmax over scores
            let scores: Vec<Vec<f32>> = self.networks_.iter()
                .map(|net| net.predict(x).data_f32().clone())
                .collect();

            let mut result = vec![0.0f32; n_samples * n_classes];
            for i in 0..n_samples {
                let max_score: f32 = (0..n_classes).map(|j| scores[j][i]).fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = (0..n_classes).map(|j| (scores[j][i] - max_score).exp()).sum();
                
                for j in 0..n_classes {
                    result[i * n_classes + j] = (scores[j][i] - max_score).exp() / exp_sum;
                }
            }

            Tensor::from_slice(&result, &[n_samples, n_classes]).unwrap()
        }
    }
}

fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut aug = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k * (n + 1) + i].abs() > aug[max_row * (n + 1) + i].abs() {
                max_row = k;
            }
        }

        for j in 0..=n {
            let tmp = aug[i * (n + 1) + j];
            aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j];
            aug[max_row * (n + 1) + j] = tmp;
        }

        let pivot = aug[i * (n + 1) + i];
        if pivot.abs() < 1e-10 { continue; }

        for j in i..=n {
            aug[i * (n + 1) + j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k * (n + 1) + i];
                for j in i..=n {
                    aug[k * (n + 1) + j] -= factor * aug[i * (n + 1) + j];
                }
            }
        }
    }

    (0..n).map(|i| aug[i * (n + 1) + n]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_network_regression() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4]).unwrap();

        let mut rbf = RBFNetwork::new(4).gamma(1.0);
        rbf.fit(&x, &y);
        let pred = rbf.predict(&x);
        
        assert_eq!(pred.dims(), &[4]);
    }

    #[test]
    fn test_rbf_classifier() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4]).unwrap();

        let mut clf = RBFClassifier::new(4).gamma(1.0);
        clf.fit(&x, &y);
        let pred = clf.predict(&x);
        
        assert_eq!(pred.dims(), &[4]);
    }
}

//! Semi-Supervised Learning - Label Propagation, Label Spreading, Self-Training

use ghostflow_core::Tensor;

/// Label Propagation for semi-supervised classification
pub struct LabelPropagation {
    pub kernel: LPKernel,
    pub gamma: f32,
    pub n_neighbors: usize,
    pub max_iter: usize,
    pub tol: f32,
    label_distributions_: Option<Vec<Vec<f32>>>,
    classes_: Option<Vec<usize>>,
    n_classes_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum LPKernel {
    RBF,
    KNN,
}

impl LabelPropagation {
    pub fn new() -> Self {
        LabelPropagation {
            kernel: LPKernel::RBF,
            gamma: 20.0,
            n_neighbors: 7,
            max_iter: 1000,
            tol: 1e-3,
            label_distributions_: None,
            classes_: None,
            n_classes_: 0,
        }
    }

    pub fn kernel(mut self, k: LPKernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    fn compute_affinity(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut affinity = vec![vec![0.0f32; n_samples]; n_samples];

        match self.kernel {
            LPKernel::RBF => {
                for i in 0..n_samples {
                    for j in i..n_samples {
                        let mut dist_sq = 0.0f32;
                        for k in 0..n_features {
                            let diff = x[i * n_features + k] - x[j * n_features + k];
                            dist_sq += diff * diff;
                        }
                        let a = (-self.gamma * dist_sq).exp();
                        affinity[i][j] = a;
                        affinity[j][i] = a;
                    }
                }
            }
            LPKernel::KNN => {
                for i in 0..n_samples {
                    let mut distances: Vec<(usize, f32)> = (0..n_samples)
                        .filter(|&j| j != i)
                        .map(|j| {
                            let mut dist_sq = 0.0f32;
                            for k in 0..n_features {
                                let diff = x[i * n_features + k] - x[j * n_features + k];
                                dist_sq += diff * diff;
                            }
                            (j, dist_sq.sqrt())
                        })
                        .collect();

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    for (j, _) in distances.into_iter().take(self.n_neighbors) {
                        affinity[i][j] = 1.0;
                    }
                }

                // Symmetrize
                for i in 0..n_samples {
                    for j in (i + 1)..n_samples {
                        let sym = (affinity[i][j] + affinity[j][i]) / 2.0;
                        affinity[i][j] = sym;
                        affinity[j][i] = sym;
                    }
                }
            }
        }

        affinity
    }

    fn normalize_affinity(&self, affinity: &mut [Vec<f32>], n_samples: usize) {
        // Row normalization: T = D^-1 * W
        for i in 0..n_samples {
            let row_sum: f32 = affinity[i].iter().sum();
            if row_sum > 1e-10 {
                for j in 0..n_samples {
                    affinity[i][j] /= row_sum;
                }
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Find classes (excluding -1 for unlabeled)
        let mut classes: Vec<usize> = y_data.iter()
            .filter(|&&yi| yi >= 0.0)
            .map(|&yi| yi as usize)
            .collect();
        classes.sort();
        classes.dedup();
        self.n_classes_ = classes.len();
        self.classes_ = Some(classes.clone());

        // Initialize label distributions
        let mut y_dist = vec![vec![0.0f32; self.n_classes_]; n_samples];
        let mut labeled_mask = vec![false; n_samples];

        for i in 0..n_samples {
            if y_data[i] >= 0.0 {
                let class_idx = classes.iter().position(|&c| c == y_data[i] as usize).unwrap();
                y_dist[i][class_idx] = 1.0;
                labeled_mask[i] = true;
            } else {
                // Uniform distribution for unlabeled
                for c in 0..self.n_classes_ {
                    y_dist[i][c] = 1.0 / self.n_classes_ as f32;
                }
            }
        }

        // Compute and normalize affinity matrix
        let mut affinity = self.compute_affinity(&x_data, n_samples, n_features);
        self.normalize_affinity(&mut affinity, n_samples);

        // Label propagation iterations
        for _ in 0..self.max_iter {
            let y_dist_old = y_dist.clone();

            // Propagate: Y = T * Y
            for i in 0..n_samples {
                if !labeled_mask[i] {
                    for c in 0..self.n_classes_ {
                        y_dist[i][c] = 0.0;
                        for j in 0..n_samples {
                            y_dist[i][c] += affinity[i][j] * y_dist_old[j][c];
                        }
                    }

                    // Normalize
                    let sum: f32 = y_dist[i].iter().sum();
                    if sum > 1e-10 {
                        for c in 0..self.n_classes_ {
                            y_dist[i][c] /= sum;
                        }
                    }
                }
            }

            // Clamp labeled points
            for i in 0..n_samples {
                if labeled_mask[i] {
                    let class_idx = classes.iter().position(|&c| c == y_data[i] as usize).unwrap();
                    for c in 0..self.n_classes_ {
                        y_dist[i][c] = if c == class_idx { 1.0 } else { 0.0 };
                    }
                }
            }

            // Check convergence
            let mut max_diff = 0.0f32;
            for i in 0..n_samples {
                for c in 0..self.n_classes_ {
                    max_diff = max_diff.max((y_dist[i][c] - y_dist_old[i][c]).abs());
                }
            }

            if max_diff < self.tol {
                break;
            }
        }

        self.label_distributions_ = Some(y_dist);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];

        let classes = self.classes_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let start = i * self.n_classes_;
                let probs = &proba_data[start..start + self.n_classes_];
                let max_idx = probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx] as f32
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, _x: &Tensor) -> Tensor {
        // For transductive learning, return stored distributions
        let y_dist = self.label_distributions_.as_ref().expect("Model not fitted");
        let n_samples = y_dist.len();

        let mut probs: Vec<f32> = Vec::with_capacity(n_samples * self.n_classes_);
        for dist in y_dist {
            probs.extend(dist);
        }

        Tensor::from_slice(&probs, &[n_samples, self.n_classes_]).unwrap()
    }
}

impl Default for LabelPropagation {
    fn default() -> Self {
        Self::new()
    }
}

/// Label Spreading - similar to Label Propagation but with clamping factor
pub struct LabelSpreading {
    pub kernel: LPKernel,
    pub gamma: f32,
    pub n_neighbors: usize,
    pub alpha: f32,  // Clamping factor
    pub max_iter: usize,
    pub tol: f32,
    label_distributions_: Option<Vec<Vec<f32>>>,
    classes_: Option<Vec<usize>>,
    n_classes_: usize,
}

impl LabelSpreading {
    pub fn new() -> Self {
        LabelSpreading {
            kernel: LPKernel::RBF,
            gamma: 20.0,
            n_neighbors: 7,
            alpha: 0.2,
            max_iter: 30,
            tol: 1e-3,
            label_distributions_: None,
            classes_: None,
            n_classes_: 0,
        }
    }

    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a.clamp(0.0, 1.0);
        self
    }

    fn compute_affinity(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut affinity = vec![vec![0.0f32; n_samples]; n_samples];

        for i in 0..n_samples {
            for j in i..n_samples {
                let mut dist_sq = 0.0f32;
                for k in 0..n_features {
                    let diff = x[i * n_features + k] - x[j * n_features + k];
                    dist_sq += diff * diff;
                }
                let a = (-self.gamma * dist_sq).exp();
                affinity[i][j] = a;
                affinity[j][i] = a;
            }
        }

        affinity
    }

    fn normalize_laplacian(&self, affinity: &[Vec<f32>], n_samples: usize) -> Vec<Vec<f32>> {
        // Normalized graph Laplacian: S = D^(-1/2) * W * D^(-1/2)
        let degrees: Vec<f32> = (0..n_samples)
            .map(|i| affinity[i].iter().sum::<f32>())
            .collect();

        let mut s = vec![vec![0.0f32; n_samples]; n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                let d_i = degrees[i].max(1e-10).sqrt();
                let d_j = degrees[j].max(1e-10).sqrt();
                s[i][j] = affinity[i][j] / (d_i * d_j);
            }
        }

        s
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Find classes
        let mut classes: Vec<usize> = y_data.iter()
            .filter(|&&yi| yi >= 0.0)
            .map(|&yi| yi as usize)
            .collect();
        classes.sort();
        classes.dedup();
        self.n_classes_ = classes.len();
        self.classes_ = Some(classes.clone());

        // Initialize label distributions
        let mut y_dist = vec![vec![0.0f32; self.n_classes_]; n_samples];
        let mut y_static = vec![vec![0.0f32; self.n_classes_]; n_samples];

        for i in 0..n_samples {
            if y_data[i] >= 0.0 {
                let class_idx = classes.iter().position(|&c| c == y_data[i] as usize).unwrap();
                y_dist[i][class_idx] = 1.0;
                y_static[i][class_idx] = 1.0;
            }
        }

        // Compute normalized Laplacian
        let affinity = self.compute_affinity(&x_data, n_samples, n_features);
        let s = self.normalize_laplacian(&affinity, n_samples);

        // Label spreading iterations: Y = alpha * S * Y + (1 - alpha) * Y_static
        for _ in 0..self.max_iter {
            let y_dist_old = y_dist.clone();

            for i in 0..n_samples {
                for c in 0..self.n_classes_ {
                    let mut propagated = 0.0f32;
                    for j in 0..n_samples {
                        propagated += s[i][j] * y_dist_old[j][c];
                    }
                    y_dist[i][c] = self.alpha * propagated + (1.0 - self.alpha) * y_static[i][c];
                }

                // Normalize
                let sum: f32 = y_dist[i].iter().sum();
                if sum > 1e-10 {
                    for c in 0..self.n_classes_ {
                        y_dist[i][c] /= sum;
                    }
                }
            }

            // Check convergence
            let mut max_diff = 0.0f32;
            for i in 0..n_samples {
                for c in 0..self.n_classes_ {
                    max_diff = max_diff.max((y_dist[i][c] - y_dist_old[i][c]).abs());
                }
            }

            if max_diff < self.tol {
                break;
            }
        }

        self.label_distributions_ = Some(y_dist);
    }

    pub fn predict(&self, _x: &Tensor) -> Tensor {
        let y_dist = self.label_distributions_.as_ref().expect("Model not fitted");
        let classes = self.classes_.as_ref().expect("Model not fitted");
        let n_samples = y_dist.len();

        let predictions: Vec<f32> = y_dist.iter()
            .map(|dist| {
                let max_idx = dist.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx] as f32
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for LabelSpreading {
    fn default() -> Self {
        Self::new()
    }
}

/// Self-Training Classifier
pub struct SelfTrainingClassifier {
    pub threshold: f32,
    pub max_iter: usize,
    pub criterion: SelfTrainingCriterion,
    #[allow(dead_code)]
    n_classes_: usize,
    #[allow(dead_code)]
    n_iter_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum SelfTrainingCriterion {
    Threshold,
    KBest(usize),
}

impl SelfTrainingClassifier {
    pub fn new(threshold: f32) -> Self {
        SelfTrainingClassifier {
            threshold,
            max_iter: 10,
            criterion: SelfTrainingCriterion::Threshold,
            n_classes_: 0,
            n_iter_: 0,
        }
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    pub fn criterion(mut self, c: SelfTrainingCriterion) -> Self {
        self.criterion = c;
        self
    }
}

impl Default for SelfTrainingClassifier {
    fn default() -> Self {
        Self::new(0.75)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_propagation() {
        // Some labeled, some unlabeled (-1)
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            1.0, 1.0,
            1.1, 1.1,
            0.5, 0.5,  // Unlabeled
        ], &[5, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0, -1.0], &[5]).unwrap();

        let mut lp = LabelPropagation::new().gamma(10.0);
        lp.fit(&x, &y);

        let predictions = lp.predict(&x);
        assert_eq!(predictions.dims(), &[5]);
    }

    #[test]
    fn test_label_spreading() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            1.0, 1.0,
            1.1, 1.1,
            0.5, 0.5,
        ], &[5, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0, -1.0], &[5]).unwrap();

        let mut ls = LabelSpreading::new().alpha(0.2);
        ls.fit(&x, &y);

        let predictions = ls.predict(&x);
        assert_eq!(predictions.dims(), &[5]);
    }
}

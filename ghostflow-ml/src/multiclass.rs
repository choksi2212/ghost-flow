//! Multiclass and Multilabel Classification Strategies

use ghostflow_core::Tensor;

/// One-vs-Rest (One-vs-All) Classifier
pub struct OneVsRestClassifier {
    classifiers_: Vec<BinaryClassifier>,
    classes_: Vec<i32>,
    n_features_: usize,
}

#[derive(Clone)]
struct BinaryClassifier {
    coef: Vec<f32>,
    intercept: f32,
}

impl BinaryClassifier {
    fn new(n_features: usize) -> Self {
        BinaryClassifier {
            coef: vec![0.0; n_features],
            intercept: 0.0,
        }
    }

    fn fit(&mut self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize) {
        let lr = 0.1f32;
        let max_iter = 100;
        let alpha = 0.01f32;

        for _ in 0..max_iter {
            let mut grad_coef = vec![0.0f32; n_features];
            let mut grad_intercept = 0.0f32;

            for i in 0..n_samples {
                let mut z = self.intercept;
                for j in 0..n_features {
                    z += self.coef[j] * x[i * n_features + j];
                }
                let pred = 1.0 / (1.0 + (-z).exp());
                let error = pred - y[i];

                grad_intercept += error;
                for j in 0..n_features {
                    grad_coef[j] += error * x[i * n_features + j];
                }
            }

            self.intercept -= lr * grad_intercept / n_samples as f32;
            for j in 0..n_features {
                self.coef[j] -= lr * (grad_coef[j] / n_samples as f32 + alpha * self.coef[j]);
            }
        }
    }

    fn predict_proba(&self, x: &[f32], n_features: usize) -> f32 {
        let mut z = self.intercept;
        for j in 0..n_features {
            z += self.coef[j] * x[j];
        }
        1.0 / (1.0 + (-z).exp())
    }
}

impl OneVsRestClassifier {
    pub fn new() -> Self {
        OneVsRestClassifier {
            classifiers_: Vec::new(),
            classes_: Vec::new(),
            n_features_: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Find unique classes
        let mut classes: Vec<i32> = y_data.iter().map(|&v| v as i32).collect();
        classes.sort();
        classes.dedup();
        self.classes_ = classes.clone();

        // Train one classifier per class
        self.classifiers_ = classes.iter()
            .map(|&class| {
                let y_binary: Vec<f32> = y_data.iter()
                    .map(|&v| if v as i32 == class { 1.0 } else { 0.0 })
                    .collect();
                
                let mut clf = BinaryClassifier::new(n_features);
                clf.fit(&x_data, &y_binary, n_samples, n_features);
                clf
            })
            .collect();
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];
        let n_classes = self.classes_.len();

        let labels: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut max_prob = f32::NEG_INFINITY;
                let mut max_class = self.classes_[0];
                for (k, &class) in self.classes_.iter().enumerate() {
                    if proba_data[i * n_classes + k] > max_prob {
                        max_prob = proba_data[i * n_classes + k];
                        max_class = class;
                    }
                }
                max_class as f32
            })
            .collect();

        Tensor::from_slice(&labels, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let n_classes = self.classes_.len();

        let mut proba = vec![0.0f32; n_samples * n_classes];

        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            let mut sum = 0.0f32;
            
            for (k, clf) in self.classifiers_.iter().enumerate() {
                let p = clf.predict_proba(xi, n_features);
                proba[i * n_classes + k] = p;
                sum += p;
            }

            // Normalize
            if sum > 0.0 {
                for k in 0..n_classes {
                    proba[i * n_classes + k] /= sum;
                }
            }
        }

        Tensor::from_slice(&proba, &[n_samples, n_classes]).unwrap()
    }
}

impl Default for OneVsRestClassifier {
    fn default() -> Self { Self::new() }
}

/// One-vs-One Classifier
pub struct OneVsOneClassifier {
    classifiers_: Vec<(i32, i32, BinaryClassifier)>,
    classes_: Vec<i32>,
    n_features_: usize,
}

impl OneVsOneClassifier {
    pub fn new() -> Self {
        OneVsOneClassifier {
            classifiers_: Vec::new(),
            classes_: Vec::new(),
            n_features_: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Find unique classes
        let mut classes: Vec<i32> = y_data.iter().map(|&v| v as i32).collect();
        classes.sort();
        classes.dedup();
        self.classes_ = classes.clone();

        // Train one classifier per pair of classes
        self.classifiers_.clear();
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                let class_i = classes[i];
                let class_j = classes[j];

                // Extract samples for these two classes
                let indices: Vec<usize> = (0..n_samples)
                    .filter(|&k| y_data[k] as i32 == class_i || y_data[k] as i32 == class_j)
                    .collect();

                let x_subset: Vec<f32> = indices.iter()
                    .flat_map(|&k| x_data[k * n_features..(k + 1) * n_features].to_vec())
                    .collect();
                let y_subset: Vec<f32> = indices.iter()
                    .map(|&k| if y_data[k] as i32 == class_j { 1.0 } else { 0.0 })
                    .collect();

                let mut clf = BinaryClassifier::new(n_features);
                clf.fit(&x_subset, &y_subset, indices.len(), n_features);
                self.classifiers_.push((class_i, class_j, clf));
            }
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let labels: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                
                // Vote counting
                let mut votes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
                
                for (class_i, class_j, clf) in &self.classifiers_ {
                    let prob = clf.predict_proba(xi, n_features);
                    let winner = if prob >= 0.5 { *class_j } else { *class_i };
                    *votes.entry(winner).or_insert(0) += 1;
                }

                // Return class with most votes
                votes.into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(class, _)| class)
                    .unwrap_or(self.classes_[0]) as f32
            })
            .collect();

        Tensor::from_slice(&labels, &[n_samples]).unwrap()
    }
}

impl Default for OneVsOneClassifier {
    fn default() -> Self { Self::new() }
}

/// Output Code Classifier (Error-Correcting Output Codes)
pub struct OutputCodeClassifier {
    pub code_size: f32,
    classifiers_: Vec<BinaryClassifier>,
    code_book_: Vec<Vec<i8>>,
    classes_: Vec<i32>,
    n_features_: usize,
}

impl OutputCodeClassifier {
    pub fn new() -> Self {
        OutputCodeClassifier {
            code_size: 1.5,
            classifiers_: Vec::new(),
            code_book_: Vec::new(),
            classes_: Vec::new(),
            n_features_: 0,
        }
    }

    pub fn code_size(mut self, size: f32) -> Self {
        self.code_size = size;
        self
    }

    fn generate_code_book(&self, n_classes: usize) -> Vec<Vec<i8>> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let n_classifiers = (self.code_size * n_classes as f32).ceil() as usize;
        
        // Generate random code book
        (0..n_classes)
            .map(|_| {
                (0..n_classifiers)
                    .map(|_| if rng.gen::<bool>() { 1i8 } else { -1i8 })
                    .collect()
            })
            .collect()
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;

        // Find unique classes
        let mut classes: Vec<i32> = y_data.iter().map(|&v| v as i32).collect();
        classes.sort();
        classes.dedup();
        self.classes_ = classes.clone();

        // Generate code book
        self.code_book_ = self.generate_code_book(classes.len());
        let n_classifiers = self.code_book_[0].len();

        // Train classifiers
        self.classifiers_ = (0..n_classifiers)
            .map(|c| {
                // Create binary labels based on code book
                let y_binary: Vec<f32> = y_data.iter()
                    .map(|&v| {
                        let class_idx = classes.iter().position(|&c| c == v as i32).unwrap();
                        if self.code_book_[class_idx][c] == 1 { 1.0 } else { 0.0 }
                    })
                    .collect();

                let mut clf = BinaryClassifier::new(n_features);
                clf.fit(&x_data, &y_binary, n_samples, n_features);
                clf
            })
            .collect();
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let labels: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                
                // Get predictions from all classifiers
                let predictions: Vec<i8> = self.classifiers_.iter()
                    .map(|clf| {
                        let prob = clf.predict_proba(xi, n_features);
                        if prob >= 0.5 { 1i8 } else { -1i8 }
                    })
                    .collect();

                // Find closest code word (Hamming distance)
                let mut min_dist = usize::MAX;
                let mut best_class = self.classes_[0];

                for (class_idx, code) in self.code_book_.iter().enumerate() {
                    let dist: usize = predictions.iter().zip(code.iter())
                        .filter(|(&p, &c)| p != c)
                        .count();
                    if dist < min_dist {
                        min_dist = dist;
                        best_class = self.classes_[class_idx];
                    }
                }

                best_class as f32
            })
            .collect();

        Tensor::from_slice(&labels, &[n_samples]).unwrap()
    }
}

impl Default for OutputCodeClassifier {
    fn default() -> Self { Self::new() }
}

/// Classifier Chain for multi-label classification
pub struct ClassifierChain {
    pub order: Option<Vec<usize>>,
    classifiers_: Vec<BinaryClassifier>,
    n_labels_: usize,
    n_features_: usize,
}

impl ClassifierChain {
    pub fn new() -> Self {
        ClassifierChain {
            order: None,
            classifiers_: Vec::new(),
            n_labels_: 0,
            n_features_: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let n_labels = y.dims()[1];

        self.n_features_ = n_features;
        self.n_labels_ = n_labels;

        // Determine order
        let order: Vec<usize> = self.order.clone()
            .unwrap_or_else(|| (0..n_labels).collect());

        // Train classifiers in chain order
        self.classifiers_ = Vec::with_capacity(n_labels);
        
        for (chain_idx, &label_idx) in order.iter().enumerate() {
            // Features = original features + predictions from previous classifiers
            let augmented_features = n_features + chain_idx;
            
            let x_augmented: Vec<f32> = (0..n_samples)
                .flat_map(|i| {
                    let mut row = x_data[i * n_features..(i + 1) * n_features].to_vec();
                    // Add previous predictions (using true labels during training)
                    for &prev_label in &order[..chain_idx] {
                        row.push(y_data[i * n_labels + prev_label]);
                    }
                    row
                })
                .collect();

            let y_label: Vec<f32> = (0..n_samples)
                .map(|i| y_data[i * n_labels + label_idx])
                .collect();

            let mut clf = BinaryClassifier::new(augmented_features);
            clf.fit(&x_augmented, &y_label, n_samples, augmented_features);
            self.classifiers_.push(clf);
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let order: Vec<usize> = self.order.clone()
            .unwrap_or_else(|| (0..self.n_labels_).collect());

        let mut predictions = vec![0.0f32; n_samples * self.n_labels_];

        for i in 0..n_samples {
            let mut augmented = x_data[i * n_features..(i + 1) * n_features].to_vec();

            for (chain_idx, &label_idx) in order.iter().enumerate() {
                let prob = self.classifiers_[chain_idx].predict_proba(&augmented, n_features + chain_idx);
                let pred = if prob >= 0.5 { 1.0 } else { 0.0 };
                predictions[i * self.n_labels_ + label_idx] = pred;
                augmented.push(pred);
            }
        }

        Tensor::from_slice(&predictions, &[n_samples, self.n_labels_]).unwrap()
    }
}

impl Default for ClassifierChain {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_vs_rest() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 1.0], &[4]).unwrap();

        let mut clf = OneVsRestClassifier::new();
        clf.fit(&x, &y);
        let pred = clf.predict(&x);
        assert_eq!(pred.dims(), &[4]);
    }

    #[test]
    fn test_one_vs_one() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 1.0], &[4]).unwrap();

        let mut clf = OneVsOneClassifier::new();
        clf.fit(&x, &y);
        let pred = clf.predict(&x);
        assert_eq!(pred.dims(), &[4]);
    }
}



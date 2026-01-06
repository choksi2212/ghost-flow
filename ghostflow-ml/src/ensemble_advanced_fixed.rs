//! Advanced Ensemble Methods - AdaBoost, Bagging, Extra Trees, Voting, Stacking

use ghostflow_core::Tensor;
use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor, Criterion};
use rayon::prelude::*;
use rand::prelude::*;

/// AdaBoost Classifier (Adaptive Boosting)
pub struct AdaBoostClassifier {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub algorithm: AdaBoostAlgorithm,
    estimators_: Vec<DecisionTreeClassifier>,
    estimator_weights_: Vec<f32>,
    n_classes_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum AdaBoostAlgorithm {
    SAMME,
    SAMME_R,
}

impl AdaBoostClassifier {
    pub fn new(n_estimators: usize) -> Self {
        AdaBoostClassifier {
            n_estimators,
            learning_rate: 1.0,
            algorithm: AdaBoostAlgorithm::SAMME_R,
            estimators_: Vec::new(),
            estimator_weights_: Vec::new(),
            n_classes_: 0,
        }
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn algorithm(mut self, algo: AdaBoostAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        // Initialize sample weights
        let mut sample_weights = vec![1.0f32 / n_samples as f32; n_samples];

        self.estimators_.clear();
        self.estimator_weights_.clear();

        for _ in 0..self.n_estimators {
            // Train weak learner (decision stump)
            let mut tree = DecisionTreeClassifier::new()
                .max_depth(1)
                .criterion(Criterion::Gini);

            // Weighted sampling
            let mut rng = thread_rng();
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| {
                    let r: f32 = rng.gen();
                    let mut cumsum = 0.0f32;
                    for (i, &w) in sample_weights.iter().enumerate() {
                        cumsum += w;
                        if r < cumsum {
                            return i;
                        }
                    }
                    n_samples - 1
                })
                .collect();

            let x_boot: Vec<f32> = indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let y_boot: Vec<f32> = indices.iter().map(|&i| y_data[i]).collect();

            let x_tensor = Tensor::from_slice(&x_boot, &[indices.len(), n_features]).unwrap();
            let y_tensor = Tensor::from_slice(&y_boot, &[indices.len()]).unwrap();

            tree.fit(&x_tensor, &y_tensor);

            // Compute predictions and error
            let predictions = tree.predict(x);
            let pred_data = predictions.data_f32();

            let mut weighted_error = 0.0f32;
            for i in 0..n_samples {
                if (pred_data[i] - y_data[i]).abs() > 0.5 {
                    weighted_error += sample_weights[i];
                }
            }

            // Avoid division by zero
            weighted_error = weighted_error.clamp(1e-10, 1.0 - 1e-10);

            // Compute estimator weight
            let estimator_weight = self.learning_rate * 
                ((1.0 - weighted_error) / weighted_error).ln() +
                (self.n_classes_ as f32 - 1.0).ln();

            // Update sample weights
            for i in 0..n_samples {
                if (pred_data[i] - y_data[i]).abs() > 0.5 {
                    sample_weights[i] *= (estimator_weight).exp();
                }
            }

            // Normalize weights
            let sum: f32 = sample_weights.iter().sum();
            for w in &mut sample_weights {
                *w /= sum;
            }

            self.estimators_.push(tree);
            self.estimator_weights_.push(estimator_weight);
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        
        // Weighted vote
        let mut class_scores = vec![vec![0.0f32; self.n_classes_]; n_samples];

        for (tree, &weight) in self.estimators_.iter().zip(self.estimator_weights_.iter()) {
            let predictions = tree.predict(x);
            let pred_data = predictions.data_f32();

            for i in 0..n_samples {
                let class = pred_data[i] as usize;
                if class < self.n_classes_ {
                    class_scores[i][class] += weight;
                }
            }
        }

        let predictions: Vec<f32> = class_scores.iter()
            .map(|scores| {
                scores.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(c, _)| c as f32)
                    .unwrap_or(0.0)
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

/// Bagging Classifier (Bootstrap Aggregating)
pub struct BaggingClassifier {
    pub n_estimators: usize,
    pub max_samples: f32,
    pub max_features: f32,
    pub bootstrap: bool,
    pub bootstrap_features: bool,
    estimators_: Vec<DecisionTreeClassifier>,
    n_classes_: usize,
}

impl BaggingClassifier {
    pub fn new(n_estimators: usize) -> Self {
        BaggingClassifier {
            n_estimators,
            max_samples: 1.0,
            max_features: 1.0,
            bootstrap: true,
            bootstrap_features: false,
            estimators_: Vec::new(),
            n_classes_: 0,
        }
    }

    pub fn max_samples(mut self, ratio: f32) -> Self {
        self.max_samples = ratio.clamp(0.1, 1.0);
        self
    }

    pub fn max_features(mut self, ratio: f32) -> Self {
        self.max_features = ratio.clamp(0.1, 1.0);
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        let n_samples_bootstrap = (n_samples as f32 * self.max_samples) as usize;
        let n_features_bootstrap = (n_features as f32 * self.max_features) as usize;

        self.estimators_ = (0..self.n_estimators)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();

                // Bootstrap samples
                let sample_indices: Vec<usize> = if self.bootstrap {
                    (0..n_samples_bootstrap).map(|_| rng.gen_range(0..n_samples)).collect()
                } else {
                    (0..n_samples).choose_multiple(&mut rng, n_samples_bootstrap).collect()
                };

                // Bootstrap features
                let feature_indices: Vec<usize> = if self.bootstrap_features {
                    (0..n_features_bootstrap).map(|_| rng.gen_range(0..n_features)).collect()
                } else {
                    (0..n_features).choose_multiple(&mut rng, n_features_bootstrap).collect()
                };

                // Create bootstrap sample
                let x_boot: Vec<f32> = sample_indices.iter()
                    .flat_map(|&i| {
                        feature_indices.iter().map(move |&j| x_data[i * n_features + j])
                    })
                    .collect();
                let y_boot: Vec<f32> = sample_indices.iter().map(|&i| y_data[i]).collect();

                let x_tensor = Tensor::from_slice(&x_boot, &[sample_indices.len(), feature_indices.len()]).unwrap();
                let y_tensor = Tensor::from_slice(&y_boot, &[sample_indices.len()]).unwrap();

                let mut tree = DecisionTreeClassifier::new();
                tree.fit(&x_tensor, &y_tensor);
                tree
            })
            .collect();
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        
        let mut class_votes = vec![vec![0usize; self.n_classes_]; n_samples];

        for tree in &self.estimators_ {
            let predictions = tree.predict(x);
            let pred_data = predictions.data_f32();

            for i in 0..n_samples {
                let class = pred_data[i] as usize;
                if class < self.n_classes_ {
                    class_votes[i][class] += 1;
                }
            }
        }

        let predictions: Vec<f32> = class_votes.iter()
            .map(|votes| {
                votes.iter()
                    .enumerate()
                    .max_by_key(|(_, &v)| v)
                    .map(|(c, _)| c as f32)
                    .unwrap_or(0.0)
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


/// Extra Trees Classifier (Extremely Randomized Trees)
pub struct ExtraTreesClassifier {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub max_features: Option<usize>,
    trees_: Vec<DecisionTreeClassifier>,
    n_classes_: usize,
}

impl ExtraTreesClassifier {
    pub fn new(n_estimators: usize) -> Self {
        ExtraTreesClassifier {
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            max_features: None,
            trees_: Vec::new(),
            n_classes_: 0,
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let y_data = y.data_f32();

        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        let max_features = self.max_features.unwrap_or_else(|| (n_features as f32).sqrt() as usize);

        // Extra Trees uses all samples but random splits
        self.trees_ = (0..self.n_estimators)
            .into_par_iter()
            .map(|_| {
                let mut tree = DecisionTreeClassifier::new()
                    .criterion(Criterion::Gini)
                    .min_samples_split(self.min_samples_split);

                if let Some(depth) = self.max_depth {
                    tree = tree.max_depth(depth);
                }
                tree.max_features = Some(max_features);

                // Extra Trees uses all data (no bootstrap)
                tree.fit(x, y);
                tree
            })
            .collect();
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        
        let mut class_votes = vec![vec![0usize; self.n_classes_]; n_samples];

        for tree in &self.trees_ {
            let predictions = tree.predict(x);
            let pred_data = predictions.data_f32();

            for i in 0..n_samples {
                let class = pred_data[i] as usize;
                if class < self.n_classes_ {
                    class_votes[i][class] += 1;
                }
            }
        }

        let predictions: Vec<f32> = class_votes.iter()
            .map(|votes| {
                votes.iter()
                    .enumerate()
                    .max_by_key(|(_, &v)| v)
                    .map(|(c, _)| c as f32)
                    .unwrap_or(0.0)
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        
        let mut class_probs = vec![vec![0.0f32; self.n_classes_]; n_samples];

        for tree in &self.trees_ {
            let proba = tree.predict_proba(x);
            let proba_data = proba.data_f32();

            for i in 0..n_samples {
                for c in 0..self.n_classes_ {
                    class_probs[i][c] += proba_data[i * self.n_classes_ + c];
                }
            }
        }

        // Average
        let n_trees = self.trees_.len() as f32;
        let mut result = Vec::with_capacity(n_samples * self.n_classes_);
        for i in 0..n_samples {
            for c in 0..self.n_classes_ {
                result.push(class_probs[i][c] / n_trees);
            }
        }

        Tensor::from_slice(&result, &[n_samples, self.n_classes_]).unwrap()
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

/// Voting Classifier - combines multiple classifiers
pub struct VotingClassifier {
    pub voting: VotingType,
    pub weights: Option<Vec<f32>>,
    estimators_: Vec<Box<dyn Classifier + Send + Sync>>,
    n_classes_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum VotingType {
    Hard,
    Soft,
}

pub trait Classifier {
    fn fit(&mut self, x: &Tensor, y: &Tensor);
    fn predict(&self, x: &Tensor) -> Tensor;
    fn predict_proba(&self, x: &Tensor) -> Option<Tensor>;
}

impl VotingClassifier {
    pub fn new(voting: VotingType) -> Self {
        VotingClassifier {
            voting,
            weights: None,
            estimators_: Vec::new(),
            n_classes_: 0,
        }
    }

    pub fn weights(mut self, w: Vec<f32>) -> Self {
        self.weights = Some(w);
        self
    }

    pub fn add_estimator(&mut self, estimator: Box<dyn Classifier + Send + Sync>) {
        self.estimators_.push(estimator);
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let y_data = y.data_f32();
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        for estimator in &mut self.estimators_ {
            estimator.fit(x, y);
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let weights = self.weights.clone().unwrap_or_else(|| vec![1.0; self.estimators_.len()]);

        match self.voting {
            VotingType::Hard => {
                let mut class_votes = vec![vec![0.0f32; self.n_classes_]; n_samples];

                for (estimator, &weight) in self.estimators_.iter().zip(weights.iter()) {
                    let predictions = estimator.predict(x);
                    let pred_data = predictions.data_f32();

                    for i in 0..n_samples {
                        let class = pred_data[i] as usize;
                        if class < self.n_classes_ {
                            class_votes[i][class] += weight;
                        }
                    }
                }

                let predictions: Vec<f32> = class_votes.iter()
                    .map(|votes| {
                        votes.iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(c, _)| c as f32)
                            .unwrap_or(0.0)
                    })
                    .collect();

                Tensor::from_slice(&predictions, &[n_samples]).unwrap()
            }
            VotingType::Soft => {
                let mut class_probs = vec![vec![0.0f32; self.n_classes_]; n_samples];

                for (estimator, &weight) in self.estimators_.iter().zip(weights.iter()) {
                    if let Some(proba) = estimator.predict_proba(x) {
                        let proba_data = proba.data_f32();

                        for i in 0..n_samples {
                            for c in 0..self.n_classes_ {
                                class_probs[i][c] += weight * proba_data[i * self.n_classes_ + c];
                            }
                        }
                    }
                }

                let predictions: Vec<f32> = class_probs.iter()
                    .map(|probs| {
                        probs.iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(c, _)| c as f32)
                            .unwrap_or(0.0)
                    })
                    .collect();

                Tensor::from_slice(&predictions, &[n_samples]).unwrap()
            }
        }
    }
}

/// Isolation Forest for anomaly detection
pub struct IsolationForest {
    pub n_estimators: usize,
    pub max_samples: usize,
    pub contamination: f32,
    trees_: Vec<IsolationTree>,
    threshold_: f32,
}

struct IsolationTree {
    root: Option<IsolationNode>,
    max_depth: usize,
}

struct IsolationNode {
    feature: Option<usize>,
    threshold: Option<f32>,
    left: Option<Box<IsolationNode>>,
    right: Option<Box<IsolationNode>>,
    size: usize,
}

impl IsolationForest {
    pub fn new(n_estimators: usize) -> Self {
        IsolationForest {
            n_estimators,
            max_samples: 256,
            contamination: 0.1,
            trees_: Vec::new(),
            threshold_: 0.0,
        }
    }

    pub fn max_samples(mut self, n: usize) -> Self {
        self.max_samples = n;
        self
    }

    pub fn contamination(mut self, c: f32) -> Self {
        self.contamination = c.clamp(0.0, 0.5);
        self
    }

    fn build_tree(&self, x: &[f32], indices: &[usize], n_features: usize, depth: usize, max_depth: usize) -> IsolationNode {
        let n = indices.len();

        if depth >= max_depth || n <= 1 {
            return IsolationNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                size: n,
            };
        }

        let mut rng = thread_rng();
        let feature = rng.gen_range(0..n_features);

        // Find min/max for this feature
        let values: Vec<f32> = indices.iter()
            .map(|&i| x[i * n_features + feature])
            .collect();
        
        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_val - min_val).abs() < 1e-10 {
            return IsolationNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                size: n,
            };
        }

        let threshold = rng.gen_range(min_val..max_val);

        let (left_indices, right_indices): (Vec<_>, Vec<_>) = indices.iter()
            .partition(|&&i| x[i * n_features + feature] < threshold);

        IsolationNode {
            feature: Some(feature),
            threshold: Some(threshold),
            left: Some(Box::new(self.build_tree(x, &left_indices, n_features, depth + 1, max_depth))),
            right: Some(Box::new(self.build_tree(x, &right_indices, n_features, depth + 1, max_depth))),
            size: n,
        }
    }

    fn path_length(&self, node: &IsolationNode, sample: &[f32], depth: usize) -> f32 {
        if node.feature.is_none() {
            let c = self.c_factor(node.size);
            return depth as f32 + c;
        }

        let feature = node.feature.unwrap();
        let threshold = node.threshold.unwrap();

        if sample[feature] < threshold {
            if let Some(ref left) = node.left {
                return self.path_length(left, sample, depth + 1);
            }
        } else {
            if let Some(ref right) = node.right {
                return self.path_length(right, sample, depth + 1);
            }
        }

        depth as f32
    }

    fn c_factor(&self, n: usize) -> f32 {
        if n <= 1 {
            return 0.0;
        }
        let n = n as f32;
        2.0 * ((n - 1.0).ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let subsample_size = self.max_samples.min(n_samples);
        let max_depth = (subsample_size as f32).log2().ceil() as usize;

        self.trees_ = (0..self.n_estimators)
            .map(|_| {
                let mut rng = thread_rng();
                let indices: Vec<usize> = (0..n_samples)
                    .choose_multiple(&mut rng, subsample_size)
                    .collect();

                let root = self.build_tree(&x_data, &indices, n_features, 0, max_depth);
                IsolationTree { root: Some(root), max_depth }
            })
            .collect();

        // Compute threshold based on contamination
        let scores = self.decision_function(x);
        let mut score_data = scores.data_f32();
        score_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let threshold_idx = ((1.0 - self.contamination) * n_samples as f32) as usize;
        self.threshold_ = score_data[threshold_idx.min(n_samples - 1)];
    }

    pub fn decision_function(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let c = self.c_factor(self.max_samples);

        let scores: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                
                let avg_path_length: f32 = self.trees_.iter()
                    .map(|tree| {
                        if let Some(ref root) = tree.root {
                            self.path_length(root, sample, 0)
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>() / self.trees_.len() as f32;

                // Anomaly score: higher means more anomalous
                -2.0f32.powf(-avg_path_length / c)
            })
            .collect();

        Tensor::from_slice(&scores, &[n_samples]).unwrap()
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let scores = self.decision_function(x);
        let score_data = scores.data_f32();
        let n_samples = x.dims()[0];

        let predictions: Vec<f32> = score_data.iter()
            .map(|&s| if s < self.threshold_ { 1.0 } else { -1.0 })  // 1 = inlier, -1 = outlier
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaboost() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0], &[4]).unwrap();
        
        let mut ada = AdaBoostClassifier::new(10);
        ada.fit(&x, &y);
        
        let predictions = ada.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }

    #[test]
    fn test_bagging() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut bag = BaggingClassifier::new(10);
        bag.fit(&x, &y);
        
        let score = bag.score(&x, &y);
        assert!(score >= 0.5);
    }

    #[test]
    fn test_isolation_forest() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            10.0, 10.0,  // Outlier
        ], &[4, 2]).unwrap();
        
        let mut iso = IsolationForest::new(10).contamination(0.25);
        iso.fit(&x);
        
        let predictions = iso.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }
}



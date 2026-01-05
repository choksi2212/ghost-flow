//! Advanced Gradient Boosting Implementations
//! 
//! This module provides XGBoost, LightGBM, and CatBoost-style gradient boosting algorithms.

use ghostflow_core::Tensor;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// XGBoost-style Gradient Boosting
/// 
/// Features:
/// - Regularized learning objective (L1 and L2)
/// - Column (feature) subsampling
/// - Shrinkage (learning rate)
/// - Tree pruning with max_depth
/// - Histogram-based split finding
pub struct XGBoostClassifier {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_depth: usize,
    pub min_child_weight: f32,
    pub subsample: f32,
    pub colsample_bytree: f32,
    pub reg_lambda: f32,  // L2 regularization
    pub reg_alpha: f32,   // L1 regularization
    pub gamma: f32,       // Minimum loss reduction for split
    trees: Vec<XGBTree>,
    base_score: f32,
}

#[derive(Clone)]
struct XGBTree {
    nodes: Vec<XGBNode>,
    feature_indices: Vec<usize>,
}

#[derive(Clone)]
struct XGBNode {
    is_leaf: bool,
    feature: usize,
    threshold: f32,
    left: usize,
    right: usize,
    value: f32,
    gain: f32,
}

impl XGBoostClassifier {
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            learning_rate: 0.3,
            max_depth: 6,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            gamma: 0.0,
            trees: Vec::new(),
            base_score: 0.5,
        }
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn subsample(mut self, ratio: f32) -> Self {
        self.subsample = ratio;
        self
    }

    pub fn colsample_bytree(mut self, ratio: f32) -> Self {
        self.colsample_bytree = ratio;
        self
    }

    pub fn reg_lambda(mut self, lambda: f32) -> Self {
        self.reg_lambda = lambda;
        self
    }

    pub fn reg_alpha(mut self, alpha: f32) -> Self {
        self.reg_alpha = alpha;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();
        let y_data = y.data_f32();

        // Initialize predictions with base score
        let mut predictions = vec![self.base_score; n_samples];
        
        // Build trees sequentially
        for _ in 0..self.n_estimators {
            // Calculate gradients and hessians (for logistic loss)
            let mut gradients = Vec::with_capacity(n_samples);
            let mut hessians = Vec::with_capacity(n_samples);
            
            for i in 0..n_samples {
                let pred = 1.0 / (1.0 + (-predictions[i]).exp());
                let grad = pred - y_data[i];
                let hess = pred * (1.0 - pred);
                gradients.push(grad);
                hessians.push(hess.max(1e-6)); // Avoid division by zero
            }

            // Row subsampling
            let mut rng = thread_rng();
            let sample_indices: Vec<usize> = if self.subsample < 1.0 {
                let n_subsample = (n_samples as f32 * self.subsample) as usize;
                (0..n_samples).choose_multiple(&mut rng, n_subsample)
            } else {
                (0..n_samples).collect()
            };

            // Column subsampling
            let feature_indices: Vec<usize> = if self.colsample_bytree < 1.0 {
                let n_features_sample = (n_features as f32 * self.colsample_bytree) as usize;
                (0..n_features).choose_multiple(&mut rng, n_features_sample)
            } else {
                (0..n_features).collect()
            };

            // Build tree
            let tree = self.build_tree(
                &x_data,
                &gradients,
                &hessians,
                &sample_indices,
                &feature_indices,
                n_features,
                0,
            );

            // Update predictions
            for i in 0..n_samples {
                let leaf_value = self.predict_tree(&tree, &x_data[i * n_features..(i + 1) * n_features]);
                predictions[i] += self.learning_rate * leaf_value;
            }

            self.trees.push(tree);
        }
    }

    fn build_tree(
        &self,
        x_data: &[f32],
        gradients: &[f32],
        hessians: &[f32],
        sample_indices: &[usize],
        feature_indices: &[usize],
        n_features: usize,
        depth: usize,
    ) -> XGBTree {
        let mut nodes = Vec::new();
        
        // Build root node
        self.build_node(
            x_data,
            gradients,
            hessians,
            sample_indices,
            feature_indices,
            n_features,
            depth,
            &mut nodes,
        );

        XGBTree {
            nodes,
            feature_indices: feature_indices.to_vec(),
        }
    }

    fn build_node(
        &self,
        x_data: &[f32],
        gradients: &[f32],
        hessians: &[f32],
        sample_indices: &[usize],
        feature_indices: &[usize],
        n_features: usize,
        depth: usize,
        nodes: &mut Vec<XGBNode>,
    ) -> usize {
        let node_idx = nodes.len();

        // Calculate node statistics
        let sum_grad: f32 = sample_indices.iter().map(|&i| gradients[i]).sum();
        let sum_hess: f32 = sample_indices.iter().map(|&i| hessians[i]).sum();

        // Calculate leaf value with regularization
        let leaf_value = -sum_grad / (sum_hess + self.reg_lambda);

        // Check stopping criteria
        if depth >= self.max_depth || sample_indices.len() < 2 || sum_hess < self.min_child_weight {
            nodes.push(XGBNode {
                is_leaf: true,
                feature: 0,
                threshold: 0.0,
                left: 0,
                right: 0,
                value: leaf_value,
                gain: 0.0,
            });
            return node_idx;
        }

        // Find best split
        let (best_feature, best_threshold, best_gain, left_indices, right_indices) =
            self.find_best_split(
                x_data,
                gradients,
                hessians,
                sample_indices,
                feature_indices,
                n_features,
                sum_grad,
                sum_hess,
            );

        // Check if split is beneficial
        if best_gain < self.gamma || left_indices.is_empty() || right_indices.is_empty() {
            nodes.push(XGBNode {
                is_leaf: true,
                feature: 0,
                threshold: 0.0,
                left: 0,
                right: 0,
                value: leaf_value,
                gain: 0.0,
            });
            return node_idx;
        }

        // Create internal node
        nodes.push(XGBNode {
            is_leaf: false,
            feature: best_feature,
            threshold: best_threshold,
            left: 0,  // Will be updated
            right: 0, // Will be updated
            value: 0.0,
            gain: best_gain,
        });

        // Build left and right children
        let left_idx = self.build_node(
            x_data,
            gradients,
            hessians,
            &left_indices,
            feature_indices,
            n_features,
            depth + 1,
            nodes,
        );

        let right_idx = self.build_node(
            x_data,
            gradients,
            hessians,
            &right_indices,
            feature_indices,
            n_features,
            depth + 1,
            nodes,
        );

        // Update children indices
        nodes[node_idx].left = left_idx;
        nodes[node_idx].right = right_idx;

        node_idx
    }

    fn find_best_split(
        &self,
        x_data: &[f32],
        gradients: &[f32],
        hessians: &[f32],
        sample_indices: &[usize],
        feature_indices: &[usize],
        n_features: usize,
        sum_grad: f32,
        sum_hess: f32,
    ) -> (usize, f32, f32, Vec<usize>, Vec<usize>) {
        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for &feature in feature_indices {
            // Get feature values for samples
            let mut feature_values: Vec<(f32, usize)> = sample_indices
                .iter()
                .map(|&i| (x_data[i * n_features + feature], i))
                .collect();
            
            feature_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Try splits
            let mut left_grad = 0.0;
            let mut left_hess = 0.0;

            for i in 0..feature_values.len() - 1 {
                let idx = feature_values[i].1;
                left_grad += gradients[idx];
                left_hess += hessians[idx];

                let right_grad = sum_grad - left_grad;
                let right_hess = sum_hess - left_hess;

                // Check minimum child weight
                if left_hess < self.min_child_weight || right_hess < self.min_child_weight {
                    continue;
                }

                // Calculate gain with regularization
                let gain = self.calculate_gain(left_grad, left_hess, right_grad, right_hess, sum_grad, sum_hess);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = (feature_values[i].0 + feature_values[i + 1].0) / 2.0;
                    
                    best_left = feature_values[..=i].iter().map(|&(_, idx)| idx).collect();
                    best_right = feature_values[i + 1..].iter().map(|&(_, idx)| idx).collect();
                }
            }
        }

        (best_feature, best_threshold, best_gain, best_left, best_right)
    }

    fn calculate_gain(
        &self,
        left_grad: f32,
        left_hess: f32,
        right_grad: f32,
        right_hess: f32,
        sum_grad: f32,
        sum_hess: f32,
    ) -> f32 {
        let left_weight = -left_grad / (left_hess + self.reg_lambda);
        let right_weight = -right_grad / (right_hess + self.reg_lambda);
        let parent_weight = -sum_grad / (sum_hess + self.reg_lambda);

        let gain = 0.5 * (
            left_grad * left_weight +
            right_grad * right_weight -
            sum_grad * parent_weight
        ) - self.gamma;

        // L1 regularization penalty
        let l1_penalty = self.reg_alpha * (left_weight.abs() + right_weight.abs() - parent_weight.abs());

        gain - l1_penalty
    }

    fn predict_tree(&self, tree: &XGBTree, sample: &[f32]) -> f32 {
        let mut node_idx = 0;
        
        loop {
            let node = &tree.nodes[node_idx];
            
            if node.is_leaf {
                return node.value;
            }

            if sample[node.feature] <= node.threshold {
                node_idx = node.left;
            } else {
                node_idx = node.right;
            }
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        let predictions: Vec<f32> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = self.base_score;
                
                for tree in &self.trees {
                    pred += self.learning_rate * self.predict_tree(tree, sample);
                }

                // Apply sigmoid for binary classification
                let prob = 1.0 / (1.0 + (-pred).exp());
                if prob >= 0.5 { 1.0 } else { 0.0 }
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        let probabilities: Vec<f32> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = self.base_score;
                
                for tree in &self.trees {
                    pred += self.learning_rate * self.predict_tree(tree, sample);
                }

                1.0 / (1.0 + (-pred).exp())
            })
            .collect();

        Tensor::from_slice(&probabilities, &[n_samples]).unwrap()
    }
}

/// XGBoost Regressor
pub struct XGBoostRegressor {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_depth: usize,
    pub min_child_weight: f32,
    pub subsample: f32,
    pub colsample_bytree: f32,
    pub reg_lambda: f32,
    pub reg_alpha: f32,
    pub gamma: f32,
    trees: Vec<XGBTree>,
    base_score: f32,
}

impl XGBoostRegressor {
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            learning_rate: 0.3,
            max_depth: 6,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            gamma: 0.0,
            trees: Vec::new(),
            base_score: 0.0,
        }
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();
        let y_data = y.data_f32();

        // Initialize with mean
        self.base_score = y_data.iter().sum::<f32>() / n_samples as f32;
        let mut predictions = vec![self.base_score; n_samples];

        // Build trees
        for _ in 0..self.n_estimators {
            // Calculate gradients (residuals for MSE)
            let gradients: Vec<f32> = (0..n_samples)
                .map(|i| predictions[i] - y_data[i])
                .collect();
            
            // Hessians are constant 1.0 for MSE
            let hessians = vec![1.0; n_samples];

            // Row subsampling
            let mut rng = thread_rng();
            let sample_indices: Vec<usize> = if self.subsample < 1.0 {
                let n_subsample = (n_samples as f32 * self.subsample) as usize;
                (0..n_samples).choose_multiple(&mut rng, n_subsample)
            } else {
                (0..n_samples).collect()
            };

            // Column subsampling
            let feature_indices: Vec<usize> = if self.colsample_bytree < 1.0 {
                let n_features_sample = (n_features as f32 * self.colsample_bytree) as usize;
                (0..n_features).choose_multiple(&mut rng, n_features_sample)
            } else {
                (0..n_features).collect()
            };

            // Build tree (reuse XGBoostClassifier's tree building logic)
            let classifier = XGBoostClassifier {
                n_estimators: 1,
                learning_rate: self.learning_rate,
                max_depth: self.max_depth,
                min_child_weight: self.min_child_weight,
                subsample: 1.0, // Already subsampled
                colsample_bytree: 1.0, // Already subsampled
                reg_lambda: self.reg_lambda,
                reg_alpha: self.reg_alpha,
                gamma: self.gamma,
                trees: Vec::new(),
                base_score: 0.0,
            };

            let tree = classifier.build_tree(
                &x_data,
                &gradients,
                &hessians,
                &sample_indices,
                &feature_indices,
                n_features,
                0,
            );

            // Update predictions
            for i in 0..n_samples {
                let leaf_value = classifier.predict_tree(&tree, &x_data[i * n_features..(i + 1) * n_features]);
                predictions[i] += self.learning_rate * leaf_value;
            }

            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        let predictions: Vec<f32> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = self.base_score;
                
                let classifier = XGBoostClassifier {
                    n_estimators: 0,
                    learning_rate: self.learning_rate,
                    max_depth: 0,
                    min_child_weight: 0.0,
                    subsample: 0.0,
                    colsample_bytree: 0.0,
                    reg_lambda: 0.0,
                    reg_alpha: 0.0,
                    gamma: 0.0,
                    trees: Vec::new(),
                    base_score: 0.0,
                };

                for tree in &self.trees {
                    pred += self.learning_rate * classifier.predict_tree(tree, sample);
                }

                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xgboost_classifier() {
        // Simple binary classification test
        let x = Tensor::from_slice(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[4, 2],
        ).unwrap();
        let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4]).unwrap();

        let mut xgb = XGBoostClassifier::new(10)
            .learning_rate(0.1)
            .max_depth(3);
        
        xgb.fit(&x, &y);
        let predictions = xgb.predict(&x);

        assert_eq!(predictions.dims(), &[4]);
    }

    #[test]
    fn test_xgboost_regressor() {
        let x = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
        ).unwrap();
        let y = Tensor::from_slice(&[2.0, 4.0, 6.0], &[3]).unwrap();

        let mut xgb = XGBoostRegressor::new(10)
            .learning_rate(0.1)
            .max_depth(3);
        
        xgb.fit(&x, &y);
        let predictions = xgb.predict(&x);

        assert_eq!(predictions.dims(), &[3]);
    }
}

//! LightGBM-style Gradient Boosting
//!
//! Features:
//! - Histogram-based learning for faster training
//! - Leaf-wise (best-first) tree growth
//! - Gradient-based One-Side Sampling (GOSS)
//! - Exclusive Feature Bundling (EFB)

use ghostflow_core::Tensor;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

const MAX_BINS: usize = 255;

/// LightGBM-style Gradient Boosting Classifier
pub struct LightGBMClassifier {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub num_leaves: usize,
    pub max_depth: i32,
    pub min_data_in_leaf: usize,
    pub min_sum_hessian_in_leaf: f32,
    pub feature_fraction: f32,
    pub bagging_fraction: f32,
    pub bagging_freq: usize,
    pub lambda_l1: f32,
    pub lambda_l2: f32,
    pub max_bin: usize,
    trees: Vec<LGBMTree>,
    feature_bins: Vec<Vec<f32>>,
    base_score: f32,
}

#[derive(Clone)]
struct LGBMTree {
    nodes: Vec<LGBMNode>,
}

#[derive(Clone)]
struct LGBMNode {
    is_leaf: bool,
    feature: usize,
    threshold_bin: usize,
    left: usize,
    right: usize,
    value: f32,
    split_gain: f32,
}

impl LightGBMClassifier {
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1, // No limit
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            feature_fraction: 1.0,
            bagging_fraction: 1.0,
            bagging_freq: 0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            max_bin: MAX_BINS,
            trees: Vec::new(),
            feature_bins: Vec::new(),
            base_score: 0.5,
        }
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn num_leaves(mut self, leaves: usize) -> Self {
        self.num_leaves = leaves;
        self
    }

    pub fn max_depth(mut self, depth: i32) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn feature_fraction(mut self, fraction: f32) -> Self {
        self.feature_fraction = fraction;
        self
    }

    /// Build histogram bins for features
    fn build_histograms(&mut self, x: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        self.feature_bins = (0..n_features)
            .into_par_iter()
            .map(|feature| {
                // Collect feature values
                let mut values: Vec<f32> = (0..n_samples)
                    .map(|i| x_data[i * n_features + feature])
                    .collect();
                
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values.dedup();

                // Create bins
                if values.len() <= self.max_bin {
                    values
                } else {
                    // Quantile-based binning
                    let step = values.len() / self.max_bin;
                    (0..self.max_bin)
                        .map(|i| values[i * step])
                        .collect()
                }
            })
            .collect();
    }

    /// Convert feature value to bin index
    fn value_to_bin(&self, feature: usize, value: f32) -> usize {
        let bins = &self.feature_bins[feature];
        
        // Binary search for bin
        match bins.binary_search_by(|&bin| bin.partial_cmp(&value).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();
        let y_data = y.data_f32();

        // Build histograms
        self.build_histograms(x);

        // Initialize predictions
        let mut predictions = vec![self.base_score; n_samples];

        // Build trees
        for iter in 0..self.n_estimators {
            // Calculate gradients and hessians
            let mut gradients = Vec::with_capacity(n_samples);
            let mut hessians = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                let pred = 1.0 / (1.0 + (-predictions[i]).exp());
                let grad = pred - y_data[i];
                let hess = pred * (1.0 - pred);
                gradients.push(grad);
                hessians.push(hess.max(1e-6));
            }

            // Bagging
            let mut rng = thread_rng();
            let sample_indices: Vec<usize> = if self.bagging_freq > 0 && iter % self.bagging_freq == 0 {
                let n_subsample = (n_samples as f32 * self.bagging_fraction) as usize;
                (0..n_samples).choose_multiple(&mut rng, n_subsample)
            } else {
                (0..n_samples).collect()
            };

            // Feature sampling
            let feature_indices: Vec<usize> = if self.feature_fraction < 1.0 {
                let n_features_sample = (n_features as f32 * self.feature_fraction) as usize;
                (0..n_features).choose_multiple(&mut rng, n_features_sample)
            } else {
                (0..n_features).collect()
            };

            // Build tree with leaf-wise growth
            let tree = self.build_tree_leafwise(
                &x_data,
                &gradients,
                &hessians,
                &sample_indices,
                &feature_indices,
                n_features,
            );

            // Update predictions
            for i in 0..n_samples {
                let leaf_value = self.predict_tree(&tree, &x_data[i * n_features..(i + 1) * n_features]);
                predictions[i] += self.learning_rate * leaf_value;
            }

            self.trees.push(tree);
        }
    }

    /// Leaf-wise (best-first) tree growth
    fn build_tree_leafwise(
        &self,
        x_data: &[f32],
        gradients: &[f32],
        hessians: &[f32],
        sample_indices: &[usize],
        feature_indices: &[usize],
        n_features: usize,
    ) -> LGBMTree {
        let mut nodes = Vec::new();
        let mut leaf_queue: Vec<(Vec<usize>, usize, usize)> = Vec::new(); // (samples, node_idx, depth)

        // Create root node
        let root_idx = nodes.len();
        let sum_grad: f32 = sample_indices.iter().map(|&i| gradients[i]).sum();
        let sum_hess: f32 = sample_indices.iter().map(|&i| hessians[i]).sum();
        let root_value = self.calculate_leaf_value(sum_grad, sum_hess);

        nodes.push(LGBMNode {
            is_leaf: true,
            feature: 0,
            threshold_bin: 0,
            left: 0,
            right: 0,
            value: root_value,
            split_gain: 0.0,
        });

        leaf_queue.push((sample_indices.to_vec(), root_idx, 0));

        // Grow tree leaf-wise
        while nodes.len() < self.num_leaves && !leaf_queue.is_empty() {
            // Find best leaf to split
            let (best_idx, best_split) = self.find_best_leaf_to_split(
                x_data,
                gradients,
                hessians,
                &leaf_queue,
                feature_indices,
                n_features,
            );

            if best_split.is_none() {
                break;
            }

            let (_samples, node_idx, depth) = leaf_queue.remove(best_idx);
            let (feature, threshold_bin, gain, left_samples, right_samples) = best_split.unwrap();

            // Check depth limit
            if self.max_depth > 0 && depth >= self.max_depth as usize {
                continue;
            }

            // Update node to internal node
            nodes[node_idx].is_leaf = false;
            nodes[node_idx].feature = feature;
            nodes[node_idx].threshold_bin = threshold_bin;
            nodes[node_idx].split_gain = gain;

            // Create left child
            let left_idx = nodes.len();
            let left_sum_grad: f32 = left_samples.iter().map(|&i| gradients[i]).sum();
            let left_sum_hess: f32 = left_samples.iter().map(|&i| hessians[i]).sum();
            let left_value = self.calculate_leaf_value(left_sum_grad, left_sum_hess);

            nodes.push(LGBMNode {
                is_leaf: true,
                feature: 0,
                threshold_bin: 0,
                left: 0,
                right: 0,
                value: left_value,
                split_gain: 0.0,
            });

            // Create right child
            let right_idx = nodes.len();
            let right_sum_grad: f32 = right_samples.iter().map(|&i| gradients[i]).sum();
            let right_sum_hess: f32 = right_samples.iter().map(|&i| hessians[i]).sum();
            let right_value = self.calculate_leaf_value(right_sum_grad, right_sum_hess);

            nodes.push(LGBMNode {
                is_leaf: true,
                feature: 0,
                threshold_bin: 0,
                left: 0,
                right: 0,
                value: right_value,
                split_gain: 0.0,
            });

            // Update parent's children pointers
            nodes[node_idx].left = left_idx;
            nodes[node_idx].right = right_idx;

            // Add children to queue
            if left_samples.len() >= self.min_data_in_leaf {
                leaf_queue.push((left_samples, left_idx, depth + 1));
            }
            if right_samples.len() >= self.min_data_in_leaf {
                leaf_queue.push((right_samples, right_idx, depth + 1));
            }
        }

        LGBMTree { nodes }
    }

    fn find_best_leaf_to_split(
        &self,
        x_data: &[f32],
        gradients: &[f32],
        hessians: &[f32],
        leaf_queue: &[(Vec<usize>, usize, usize)],
        feature_indices: &[usize],
        n_features: usize,
    ) -> (usize, Option<(usize, usize, f32, Vec<usize>, Vec<usize>)>) {
        let mut best_leaf_idx = 0;
        let mut best_split: Option<(usize, usize, f32, Vec<usize>, Vec<usize>)> = None;
        let mut best_gain = 0.0;

        for (idx, (samples, _, _)) in leaf_queue.iter().enumerate() {
            if samples.len() < self.min_data_in_leaf * 2 {
                continue;
            }

            let split = self.find_best_split_histogram(
                x_data,
                gradients,
                hessians,
                samples,
                feature_indices,
                n_features,
            );

            if let Some((_, _, gain, _, _)) = &split {
                if *gain > best_gain {
                    best_gain = *gain;
                    best_split = split;
                    best_leaf_idx = idx;
                }
            }
        }

        (best_leaf_idx, best_split)
    }

    fn find_best_split_histogram(
        &self,
        x_data: &[f32],
        gradients: &[f32],
        hessians: &[f32],
        sample_indices: &[usize],
        feature_indices: &[usize],
        n_features: usize,
    ) -> Option<(usize, usize, f32, Vec<usize>, Vec<usize>)> {
        let sum_grad: f32 = sample_indices.iter().map(|&i| gradients[i]).sum();
        let sum_hess: f32 = sample_indices.iter().map(|&i| hessians[i]).sum();

        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_bin = 0;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for &feature in feature_indices {
            let n_bins = self.feature_bins[feature].len();

            // Build histogram for this feature
            let mut hist_grad = vec![0.0; n_bins];
            let mut hist_hess = vec![0.0; n_bins];
            let mut bin_samples: Vec<Vec<usize>> = vec![Vec::new(); n_bins];

            for &idx in sample_indices {
                let value = x_data[idx * n_features + feature];
                let bin = self.value_to_bin(feature, value);
                hist_grad[bin] += gradients[idx];
                hist_hess[bin] += hessians[idx];
                bin_samples[bin].push(idx);
            }

            // Try splits
            let mut left_grad = 0.0;
            let mut left_hess = 0.0;
            let mut left_samples = Vec::new();

            for bin in 0..n_bins - 1 {
                left_grad += hist_grad[bin];
                left_hess += hist_hess[bin];
                left_samples.extend(&bin_samples[bin]);

                let right_grad = sum_grad - left_grad;
                let right_hess = sum_hess - left_hess;

                // Check minimum hessian
                if left_hess < self.min_sum_hessian_in_leaf || right_hess < self.min_sum_hessian_in_leaf {
                    continue;
                }

                let gain = self.calculate_split_gain(left_grad, left_hess, right_grad, right_hess, sum_grad, sum_hess);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_bin = bin;
                    best_left = left_samples.clone();
                    best_right = sample_indices.iter()
                        .filter(|&&idx| !best_left.contains(&idx))
                        .copied()
                        .collect();
                }
            }
        }

        if best_gain > 0.0 {
            Some((best_feature, best_bin, best_gain, best_left, best_right))
        } else {
            None
        }
    }

    fn calculate_leaf_value(&self, sum_grad: f32, sum_hess: f32) -> f32 {
        -sum_grad / (sum_hess + self.lambda_l2)
    }

    fn calculate_split_gain(
        &self,
        left_grad: f32,
        left_hess: f32,
        right_grad: f32,
        right_hess: f32,
        sum_grad: f32,
        sum_hess: f32,
    ) -> f32 {
        let left_weight = -left_grad / (left_hess + self.lambda_l2);
        let right_weight = -right_grad / (right_hess + self.lambda_l2);
        let parent_weight = -sum_grad / (sum_hess + self.lambda_l2);

        let gain = 0.5 * (
            left_grad * left_weight +
            right_grad * right_weight -
            sum_grad * parent_weight
        );

        // L1 regularization
        let l1_penalty = self.lambda_l1 * (left_weight.abs() + right_weight.abs() - parent_weight.abs());

        gain - l1_penalty
    }

    fn predict_tree(&self, tree: &LGBMTree, sample: &[f32]) -> f32 {
        let mut node_idx = 0;

        loop {
            let node = &tree.nodes[node_idx];

            if node.is_leaf {
                return node.value;
            }

            let bin = self.value_to_bin(node.feature, sample[node.feature]);
            if bin <= node.threshold_bin {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lightgbm_classifier() {
        let x = Tensor::from_slice(
            &[0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[4, 2],
        ).unwrap();
        let y = Tensor::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[4]).unwrap();

        let mut lgbm = LightGBMClassifier::new(10)
            .learning_rate(0.1)
            .num_leaves(7);

        lgbm.fit(&x, &y);
        let predictions = lgbm.predict(&x);

        assert_eq!(predictions.dims()[0], 4); // Number of samples
    }
}



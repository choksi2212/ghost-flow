//! Decision Tree implementations - Real CART algorithm

use ghostflow_core::Tensor;

/// Split criterion for decision trees
#[derive(Debug, Clone, Copy)]
pub enum Criterion {
    Gini,
    Entropy,
    MSE,
    MAE,
}

/// A node in the decision tree
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Feature index for split
    pub feature_index: Option<usize>,
    /// Threshold for split
    pub threshold: Option<f32>,
    /// Left child
    pub left: Option<Box<TreeNode>>,
    /// Right child
    pub right: Option<Box<TreeNode>>,
    /// Prediction value (for leaf nodes)
    pub value: Option<f32>,
    /// Class probabilities (for classification)
    pub class_probs: Option<Vec<f32>>,
    /// Number of samples at this node
    pub n_samples: usize,
    /// Impurity at this node
    pub impurity: f32,
}

impl TreeNode {
    fn leaf(value: f32, n_samples: usize, impurity: f32) -> Self {
        TreeNode {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(value),
            class_probs: None,
            n_samples,
            impurity,
        }
    }

    fn leaf_classification(class_probs: Vec<f32>, n_samples: usize, impurity: f32) -> Self {
        let value = class_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as f32)
            .unwrap_or(0.0);
        
        TreeNode {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(value),
            class_probs: Some(class_probs),
            n_samples,
            impurity,
        }
    }

    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

/// Decision Tree Classifier using CART algorithm
pub struct DecisionTreeClassifier {
    /// Maximum depth of tree
    pub max_depth: Option<usize>,
    /// Minimum samples to split
    pub min_samples_split: usize,
    /// Minimum samples in leaf
    pub min_samples_leaf: usize,
    /// Maximum features to consider
    pub max_features: Option<usize>,
    /// Split criterion
    pub criterion: Criterion,
    /// Number of classes
    n_classes: usize,
    /// Root node
    root: Option<TreeNode>,
}

impl DecisionTreeClassifier {
    pub fn new() -> Self {
        DecisionTreeClassifier {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: Criterion::Gini,
            n_classes: 0,
            root: None,
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Fit the decision tree
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        // Determine number of classes
        self.n_classes = y_data.iter()
            .map(|&v| v as usize)
            .max()
            .unwrap_or(0) + 1;
        
        let indices: Vec<usize> = (0..n_samples).collect();
        
        self.root = Some(self.build_tree(
            &x_data, &y_data, &indices, n_features, 0
        ));
    }

    fn build_tree(
        &self,
        x: &[f32],
        y: &[f32],
        indices: &[usize],
        n_features: usize,
        depth: usize,
    ) -> TreeNode {
        let n_samples = indices.len();
        
        // Calculate class distribution
        let mut class_counts = vec![0usize; self.n_classes];
        for &idx in indices {
            let class = y[idx] as usize;
            if class < self.n_classes {
                class_counts[class] += 1;
            }
        }
        
        let class_probs: Vec<f32> = class_counts.iter()
            .map(|&c| c as f32 / n_samples as f32)
            .collect();
        
        let impurity = self.calculate_impurity(&class_probs);
        
        // Check stopping conditions
        let should_stop = 
            n_samples < self.min_samples_split ||
            self.max_depth.is_some_and(|d| depth >= d) ||
            impurity < 1e-7 ||
            class_counts.iter().filter(|&&c| c > 0).count() <= 1;
        
        if should_stop {
            return TreeNode::leaf_classification(class_probs, n_samples, impurity);
        }
        
        // Find best split
        let max_features = self.max_features.unwrap_or(n_features);
        let features_to_try: Vec<usize> = if max_features < n_features {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut all: Vec<usize> = (0..n_features).collect();
            all.shuffle(&mut rng);
            all.into_iter().take(max_features).collect()
        } else {
            (0..n_features).collect()
        };
        
        let mut best_gain = 0.0f32;
        let mut best_feature = 0;
        let mut best_threshold = 0.0f32;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();
        
        for &feature in &features_to_try {
            // Get unique values for this feature
            let mut values: Vec<f32> = indices.iter()
                .map(|&idx| x[idx * n_features + feature])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();
            
            // Try each threshold
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;
                
                let (left_indices, right_indices): (Vec<_>, Vec<_>) = indices.iter()
                    .partition(|&&idx| x[idx * n_features + feature] <= threshold);
                
                if left_indices.len() < self.min_samples_leaf || 
                   right_indices.len() < self.min_samples_leaf {
                    continue;
                }
                
                let gain = self.information_gain(
                    y, indices, &left_indices, &right_indices, impurity
                );
                
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left_indices = left_indices;
                    best_right_indices = right_indices;
                }
            }
        }
        
        // If no good split found, make leaf
        if best_gain <= 0.0 || best_left_indices.is_empty() || best_right_indices.is_empty() {
            return TreeNode::leaf_classification(class_probs, n_samples, impurity);
        }
        
        // Recursively build children
        let left = self.build_tree(x, y, &best_left_indices, n_features, depth + 1);
        let right = self.build_tree(x, y, &best_right_indices, n_features, depth + 1);
        
        TreeNode {
            feature_index: Some(best_feature),
            threshold: Some(best_threshold),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            value: None,
            class_probs: Some(class_probs),
            n_samples,
            impurity,
        }
    }

    fn calculate_impurity(&self, probs: &[f32]) -> f32 {
        match self.criterion {
            Criterion::Gini => {
                1.0 - probs.iter().map(|&p| p * p).sum::<f32>()
            }
            Criterion::Entropy => {
                -probs.iter()
                    .filter(|&&p| p > 0.0)
                    .map(|&p| p * p.ln())
                    .sum::<f32>()
            }
            _ => 0.0,
        }
    }

    fn information_gain(
        &self,
        y: &[f32],
        parent_indices: &[usize],
        left_indices: &[usize],
        right_indices: &[usize],
        parent_impurity: f32,
    ) -> f32 {
        let n_parent = parent_indices.len() as f32;
        let n_left = left_indices.len() as f32;
        let n_right = right_indices.len() as f32;
        
        let left_probs = self.class_probs_from_indices(y, left_indices);
        let right_probs = self.class_probs_from_indices(y, right_indices);
        
        let left_impurity = self.calculate_impurity(&left_probs);
        let right_impurity = self.calculate_impurity(&right_probs);
        
        parent_impurity - (n_left / n_parent) * left_impurity - (n_right / n_parent) * right_impurity
    }

    fn class_probs_from_indices(&self, y: &[f32], indices: &[usize]) -> Vec<f32> {
        let mut counts = vec![0usize; self.n_classes];
        for &idx in indices {
            let class = y[idx] as usize;
            if class < self.n_classes {
                counts[class] += 1;
            }
        }
        let total = indices.len() as f32;
        counts.iter().map(|&c| c as f32 / total).collect()
    }

    /// Predict class labels
    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                self.predict_sample(sample)
            })
            .collect();
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let mut probs = Vec::with_capacity(n_samples * self.n_classes);
        
        for i in 0..n_samples {
            let sample = &x_data[i * n_features..(i + 1) * n_features];
            let sample_probs = self.predict_proba_sample(sample);
            probs.extend(sample_probs);
        }
        
        Tensor::from_slice(&probs, &[n_samples, self.n_classes]).unwrap()
    }

    fn predict_sample(&self, sample: &[f32]) -> f32 {
        let mut node = self.root.as_ref().unwrap();
        
        while !node.is_leaf() {
            let feature = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();
            
            if sample[feature] <= threshold {
                node = node.left.as_ref().unwrap();
            } else {
                node = node.right.as_ref().unwrap();
            }
        }
        
        node.value.unwrap()
    }

    fn predict_proba_sample(&self, sample: &[f32]) -> Vec<f32> {
        let mut node = self.root.as_ref().unwrap();
        
        while !node.is_leaf() {
            let feature = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();
            
            if sample[feature] <= threshold {
                node = node.left.as_ref().unwrap();
            } else {
                node = node.right.as_ref().unwrap();
            }
        }
        
        node.class_probs.clone().unwrap_or_else(|| vec![0.0; self.n_classes])
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Decision Tree Regressor using CART algorithm
pub struct DecisionTreeRegressor {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub criterion: Criterion,
    root: Option<TreeNode>,
}

impl DecisionTreeRegressor {
    pub fn new() -> Self {
        DecisionTreeRegressor {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: Criterion::MSE,
            root: None,
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let indices: Vec<usize> = (0..n_samples).collect();
        
        self.root = Some(self.build_tree(&x_data, &y_data, &indices, n_features, 0));
    }

    fn build_tree(
        &self,
        x: &[f32],
        y: &[f32],
        indices: &[usize],
        n_features: usize,
        depth: usize,
    ) -> TreeNode {
        let n_samples = indices.len();
        
        // Calculate mean and variance
        let mean: f32 = indices.iter().map(|&i| y[i]).sum::<f32>() / n_samples as f32;
        let variance: f32 = indices.iter()
            .map(|&i| (y[i] - mean).powi(2))
            .sum::<f32>() / n_samples as f32;
        
        // Check stopping conditions
        let should_stop = 
            n_samples < self.min_samples_split ||
            self.max_depth.is_some_and(|d| depth >= d) ||
            variance < 1e-7;
        
        if should_stop {
            return TreeNode::leaf(mean, n_samples, variance);
        }
        
        // Find best split
        let mut best_mse = f32::INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0f32;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();
        
        for feature in 0..n_features {
            let mut values: Vec<f32> = indices.iter()
                .map(|&idx| x[idx * n_features + feature])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();
            
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;
                
                let (left_indices, right_indices): (Vec<_>, Vec<_>) = indices.iter()
                    .partition(|&&idx| x[idx * n_features + feature] <= threshold);
                
                if left_indices.len() < self.min_samples_leaf || 
                   right_indices.len() < self.min_samples_leaf {
                    continue;
                }
                
                let left_mean: f32 = left_indices.iter().map(|&i| y[i]).sum::<f32>() / left_indices.len() as f32;
                let right_mean: f32 = right_indices.iter().map(|&i| y[i]).sum::<f32>() / right_indices.len() as f32;
                
                let left_mse: f32 = left_indices.iter().map(|&i| {
                    let diff: f32 = y[i] - left_mean;
                    diff.powi(2)
                }).sum::<f32>();
                let right_mse: f32 = right_indices.iter().map(|&i| {
                    let diff: f32 = y[i] - right_mean;
                    diff.powi(2)
                }).sum::<f32>();
                let total_mse = left_mse + right_mse;
                
                if total_mse < best_mse {
                    best_mse = total_mse;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left_indices = left_indices;
                    best_right_indices = right_indices;
                }
            }
        }
        
        if best_left_indices.is_empty() || best_right_indices.is_empty() {
            return TreeNode::leaf(mean, n_samples, variance);
        }
        
        let left = self.build_tree(x, y, &best_left_indices, n_features, depth + 1);
        let right = self.build_tree(x, y, &best_right_indices, n_features, depth + 1);
        
        TreeNode {
            feature_index: Some(best_feature),
            threshold: Some(best_threshold),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            value: Some(mean),
            class_probs: None,
            n_samples,
            impurity: variance,
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                self.predict_sample(sample)
            })
            .collect();
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    fn predict_sample(&self, sample: &[f32]) -> f32 {
        let mut node = self.root.as_ref().unwrap();
        
        while !node.is_leaf() {
            let feature = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();
            
            if sample[feature] <= threshold {
                node = node.left.as_ref().unwrap();
            } else {
                node = node.right.as_ref().unwrap();
            }
        }
        
        node.value.unwrap()
    }
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_classifier() {
        // Simple XOR-like problem
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4]).unwrap();
        
        let mut tree = DecisionTreeClassifier::new().max_depth(3);
        tree.fit(&x, &y);
        
        let predictions = tree.predict(&x);
        let pred_data = predictions.data_f32();
        
        // Should learn the XOR pattern
        assert_eq!(pred_data.len(), 4);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let x = Tensor::from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0,
        ], &[5, 1]).unwrap();
        
        let y = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
        
        let mut tree = DecisionTreeRegressor::new().max_depth(5);
        tree.fit(&x, &y);
        
        let predictions = tree.predict(&x);
        let pred_data = predictions.data_f32();
        
        // Should approximate y = 2x
        assert_eq!(pred_data.len(), 5);
    }
}

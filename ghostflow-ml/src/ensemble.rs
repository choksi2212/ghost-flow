//! Ensemble methods - Random Forest and Gradient Boosting

use ghostflow_core::Tensor;
use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor, Criterion};
use rayon::prelude::*;
use rand::prelude::*;

/// Random Forest Classifier
pub struct RandomForestClassifier {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub bootstrap: bool,
    pub n_jobs: Option<usize>,
    trees: Vec<DecisionTreeClassifier>,
    n_classes: usize,
}

impl RandomForestClassifier {
    pub fn new(n_estimators: usize) -> Self {
        RandomForestClassifier {
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None, // sqrt(n_features) by default
            bootstrap: true,
            n_jobs: None,
            trees: Vec::new(),
            n_classes: 0,
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

    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        // Determine number of classes
        let y_data = y.data_f32();
        self.n_classes = y_data.iter()
            .map(|&v| v as usize)
            .max()
            .unwrap_or(0) + 1;
        
        // Default max_features = sqrt(n_features)
        let max_features = self.max_features
            .unwrap_or_else(|| (n_features as f32).sqrt() as usize);
        
        // Build trees in parallel
        self.trees = (0..self.n_estimators)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                
                // Bootstrap sampling
                let indices: Vec<usize> = if self.bootstrap {
                    (0..n_samples).map(|_| rng.gen_range(0..n_samples)).collect()
                } else {
                    (0..n_samples).collect()
                };
                
                // Create bootstrap sample
                let x_data = x.data_f32();
                let y_data = y.data_f32();
                
                let x_boot: Vec<f32> = indices.iter()
                    .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                    .collect();
                let y_boot: Vec<f32> = indices.iter()
                    .map(|&i| y_data[i])
                    .collect();
                
                let x_tensor = Tensor::from_slice(&x_boot, &[indices.len(), n_features]).unwrap();
                let y_tensor = Tensor::from_slice(&y_boot, &[indices.len()]).unwrap();
                
                // Build tree
                let mut tree = DecisionTreeClassifier::new()
                    .criterion(Criterion::Gini);
                
                if let Some(depth) = self.max_depth {
                    tree = tree.max_depth(depth);
                }
                tree = tree.min_samples_split(self.min_samples_split)
                    .min_samples_leaf(self.min_samples_leaf);
                
                tree.max_features = Some(max_features);
                tree.fit(&x_tensor, &y_tensor);
                tree
            })
            .collect();
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];
        
        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let start = i * self.n_classes;
                let sample_probs = &proba_data[start..start + self.n_classes];
                sample_probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as f32)
                    .unwrap_or(0.0)
            })
            .collect();
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        
        // Collect predictions from all trees
        let all_probs: Vec<Vec<f32>> = self.trees.par_iter()
            .map(|tree| tree.predict_proba(x).data_f32())
            .collect();
        
        // Average probabilities
        let mut avg_probs = vec![0.0f32; n_samples * self.n_classes];
        
        for probs in &all_probs {
            for (i, &p) in probs.iter().enumerate() {
                avg_probs[i] += p;
            }
        }
        
        let n_trees = self.trees.len() as f32;
        for p in &mut avg_probs {
            *p /= n_trees;
        }
        
        Tensor::from_slice(&avg_probs, &[n_samples, self.n_classes]).unwrap()
    }

    /// Feature importance based on impurity decrease
    pub fn feature_importances(&self, n_features: usize) -> Vec<f32> {
        let mut importances = vec![0.0f32; n_features];
        
        // This would require tracking impurity decrease during tree building
        // For now, return uniform importances
        let uniform = 1.0 / n_features as f32;
        importances.iter_mut().for_each(|x| *x = uniform);
        
        importances
    }
}

/// Random Forest Regressor
pub struct RandomForestRegressor {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub bootstrap: bool,
    trees: Vec<DecisionTreeRegressor>,
}

impl RandomForestRegressor {
    pub fn new(n_estimators: usize) -> Self {
        RandomForestRegressor {
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            bootstrap: true,
            trees: Vec::new(),
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let max_features = self.max_features
            .unwrap_or(n_features / 3);
        
        self.trees = (0..self.n_estimators)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                
                let indices: Vec<usize> = if self.bootstrap {
                    (0..n_samples).map(|_| rng.gen_range(0..n_samples)).collect()
                } else {
                    (0..n_samples).collect()
                };
                
                let x_data = x.data_f32();
                let y_data = y.data_f32();
                
                let x_boot: Vec<f32> = indices.iter()
                    .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                    .collect();
                let y_boot: Vec<f32> = indices.iter()
                    .map(|&i| y_data[i])
                    .collect();
                
                let x_tensor = Tensor::from_slice(&x_boot, &[indices.len(), n_features]).unwrap();
                let y_tensor = Tensor::from_slice(&y_boot, &[indices.len()]).unwrap();
                
                let mut tree = DecisionTreeRegressor::new();
                if let Some(depth) = self.max_depth {
                    tree = tree.max_depth(depth);
                }
                tree.max_features = Some(max_features);
                tree.fit(&x_tensor, &y_tensor);
                tree
            })
            .collect();
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        
        let all_preds: Vec<Vec<f32>> = self.trees.par_iter()
            .map(|tree| tree.predict(x).data_f32())
            .collect();
        
        let mut avg_preds = vec![0.0f32; n_samples];
        
        for preds in &all_preds {
            for (i, &p) in preds.iter().enumerate() {
                avg_preds[i] += p;
            }
        }
        
        let n_trees = self.trees.len() as f32;
        for p in &mut avg_preds {
            *p /= n_trees;
        }
        
        Tensor::from_slice(&avg_preds, &[n_samples]).unwrap()
    }
}

/// Gradient Boosting Classifier
pub struct GradientBoostingClassifier {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub subsample: f32,
    trees: Vec<Vec<DecisionTreeRegressor>>, // One set of trees per class
    initial_predictions: Vec<f32>,
    n_classes: usize,
}

impl GradientBoostingClassifier {
    pub fn new(n_estimators: usize) -> Self {
        GradientBoostingClassifier {
            n_estimators,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            subsample: 1.0,
            trees: Vec::new(),
            initial_predictions: Vec::new(),
            n_classes: 0,
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
        
        self.n_classes = y_data.iter()
            .map(|&v| v as usize)
            .max()
            .unwrap_or(0) + 1;
        
        // Initialize with log-odds
        self.initial_predictions = vec![0.0; self.n_classes];
        for c in 0..self.n_classes {
            let count = y_data.iter().filter(|&&v| v as usize == c).count();
            let prob = count as f32 / n_samples as f32;
            self.initial_predictions[c] = (prob / (1.0 - prob + 1e-10)).ln();
        }
        
        // Initialize predictions
        let mut f = vec![vec![0.0f32; n_samples]; self.n_classes];
        #[allow(clippy::needless_range_loop)]
        for c in 0..self.n_classes {
            for i in 0..n_samples {
                f[c][i] = self.initial_predictions[c];
            }
        }
        
        self.trees = (0..self.n_classes).map(|_| Vec::new()).collect();
        
        // Boosting iterations
        for _ in 0..self.n_estimators {
            // Compute probabilities using softmax
            let probs = self.softmax_predictions(&f);
            
            // For each class, fit a tree to the negative gradient
            for c in 0..self.n_classes {
                // Compute residuals (negative gradient)
                let residuals: Vec<f32> = (0..n_samples)
                    .map(|i| {
                        let y_true = if y_data[i] as usize == c { 1.0 } else { 0.0 };
                        y_true - probs[c][i]
                    })
                    .collect();
                
                let _residual_tensor = Tensor::from_slice(&residuals, &[n_samples]).unwrap();
                
                // Subsample
                let sample_indices: Vec<usize> = if self.subsample < 1.0 {
                    let mut rng = thread_rng();
                    let n_subsample = (n_samples as f32 * self.subsample) as usize;
                    (0..n_samples).choose_multiple(&mut rng, n_subsample)
                } else {
                    (0..n_samples).collect()
                };
                
                let x_sub: Vec<f32> = sample_indices.iter()
                    .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                    .collect();
                let r_sub: Vec<f32> = sample_indices.iter()
                    .map(|&i| residuals[i])
                    .collect();
                
                let x_tensor = Tensor::from_slice(&x_sub, &[sample_indices.len(), n_features]).unwrap();
                let r_tensor = Tensor::from_slice(&r_sub, &[sample_indices.len()]).unwrap();
                
                // Fit tree
                let mut tree = DecisionTreeRegressor::new()
                    .max_depth(self.max_depth);
                tree.min_samples_split = self.min_samples_split;
                tree.fit(&x_tensor, &r_tensor);
                
                // Update predictions
                let tree_preds = tree.predict(x);
                let tree_pred_data = tree_preds.data_f32();
                
                #[allow(clippy::needless_range_loop)]
                for i in 0..n_samples {
                    f[c][i] += self.learning_rate * tree_pred_data[i];
                }
                
                self.trees[c].push(tree);
            }
        }
    }

    fn softmax_predictions(&self, f: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_samples = f[0].len();
        let mut probs = vec![vec![0.0f32; n_samples]; self.n_classes];
        
        for i in 0..n_samples {
            let max_f = (0..self.n_classes).map(|c| f[c][i]).fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = (0..self.n_classes).map(|c| (f[c][i] - max_f).exp()).sum();
            
            for c in 0..self.n_classes {
                probs[c][i] = (f[c][i] - max_f).exp() / sum_exp;
            }
        }
        
        probs
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let proba = self.predict_proba(x);
        let proba_data = proba.data_f32();
        let n_samples = x.dims()[0];
        
        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let start = i * self.n_classes;
                let sample_probs = &proba_data[start..start + self.n_classes];
                sample_probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as f32)
                    .unwrap_or(0.0)
            })
            .collect();
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        
        // Initialize with initial predictions
        let mut f = vec![vec![0.0f32; n_samples]; self.n_classes];
        #[allow(clippy::needless_range_loop)]
        for c in 0..self.n_classes {
            for i in 0..n_samples {
                f[c][i] = self.initial_predictions[c];
            }
        }
        
        // Add tree predictions
        for c in 0..self.n_classes {
            for tree in &self.trees[c] {
                let preds = tree.predict(x);
                let pred_data = preds.data_f32();
                for i in 0..n_samples {
                    f[c][i] += self.learning_rate * pred_data[i];
                }
            }
        }
        
        // Softmax
        let probs = self.softmax_predictions(&f);
        
        let mut result = Vec::with_capacity(n_samples * self.n_classes);
        for i in 0..n_samples {
            for c in 0..self.n_classes {
                result.push(probs[c][i]);
            }
        }
        
        Tensor::from_slice(&result, &[n_samples, self.n_classes]).unwrap()
    }
}

/// Gradient Boosting Regressor
pub struct GradientBoostingRegressor {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub subsample: f32,
    pub loss: GBLoss,
    trees: Vec<DecisionTreeRegressor>,
    initial_prediction: f32,
}

#[derive(Clone, Copy)]
pub enum GBLoss {
    SquaredError,
    AbsoluteError,
    Huber,
}

impl GradientBoostingRegressor {
    pub fn new(n_estimators: usize) -> Self {
        GradientBoostingRegressor {
            n_estimators,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            subsample: 1.0,
            loss: GBLoss::SquaredError,
            trees: Vec::new(),
            initial_prediction: 0.0,
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

    pub fn loss(mut self, loss: GBLoss) -> Self {
        self.loss = loss;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        
        // Initialize with mean
        self.initial_prediction = y_data.iter().sum::<f32>() / n_samples as f32;
        
        let mut predictions = vec![self.initial_prediction; n_samples];
        
        for _ in 0..self.n_estimators {
            // Compute residuals
            let residuals: Vec<f32> = match self.loss {
                GBLoss::SquaredError => {
                    (0..n_samples).map(|i| y_data[i] - predictions[i]).collect()
                }
                GBLoss::AbsoluteError => {
                    (0..n_samples).map(|i| {
                        let diff = y_data[i] - predictions[i];
                        if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 }
                    }).collect()
                }
                GBLoss::Huber => {
                    let delta = 1.0;
                    (0..n_samples).map(|i| {
                        let diff = y_data[i] - predictions[i];
                        if diff.abs() <= delta {
                            diff
                        } else {
                            delta * diff.signum()
                        }
                    }).collect()
                }
            };
            
            // Subsample
            let sample_indices: Vec<usize> = if self.subsample < 1.0 {
                let mut rng = thread_rng();
                let n_subsample = (n_samples as f32 * self.subsample) as usize;
                (0..n_samples).choose_multiple(&mut rng, n_subsample)
            } else {
                (0..n_samples).collect()
            };
            
            let x_sub: Vec<f32> = sample_indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let r_sub: Vec<f32> = sample_indices.iter()
                .map(|&i| residuals[i])
                .collect();
            
            let x_tensor = Tensor::from_slice(&x_sub, &[sample_indices.len(), n_features]).unwrap();
            let r_tensor = Tensor::from_slice(&r_sub, &[sample_indices.len()]).unwrap();
            
            // Fit tree
            let mut tree = DecisionTreeRegressor::new()
                .max_depth(self.max_depth);
            tree.min_samples_split = self.min_samples_split;
            tree.fit(&x_tensor, &r_tensor);
            
            // Update predictions
            let tree_preds = tree.predict(x);
            let tree_pred_data = tree_preds.data_f32();
            
            for i in 0..n_samples {
                predictions[i] += self.learning_rate * tree_pred_data[i];
            }
            
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let mut predictions = vec![self.initial_prediction; n_samples];
        
        for tree in &self.trees {
            let tree_preds = tree.predict(x);
            let tree_pred_data = tree_preds.data_f32();
            
            for i in 0..n_samples {
                predictions[i] += self.learning_rate * tree_pred_data[i];
            }
        }
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_classifier() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
            0.5, 0.5,
        ], &[5, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0, 0.0], &[5]).unwrap();
        
        let mut rf = RandomForestClassifier::new(10).max_depth(3);
        rf.fit(&x, &y);
        
        let predictions = rf.predict(&x);
        assert_eq!(predictions.dims(), &[5]);
    }

    #[test]
    fn test_gradient_boosting_regressor() {
        let x = Tensor::from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0,
        ], &[5, 1]).unwrap();
        
        let y = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
        
        let mut gb = GradientBoostingRegressor::new(50)
            .learning_rate(0.1)
            .max_depth(3);
        gb.fit(&x, &y);
        
        let predictions = gb.predict(&x);
        assert_eq!(predictions.dims(), &[5]);
    }
}

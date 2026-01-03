//! Stacking Ensemble - Stacking Classifier and Regressor

use ghostflow_core::Tensor;

/// Base estimator trait for stacking
pub trait BaseEstimator: Send + Sync {
    fn fit(&mut self, x: &Tensor, y: &Tensor);
    fn predict(&self, x: &Tensor) -> Tensor;
    fn clone_box(&self) -> Box<dyn BaseEstimator>;
}

/// Simple linear model for meta-learner
#[derive(Clone)]
pub struct LinearMeta {
    coef_: Option<Vec<f32>>,
    intercept_: f32,
}

impl LinearMeta {
    pub fn new() -> Self {
        LinearMeta {
            coef_: None,
            intercept_: 0.0,
        }
    }

    pub fn fit(&mut self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize) {
        // Ridge regression
        let alpha = 1.0f32;

        let x_mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();
        let y_mean = y.iter().sum::<f32>() / n_samples as f32;

        let mut xtx = vec![0.0f32; n_features * n_features];
        let mut xty = vec![0.0f32; n_features];

        for i in 0..n_features {
            for k in 0..n_samples {
                let xki = x[k * n_features + i] - x_mean[i];
                xty[i] += xki * (y[k] - y_mean);
            }
            for j in 0..n_features {
                for k in 0..n_samples {
                    xtx[i * n_features + j] += 
                        (x[k * n_features + i] - x_mean[i]) * (x[k * n_features + j] - x_mean[j]);
                }
            }
            xtx[i * n_features + i] += alpha;
        }

        let coef = solve_linear(&xtx, &xty, n_features);
        let intercept = y_mean - coef.iter().zip(x_mean.iter()).map(|(&c, &m)| c * m).sum::<f32>();

        self.coef_ = Some(coef);
        self.intercept_ = intercept;
    }

    pub fn predict(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let coef = self.coef_.as_ref().expect("Not fitted");
        (0..n_samples)
            .map(|i| {
                let mut pred = self.intercept_;
                for j in 0..n_features {
                    pred += coef[j] * x[i * n_features + j];
                }
                pred
            })
            .collect()
    }
}

impl Default for LinearMeta {
    fn default() -> Self { Self::new() }
}

/// Logistic meta-learner for classification
#[derive(Clone)]
pub struct LogisticMeta {
    coef_: Option<Vec<f32>>,
    intercept_: f32,
    max_iter: usize,
}

impl LogisticMeta {
    pub fn new() -> Self {
        LogisticMeta {
            coef_: None,
            intercept_: 0.0,
            max_iter: 100,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn fit(&mut self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize) {
        let mut coef = vec![0.0f32; n_features];
        let mut intercept = 0.0f32;
        let lr = 0.1f32;
        let alpha = 0.01f32;

        for _ in 0..self.max_iter {
            let mut grad_coef = vec![0.0f32; n_features];
            let mut grad_intercept = 0.0f32;

            for i in 0..n_samples {
                let mut z = intercept;
                for j in 0..n_features {
                    z += coef[j] * x[i * n_features + j];
                }
                let pred = Self::sigmoid(z);
                let error = pred - y[i];

                grad_intercept += error;
                for j in 0..n_features {
                    grad_coef[j] += error * x[i * n_features + j];
                }
            }

            intercept -= lr * grad_intercept / n_samples as f32;
            for j in 0..n_features {
                coef[j] -= lr * (grad_coef[j] / n_samples as f32 + alpha * coef[j]);
            }
        }

        self.coef_ = Some(coef);
        self.intercept_ = intercept;
    }

    pub fn predict_proba(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let coef = self.coef_.as_ref().expect("Not fitted");
        (0..n_samples)
            .map(|i| {
                let mut z = self.intercept_;
                for j in 0..n_features {
                    z += coef[j] * x[i * n_features + j];
                }
                Self::sigmoid(z)
            })
            .collect()
    }

    pub fn predict(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        self.predict_proba(x, n_samples, n_features)
            .into_iter()
            .map(|p| if p >= 0.5 { 1.0 } else { 0.0 })
            .collect()
    }
}

impl Default for LogisticMeta {
    fn default() -> Self { Self::new() }
}

/// Stacking Regressor
pub struct StackingRegressor {
    pub cv: usize,
    pub passthrough: bool,
    estimators_: Vec<Box<dyn Fn() -> Box<dyn StackingEstimator>>>,
    fitted_estimators_: Vec<Vec<Box<dyn StackingEstimator>>>,
    final_estimator_: LinearMeta,
    n_features_: usize,
}

pub trait StackingEstimator: Send {
    fn fit(&mut self, x: &Tensor, y: &Tensor);
    fn predict(&self, x: &Tensor) -> Tensor;
}

impl StackingRegressor {
    pub fn new() -> Self {
        StackingRegressor {
            cv: 5,
            passthrough: false,
            estimators_: Vec::new(),
            fitted_estimators_: Vec::new(),
            final_estimator_: LinearMeta::new(),
            n_features_: 0,
        }
    }

    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    pub fn passthrough(mut self, p: bool) -> Self {
        self.passthrough = p;
        self
    }

    pub fn add_estimator<F>(&mut self, factory: F) 
    where 
        F: Fn() -> Box<dyn StackingEstimator> + 'static 
    {
        self.estimators_.push(Box::new(factory));
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let n_estimators = self.estimators_.len();

        self.n_features_ = n_features;

        if n_estimators == 0 {
            panic!("No estimators added");
        }

        // Generate cross-validation predictions
        let fold_size = n_samples / self.cv;
        let mut meta_features = vec![0.0f32; n_samples * n_estimators];

        self.fitted_estimators_ = (0..n_estimators).map(|_| Vec::new()).collect();

        for fold in 0..self.cv {
            let val_start = fold * fold_size;
            let val_end = if fold == self.cv - 1 { n_samples } else { (fold + 1) * fold_size };

            // Split data
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| i < val_start || i >= val_end)
                .collect();
            let val_indices: Vec<usize> = (val_start..val_end).collect();

            let x_train: Vec<f32> = train_indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let y_train: Vec<f32> = train_indices.iter().map(|&i| y_data[i]).collect();

            let x_val: Vec<f32> = val_indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();

            let x_train_tensor = Tensor::from_slice(&x_train, &[train_indices.len(), n_features]).unwrap();
            let y_train_tensor = Tensor::from_slice(&y_train, &[train_indices.len()]).unwrap();
            let x_val_tensor = Tensor::from_slice(&x_val, &[val_indices.len(), n_features]).unwrap();

            // Train each estimator and get predictions
            for (est_idx, factory) in self.estimators_.iter().enumerate() {
                let mut estimator = factory();
                estimator.fit(&x_train_tensor, &y_train_tensor);
                
                let predictions = estimator.predict(&x_val_tensor);
                let pred_data = predictions.data_f32();

                for (i, &val_idx) in val_indices.iter().enumerate() {
                    meta_features[val_idx * n_estimators + est_idx] = pred_data[i];
                }

                if fold == self.cv - 1 {
                    self.fitted_estimators_[est_idx].push(estimator);
                }
            }
        }

        // Fit final estimator on all data
        for (est_idx, factory) in self.estimators_.iter().enumerate() {
            let mut estimator = factory();
            estimator.fit(x, y);
            self.fitted_estimators_[est_idx] = vec![estimator];
        }

        // Prepare meta features
        let meta_n_features = if self.passthrough {
            n_estimators + n_features
        } else {
            n_estimators
        };

        let final_meta: Vec<f32> = if self.passthrough {
            (0..n_samples)
                .flat_map(|i| {
                    let mut row = Vec::with_capacity(meta_n_features);
                    for j in 0..n_estimators {
                        row.push(meta_features[i * n_estimators + j]);
                    }
                    for j in 0..n_features {
                        row.push(x_data[i * n_features + j]);
                    }
                    row
                })
                .collect()
        } else {
            meta_features
        };

        self.final_estimator_.fit(&final_meta, &y_data, n_samples, meta_n_features);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let n_estimators = self.fitted_estimators_.len();

        // Get predictions from each estimator
        let mut meta_features = vec![0.0f32; n_samples * n_estimators];

        for (est_idx, estimators) in self.fitted_estimators_.iter().enumerate() {
            let predictions = estimators[0].predict(x);
            let pred_data = predictions.data_f32();
            for i in 0..n_samples {
                meta_features[i * n_estimators + est_idx] = pred_data[i];
            }
        }

        // Prepare meta features
        let meta_n_features = if self.passthrough {
            n_estimators + n_features
        } else {
            n_estimators
        };

        let final_meta: Vec<f32> = if self.passthrough {
            (0..n_samples)
                .flat_map(|i| {
                    let mut row = Vec::with_capacity(meta_n_features);
                    for j in 0..n_estimators {
                        row.push(meta_features[i * n_estimators + j]);
                    }
                    for j in 0..n_features {
                        row.push(x_data[i * n_features + j]);
                    }
                    row
                })
                .collect()
        } else {
            meta_features
        };

        let predictions = self.final_estimator_.predict(&final_meta, n_samples, meta_n_features);
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for StackingRegressor {
    fn default() -> Self { Self::new() }
}

/// Stacking Classifier
pub struct StackingClassifier {
    pub cv: usize,
    pub passthrough: bool,
    pub stack_method: StackMethod,
    estimators_: Vec<Box<dyn Fn() -> Box<dyn StackingClassifierEstimator>>>,
    fitted_estimators_: Vec<Vec<Box<dyn StackingClassifierEstimator>>>,
    final_estimator_: LogisticMeta,
    classes_: Vec<i32>,
    n_features_: usize,
}

#[derive(Clone, Copy)]
pub enum StackMethod {
    Predict,
    PredictProba,
}

pub trait StackingClassifierEstimator: Send {
    fn fit(&mut self, x: &Tensor, y: &Tensor);
    fn predict(&self, x: &Tensor) -> Tensor;
    fn predict_proba(&self, x: &Tensor) -> Option<Tensor>;
}

impl StackingClassifier {
    pub fn new() -> Self {
        StackingClassifier {
            cv: 5,
            passthrough: false,
            stack_method: StackMethod::PredictProba,
            estimators_: Vec::new(),
            fitted_estimators_: Vec::new(),
            final_estimator_: LogisticMeta::new(),
            classes_: Vec::new(),
            n_features_: 0,
        }
    }

    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    pub fn stack_method(mut self, m: StackMethod) -> Self {
        self.stack_method = m;
        self
    }

    pub fn add_estimator<F>(&mut self, factory: F) 
    where 
        F: Fn() -> Box<dyn StackingClassifierEstimator> + 'static 
    {
        self.estimators_.push(Box::new(factory));
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let n_estimators = self.estimators_.len();

        self.n_features_ = n_features;

        // Find classes
        let mut classes: Vec<i32> = y_data.iter().map(|&v| v as i32).collect();
        classes.sort();
        classes.dedup();
        self.classes_ = classes;

        if n_estimators == 0 {
            panic!("No estimators added");
        }

        // For binary classification, we just need one probability column per estimator
        let meta_cols_per_est = 1;
        let fold_size = n_samples / self.cv;
        let mut meta_features = vec![0.0f32; n_samples * n_estimators * meta_cols_per_est];

        self.fitted_estimators_ = (0..n_estimators).map(|_| Vec::new()).collect();

        for fold in 0..self.cv {
            let val_start = fold * fold_size;
            let val_end = if fold == self.cv - 1 { n_samples } else { (fold + 1) * fold_size };

            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| i < val_start || i >= val_end)
                .collect();
            let val_indices: Vec<usize> = (val_start..val_end).collect();

            let x_train: Vec<f32> = train_indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let y_train: Vec<f32> = train_indices.iter().map(|&i| y_data[i]).collect();

            let x_val: Vec<f32> = val_indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();

            let x_train_tensor = Tensor::from_slice(&x_train, &[train_indices.len(), n_features]).unwrap();
            let y_train_tensor = Tensor::from_slice(&y_train, &[train_indices.len()]).unwrap();
            let x_val_tensor = Tensor::from_slice(&x_val, &[val_indices.len(), n_features]).unwrap();

            for (est_idx, factory) in self.estimators_.iter().enumerate() {
                let mut estimator = factory();
                estimator.fit(&x_train_tensor, &y_train_tensor);
                
                let pred_data = match self.stack_method {
                    StackMethod::PredictProba => {
                        if let Some(proba) = estimator.predict_proba(&x_val_tensor) {
                            proba.data_f32().clone()
                        } else {
                            estimator.predict(&x_val_tensor).data_f32().clone()
                        }
                    }
                    StackMethod::Predict => {
                        estimator.predict(&x_val_tensor).data_f32().clone()
                    }
                };

                for (i, &val_idx) in val_indices.iter().enumerate() {
                    meta_features[val_idx * n_estimators + est_idx] = pred_data[i];
                }
            }
        }

        // Fit final estimators on all data
        for (est_idx, factory) in self.estimators_.iter().enumerate() {
            let mut estimator = factory();
            estimator.fit(x, y);
            self.fitted_estimators_[est_idx] = vec![estimator];
        }

        // Fit final meta-learner
        let meta_n_features = if self.passthrough {
            n_estimators + n_features
        } else {
            n_estimators
        };

        let final_meta: Vec<f32> = if self.passthrough {
            (0..n_samples)
                .flat_map(|i| {
                    let mut row = Vec::with_capacity(meta_n_features);
                    for j in 0..n_estimators {
                        row.push(meta_features[i * n_estimators + j]);
                    }
                    for j in 0..n_features {
                        row.push(x_data[i * n_features + j]);
                    }
                    row
                })
                .collect()
        } else {
            meta_features
        };

        self.final_estimator_.fit(&final_meta, &y_data, n_samples, meta_n_features);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let n_estimators = self.fitted_estimators_.len();

        let mut meta_features = vec![0.0f32; n_samples * n_estimators];

        for (est_idx, estimators) in self.fitted_estimators_.iter().enumerate() {
            let pred_data = match self.stack_method {
                StackMethod::PredictProba => {
                    if let Some(proba) = estimators[0].predict_proba(x) {
                        proba.data_f32().clone()
                    } else {
                        estimators[0].predict(x).data_f32().clone()
                    }
                }
                StackMethod::Predict => {
                    estimators[0].predict(x).data_f32().clone()
                }
            };

            for i in 0..n_samples {
                meta_features[i * n_estimators + est_idx] = pred_data[i];
            }
        }

        let meta_n_features = if self.passthrough {
            n_estimators + n_features
        } else {
            n_estimators
        };

        let final_meta: Vec<f32> = if self.passthrough {
            (0..n_samples)
                .flat_map(|i| {
                    let mut row = Vec::with_capacity(meta_n_features);
                    for j in 0..n_estimators {
                        row.push(meta_features[i * n_estimators + j]);
                    }
                    for j in 0..n_features {
                        row.push(x_data[i * n_features + j]);
                    }
                    row
                })
                .collect()
        } else {
            meta_features
        };

        let predictions = self.final_estimator_.predict(&final_meta, n_samples, meta_n_features);
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let n_estimators = self.fitted_estimators_.len();

        let mut meta_features = vec![0.0f32; n_samples * n_estimators];

        for (est_idx, estimators) in self.fitted_estimators_.iter().enumerate() {
            let pred_data = match self.stack_method {
                StackMethod::PredictProba => {
                    if let Some(proba) = estimators[0].predict_proba(x) {
                        proba.data_f32().clone()
                    } else {
                        estimators[0].predict(x).data_f32().clone()
                    }
                }
                StackMethod::Predict => {
                    estimators[0].predict(x).data_f32().clone()
                }
            };

            for i in 0..n_samples {
                meta_features[i * n_estimators + est_idx] = pred_data[i];
            }
        }

        let meta_n_features = if self.passthrough {
            n_estimators + n_features
        } else {
            n_estimators
        };

        let final_meta: Vec<f32> = if self.passthrough {
            (0..n_samples)
                .flat_map(|i| {
                    let mut row = Vec::with_capacity(meta_n_features);
                    for j in 0..n_estimators {
                        row.push(meta_features[i * n_estimators + j]);
                    }
                    for j in 0..n_features {
                        row.push(x_data[i * n_features + j]);
                    }
                    row
                })
                .collect()
        } else {
            meta_features
        };

        let proba = self.final_estimator_.predict_proba(&final_meta, n_samples, meta_n_features);
        
        // Return as [n_samples, 2] for binary
        let result: Vec<f32> = proba.iter()
            .flat_map(|&p| vec![1.0 - p, p])
            .collect();
        
        Tensor::from_slice(&result, &[n_samples, 2]).unwrap()
    }
}

impl Default for StackingClassifier {
    fn default() -> Self { Self::new() }
}

fn solve_linear(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
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
            aug.swap(i * (n + 1) + j, max_row * (n + 1) + j);
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
    fn test_linear_meta() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![1.0, 2.0, 3.0];
        
        let mut meta = LinearMeta::new();
        meta.fit(&x, &y, 3, 2);
        let pred = meta.predict(&x, 3, 2);
        
        assert_eq!(pred.len(), 3);
    }

    #[test]
    fn test_logistic_meta() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        
        let mut meta = LogisticMeta::new();
        meta.fit(&x, &y, 4, 2);
        let pred = meta.predict(&x, 4, 2);
        
        assert_eq!(pred.len(), 4);
    }
}

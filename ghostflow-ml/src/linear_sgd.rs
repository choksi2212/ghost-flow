//! SGD-based Linear Models - SGDClassifier, SGDRegressor, RidgeClassifier

use ghostflow_core::Tensor;
use rand::prelude::*;

/// SGD Classifier - Linear classifier with Stochastic Gradient Descent
pub struct SGDClassifier {
    pub loss: SGDLoss,
    pub penalty: Penalty,
    pub alpha: f32,
    pub l1_ratio: f32,
    pub max_iter: usize,
    pub tol: f32,
    pub learning_rate: LearningRate,
    pub eta0: f32,
    pub power_t: f32,
    pub shuffle: bool,
    pub warm_start: bool,
    pub class_weight: Option<Vec<f32>>,
    coef_: Option<Vec<Vec<f32>>>,
    intercept_: Option<Vec<f32>>,
    classes_: Vec<i32>,
    n_iter_: usize,
}

#[derive(Clone, Copy)]
pub enum SGDLoss {
    Hinge,           // Linear SVM
    Log,             // Logistic Regression
    ModifiedHuber,   // Smoothed hinge
    SquaredHinge,    // Squared hinge
    Perceptron,      // Perceptron
}

#[derive(Clone, Copy)]
pub enum Penalty {
    L2,
    L1,
    ElasticNet,
    None,
}

#[derive(Clone, Copy)]
pub enum LearningRate {
    Constant,
    Optimal,
    InvScaling,
    Adaptive,
}

impl SGDClassifier {
    pub fn new() -> Self {
        SGDClassifier {
            loss: SGDLoss::Hinge,
            penalty: Penalty::L2,
            alpha: 0.0001,
            l1_ratio: 0.15,
            max_iter: 1000,
            tol: 1e-3,
            learning_rate: LearningRate::Optimal,
            eta0: 0.0,
            power_t: 0.5,
            shuffle: true,
            warm_start: false,
            class_weight: None,
            coef_: None,
            intercept_: None,
            classes_: Vec::new(),
            n_iter_: 0,
        }
    }

    pub fn loss(mut self, l: SGDLoss) -> Self { self.loss = l; self }
    pub fn penalty(mut self, p: Penalty) -> Self { self.penalty = p; self }
    pub fn alpha(mut self, a: f32) -> Self { self.alpha = a; self }
    pub fn max_iter(mut self, n: usize) -> Self { self.max_iter = n; self }

    fn compute_loss_gradient(&self, margin: f32, y: f32) -> f32 {
        match self.loss {
            SGDLoss::Hinge => {
                if y * margin < 1.0 { -y } else { 0.0 }
            }
            SGDLoss::Log => {
                let p = 1.0 / (1.0 + (-margin).exp());
                p - (y + 1.0) / 2.0  // Convert y from {-1,1} to {0,1}
            }
            SGDLoss::ModifiedHuber => {
                let z = y * margin;
                if z >= 1.0 { 0.0 }
                else if z >= -1.0 { -2.0 * y * (1.0 - z) }
                else { -4.0 * y }
            }
            SGDLoss::SquaredHinge => {
                let z = y * margin;
                if z < 1.0 { -2.0 * y * (1.0 - z) } else { 0.0 }
            }
            SGDLoss::Perceptron => {
                if y * margin <= 0.0 { -y } else { 0.0 }
            }
        }
    }

    fn get_learning_rate(&self, t: usize) -> f32 {
        match self.learning_rate {
            LearningRate::Constant => self.eta0,
            LearningRate::Optimal => 1.0 / (self.alpha * (t as f32 + 1.0)),
            LearningRate::InvScaling => self.eta0 / (t as f32 + 1.0).powf(self.power_t),
            LearningRate::Adaptive => self.eta0,
        }
    }

    fn apply_penalty(&self, coef: &mut [f32], lr: f32) {
        match self.penalty {
            Penalty::L2 => {
                for c in coef.iter_mut() {
                    *c *= 1.0 - lr * self.alpha;
                }
            }
            Penalty::L1 => {
                for c in coef.iter_mut() {
                    let sign = c.signum();
                    *c = (*c - lr * self.alpha * sign).max(0.0) * sign;
                }
            }
            Penalty::ElasticNet => {
                let l1_penalty = self.alpha * self.l1_ratio;
                let l2_penalty = self.alpha * (1.0 - self.l1_ratio);
                for c in coef.iter_mut() {
                    *c *= 1.0 - lr * l2_penalty;
                    let sign = c.signum();
                    *c = (*c - lr * l1_penalty * sign).max(0.0) * sign;
                }
            }
            Penalty::None => {}
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Find unique classes
        let mut classes: Vec<i32> = y_data.iter().map(|&v| v as i32).collect();
        classes.sort();
        classes.dedup();
        self.classes_ = classes.clone();

        let n_classes = self.classes_.len();
        let mut rng = thread_rng();

        // Initialize coefficients
        let mut coef = if self.warm_start && self.coef_.is_some() {
            self.coef_.clone().unwrap()
        } else if n_classes == 2 {
            vec![vec![0.0f32; n_features]]
        } else {
            vec![vec![0.0f32; n_features]; n_classes]
        };

        let mut intercept = if self.warm_start && self.intercept_.is_some() {
            self.intercept_.clone().unwrap()
        } else if n_classes == 2 {
            vec![0.0f32]
        } else {
            vec![0.0f32; n_classes]
        };

        // Training loop
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut t = 0usize;

        for epoch in 0..self.max_iter {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut total_loss = 0.0f32;

            for &i in &indices {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let yi = y_data[i] as i32;

                let lr = self.get_learning_rate(t);
                t += 1;

                if n_classes == 2 {
                    // Binary classification
                    let y_signed = if yi == self.classes_[1] { 1.0f32 } else { -1.0f32 };
                    
                    let mut margin = intercept[0];
                    for j in 0..n_features {
                        margin += coef[0][j] * xi[j];
                    }

                    let grad = self.compute_loss_gradient(margin, y_signed);
                    total_loss += grad.abs();

                    // Update weights
                    for j in 0..n_features {
                        coef[0][j] -= lr * grad * xi[j];
                    }
                    intercept[0] -= lr * grad;

                    self.apply_penalty(&mut coef[0], lr);
                } else {
                    // Multi-class (one-vs-all)
                    for k in 0..n_classes {
                        let y_signed = if yi == self.classes_[k] { 1.0f32 } else { -1.0f32 };
                        
                        let mut margin = intercept[k];
                        for j in 0..n_features {
                            margin += coef[k][j] * xi[j];
                        }

                        let grad = self.compute_loss_gradient(margin, y_signed);
                        total_loss += grad.abs();

                        for j in 0..n_features {
                            coef[k][j] -= lr * grad * xi[j];
                        }
                        intercept[k] -= lr * grad;

                        self.apply_penalty(&mut coef[k], lr);
                    }
                }
            }

            self.n_iter_ = epoch + 1;

            if total_loss / (n_samples as f32) < self.tol {
                break;
            }
        }

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.as_ref().unwrap();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];

                if self.classes_.len() == 2 {
                    let mut score = intercept[0];
                    for j in 0..n_features {
                        score += coef[0][j] * xi[j];
                    }
                    if score >= 0.0 { self.classes_[1] as f32 } else { self.classes_[0] as f32 }
                } else {
                    let mut best_score = f32::NEG_INFINITY;
                    let mut best_class = self.classes_[0];
                    for k in 0..self.classes_.len() {
                        let mut score = intercept[k];
                        for j in 0..n_features {
                            score += coef[k][j] * xi[j];
                        }
                        if score > best_score {
                            best_score = score;
                            best_class = self.classes_[k];
                        }
                    }
                    best_class as f32
                }
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn decision_function(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.as_ref().unwrap();

        if self.classes_.len() == 2 {
            let scores: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let xi = &x_data[i * n_features..(i + 1) * n_features];
                    let mut score = intercept[0];
                    for j in 0..n_features {
                        score += coef[0][j] * xi[j];
                    }
                    score
                })
                .collect();
            Tensor::from_slice(&scores, &[n_samples]).unwrap()
        } else {
            let n_classes = self.classes_.len();
            let scores: Vec<f32> = (0..n_samples)
                .flat_map(|i| {
                    let xi = &x_data[i * n_features..(i + 1) * n_features];
                    (0..n_classes).map(move |k| {
                        let mut score = intercept[k];
                        for j in 0..n_features {
                            score += coef[k][j] * xi[j];
                        }
                        score
                    })
                })
                .collect();
            Tensor::from_slice(&scores, &[n_samples, n_classes]).unwrap()
        }
    }
}

impl Default for SGDClassifier {
    fn default() -> Self { Self::new() }
}

/// SGD Regressor - Linear regressor with Stochastic Gradient Descent
pub struct SGDRegressor {
    pub loss: SGDRegressorLoss,
    pub penalty: Penalty,
    pub alpha: f32,
    pub l1_ratio: f32,
    pub max_iter: usize,
    pub tol: f32,
    pub learning_rate: LearningRate,
    pub eta0: f32,
    pub power_t: f32,
    pub epsilon: f32,
    pub shuffle: bool,
    coef_: Option<Vec<f32>>,
    intercept_: f32,
    n_iter_: usize,
}

#[derive(Clone, Copy)]
pub enum SGDRegressorLoss {
    SquaredError,
    Huber,
    EpsilonInsensitive,
    SquaredEpsilonInsensitive,
}

impl SGDRegressor {
    pub fn new() -> Self {
        SGDRegressor {
            loss: SGDRegressorLoss::SquaredError,
            penalty: Penalty::L2,
            alpha: 0.0001,
            l1_ratio: 0.15,
            max_iter: 1000,
            tol: 1e-3,
            learning_rate: LearningRate::InvScaling,
            eta0: 0.01,
            power_t: 0.25,
            epsilon: 0.1,
            shuffle: true,
            coef_: None,
            intercept_: 0.0,
            n_iter_: 0,
        }
    }

    pub fn loss(mut self, l: SGDRegressorLoss) -> Self { self.loss = l; self }
    pub fn penalty(mut self, p: Penalty) -> Self { self.penalty = p; self }
    pub fn alpha(mut self, a: f32) -> Self { self.alpha = a; self }

    fn compute_loss_gradient(&self, pred: f32, y: f32) -> f32 {
        let residual = pred - y;
        match self.loss {
            SGDRegressorLoss::SquaredError => residual,
            SGDRegressorLoss::Huber => {
                if residual.abs() <= self.epsilon {
                    residual
                } else {
                    self.epsilon * residual.signum()
                }
            }
            SGDRegressorLoss::EpsilonInsensitive => {
                if residual.abs() <= self.epsilon {
                    0.0
                } else {
                    residual.signum()
                }
            }
            SGDRegressorLoss::SquaredEpsilonInsensitive => {
                if residual.abs() <= self.epsilon {
                    0.0
                } else {
                    residual - self.epsilon * residual.signum()
                }
            }
        }
    }

    fn get_learning_rate(&self, t: usize) -> f32 {
        match self.learning_rate {
            LearningRate::Constant => self.eta0,
            LearningRate::Optimal => 1.0 / (self.alpha * (t as f32 + 1.0)),
            LearningRate::InvScaling => self.eta0 / (t as f32 + 1.0).powf(self.power_t),
            LearningRate::Adaptive => self.eta0,
        }
    }

    fn apply_penalty(&self, coef: &mut [f32], lr: f32) {
        match self.penalty {
            Penalty::L2 => {
                for c in coef.iter_mut() {
                    *c *= 1.0 - lr * self.alpha;
                }
            }
            Penalty::L1 => {
                for c in coef.iter_mut() {
                    let sign = c.signum();
                    *c = (*c - lr * self.alpha * sign).max(0.0) * sign;
                }
            }
            Penalty::ElasticNet => {
                let l1 = self.alpha * self.l1_ratio;
                let l2 = self.alpha * (1.0 - self.l1_ratio);
                for c in coef.iter_mut() {
                    *c *= 1.0 - lr * l2;
                    let sign = c.signum();
                    *c = (*c - lr * l1 * sign).max(0.0) * sign;
                }
            }
            Penalty::None => {}
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut rng = thread_rng();
        let mut coef = vec![0.0f32; n_features];
        let mut intercept = 0.0f32;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut t = 0usize;

        for epoch in 0..self.max_iter {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut total_loss = 0.0f32;

            for &i in &indices {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let yi = y_data[i];

                let lr = self.get_learning_rate(t);
                t += 1;

                // Compute prediction
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += coef[j] * xi[j];
                }

                let grad = self.compute_loss_gradient(pred, yi);
                total_loss += grad.abs();

                // Update weights
                for j in 0..n_features {
                    coef[j] -= lr * grad * xi[j];
                }
                intercept -= lr * grad;

                self.apply_penalty(&mut coef, lr);
            }

            self.n_iter_ = epoch + 1;

            if total_loss / (n_samples as f32) < self.tol {
                break;
            }
        }

        self.coef_ = Some(coef);
        self.intercept_ = intercept;
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];
                let mut pred = self.intercept_;
                for j in 0..n_features {
                    pred += coef[j] * xi[j];
                }
                pred
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for SGDRegressor {
    fn default() -> Self { Self::new() }
}

/// Ridge Classifier - Classification using Ridge regression
pub struct RidgeClassifier {
    pub alpha: f32,
    pub fit_intercept: bool,
    pub normalize: bool,
    pub class_weight: Option<Vec<f32>>,
    coef_: Option<Vec<Vec<f32>>>,
    intercept_: Option<Vec<f32>>,
    classes_: Vec<i32>,
}

impl RidgeClassifier {
    pub fn new() -> Self {
        RidgeClassifier {
            alpha: 1.0,
            fit_intercept: true,
            normalize: false,
            class_weight: None,
            coef_: None,
            intercept_: None,
            classes_: Vec::new(),
        }
    }

    pub fn alpha(mut self, a: f32) -> Self { self.alpha = a; self }

    fn solve_ridge(&self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize) -> (Vec<f32>, f32) {
        // Center data
        let x_mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();
        let y_mean = y.iter().sum::<f32>() / n_samples as f32;

        // Compute X^T X + alpha * I
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
            xtx[i * n_features + i] += self.alpha;
        }

        // Solve using Cholesky
        let coef = solve_cholesky(&xtx, &xty, n_features);
        let intercept = y_mean - coef.iter().zip(x_mean.iter()).map(|(&c, &m)| c * m).sum::<f32>();

        (coef, intercept)
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Find unique classes
        let mut classes: Vec<i32> = y_data.iter().map(|&v| v as i32).collect();
        classes.sort();
        classes.dedup();
        self.classes_ = classes.clone();

        if self.classes_.len() == 2 {
            // Binary: use {-1, 1} encoding
            let y_encoded: Vec<f32> = y_data.iter()
                .map(|&v| if v as i32 == self.classes_[1] { 1.0 } else { -1.0 })
                .collect();

            let (coef, intercept) = self.solve_ridge(&x_data, &y_encoded, n_samples, n_features);
            self.coef_ = Some(vec![coef]);
            self.intercept_ = Some(vec![intercept]);
        } else {
            // Multi-class: one-vs-rest
            let mut all_coef = Vec::new();
            let mut all_intercept = Vec::new();

            for &class in &self.classes_ {
                let y_encoded: Vec<f32> = y_data.iter()
                    .map(|&v| if v as i32 == class { 1.0 } else { -1.0 })
                    .collect();

                let (coef, intercept) = self.solve_ridge(&x_data, &y_encoded, n_samples, n_features);
                all_coef.push(coef);
                all_intercept.push(intercept);
            }

            self.coef_ = Some(all_coef);
            self.intercept_ = Some(all_intercept);
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.as_ref().unwrap();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let xi = &x_data[i * n_features..(i + 1) * n_features];

                if self.classes_.len() == 2 {
                    let mut score = intercept[0];
                    for j in 0..n_features {
                        score += coef[0][j] * xi[j];
                    }
                    if score >= 0.0 { self.classes_[1] as f32 } else { self.classes_[0] as f32 }
                } else {
                    let mut best_score = f32::NEG_INFINITY;
                    let mut best_class = self.classes_[0];
                    for k in 0..self.classes_.len() {
                        let mut score = intercept[k];
                        for j in 0..n_features {
                            score += coef[k][j] * xi[j];
                        }
                        if score > best_score {
                            best_score = score;
                            best_class = self.classes_[k];
                        }
                    }
                    best_class as f32
                }
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn decision_function(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.as_ref().unwrap();

        if self.classes_.len() == 2 {
            let scores: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let xi = &x_data[i * n_features..(i + 1) * n_features];
                    let mut score = intercept[0];
                    for j in 0..n_features {
                        score += coef[0][j] * xi[j];
                    }
                    score
                })
                .collect();
            Tensor::from_slice(&scores, &[n_samples]).unwrap()
        } else {
            let n_classes = self.classes_.len();
            let scores: Vec<f32> = (0..n_samples)
                .flat_map(|i| {
                    let xi = &x_data[i * n_features..(i + 1) * n_features];
                    (0..n_classes).map(move |k| {
                        let mut score = intercept[k];
                        for j in 0..n_features {
                            score += coef[k][j] * xi[j];
                        }
                        score
                    })
                })
                .collect();
            Tensor::from_slice(&scores, &[n_samples, n_classes]).unwrap()
        }
    }
}

impl Default for RidgeClassifier {
    fn default() -> Self { Self::new() }
}

fn solve_cholesky(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n * n];
    
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = sum.max(1e-10).sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j].max(1e-10);
            }
        }
    }

    // Forward substitution
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i].max(1e-10);
    }

    // Backward substitution
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i].max(1e-10);
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_classifier() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut clf = SGDClassifier::new().max_iter(100);
        clf.fit(&x, &y);
        let pred = clf.predict(&x);
        assert_eq!(pred.dims(), &[4]);
    }

    #[test]
    fn test_sgd_regressor() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        
        let mut reg = SGDRegressor::new().max_iter(100);
        reg.fit(&x, &y);
        let pred = reg.predict(&x);
        assert_eq!(pred.dims(), &[3]);
    }

    #[test]
    fn test_ridge_classifier() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut clf = RidgeClassifier::new();
        clf.fit(&x, &y);
        let pred = clf.predict(&x);
        assert_eq!(pred.dims(), &[4]);
    }
}

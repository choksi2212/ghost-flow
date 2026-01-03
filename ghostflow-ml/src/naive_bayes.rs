//! Naive Bayes classifiers

use ghostflow_core::Tensor;
use std::f32::consts::PI;

/// Gaussian Naive Bayes classifier
pub struct GaussianNB {
    pub var_smoothing: f32,
    class_prior_: Option<Vec<f32>>,
    class_count_: Option<Vec<usize>>,
    theta_: Option<Vec<Vec<f32>>>,  // Mean of each feature per class
    var_: Option<Vec<Vec<f32>>>,    // Variance of each feature per class
    n_classes_: usize,
    n_features_: usize,
}

impl GaussianNB {
    pub fn new() -> Self {
        GaussianNB {
            var_smoothing: 1e-9,
            class_prior_: None,
            class_count_: None,
            theta_: None,
            var_: None,
            n_classes_: 0,
            n_features_: 0,
        }
    }

    pub fn var_smoothing(mut self, smoothing: f32) -> Self {
        self.var_smoothing = smoothing;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        // Count samples per class
        let mut class_count = vec![0usize; self.n_classes_];
        for &label in &y_data {
            class_count[label as usize] += 1;
        }

        // Compute class priors
        let class_prior: Vec<f32> = class_count.iter()
            .map(|&c| c as f32 / n_samples as f32)
            .collect();

        // Compute mean and variance per class per feature
        let mut theta = vec![vec![0.0f32; n_features]; self.n_classes_];
        let mut var = vec![vec![0.0f32; n_features]; self.n_classes_];

        // Compute means
        for i in 0..n_samples {
            let class = y_data[i] as usize;
            for j in 0..n_features {
                theta[class][j] += x_data[i * n_features + j];
            }
        }
        for c in 0..self.n_classes_ {
            if class_count[c] > 0 {
                for j in 0..n_features {
                    theta[c][j] /= class_count[c] as f32;
                }
            }
        }

        // Compute variances
        for i in 0..n_samples {
            let class = y_data[i] as usize;
            for j in 0..n_features {
                let diff = x_data[i * n_features + j] - theta[class][j];
                var[class][j] += diff * diff;
            }
        }
        for c in 0..self.n_classes_ {
            if class_count[c] > 0 {
                for j in 0..n_features {
                    var[c][j] = var[c][j] / class_count[c] as f32 + self.var_smoothing;
                }
            }
        }

        self.class_prior_ = Some(class_prior);
        self.class_count_ = Some(class_count);
        self.theta_ = Some(theta);
        self.var_ = Some(var);
    }

    fn log_likelihood(&self, x: &[f32], class: usize) -> f32 {
        let theta = self.theta_.as_ref().unwrap();
        let var = self.var_.as_ref().unwrap();
        let prior = self.class_prior_.as_ref().unwrap();

        let mut log_prob = prior[class].ln();

        for j in 0..self.n_features_ {
            let mean = theta[class][j];
            let variance = var[class][j];
            let diff = x[j] - mean;
            
            // Log of Gaussian PDF
            log_prob += -0.5 * (2.0 * PI * variance).ln() - (diff * diff) / (2.0 * variance);
        }

        log_prob
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                
                let mut best_class = 0;
                let mut best_log_prob = f32::NEG_INFINITY;

                for c in 0..self.n_classes_ {
                    let log_prob = self.log_likelihood(sample, c);
                    if log_prob > best_log_prob {
                        best_log_prob = log_prob;
                        best_class = c;
                    }
                }

                best_class as f32
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut probs = Vec::with_capacity(n_samples * self.n_classes_);

        for i in 0..n_samples {
            let sample = &x_data[i * n_features..(i + 1) * n_features];
            
            let log_probs: Vec<f32> = (0..self.n_classes_)
                .map(|c| self.log_likelihood(sample, c))
                .collect();

            // Softmax normalization
            let max_log = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
            
            for &lp in &log_probs {
                probs.push((lp - max_log).exp() / exp_sum);
            }
        }

        Tensor::from_slice(&probs, &[n_samples, self.n_classes_]).unwrap()
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

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}


/// Multinomial Naive Bayes for discrete/count data
pub struct MultinomialNB {
    pub alpha: f32,  // Laplace smoothing
    class_log_prior_: Option<Vec<f32>>,
    feature_log_prob_: Option<Vec<Vec<f32>>>,
    class_count_: Option<Vec<f32>>,
    feature_count_: Option<Vec<Vec<f32>>>,
    n_classes_: usize,
    n_features_: usize,
}

impl MultinomialNB {
    pub fn new() -> Self {
        MultinomialNB {
            alpha: 1.0,
            class_log_prior_: None,
            feature_log_prob_: None,
            class_count_: None,
            feature_count_: None,
            n_classes_: 0,
            n_features_: 0,
        }
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        // Count features per class
        let mut class_count = vec![0.0f32; self.n_classes_];
        let mut feature_count = vec![vec![0.0f32; n_features]; self.n_classes_];

        for i in 0..n_samples {
            let class = y_data[i] as usize;
            class_count[class] += 1.0;
            for j in 0..n_features {
                feature_count[class][j] += x_data[i * n_features + j];
            }
        }

        // Compute log probabilities with Laplace smoothing
        let mut class_log_prior = vec![0.0f32; self.n_classes_];
        let mut feature_log_prob = vec![vec![0.0f32; n_features]; self.n_classes_];

        let total_samples = n_samples as f32;
        for c in 0..self.n_classes_ {
            class_log_prior[c] = (class_count[c] / total_samples).ln();

            let smoothed_total: f32 = feature_count[c].iter().sum::<f32>() + self.alpha * n_features as f32;
            for j in 0..n_features {
                feature_log_prob[c][j] = ((feature_count[c][j] + self.alpha) / smoothed_total).ln();
            }
        }

        self.class_log_prior_ = Some(class_log_prior);
        self.feature_log_prob_ = Some(feature_log_prob);
        self.class_count_ = Some(class_count);
        self.feature_count_ = Some(feature_count);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let class_log_prior = self.class_log_prior_.as_ref().unwrap();
        let feature_log_prob = self.feature_log_prob_.as_ref().unwrap();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                
                let mut best_class = 0;
                let mut best_log_prob = f32::NEG_INFINITY;

                for c in 0..self.n_classes_ {
                    let mut log_prob = class_log_prior[c];
                    for j in 0..n_features {
                        log_prob += sample[j] * feature_log_prob[c][j];
                    }
                    
                    if log_prob > best_log_prob {
                        best_log_prob = log_prob;
                        best_class = c;
                    }
                }

                best_class as f32
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

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Bernoulli Naive Bayes for binary features
pub struct BernoulliNB {
    pub alpha: f32,
    pub binarize: Option<f32>,
    class_log_prior_: Option<Vec<f32>>,
    feature_log_prob_: Option<Vec<Vec<f32>>>,
    n_classes_: usize,
    n_features_: usize,
}

impl BernoulliNB {
    pub fn new() -> Self {
        BernoulliNB {
            alpha: 1.0,
            binarize: Some(0.0),
            class_log_prior_: None,
            feature_log_prob_: None,
            n_classes_: 0,
            n_features_: 0,
        }
    }

    pub fn binarize(mut self, threshold: f32) -> Self {
        self.binarize = Some(threshold);
        self
    }

    fn binarize_data(&self, x: &[f32]) -> Vec<f32> {
        if let Some(threshold) = self.binarize {
            x.iter().map(|&v| if v > threshold { 1.0 } else { 0.0 }).collect()
        } else {
            x.to_vec()
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = self.binarize_data(&x.data_f32());
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        let mut class_count = vec![0.0f32; self.n_classes_];
        let mut feature_count = vec![vec![0.0f32; n_features]; self.n_classes_];

        for i in 0..n_samples {
            let class = y_data[i] as usize;
            class_count[class] += 1.0;
            for j in 0..n_features {
                feature_count[class][j] += x_data[i * n_features + j];
            }
        }

        let mut class_log_prior = vec![0.0f32; self.n_classes_];
        let mut feature_log_prob = vec![vec![0.0f32; n_features * 2]; self.n_classes_];

        let total_samples = n_samples as f32;
        for c in 0..self.n_classes_ {
            class_log_prior[c] = (class_count[c] / total_samples).ln();

            for j in 0..n_features {
                let p = (feature_count[c][j] + self.alpha) / (class_count[c] + 2.0 * self.alpha);
                feature_log_prob[c][j * 2] = p.ln();           // log P(x_j=1|c)
                feature_log_prob[c][j * 2 + 1] = (1.0 - p).ln(); // log P(x_j=0|c)
            }
        }

        self.class_log_prior_ = Some(class_log_prior);
        self.feature_log_prob_ = Some(feature_log_prob);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = self.binarize_data(&x.data_f32());
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let class_log_prior = self.class_log_prior_.as_ref().unwrap();
        let feature_log_prob = self.feature_log_prob_.as_ref().unwrap();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                
                let mut best_class = 0;
                let mut best_log_prob = f32::NEG_INFINITY;

                for c in 0..self.n_classes_ {
                    let mut log_prob = class_log_prior[c];
                    for j in 0..n_features {
                        if sample[j] > 0.5 {
                            log_prob += feature_log_prob[c][j * 2];
                        } else {
                            log_prob += feature_log_prob[c][j * 2 + 1];
                        }
                    }
                    
                    if log_prob > best_log_prob {
                        best_log_prob = log_prob;
                        best_class = c;
                    }
                }

                best_class as f32
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

impl Default for BernoulliNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Complement Naive Bayes - good for imbalanced datasets
pub struct ComplementNB {
    pub alpha: f32,
    pub norm: bool,
    class_log_prior_: Option<Vec<f32>>,
    feature_log_prob_: Option<Vec<Vec<f32>>>,
    n_classes_: usize,
    n_features_: usize,
}

impl ComplementNB {
    pub fn new() -> Self {
        ComplementNB {
            alpha: 1.0,
            norm: false,
            class_log_prior_: None,
            feature_log_prob_: None,
            n_classes_: 0,
            n_features_: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        // Count features NOT in each class (complement)
        let mut total_feature_count = vec![0.0f32; n_features];
        let mut class_feature_count = vec![vec![0.0f32; n_features]; self.n_classes_];
        let mut class_count = vec![0.0f32; self.n_classes_];

        for i in 0..n_samples {
            let class = y_data[i] as usize;
            class_count[class] += 1.0;
            for j in 0..n_features {
                let val = x_data[i * n_features + j];
                total_feature_count[j] += val;
                class_feature_count[class][j] += val;
            }
        }

        // Compute complement feature counts
        let mut feature_log_prob = vec![vec![0.0f32; n_features]; self.n_classes_];
        
        for c in 0..self.n_classes_ {
            let mut complement_sum = 0.0f32;
            for j in 0..n_features {
                let complement_count = total_feature_count[j] - class_feature_count[c][j] + self.alpha;
                complement_sum += complement_count;
            }

            for j in 0..n_features {
                let complement_count = total_feature_count[j] - class_feature_count[c][j] + self.alpha;
                feature_log_prob[c][j] = (complement_count / complement_sum).ln();
            }

            // Normalize if requested
            if self.norm {
                let norm: f32 = feature_log_prob[c].iter().map(|&x| x.abs()).sum();
                if norm > 0.0 {
                    for j in 0..n_features {
                        feature_log_prob[c][j] /= norm;
                    }
                }
            }
        }

        let class_log_prior: Vec<f32> = class_count.iter()
            .map(|&c| (c / n_samples as f32).ln())
            .collect();

        self.class_log_prior_ = Some(class_log_prior);
        self.feature_log_prob_ = Some(feature_log_prob);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let feature_log_prob = self.feature_log_prob_.as_ref().unwrap();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let sample = &x_data[i * n_features..(i + 1) * n_features];
                
                let mut best_class = 0;
                let mut best_score = f32::NEG_INFINITY;

                for c in 0..self.n_classes_ {
                    // Complement NB uses negative of complement log prob
                    let mut score = 0.0f32;
                    for j in 0..n_features {
                        score -= sample[j] * feature_log_prob[c][j];
                    }
                    
                    if score > best_score {
                        best_score = score;
                        best_class = c;
                    }
                }

                best_class as f32
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for ComplementNB {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb() {
        let x = Tensor::from_slice(&[
            1.0, 2.0,
            1.5, 1.8,
            5.0, 8.0,
            6.0, 9.0,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut gnb = GaussianNB::new();
        gnb.fit(&x, &y);
        
        let score = gnb.score(&x, &y);
        assert!(score >= 0.5);
    }

    #[test]
    fn test_multinomial_nb() {
        let x = Tensor::from_slice(&[
            2.0, 1.0, 0.0,
            1.0, 2.0, 0.0,
            0.0, 1.0, 2.0,
            0.0, 0.0, 3.0,
        ], &[4, 3]).unwrap();
        
        let y = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut mnb = MultinomialNB::new();
        mnb.fit(&x, &y);
        
        let predictions = mnb.predict(&x);
        assert_eq!(predictions.dims(), &[4]);
    }
}

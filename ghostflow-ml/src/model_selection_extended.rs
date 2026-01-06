//! Extended Model Selection - RandomizedSearchCV, GroupKFold, RepeatedKFold

use ghostflow_core::Tensor;
use rand::prelude::*;
use std::collections::HashMap;

/// Parameter distribution for randomized search
#[derive(Clone)]
pub enum ParamDistribution {
    /// Uniform distribution over continuous range
    Uniform { low: f32, high: f32 },
    /// Log-uniform distribution (for learning rates, regularization)
    LogUniform { low: f32, high: f32 },
    /// Discrete uniform over integers
    IntUniform { low: i32, high: i32 },
    /// Choice from a list of values
    Choice(Vec<f32>),
    /// Choice from a list of integers
    IntChoice(Vec<i32>),
}

impl ParamDistribution {
    pub fn sample(&self, rng: &mut impl Rng) -> f32 {
        match self {
            ParamDistribution::Uniform { low, high } => {
                rng.gen::<f32>() * (high - low) + low
            }
            ParamDistribution::LogUniform { low, high } => {
                let log_low = low.ln();
                let log_high = high.ln();
                (rng.gen::<f32>() * (log_high - log_low) + log_low).exp()
            }
            ParamDistribution::IntUniform { low, high } => {
                rng.gen_range(*low..=*high) as f32
            }
            ParamDistribution::Choice(values) => {
                values[rng.gen_range(0..values.len())]
            }
            ParamDistribution::IntChoice(values) => {
                values[rng.gen_range(0..values.len())] as f32
            }
        }
    }
}

/// Result of randomized search
#[derive(Clone)]
pub struct RandomizedSearchResult {
    pub best_params: HashMap<String, f32>,
    pub best_score: f32,
    pub cv_results: Vec<CVResult>,
}

#[derive(Clone)]
pub struct CVResult {
    pub params: HashMap<String, f32>,
    pub mean_score: f32,
    pub std_score: f32,
    pub scores: Vec<f32>,
}

/// Randomized Search Cross-Validation
pub struct RandomizedSearchCV {
    pub param_distributions: HashMap<String, ParamDistribution>,
    pub n_iter: usize,
    pub cv: usize,
    pub scoring: Scoring,
    pub random_state: Option<u64>,
    pub refit: bool,
    pub n_jobs: usize,
    best_params_: Option<HashMap<String, f32>>,
    best_score_: f32,
    cv_results_: Vec<CVResult>,
}

#[derive(Clone, Copy)]
pub enum Scoring {
    Accuracy,
    F1,
    Precision,
    Recall,
    R2,
    NegMSE,
    NegMAE,
}

impl RandomizedSearchCV {
    pub fn new(param_distributions: HashMap<String, ParamDistribution>, n_iter: usize) -> Self {
        RandomizedSearchCV {
            param_distributions,
            n_iter,
            cv: 5,
            scoring: Scoring::Accuracy,
            random_state: None,
            refit: true,
            n_jobs: 1,
            best_params_: None,
            best_score_: f32::NEG_INFINITY,
            cv_results_: Vec::new(),
        }
    }

    pub fn cv(mut self, cv: usize) -> Self { self.cv = cv; self }
    pub fn scoring(mut self, s: Scoring) -> Self { self.scoring = s; self }
    pub fn random_state(mut self, seed: u64) -> Self { self.random_state = Some(seed); self }

    /// Run randomized search with a generic model
    /// Returns the best parameters found
    pub fn search<F>(&mut self, x: &Tensor, y: &Tensor, mut fit_and_score: F) -> RandomizedSearchResult
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, &Tensor, &HashMap<String, f32>) -> f32,
    {
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();
        let y_data = y.data_f32();

        self.cv_results_.clear();
        self.best_score_ = f32::NEG_INFINITY;

        for _ in 0..self.n_iter {
            // Sample parameters
            let params: HashMap<String, f32> = self.param_distributions.iter()
                .map(|(name, dist)| (name.clone(), dist.sample(&mut rng)))
                .collect();

            // Cross-validation
            let fold_size = n_samples / self.cv;
            let mut scores = Vec::with_capacity(self.cv);

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
                let y_val: Vec<f32> = val_indices.iter().map(|&i| y_data[i]).collect();

                let x_train_t = Tensor::from_slice(&x_train, &[train_indices.len(), n_features]).unwrap();
                let y_train_t = Tensor::from_slice(&y_train, &[train_indices.len()]).unwrap();
                let x_val_t = Tensor::from_slice(&x_val, &[val_indices.len(), n_features]).unwrap();
                let y_val_t = Tensor::from_slice(&y_val, &[val_indices.len()]).unwrap();

                let score = fit_and_score(&x_train_t, &y_train_t, &x_val_t, &y_val_t, &params);
                scores.push(score);
            }

            let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
            let std_score = (scores.iter().map(|&s| (s - mean_score).powi(2)).sum::<f32>() 
                / scores.len() as f32).sqrt();

            let cv_result = CVResult {
                params: params.clone(),
                mean_score,
                std_score,
                scores,
            };
            self.cv_results_.push(cv_result);

            if mean_score > self.best_score_ {
                self.best_score_ = mean_score;
                self.best_params_ = Some(params);
            }
        }

        RandomizedSearchResult {
            best_params: self.best_params_.clone().unwrap_or_default(),
            best_score: self.best_score_,
            cv_results: self.cv_results_.clone(),
        }
    }

    pub fn best_params(&self) -> Option<&HashMap<String, f32>> {
        self.best_params_.as_ref()
    }

    pub fn best_score(&self) -> f32 {
        self.best_score_
    }

    pub fn cv_results(&self) -> &[CVResult] {
        &self.cv_results_
    }
}

/// Group K-Fold Cross-Validation
pub struct GroupKFold {
    pub n_splits: usize,
}

impl GroupKFold {
    pub fn new(n_splits: usize) -> Self {
        GroupKFold { n_splits }
    }

    /// Split data ensuring groups are not split across folds
    pub fn split(&self, n_samples: usize, groups: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
        // Find unique groups
        let mut unique_groups: Vec<usize> = groups.to_vec();
        unique_groups.sort();
        unique_groups.dedup();

        let n_groups = unique_groups.len();
        let groups_per_fold = (n_groups + self.n_splits - 1) / self.n_splits;

        let mut folds = Vec::with_capacity(self.n_splits);

        for fold in 0..self.n_splits {
            let fold_groups_start = fold * groups_per_fold;
            let fold_groups_end = ((fold + 1) * groups_per_fold).min(n_groups);
            let fold_groups: std::collections::HashSet<usize> = 
                unique_groups[fold_groups_start..fold_groups_end].iter().cloned().collect();

            let test_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| fold_groups.contains(&groups[i]))
                .collect();
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| !fold_groups.contains(&groups[i]))
                .collect();

            folds.push((train_indices, test_indices));
        }

        folds
    }
}

/// Repeated K-Fold Cross-Validation
pub struct RepeatedKFold {
    pub n_splits: usize,
    pub n_repeats: usize,
    pub random_state: Option<u64>,
}

impl RepeatedKFold {
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        RepeatedKFold {
            n_splits,
            n_repeats,
            random_state: None,
        }
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let mut all_folds = Vec::with_capacity(self.n_splits * self.n_repeats);

        for _ in 0..self.n_repeats {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let fold_size = n_samples / self.n_splits;

            for fold in 0..self.n_splits {
                let test_start = fold * fold_size;
                let test_end = if fold == self.n_splits - 1 { n_samples } else { (fold + 1) * fold_size };

                let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
                let train_indices: Vec<usize> = indices[..test_start].iter()
                    .chain(indices[test_end..].iter())
                    .cloned()
                    .collect();

                all_folds.push((train_indices, test_indices));
            }
        }

        all_folds
    }

    pub fn get_n_splits(&self) -> usize {
        self.n_splits * self.n_repeats
    }
}

/// Stratified Shuffle Split
pub struct StratifiedShuffleSplit {
    pub n_splits: usize,
    pub test_size: f32,
    pub random_state: Option<u64>,
}

impl StratifiedShuffleSplit {
    pub fn new(n_splits: usize, test_size: f32) -> Self {
        StratifiedShuffleSplit {
            n_splits,
            test_size,
            random_state: None,
        }
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn split(&self, y: &[f32]) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let _n_samples = y.len();

        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_indices.entry(label as i32).or_default().push(i);
        }

        let mut all_splits = Vec::with_capacity(self.n_splits);

        for _ in 0..self.n_splits {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for (_, indices) in &class_indices {
                let mut shuffled = indices.clone();
                shuffled.shuffle(&mut rng);

                let n_test = (indices.len() as f32 * self.test_size).ceil() as usize;
                let n_test = n_test.max(1).min(indices.len() - 1);

                test_indices.extend_from_slice(&shuffled[..n_test]);
                train_indices.extend_from_slice(&shuffled[n_test..]);
            }

            train_indices.shuffle(&mut rng);
            test_indices.shuffle(&mut rng);

            all_splits.push((train_indices, test_indices));
        }

        all_splits
    }
}

/// Learning Curve - evaluate model performance with varying training set sizes
pub fn learning_curve<F>(
    x: &Tensor,
    y: &Tensor,
    train_sizes: &[f32],
    cv: usize,
    mut fit_and_score: F,
) -> (Vec<usize>, Vec<f32>, Vec<f32>)
where
    F: FnMut(&Tensor, &Tensor, &Tensor, &Tensor) -> f32,
{
    let x_data = x.data_f32();
    let y_data = y.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];

    let mut sizes = Vec::new();
    let mut train_scores = Vec::new();
    let mut test_scores = Vec::new();

    for &size_ratio in train_sizes {
        let train_size = (n_samples as f32 * size_ratio) as usize;
        if train_size < 2 { continue; }

        let fold_size = n_samples / cv;
        let mut fold_train_scores = Vec::new();
        let mut fold_test_scores = Vec::new();

        for fold in 0..cv {
            let val_start = fold * fold_size;
            let val_end = if fold == cv - 1 { n_samples } else { (fold + 1) * fold_size };

            let all_train_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| i < val_start || i >= val_end)
                .collect();
            let val_indices: Vec<usize> = (val_start..val_end).collect();

            // Use only train_size samples
            let train_indices: Vec<usize> = all_train_indices.into_iter()
                .take(train_size)
                .collect();

            if train_indices.is_empty() { continue; }

            let x_train: Vec<f32> = train_indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let y_train: Vec<f32> = train_indices.iter().map(|&i| y_data[i]).collect();

            let x_val: Vec<f32> = val_indices.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let y_val: Vec<f32> = val_indices.iter().map(|&i| y_data[i]).collect();

            let x_train_t = Tensor::from_slice(&x_train, &[train_indices.len(), n_features]).unwrap();
            let y_train_t = Tensor::from_slice(&y_train, &[train_indices.len()]).unwrap();
            let x_val_t = Tensor::from_slice(&x_val, &[val_indices.len(), n_features]).unwrap();
            let y_val_t = Tensor::from_slice(&y_val, &[val_indices.len()]).unwrap();

            // Score on training data
            let train_score = fit_and_score(&x_train_t, &y_train_t, &x_train_t, &y_train_t);
            fold_train_scores.push(train_score);

            // Score on validation data
            let test_score = fit_and_score(&x_train_t, &y_train_t, &x_val_t, &y_val_t);
            fold_test_scores.push(test_score);
        }

        if !fold_train_scores.is_empty() {
            sizes.push(train_size);
            train_scores.push(fold_train_scores.iter().sum::<f32>() / fold_train_scores.len() as f32);
            test_scores.push(fold_test_scores.iter().sum::<f32>() / fold_test_scores.len() as f32);
        }
    }

    (sizes, train_scores, test_scores)
}

/// Validation Curve - evaluate model performance with varying hyperparameter values
pub fn validation_curve<F>(
    x: &Tensor,
    y: &Tensor,
    param_values: &[f32],
    cv: usize,
    mut fit_and_score: F,
) -> (Vec<f32>, Vec<f32>)
where
    F: FnMut(&Tensor, &Tensor, &Tensor, &Tensor, f32) -> f32,
{
    let x_data = x.data_f32();
    let y_data = y.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];

    let mut train_scores = Vec::new();
    let mut test_scores = Vec::new();

    for &param_value in param_values {
        let fold_size = n_samples / cv;
        let mut fold_train_scores = Vec::new();
        let mut fold_test_scores = Vec::new();

        for fold in 0..cv {
            let val_start = fold * fold_size;
            let val_end = if fold == cv - 1 { n_samples } else { (fold + 1) * fold_size };

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
            let y_val: Vec<f32> = val_indices.iter().map(|&i| y_data[i]).collect();

            let x_train_t = Tensor::from_slice(&x_train, &[train_indices.len(), n_features]).unwrap();
            let y_train_t = Tensor::from_slice(&y_train, &[train_indices.len()]).unwrap();
            let x_val_t = Tensor::from_slice(&x_val, &[val_indices.len(), n_features]).unwrap();
            let y_val_t = Tensor::from_slice(&y_val, &[val_indices.len()]).unwrap();

            let train_score = fit_and_score(&x_train_t, &y_train_t, &x_train_t, &y_train_t, param_value);
            fold_train_scores.push(train_score);

            let test_score = fit_and_score(&x_train_t, &y_train_t, &x_val_t, &y_val_t, param_value);
            fold_test_scores.push(test_score);
        }

        train_scores.push(fold_train_scores.iter().sum::<f32>() / fold_train_scores.len() as f32);
        test_scores.push(fold_test_scores.iter().sum::<f32>() / fold_test_scores.len() as f32);
    }

    (train_scores, test_scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_kfold() {
        let groups = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let gkf = GroupKFold::new(2);
        let splits = gkf.split(8, &groups);
        
        assert_eq!(splits.len(), 2);
        for (train, test) in &splits {
            assert!(!train.is_empty());
            assert!(!test.is_empty());
        }
    }

    #[test]
    fn test_repeated_kfold() {
        let rkf = RepeatedKFold::new(3, 2).random_state(42);
        let splits = rkf.split(9);
        
        assert_eq!(splits.len(), 6); // 3 folds * 2 repeats
    }

    #[test]
    fn test_stratified_shuffle_split() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let sss = StratifiedShuffleSplit::new(3, 0.33).random_state(42);
        let splits = sss.split(&y);
        
        assert_eq!(splits.len(), 3);
    }
}



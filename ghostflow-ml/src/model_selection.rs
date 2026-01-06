//! Model Selection - Cross-validation, Grid Search, Train-Test Split

use ghostflow_core::Tensor;
use rand::prelude::*;

/// K-Fold Cross-Validation
pub struct KFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
}

impl KFold {
    pub fn new(n_splits: usize) -> Self {
        KFold {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }

        let fold_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        let mut folds = Vec::with_capacity(self.n_splits);
        let mut start = 0;

        for i in 0..self.n_splits {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + fold_size + extra;

            let test_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[..start].iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            folds.push((train_indices, test_indices));
            start = end;
        }

        folds
    }
}

/// Stratified K-Fold Cross-Validation
pub struct StratifiedKFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
}

impl StratifiedKFold {
    pub fn new(n_splits: usize) -> Self {
        StratifiedKFold {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn split(&self, y: &Tensor) -> Vec<(Vec<usize>, Vec<usize>)> {
        let y_data = y.data_f32();
        let _n_samples = y_data.len();

        // Group indices by class
        let n_classes = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];

        for (i, &label) in y_data.iter().enumerate() {
            class_indices[label as usize].push(i);
        }

        // Shuffle within each class if needed
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_entropy(),
            };
            for indices in &mut class_indices {
                indices.shuffle(&mut rng);
            }
        }

        // Create stratified folds
        let mut folds: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_splits)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();

        for class_idx in &class_indices {
            let n_class = class_idx.len();
            let fold_size = n_class / self.n_splits;
            let remainder = n_class % self.n_splits;

            let mut start = 0;
            for i in 0..self.n_splits {
                let extra = if i < remainder { 1 } else { 0 };
                let end = start + fold_size + extra;

                // Add to test set for this fold
                folds[i].1.extend(&class_idx[start..end]);

                // Add to train set for other folds
                for j in 0..self.n_splits {
                    if j != i {
                        folds[j].0.extend(&class_idx[start..end]);
                    }
                }

                start = end;
            }
        }

        folds
    }
}

/// Leave-One-Out Cross-Validation
pub struct LeaveOneOut;

impl LeaveOneOut {
    pub fn new() -> Self {
        LeaveOneOut
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        (0..n_samples)
            .map(|i| {
                let train: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
                let test = vec![i];
                (train, test)
            })
            .collect()
    }
}

impl Default for LeaveOneOut {
    fn default() -> Self {
        Self::new()
    }
}

/// Time Series Split for temporal data
pub struct TimeSeriesSplit {
    pub n_splits: usize,
    pub max_train_size: Option<usize>,
    pub test_size: Option<usize>,
    pub gap: usize,
}

impl TimeSeriesSplit {
    pub fn new(n_splits: usize) -> Self {
        TimeSeriesSplit {
            n_splits,
            max_train_size: None,
            test_size: None,
            gap: 0,
        }
    }

    pub fn gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let test_size = self.test_size.unwrap_or(n_samples / (self.n_splits + 1));
        
        let mut folds = Vec::with_capacity(self.n_splits);

        for i in 0..self.n_splits {
            let test_start = n_samples - (self.n_splits - i) * test_size;
            let test_end = test_start + test_size;
            
            let train_end = test_start - self.gap;
            let train_start = match self.max_train_size {
                Some(max) => train_end.saturating_sub(max),
                None => 0,
            };

            if train_start < train_end && test_start < test_end {
                let train: Vec<usize> = (train_start..train_end).collect();
                let test: Vec<usize> = (test_start..test_end.min(n_samples)).collect();
                folds.push((train, test));
            }
        }

        folds
    }
}

/// Cross-validation score computation
pub fn cross_val_score<F>(
    x: &Tensor,
    y: &Tensor,
    cv: &[(Vec<usize>, Vec<usize>)],
    fit_predict_score: F,
) -> Vec<f32>
where
    F: Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> f32,
{
    let x_data = x.data_f32();
    let y_data = y.data_f32();
    let n_features = x.dims()[1];

    cv.iter()
        .map(|(train_idx, test_idx)| {
            // Create train tensors
            let x_train: Vec<f32> = train_idx.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let y_train: Vec<f32> = train_idx.iter().map(|&i| y_data[i]).collect();

            // Create test tensors
            let x_test: Vec<f32> = test_idx.iter()
                .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let y_test: Vec<f32> = test_idx.iter().map(|&i| y_data[i]).collect();

            let x_train_tensor = Tensor::from_slice(&x_train, &[train_idx.len(), n_features]).unwrap();
            let y_train_tensor = Tensor::from_slice(&y_train, &[train_idx.len()]).unwrap();
            let x_test_tensor = Tensor::from_slice(&x_test, &[test_idx.len(), n_features]).unwrap();
            let y_test_tensor = Tensor::from_slice(&y_test, &[test_idx.len()]).unwrap();

            fit_predict_score(&x_train_tensor, &y_train_tensor, &x_test_tensor, &y_test_tensor)
        })
        .collect()
}

/// Grid Search for hyperparameter tuning
pub struct GridSearchResult {
    pub best_params: Vec<(String, f32)>,
    pub best_score: f32,
    pub cv_results: Vec<(Vec<(String, f32)>, f32)>,
}

/// Generate parameter grid combinations
pub fn parameter_grid(params: &[(String, Vec<f32>)]) -> Vec<Vec<(String, f32)>> {
    if params.is_empty() {
        return vec![vec![]];
    }

    let (name, values) = &params[0];
    let rest = parameter_grid(&params[1..]);

    let mut result = Vec::new();
    for &value in values {
        for r in &rest {
            let mut combo = vec![(name.clone(), value)];
            combo.extend(r.clone());
            result.push(combo);
        }
    }

    result
}

/// Shuffle and split data
pub fn shuffle_split(
    x: &Tensor,
    y: &Tensor,
    test_size: f32,
    random_state: Option<u64>,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let x_data = x.data_f32();
    let y_data = y.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];

    let mut indices: Vec<usize> = (0..n_samples).collect();
    
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    indices.shuffle(&mut rng);

    let n_test = (n_samples as f32 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];

    let x_train: Vec<f32> = train_indices.iter()
        .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
        .collect();
    let y_train: Vec<f32> = train_indices.iter().map(|&i| y_data[i]).collect();

    let x_test: Vec<f32> = test_indices.iter()
        .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
        .collect();
    let y_test: Vec<f32> = test_indices.iter().map(|&i| y_data[i]).collect();

    (
        Tensor::from_slice(&x_train, &[n_train, n_features]).unwrap(),
        Tensor::from_slice(&x_test, &[n_test, n_features]).unwrap(),
        Tensor::from_slice(&y_train, &[n_train]).unwrap(),
        Tensor::from_slice(&y_test, &[n_test]).unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold() {
        let kf = KFold::new(5);
        let folds = kf.split(100);
        
        assert_eq!(folds.len(), 5);
        
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 100);
            assert_eq!(test.len(), 20);
        }
    }

    #[test]
    fn test_stratified_kfold() {
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[8]).unwrap();
        
        let skf = StratifiedKFold::new(2);
        let folds = skf.split(&y);
        
        assert_eq!(folds.len(), 2);
    }

    #[test]
    fn test_time_series_split() {
        let tss = TimeSeriesSplit::new(3);
        let folds = tss.split(100);
        
        assert_eq!(folds.len(), 3);
        
        // Verify temporal ordering
        for (train, test) in &folds {
            let max_train = train.iter().max().unwrap_or(&0);
            let min_test = test.iter().min().unwrap_or(&100);
            assert!(max_train < min_test);
        }
    }

    #[test]
    fn test_parameter_grid() {
        let params = vec![
            ("alpha".to_string(), vec![0.1, 1.0]),
            ("beta".to_string(), vec![0.01, 0.1]),
        ];
        
        let grid = parameter_grid(&params);
        assert_eq!(grid.len(), 4);
    }
}



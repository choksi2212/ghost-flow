//! Data preprocessing utilities

use ghostflow_core::Tensor;

/// Standard Scaler - standardize features by removing mean and scaling to unit variance
pub struct StandardScaler {
    pub mean_: Option<Vec<f32>>,
    pub std_: Option<Vec<f32>>,
    pub with_mean: bool,
    pub with_std: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler {
            mean_: None,
            std_: None,
            with_mean: true,
            with_std: true,
        }
    }

    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut mean = vec![0.0f32; n_features];
        let mut std = vec![0.0f32; n_features];

        // Compute mean
        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += x_data[i * n_features + j];
            }
        }
        for j in 0..n_features {
            mean[j] /= n_samples as f32;
        }

        // Compute std
        for i in 0..n_samples {
            for j in 0..n_features {
                let diff = x_data[i * n_features + j] - mean[j];
                std[j] += diff * diff;
            }
        }
        for j in 0..n_features {
            std[j] = (std[j] / n_samples as f32).sqrt().max(1e-10);
        }

        self.mean_ = Some(mean);
        self.std_ = Some(std);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean = self.mean_.as_ref().expect("Scaler not fitted");
        let std = self.std_.as_ref().expect("Scaler not fitted");

        let mut result = vec![0.0f32; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut val = x_data[i * n_features + j];
                if self.with_mean {
                    val -= mean[j];
                }
                if self.with_std {
                    val /= std[j];
                }
                result[i * n_features + j] = val;
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean = self.mean_.as_ref().expect("Scaler not fitted");
        let std = self.std_.as_ref().expect("Scaler not fitted");

        let mut result = vec![0.0f32; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut val = x_data[i * n_features + j];
                if self.with_std {
                    val *= std[j];
                }
                if self.with_mean {
                    val += mean[j];
                }
                result[i * n_features + j] = val;
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// MinMax Scaler - scale features to a given range
pub struct MinMaxScaler {
    pub min_: Option<Vec<f32>>,
    pub max_: Option<Vec<f32>>,
    pub feature_range: (f32, f32),
}

impl MinMaxScaler {
    pub fn new() -> Self {
        MinMaxScaler {
            min_: None,
            max_: None,
            feature_range: (0.0, 1.0),
        }
    }

    pub fn feature_range(mut self, min: f32, max: f32) -> Self {
        self.feature_range = (min, max);
        self
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut min = vec![f32::INFINITY; n_features];
        let mut max = vec![f32::NEG_INFINITY; n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x_data[i * n_features + j];
                min[j] = min[j].min(val);
                max[j] = max[j].max(val);
            }
        }

        self.min_ = Some(min);
        self.max_ = Some(max);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let min = self.min_.as_ref().expect("Scaler not fitted");
        let max = self.max_.as_ref().expect("Scaler not fitted");
        let (range_min, range_max) = self.feature_range;

        let mut result = vec![0.0f32; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x_data[i * n_features + j];
                let scale = max[j] - min[j];
                let scaled = if scale > 1e-10 {
                    (val - min[j]) / scale
                } else {
                    0.5
                };
                result[i * n_features + j] = scaled * (range_max - range_min) + range_min;
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}


/// Normalizer - normalize samples individually to unit norm
pub struct Normalizer {
    pub norm: Norm,
}

#[derive(Clone, Copy, Debug)]
pub enum Norm {
    L1,
    L2,
    Max,
}

impl Normalizer {
    pub fn new(norm: Norm) -> Self {
        Normalizer { norm }
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut result = vec![0.0f32; n_samples * n_features];

        for i in 0..n_samples {
            let row = &x_data[i * n_features..(i + 1) * n_features];
            
            let norm_val = match self.norm {
                Norm::L1 => row.iter().map(|&x| x.abs()).sum::<f32>(),
                Norm::L2 => row.iter().map(|&x| x * x).sum::<f32>().sqrt(),
                Norm::Max => row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max),
            };

            let norm_val = norm_val.max(1e-10);

            for j in 0..n_features {
                result[i * n_features + j] = row[j] / norm_val;
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }
}

/// Label Encoder - encode target labels with value between 0 and n_classes-1
pub struct LabelEncoder {
    pub classes_: Option<Vec<String>>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        LabelEncoder { classes_: None }
    }

    pub fn fit(&mut self, labels: &[String]) {
        let mut classes: Vec<String> = labels.to_vec();
        classes.sort();
        classes.dedup();
        self.classes_ = Some(classes);
    }

    pub fn transform(&self, labels: &[String]) -> Vec<usize> {
        let classes = self.classes_.as_ref().expect("Encoder not fitted");
        
        labels.iter()
            .map(|label| {
                classes.iter().position(|c| c == label).unwrap_or(0)
            })
            .collect()
    }

    pub fn fit_transform(&mut self, labels: &[String]) -> Vec<usize> {
        self.fit(labels);
        self.transform(labels)
    }

    pub fn inverse_transform(&self, encoded: &[usize]) -> Vec<String> {
        let classes = self.classes_.as_ref().expect("Encoder not fitted");
        
        encoded.iter()
            .map(|&idx| {
                classes.get(idx).cloned().unwrap_or_default()
            })
            .collect()
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// One-Hot Encoder
pub struct OneHotEncoder {
    pub n_categories_: Option<Vec<usize>>,
}

impl OneHotEncoder {
    pub fn new() -> Self {
        OneHotEncoder { n_categories_: None }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut n_categories = vec![0usize; n_features];

        for j in 0..n_features {
            let max_val = (0..n_samples)
                .map(|i| x_data[i * n_features + j] as usize)
                .max()
                .unwrap_or(0);
            n_categories[j] = max_val + 1;
        }

        self.n_categories_ = Some(n_categories);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let n_categories = self.n_categories_.as_ref().expect("Encoder not fitted");
        let total_cols: usize = n_categories.iter().sum();

        let mut result = vec![0.0f32; n_samples * total_cols];

        for i in 0..n_samples {
            let mut col_offset = 0;
            for j in 0..n_features {
                let category = x_data[i * n_features + j] as usize;
                if category < n_categories[j] {
                    result[i * total_cols + col_offset + category] = 1.0;
                }
                col_offset += n_categories[j];
            }
        }

        Tensor::from_slice(&result, &[n_samples, total_cols]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Train-test split utility
pub fn train_test_split(
    x: &Tensor,
    y: &Tensor,
    test_size: f32,
    shuffle: bool,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let x_data = x.data_f32();
    let y_data = y.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];

    let mut indices: Vec<usize> = (0..n_samples).collect();
    
    if shuffle {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }

    let n_test = (n_samples as f32 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    let mut x_train = vec![0.0f32; n_train * n_features];
    let mut x_test = vec![0.0f32; n_test * n_features];
    let mut y_train = vec![0.0f32; n_train];
    let mut y_test = vec![0.0f32; n_test];

    for (new_idx, &orig_idx) in indices.iter().enumerate() {
        if new_idx < n_train {
            for j in 0..n_features {
                x_train[new_idx * n_features + j] = x_data[orig_idx * n_features + j];
            }
            y_train[new_idx] = y_data[orig_idx];
        } else {
            let test_idx = new_idx - n_train;
            for j in 0..n_features {
                x_test[test_idx * n_features + j] = x_data[orig_idx * n_features + j];
            }
            y_test[test_idx] = y_data[orig_idx];
        }
    }

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
    fn test_standard_scaler() {
        let x = Tensor::from_slice(&[1.0f32, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ], &[3, 2]).unwrap();

        let mut scaler = StandardScaler::new();
        let scaled = scaler.fit_transform(&x);
        
        assert_eq!(scaled.dims(), &[3, 2]);
        
        // Check mean is ~0
        let scaled_data = scaled.storage().as_slice::<f32>().to_vec();
        let mean: f32 = scaled_data.iter().sum::<f32>() / scaled_data.len() as f32;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_minmax_scaler() {
        let x = Tensor::from_slice(&[0.0f32, 10.0,
            5.0, 20.0,
            10.0, 30.0,
        ], &[3, 2]).unwrap();

        let mut scaler = MinMaxScaler::new();
        let scaled = scaler.fit_transform(&x);
        
        let scaled_data = scaled.storage().as_slice::<f32>().to_vec();
        assert!(scaled_data.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn test_train_test_split() {
        let x = Tensor::from_slice(&[1.0f32, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
            9.0, 10.0,
        ], &[5, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 1.0, 0.0], &[5]).unwrap();

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.4, false);
        
        assert_eq!(x_train.dims()[0], 3);
        assert_eq!(x_test.dims()[0], 2);
        assert_eq!(y_train.dims()[0], 3);
        assert_eq!(y_test.dims()[0], 2);
    }
}



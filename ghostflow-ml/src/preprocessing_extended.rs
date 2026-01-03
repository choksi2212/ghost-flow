//! Extended Preprocessing - RobustScaler, MaxAbsScaler, OrdinalEncoder, etc.

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// Robust Scaler - scales using statistics robust to outliers
/// Uses median and IQR instead of mean and std
pub struct RobustScaler {
    pub with_centering: bool,
    pub with_scaling: bool,
    pub quantile_range: (f32, f32),
    center_: Option<Vec<f32>>,
    scale_: Option<Vec<f32>>,
}

impl RobustScaler {
    pub fn new() -> Self {
        RobustScaler {
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
            center_: None,
            scale_: None,
        }
    }

    pub fn with_centering(mut self, c: bool) -> Self { self.with_centering = c; self }
    pub fn with_scaling(mut self, s: bool) -> Self { self.with_scaling = s; self }
    pub fn quantile_range(mut self, low: f32, high: f32) -> Self { 
        self.quantile_range = (low, high); 
        self 
    }

    fn quantile(sorted: &[f32], q: f32) -> f32 {
        if sorted.is_empty() { return 0.0; }
        let idx = (q / 100.0 * (sorted.len() - 1) as f32) as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut center = vec![0.0f32; n_features];
        let mut scale = vec![1.0f32; n_features];

        for j in 0..n_features {
            let mut values: Vec<f32> = (0..n_samples)
                .map(|i| x_data[i * n_features + j])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if self.with_centering {
                center[j] = Self::quantile(&values, 50.0); // Median
            }

            if self.with_scaling {
                let q_low = Self::quantile(&values, self.quantile_range.0);
                let q_high = Self::quantile(&values, self.quantile_range.1);
                let iqr = q_high - q_low;
                scale[j] = if iqr > 1e-10 { iqr } else { 1.0 };
            }
        }

        self.center_ = Some(center);
        self.scale_ = Some(scale);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let center = self.center_.as_ref().expect("Scaler not fitted");
        let scale = self.scale_.as_ref().unwrap();

        let result: Vec<f32> = (0..n_samples)
            .flat_map(|i| {
                (0..n_features).map(|j| {
                    let mut val = x_data[i * n_features + j];
                    if self.with_centering {
                        val -= center[j];
                    }
                    if self.with_scaling {
                        val /= scale[j];
                    }
                    val
                }).collect::<Vec<_>>()
            })
            .collect();

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

        let center = self.center_.as_ref().expect("Scaler not fitted");
        let scale = self.scale_.as_ref().unwrap();

        let result: Vec<f32> = (0..n_samples)
            .flat_map(|i| {
                (0..n_features).map(|j| {
                    let mut val = x_data[i * n_features + j];
                    if self.with_scaling {
                        val *= scale[j];
                    }
                    if self.with_centering {
                        val += center[j];
                    }
                    val
                }).collect::<Vec<_>>()
            })
            .collect();

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }
}

impl Default for RobustScaler {
    fn default() -> Self { Self::new() }
}

/// MaxAbs Scaler - scales by maximum absolute value
pub struct MaxAbsScaler {
    max_abs_: Option<Vec<f32>>,
}

impl MaxAbsScaler {
    pub fn new() -> Self {
        MaxAbsScaler { max_abs_: None }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let max_abs: Vec<f32> = (0..n_features)
            .map(|j| {
                (0..n_samples)
                    .map(|i| x_data[i * n_features + j].abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-10)
            })
            .collect();

        self.max_abs_ = Some(max_abs);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let max_abs = self.max_abs_.as_ref().expect("Scaler not fitted");

        let result: Vec<f32> = (0..n_samples)
            .flat_map(|i| {
                (0..n_features).map(|j| x_data[i * n_features + j] / max_abs[j]).collect::<Vec<_>>()
            })
            .collect();

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

        let max_abs = self.max_abs_.as_ref().expect("Scaler not fitted");

        let result: Vec<f32> = (0..n_samples)
            .flat_map(|i| {
                (0..n_features).map(|j| x_data[i * n_features + j] * max_abs[j]).collect::<Vec<_>>()
            })
            .collect();

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self { Self::new() }
}

/// Ordinal Encoder - encodes categorical features as integers
pub struct OrdinalEncoder {
    pub handle_unknown: HandleUnknown,
    pub unknown_value: Option<f32>,
    categories_: Option<Vec<Vec<String>>>,
}

#[derive(Clone, Copy)]
pub enum HandleUnknown {
    Error,
    UseEncodedValue,
}

impl OrdinalEncoder {
    pub fn new() -> Self {
        OrdinalEncoder {
            handle_unknown: HandleUnknown::Error,
            unknown_value: None,
            categories_: None,
        }
    }

    pub fn handle_unknown(mut self, h: HandleUnknown, value: Option<f32>) -> Self {
        self.handle_unknown = h;
        self.unknown_value = value;
        self
    }

    pub fn fit(&mut self, x: &[Vec<String>]) {
        let n_features = if x.is_empty() { 0 } else { x[0].len() };
        
        let mut categories: Vec<Vec<String>> = vec![Vec::new(); n_features];

        for row in x {
            for (j, val) in row.iter().enumerate() {
                if !categories[j].contains(val) {
                    categories[j].push(val.clone());
                }
            }
        }

        // Sort categories for consistency
        for cats in &mut categories {
            cats.sort();
        }

        self.categories_ = Some(categories);
    }

    pub fn transform(&self, x: &[Vec<String>]) -> Vec<Vec<f32>> {
        let categories = self.categories_.as_ref().expect("Encoder not fitted");
        let _n_features = categories.len();

        x.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, val)| {
                        if let Some(idx) = categories[j].iter().position(|c| c == val) {
                            idx as f32
                        } else {
                            match self.handle_unknown {
                                HandleUnknown::UseEncodedValue => {
                                    self.unknown_value.unwrap_or(-1.0)
                                }
                                HandleUnknown::Error => {
                                    panic!("Unknown category: {}", val);
                                }
                            }
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn fit_transform(&mut self, x: &[Vec<String>]) -> Vec<Vec<f32>> {
        self.fit(x);
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &[Vec<f32>]) -> Vec<Vec<String>> {
        let categories = self.categories_.as_ref().expect("Encoder not fitted");

        x.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &val)| {
                        let idx = val as usize;
                        if idx < categories[j].len() {
                            categories[j][idx].clone()
                        } else {
                            "unknown".to_string()
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

impl Default for OrdinalEncoder {
    fn default() -> Self { Self::new() }
}

/// Target Encoder - encodes categorical features using target statistics
pub struct TargetEncoder {
    pub smooth: f32,
    pub target_type: TargetType,
    encodings_: Option<Vec<HashMap<String, f32>>>,
    global_mean_: f32,
}

#[derive(Clone, Copy)]
pub enum TargetType {
    Continuous,
    Binary,
}

impl TargetEncoder {
    pub fn new() -> Self {
        TargetEncoder {
            smooth: 1.0,
            target_type: TargetType::Continuous,
            encodings_: None,
            global_mean_: 0.0,
        }
    }

    pub fn smooth(mut self, s: f32) -> Self { self.smooth = s; self }

    pub fn fit(&mut self, x: &[Vec<String>], y: &[f32]) {
        let n_samples = x.len();
        let n_features = if x.is_empty() { 0 } else { x[0].len() };

        self.global_mean_ = y.iter().sum::<f32>() / n_samples as f32;

        let mut encodings: Vec<HashMap<String, f32>> = vec![HashMap::new(); n_features];

        for j in 0..n_features {
            // Group by category
            let mut category_stats: HashMap<String, (f32, usize)> = HashMap::new();

            for (i, row) in x.iter().enumerate() {
                let cat = &row[j];
                let entry = category_stats.entry(cat.clone()).or_insert((0.0, 0));
                entry.0 += y[i];
                entry.1 += 1;
            }

            // Compute smoothed encoding
            for (cat, (sum, count)) in category_stats {
                let cat_mean = sum / count as f32;
                // Smoothed mean: (count * cat_mean + smooth * global_mean) / (count + smooth)
                let smoothed = (count as f32 * cat_mean + self.smooth * self.global_mean_) 
                    / (count as f32 + self.smooth);
                encodings[j].insert(cat, smoothed);
            }
        }

        self.encodings_ = Some(encodings);
    }

    pub fn transform(&self, x: &[Vec<String>]) -> Vec<Vec<f32>> {
        let encodings = self.encodings_.as_ref().expect("Encoder not fitted");

        x.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, cat)| {
                        *encodings[j].get(cat).unwrap_or(&self.global_mean_)
                    })
                    .collect()
            })
            .collect()
    }

    pub fn fit_transform(&mut self, x: &[Vec<String>], y: &[f32]) -> Vec<Vec<f32>> {
        self.fit(x, y);
        self.transform(x)
    }
}

impl Default for TargetEncoder {
    fn default() -> Self { Self::new() }
}

/// KBins Discretizer - bins continuous features into discrete intervals
pub struct KBinsDiscretizer {
    pub n_bins: usize,
    pub strategy: BinStrategy,
    pub encode: BinEncode,
    bin_edges_: Option<Vec<Vec<f32>>>,
}

#[derive(Clone, Copy)]
pub enum BinStrategy {
    Uniform,
    Quantile,
    KMeans,
}

#[derive(Clone, Copy)]
pub enum BinEncode {
    Ordinal,
    OneHot,
}

impl KBinsDiscretizer {
    pub fn new(n_bins: usize) -> Self {
        KBinsDiscretizer {
            n_bins,
            strategy: BinStrategy::Quantile,
            encode: BinEncode::Ordinal,
            bin_edges_: None,
        }
    }

    pub fn strategy(mut self, s: BinStrategy) -> Self { self.strategy = s; self }
    pub fn encode(mut self, e: BinEncode) -> Self { self.encode = e; self }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut bin_edges: Vec<Vec<f32>> = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut values: Vec<f32> = (0..n_samples)
                .map(|i| x_data[i * n_features + j])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let edges = match self.strategy {
                BinStrategy::Uniform => {
                    let min_val = values[0];
                    let max_val = values[values.len() - 1];
                    let step = (max_val - min_val) / self.n_bins as f32;
                    (0..=self.n_bins).map(|i| min_val + i as f32 * step).collect()
                }
                BinStrategy::Quantile => {
                    (0..=self.n_bins)
                        .map(|i| {
                            let q = i as f32 / self.n_bins as f32;
                            let idx = ((n_samples - 1) as f32 * q) as usize;
                            values[idx]
                        })
                        .collect()
                }
                BinStrategy::KMeans => {
                    // Simplified: use uniform for now
                    let min_val = values[0];
                    let max_val = values[values.len() - 1];
                    let step = (max_val - min_val) / self.n_bins as f32;
                    (0..=self.n_bins).map(|i| min_val + i as f32 * step).collect()
                }
            };

            bin_edges.push(edges);
        }

        self.bin_edges_ = Some(bin_edges);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let bin_edges = self.bin_edges_.as_ref().expect("Discretizer not fitted");

        match self.encode {
            BinEncode::Ordinal => {
                let result: Vec<f32> = (0..n_samples)
                    .flat_map(|i| {
                        (0..n_features).map(|j| {
                            let val = x_data[i * n_features + j];
                            let edges = &bin_edges[j];
                            let mut bin = 0;
                            for k in 1..edges.len() {
                                if val >= edges[k] {
                                    bin = k;
                                } else {
                                    break;
                                }
                            }
                            bin.min(self.n_bins - 1) as f32
                        }).collect::<Vec<_>>()
                    })
                    .collect();

                Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
            }
            BinEncode::OneHot => {
                let n_output = n_features * self.n_bins;
                let mut result = vec![0.0f32; n_samples * n_output];

                for i in 0..n_samples {
                    for j in 0..n_features {
                        let val = x_data[i * n_features + j];
                        let edges = &bin_edges[j];
                        let mut bin = 0;
                        for k in 1..edges.len() {
                            if val >= edges[k] {
                                bin = k;
                            } else {
                                break;
                            }
                        }
                        bin = bin.min(self.n_bins - 1);
                        result[i * n_output + j * self.n_bins + bin] = 1.0;
                    }
                }

                Tensor::from_slice(&result, &[n_samples, n_output]).unwrap()
            }
        }
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

/// Binarizer - threshold features to binary values
pub struct Binarizer {
    pub threshold: f32,
}

impl Binarizer {
    pub fn new(threshold: f32) -> Self {
        Binarizer { threshold }
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let result: Vec<f32> = x_data.iter()
            .map(|&v| if v > self.threshold { 1.0 } else { 0.0 })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robust_scaler() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 100.0, 6.0], &[3, 2]).unwrap();
        let mut scaler = RobustScaler::new();
        let result = scaler.fit_transform(&x);
        assert_eq!(result.dims(), &[3, 2]);
    }

    #[test]
    fn test_max_abs_scaler() {
        let x = Tensor::from_slice(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]).unwrap();
        let mut scaler = MaxAbsScaler::new();
        let result = scaler.fit_transform(&x);
        
        let data = result.data_f32();
        assert!(data.iter().all(|&v| v.abs() <= 1.0));
    }

    #[test]
    fn test_kbins_discretizer() {
        let x = Tensor::from_slice(&[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], &[3, 2]).unwrap();
        let mut disc = KBinsDiscretizer::new(3);
        let result = disc.fit_transform(&x);
        assert_eq!(result.dims(), &[3, 2]);
    }

    #[test]
    fn test_binarizer() {
        let x = Tensor::from_slice(&[0.0, 0.5, 1.0, 1.5], &[2, 2]).unwrap();
        let binarizer = Binarizer::new(0.5);
        let result = binarizer.transform(&x);
        
        let data = result.data_f32();
        assert_eq!(data, &[0.0, 0.0, 1.0, 1.0]);
    }
}

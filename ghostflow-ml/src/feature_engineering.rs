//! Feature Engineering Utilities
//!
//! Tools for creating and transforming features to improve model performance.

use ghostflow_core::Tensor;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Polynomial Feature Generator
/// 
/// Generates polynomial and interaction features.
/// Example: [a, b] with degree=2 -> [1, a, b, a², ab, b²]
pub struct PolynomialFeatures {
    pub degree: usize,
    pub interaction_only: bool,
    pub include_bias: bool,
    n_input_features: usize,
    n_output_features: usize,
}

impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            interaction_only: false,
            include_bias: true,
            n_input_features: 0,
            n_output_features: 0,
        }
    }

    pub fn interaction_only(mut self, value: bool) -> Self {
        self.interaction_only = value;
        self
    }

    pub fn include_bias(mut self, value: bool) -> Self {
        self.include_bias = value;
        self
    }

    /// Fit to determine output feature count
    pub fn fit(&mut self, x: &Tensor) {
        self.n_input_features = x.dims()[1];
        self.n_output_features = self.calculate_n_output_features();
    }

    fn calculate_n_output_features(&self) -> usize {
        let n = self.n_input_features;
        let d = self.degree;
        
        let mut count = if self.include_bias { 1 } else { 0 };

        if self.interaction_only {
            // Only interaction terms
            for degree in 1..=d {
                count += Self::n_combinations(n, degree);
            }
        } else {
            // All polynomial terms
            count += Self::n_combinations_with_replacement(n + d, d) - 1;
            if !self.include_bias {
                count -= 1;
            }
        }

        count
    }

    fn n_combinations(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    fn n_combinations_with_replacement(n: usize, k: usize) -> usize {
        Self::n_combinations(n + k - 1, k)
    }

    /// Transform features to polynomial features
    pub fn transform(&self, x: &Tensor) -> Tensor {
        let n_samples = x.dims()[0];
        let x_data = x.data_f32();
        
        let mut all_features = Vec::new();
        let mut actual_n_features = 0;

        for i in 0..n_samples {
            let sample = &x_data[i * self.n_input_features..(i + 1) * self.n_input_features];
            let poly_features = self.generate_polynomial_features(sample);
            if i == 0 {
                actual_n_features = poly_features.len();
            }
            all_features.extend(poly_features);
        }

        Tensor::from_slice(&all_features, &[n_samples, actual_n_features]).unwrap()
    }

    fn generate_polynomial_features(&self, sample: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        if self.include_bias {
            features.push(1.0);
        }

        // Generate all combinations up to degree
        self.generate_combinations(sample, &mut features, &mut Vec::new(), 0, 0);

        features
    }

    fn generate_combinations(
        &self,
        sample: &[f32],
        features: &mut Vec<f32>,
        current: &mut Vec<usize>,
        start: usize,
        current_degree: usize,
    ) {
        if current_degree > 0 {
            // Calculate product of current combination
            let mut product = 1.0;
            for &idx in current.iter() {
                product *= sample[idx];
            }
            features.push(product);
        }

        if current_degree >= self.degree {
            return;
        }

        for i in start..self.n_input_features {
            current.push(i);
            
            let next_start = if self.interaction_only { i + 1 } else { i };
            self.generate_combinations(sample, features, current, next_start, current_degree + 1);
            
            current.pop();
        }
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

/// Feature Hashing (Hashing Trick)
/// 
/// Maps features to a fixed-size feature space using hashing.
/// Useful for high-dimensional sparse features.
pub struct FeatureHasher {
    pub n_features: usize,
    pub alternate_sign: bool,
}

impl FeatureHasher {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            alternate_sign: true,
        }
    }

    pub fn alternate_sign(mut self, value: bool) -> Self {
        self.alternate_sign = value;
        self
    }

    /// Transform string features to hashed features
    pub fn transform_strings(&self, features: &[Vec<String>]) -> Tensor {
        let n_samples = features.len();
        let mut output = vec![0.0f32; n_samples * self.n_features];

        for (i, sample_features) in features.iter().enumerate() {
            for feature in sample_features {
                let hash = self.hash_feature(feature);
                let idx = (hash % self.n_features as u64) as usize;
                let sign = if self.alternate_sign && (hash / self.n_features as u64) % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                output[i * self.n_features + idx] += sign;
            }
        }

        Tensor::from_slice(&output, &[n_samples, self.n_features]).unwrap()
    }

    /// Transform feature-value pairs
    pub fn transform_pairs(&self, features: &[Vec<(String, f32)>]) -> Tensor {
        let n_samples = features.len();
        let mut output = vec![0.0f32; n_samples * self.n_features];

        for (i, sample_features) in features.iter().enumerate() {
            for (feature, value) in sample_features {
                let hash = self.hash_feature(feature);
                let idx = (hash % self.n_features as u64) as usize;
                let sign = if self.alternate_sign && (hash / self.n_features as u64) % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                output[i * self.n_features + idx] += sign * value;
            }
        }

        Tensor::from_slice(&output, &[n_samples, self.n_features]).unwrap()
    }

    fn hash_feature(&self, feature: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        feature.hash(&mut hasher);
        hasher.finish()
    }
}

/// Target Encoder
/// 
/// Encodes categorical features using target statistics.
/// Useful for high-cardinality categorical features.
pub struct TargetEncoder {
    pub smoothing: f32,
    pub min_samples_leaf: usize,
    encodings: HashMap<String, f32>,
    global_mean: f32,
}

impl TargetEncoder {
    pub fn new() -> Self {
        Self {
            smoothing: 1.0,
            min_samples_leaf: 1,
            encodings: HashMap::new(),
            global_mean: 0.0,
        }
    }

    pub fn smoothing(mut self, value: f32) -> Self {
        self.smoothing = value;
        self
    }

    pub fn min_samples_leaf(mut self, value: usize) -> Self {
        self.min_samples_leaf = value;
        self
    }

    /// Fit encoder on categorical features and target
    pub fn fit(&mut self, categories: &[String], target: &[f32]) {
        assert_eq!(categories.len(), target.len());

        // Calculate global mean
        self.global_mean = target.iter().sum::<f32>() / target.len() as f32;

        // Calculate category statistics
        let mut category_stats: HashMap<String, (f32, usize)> = HashMap::new();

        for (cat, &tgt) in categories.iter().zip(target.iter()) {
            let entry = category_stats.entry(cat.clone()).or_insert((0.0, 0));
            entry.0 += tgt;
            entry.1 += 1;
        }

        // Calculate smoothed encodings
        for (category, (sum, count)) in category_stats {
            if count >= self.min_samples_leaf {
                let category_mean = sum / count as f32;
                // Smoothing formula: (count * category_mean + smoothing * global_mean) / (count + smoothing)
                let smoothed = (count as f32 * category_mean + self.smoothing * self.global_mean) 
                    / (count as f32 + self.smoothing);
                self.encodings.insert(category, smoothed);
            }
        }
    }

    /// Transform categories to encoded values
    pub fn transform(&self, categories: &[String]) -> Vec<f32> {
        categories
            .iter()
            .map(|cat| {
                *self.encodings.get(cat).unwrap_or(&self.global_mean)
            })
            .collect()
    }

    pub fn fit_transform(&mut self, categories: &[String], target: &[f32]) -> Vec<f32> {
        self.fit(categories, target);
        self.transform(categories)
    }
}

/// One-Hot Encoder
/// 
/// Converts categorical features to one-hot encoded vectors.
pub struct OneHotEncoder {
    categories: Vec<Vec<String>>,
    n_features: usize,
}

impl OneHotEncoder {
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit encoder to learn categories
    pub fn fit(&mut self, data: &[Vec<String>]) {
        if data.is_empty() {
            return;
        }

        let n_cols = data[0].len();
        self.categories = vec![Vec::new(); n_cols];

        // Collect unique categories for each column
        for sample in data {
            for (col_idx, value) in sample.iter().enumerate() {
                if !self.categories[col_idx].contains(value) {
                    self.categories[col_idx].push(value.clone());
                }
            }
        }

        // Sort categories for consistency
        for cats in &mut self.categories {
            cats.sort();
        }

        // Calculate total number of features
        self.n_features = self.categories.iter().map(|cats| cats.len()).sum();
    }

    /// Transform categorical data to one-hot encoded
    pub fn transform(&self, data: &[Vec<String>]) -> Tensor {
        let n_samples = data.len();
        let mut output = vec![0.0f32; n_samples * self.n_features];

        for (sample_idx, sample) in data.iter().enumerate() {
            let mut feature_offset = 0;

            for (col_idx, value) in sample.iter().enumerate() {
                if let Some(cat_idx) = self.categories[col_idx].iter().position(|c| c == value) {
                    let output_idx = sample_idx * self.n_features + feature_offset + cat_idx;
                    output[output_idx] = 1.0;
                }
                feature_offset += self.categories[col_idx].len();
            }
        }

        Tensor::from_slice(&output, &[n_samples, self.n_features]).unwrap()
    }

    pub fn fit_transform(&mut self, data: &[Vec<String>]) -> Tensor {
        self.fit(data);
        self.transform(data)
    }

    /// Get feature names for the one-hot encoded output
    pub fn get_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        for (col_idx, cats) in self.categories.iter().enumerate() {
            for cat in cats {
                names.push(format!("col{}_{}", col_idx, cat));
            }
        }

        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_features() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        
        let mut poly = PolynomialFeatures::new(2);
        let transformed = poly.fit_transform(&x);

        // Should have more features than input
        assert!(transformed.dims()[1] > x.dims()[1]);
        assert_eq!(transformed.dims()[0], 2); // Same number of samples
    }

    #[test]
    fn test_feature_hasher() {
        let features = vec![
            vec!["feature1".to_string(), "feature2".to_string()],
            vec!["feature3".to_string()],
        ];

        let hasher = FeatureHasher::new(10);
        let hashed = hasher.transform_strings(&features);

        assert_eq!(hashed.dims(), &[2, 10]);
    }

    #[test]
    fn test_target_encoder() {
        let categories = vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
        ];
        let target = vec![1.0, 0.0, 1.0, 0.0];

        let mut encoder = TargetEncoder::new();
        let encoded = encoder.fit_transform(&categories, &target);

        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn test_one_hot_encoder() {
        let data = vec![
            vec!["A".to_string(), "X".to_string()],
            vec!["B".to_string(), "Y".to_string()],
            vec!["A".to_string(), "X".to_string()],
        ];

        let mut encoder = OneHotEncoder::new();
        let encoded = encoder.fit_transform(&data);

        // 2 categories in col0 + 2 categories in col1 = 4 features
        assert_eq!(encoded.dims()[1], 4);
    }
}



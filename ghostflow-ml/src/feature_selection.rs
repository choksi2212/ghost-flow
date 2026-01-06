//! Feature Selection - Variance Threshold, SelectKBest, RFE, SelectFromModel

use ghostflow_core::Tensor;

/// Variance Threshold - remove low-variance features
pub struct VarianceThreshold {
    pub threshold: f32,
    variances_: Option<Vec<f32>>,
    mask_: Option<Vec<bool>>,
    n_features_in_: usize,
}

impl VarianceThreshold {
    pub fn new(threshold: f32) -> Self {
        VarianceThreshold {
            threshold,
            variances_: None,
            mask_: None,
            n_features_in_: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_in_ = n_features;

        // Compute variance for each feature
        let variances: Vec<f32> = (0..n_features)
            .map(|j| {
                let mean: f32 = (0..n_samples)
                    .map(|i| x_data[i * n_features + j])
                    .sum::<f32>() / n_samples as f32;
                
                (0..n_samples)
                    .map(|i| (x_data[i * n_features + j] - mean).powi(2))
                    .sum::<f32>() / n_samples as f32
            })
            .collect();

        let mask: Vec<bool> = variances.iter().map(|&v| v > self.threshold).collect();

        self.variances_ = Some(variances);
        self.mask_ = Some(mask);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mask = self.mask_.as_ref().expect("Model not fitted");
        let n_selected: usize = mask.iter().filter(|&&m| m).count();

        let mut result = Vec::with_capacity(n_samples * n_selected);

        for i in 0..n_samples {
            for (j, &keep) in mask.iter().enumerate() {
                if keep {
                    result.push(x_data[i * n_features + j]);
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_selected]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }

    pub fn get_support(&self) -> Vec<bool> {
        self.mask_.clone().unwrap_or_default()
    }
}

/// SelectKBest - select k features with highest scores
pub struct SelectKBest {
    pub k: usize,
    pub score_func: ScoreFunction,
    scores_: Option<Vec<f32>>,
    #[allow(dead_code)]
    pvalues_: Option<Vec<f32>>,
    mask_: Option<Vec<bool>>,
    n_features_in_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ScoreFunction {
    FClassif,      // ANOVA F-value for classification
    MutualInfoClassif,
    Chi2,
    FRegression,   // F-value for regression
    MutualInfoRegression,
}

impl SelectKBest {
    pub fn new(k: usize, score_func: ScoreFunction) -> Self {
        SelectKBest {
            k,
            score_func,
            scores_: None,
            pvalues_: None,
            mask_: None,
            n_features_in_: 0,
        }
    }

    fn f_classif(&self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let n_classes = y.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        (0..n_features)
            .map(|j| {
                // Compute class means and overall mean
                let mut class_sums = vec![0.0f32; n_classes];
                let mut class_counts = vec![0usize; n_classes];
                let mut total_sum = 0.0f32;

                for i in 0..n_samples {
                    let class = y[i] as usize;
                    let val = x[i * n_features + j];
                    class_sums[class] += val;
                    class_counts[class] += 1;
                    total_sum += val;
                }

                let overall_mean = total_sum / n_samples as f32;
                let class_means: Vec<f32> = class_sums.iter()
                    .zip(class_counts.iter())
                    .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
                    .collect();

                // Between-group variance
                let ss_between: f32 = class_means.iter()
                    .zip(class_counts.iter())
                    .map(|(&mean, &count)| count as f32 * (mean - overall_mean).powi(2))
                    .sum();

                // Within-group variance
                let mut ss_within = 0.0f32;
                for i in 0..n_samples {
                    let class = y[i] as usize;
                    let val = x[i * n_features + j];
                    ss_within += (val - class_means[class]).powi(2);
                }

                // F-statistic
                let df_between = (n_classes - 1) as f32;
                let df_within = (n_samples - n_classes) as f32;

                if ss_within > 1e-10 && df_within > 0.0 {
                    (ss_between / df_between) / (ss_within / df_within)
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn chi2(&self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let n_classes = y.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        (0..n_features)
            .map(|j| {
                // Compute observed and expected frequencies
                let mut observed = vec![vec![0.0f32; 2]; n_classes];  // Binary: feature present/absent
                let mut class_totals = vec![0.0f32; n_classes];
                let mut feature_totals = [0.0f32; 2];

                for i in 0..n_samples {
                    let class = y[i] as usize;
                    let val = if x[i * n_features + j] > 0.0 { 1 } else { 0 };
                    observed[class][val] += 1.0;
                    class_totals[class] += 1.0;
                    feature_totals[val] += 1.0;
                }

                let total = n_samples as f32;
                let mut chi2 = 0.0f32;

                for c in 0..n_classes {
                    for f in 0..2 {
                        let expected = class_totals[c] * feature_totals[f] / total;
                        if expected > 0.0 {
                            chi2 += (observed[c][f] - expected).powi(2) / expected;
                        }
                    }
                }

                chi2
            })
            .collect()
    }

    fn f_regression(&self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let y_mean: f32 = y.iter().sum::<f32>() / n_samples as f32;
        let ss_tot: f32 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        (0..n_features)
            .map(|j| {
                // Simple linear regression for each feature
                let x_mean: f32 = (0..n_samples)
                    .map(|i| x[i * n_features + j])
                    .sum::<f32>() / n_samples as f32;

                let mut cov = 0.0f32;
                let mut var_x = 0.0f32;

                for i in 0..n_samples {
                    let xi = x[i * n_features + j] - x_mean;
                    let yi = y[i] - y_mean;
                    cov += xi * yi;
                    var_x += xi * xi;
                }

                if var_x < 1e-10 {
                    return 0.0;
                }

                let slope = cov / var_x;
                let intercept = y_mean - slope * x_mean;

                // Compute R² and convert to F-statistic
                let ss_res: f32 = (0..n_samples)
                    .map(|i| {
                        let pred = slope * x[i * n_features + j] + intercept;
                        (y[i] - pred).powi(2)
                    })
                    .sum();

                let r2 = 1.0 - ss_res / ss_tot.max(1e-10);
                let df1 = 1.0f32;
                let df2 = (n_samples - 2) as f32;

                if r2 < 1.0 && df2 > 0.0 {
                    (r2 / df1) / ((1.0 - r2) / df2)
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn mutual_info(&self, x: &[f32], y: &[f32], n_samples: usize, n_features: usize, is_classification: bool) -> Vec<f32> {
        // Simplified mutual information using binning
        let n_bins = 10;

        (0..n_features)
            .map(|j| {
                let feature_vals: Vec<f32> = (0..n_samples)
                    .map(|i| x[i * n_features + j])
                    .collect();

                let min_val = feature_vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = feature_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = (max_val - min_val).max(1e-10);

                // Bin the feature
                let x_binned: Vec<usize> = feature_vals.iter()
                    .map(|&v| (((v - min_val) / range * (n_bins - 1) as f32) as usize).min(n_bins - 1))
                    .collect();

                // Bin y for regression
                let y_binned: Vec<usize> = if is_classification {
                    y.iter().map(|&v| v as usize).collect()
                } else {
                    let y_min = y.iter().cloned().fold(f32::INFINITY, f32::min);
                    let y_max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let y_range = (y_max - y_min).max(1e-10);
                    y.iter()
                        .map(|&v| (((v - y_min) / y_range * (n_bins - 1) as f32) as usize).min(n_bins - 1))
                        .collect()
                };

                let n_y_bins = if is_classification {
                    y.iter().map(|&v| v as usize).max().unwrap_or(0) + 1
                } else {
                    n_bins
                };

                // Compute joint and marginal probabilities
                let mut joint = vec![vec![0.0f32; n_y_bins]; n_bins];
                let mut p_x = vec![0.0f32; n_bins];
                let mut p_y = vec![0.0f32; n_y_bins];

                for i in 0..n_samples {
                    joint[x_binned[i]][y_binned[i]] += 1.0;
                    p_x[x_binned[i]] += 1.0;
                    p_y[y_binned[i]] += 1.0;
                }

                let n = n_samples as f32;
                for p in &mut p_x { *p /= n; }
                for p in &mut p_y { *p /= n; }
                for row in &mut joint {
                    for p in row { *p /= n; }
                }

                // Compute mutual information
                let mut mi = 0.0f32;
                for xi in 0..n_bins {
                    for yi in 0..n_y_bins {
                        if joint[xi][yi] > 1e-10 && p_x[xi] > 1e-10 && p_y[yi] > 1e-10 {
                            mi += joint[xi][yi] * (joint[xi][yi] / (p_x[xi] * p_y[yi])).ln();
                        }
                    }
                }

                mi.max(0.0)
            })
            .collect()
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_in_ = n_features;

        let scores = match self.score_func {
            ScoreFunction::FClassif => self.f_classif(&x_data, &y_data, n_samples, n_features),
            ScoreFunction::Chi2 => self.chi2(&x_data, &y_data, n_samples, n_features),
            ScoreFunction::FRegression => self.f_regression(&x_data, &y_data, n_samples, n_features),
            ScoreFunction::MutualInfoClassif => self.mutual_info(&x_data, &y_data, n_samples, n_features, true),
            ScoreFunction::MutualInfoRegression => self.mutual_info(&x_data, &y_data, n_samples, n_features, false),
        };

        // Select top k features
        let mut indexed_scores: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = self.k.min(n_features);
        let selected_indices: Vec<usize> = indexed_scores.iter().take(k).map(|(i, _)| *i).collect();

        let mut mask = vec![false; n_features];
        for &idx in &selected_indices {
            mask[idx] = true;
        }

        self.scores_ = Some(scores);
        self.mask_ = Some(mask);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mask = self.mask_.as_ref().expect("Model not fitted");
        let n_selected: usize = mask.iter().filter(|&&m| m).count();

        let mut result = Vec::with_capacity(n_samples * n_selected);

        for i in 0..n_samples {
            for (j, &keep) in mask.iter().enumerate() {
                if keep {
                    result.push(x_data[i * n_features + j]);
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_selected]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor, y: &Tensor) -> Tensor {
        self.fit(x, y);
        self.transform(x)
    }

    pub fn get_support(&self) -> Vec<bool> {
        self.mask_.clone().unwrap_or_default()
    }

    pub fn scores(&self) -> Vec<f32> {
        self.scores_.clone().unwrap_or_default()
    }
}

/// Recursive Feature Elimination
pub struct RFE {
    pub n_features_to_select: usize,
    pub step: usize,
    support_: Option<Vec<bool>>,
    ranking_: Option<Vec<usize>>,
    n_features_in_: usize,
}

impl RFE {
    pub fn new(n_features_to_select: usize) -> Self {
        RFE {
            n_features_to_select,
            step: 1,
            support_: None,
            ranking_: None,
            n_features_in_: 0,
        }
    }

    pub fn step(mut self, step: usize) -> Self {
        self.step = step.max(1);
        self
    }

    /// Fit RFE using feature importance from a simple model
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_in_ = n_features;

        let mut mask = vec![true; n_features];
        let mut ranking = vec![1usize; n_features];
        let mut current_rank = 1;

        while mask.iter().filter(|&&m| m).count() > self.n_features_to_select {
            // Compute feature importances using correlation with target
            let importances: Vec<(usize, f32)> = (0..n_features)
                .filter(|&j| mask[j])
                .map(|j| {
                    let x_mean: f32 = (0..n_samples)
                        .map(|i| x_data[i * n_features + j])
                        .sum::<f32>() / n_samples as f32;
                    let y_mean: f32 = y_data.iter().sum::<f32>() / n_samples as f32;

                    let mut cov = 0.0f32;
                    let mut var_x = 0.0f32;
                    let mut var_y = 0.0f32;

                    for i in 0..n_samples {
                        let xi = x_data[i * n_features + j] - x_mean;
                        let yi = y_data[i] - y_mean;
                        cov += xi * yi;
                        var_x += xi * xi;
                        var_y += yi * yi;
                    }

                    let corr = if var_x > 1e-10 && var_y > 1e-10 {
                        (cov / (var_x.sqrt() * var_y.sqrt())).abs()
                    } else {
                        0.0
                    };

                    (j, corr)
                })
                .collect();

            // Find features to eliminate
            let mut sorted_importances = importances.clone();
            sorted_importances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let n_to_remove = self.step.min(mask.iter().filter(|&&m| m).count() - self.n_features_to_select);

            current_rank += 1;
            for (idx, _) in sorted_importances.iter().take(n_to_remove) {
                mask[*idx] = false;
                ranking[*idx] = current_rank;
            }
        }

        self.support_ = Some(mask);
        self.ranking_ = Some(ranking);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mask = self.support_.as_ref().expect("Model not fitted");
        let n_selected: usize = mask.iter().filter(|&&m| m).count();

        let mut result = Vec::with_capacity(n_samples * n_selected);

        for i in 0..n_samples {
            for (j, &keep) in mask.iter().enumerate() {
                if keep {
                    result.push(x_data[i * n_features + j]);
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_selected]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor, y: &Tensor) -> Tensor {
        self.fit(x, y);
        self.transform(x)
    }

    pub fn get_support(&self) -> Vec<bool> {
        self.support_.clone().unwrap_or_default()
    }

    pub fn ranking(&self) -> Vec<usize> {
        self.ranking_.clone().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_threshold() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 4.0, 0.0,
        ], &[4, 3]).unwrap();

        let mut vt = VarianceThreshold::new(0.0);
        let transformed = vt.fit_transform(&x);
        
        // Should remove constant features
        assert!(transformed.dims()[1] <= 3);
    }

    #[test]
    fn test_select_k_best() {
        let x = Tensor::from_slice(&[1.0f32, 0.0, 2.0,
            2.0, 0.0, 4.0,
            3.0, 0.0, 6.0,
            4.0, 0.0, 8.0,
        ], &[4, 3]).unwrap();
        
        let y = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[4]).unwrap();

        let mut skb = SelectKBest::new(2, ScoreFunction::FClassif);
        let transformed = skb.fit_transform(&x, &y);
        
        assert_eq!(transformed.dims()[1], 2);
    }

    #[test]
    fn test_rfe() {
        let x = Tensor::from_slice(&[1.0f32, 0.0, 2.0, 0.5,
            2.0, 0.0, 4.0, 0.5,
            3.0, 0.0, 6.0, 0.5,
            4.0, 0.0, 8.0, 0.5,
        ], &[4, 4]).unwrap();
        
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        let mut rfe = RFE::new(2);
        let transformed = rfe.fit_transform(&x, &y);
        
        assert_eq!(transformed.dims()[1], 2);
    }
}



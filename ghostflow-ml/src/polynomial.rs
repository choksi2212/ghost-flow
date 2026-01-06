//! Polynomial Features and Spline Transformers

use ghostflow_core::Tensor;

/// Polynomial Features - generate polynomial and interaction features
pub struct PolynomialFeatures {
    pub degree: usize,
    pub interaction_only: bool,
    pub include_bias: bool,
    n_input_features_: usize,
    n_output_features_: usize,
    powers_: Vec<Vec<usize>>,
}

impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self {
        PolynomialFeatures {
            degree,
            interaction_only: false,
            include_bias: true,
            n_input_features_: 0,
            n_output_features_: 0,
            powers_: Vec::new(),
        }
    }

    pub fn interaction_only(mut self, io: bool) -> Self {
        self.interaction_only = io;
        self
    }

    pub fn include_bias(mut self, ib: bool) -> Self {
        self.include_bias = ib;
        self
    }

    fn generate_powers(&mut self, n_features: usize) {
        self.powers_.clear();
        
        // Generate all combinations of powers
        fn generate_combinations(
            n_features: usize,
            degree: usize,
            interaction_only: bool,
            current: &mut Vec<usize>,
            start: usize,
            remaining_degree: usize,
            result: &mut Vec<Vec<usize>>,
        ) {
            if remaining_degree == 0 {
                result.push(current.clone());
                return;
            }

            for i in start..n_features {
                let max_power = if interaction_only { 1 } else { remaining_degree };
                for p in 1..=max_power {
                    if current.iter().sum::<usize>() + p <= degree {
                        current[i] += p;
                        generate_combinations(
                            n_features,
                            degree,
                            interaction_only,
                            current,
                            if interaction_only { i + 1 } else { i },
                            remaining_degree - p,
                            result,
                        );
                        current[i] -= p;
                    }
                }
            }
        }

        // Add bias term
        if self.include_bias {
            self.powers_.push(vec![0; n_features]);
        }

        // Add all polynomial terms
        for d in 1..=self.degree {
            let mut current = vec![0usize; n_features];
            generate_combinations(
                n_features,
                self.degree,
                self.interaction_only,
                &mut current,
                0,
                d,
                &mut self.powers_,
            );
        }

        self.n_input_features_ = n_features;
        self.n_output_features_ = self.powers_.len();
    }

    pub fn fit(&mut self, x: &Tensor) {
        let n_features = x.dims()[1];
        self.generate_powers(n_features);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut result = vec![0.0f32; n_samples * self.n_output_features_];

        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            
            for (j, powers) in self.powers_.iter().enumerate() {
                let mut val = 1.0f32;
                for (k, &p) in powers.iter().enumerate() {
                    if p > 0 {
                        val *= xi[k].powi(p as i32);
                    }
                }
                result[i * self.n_output_features_ + j] = val;
            }
        }

        Tensor::from_slice(&result, &[n_samples, self.n_output_features_]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }

    pub fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let default_names: Vec<String> = (0..self.n_input_features_)
            .map(|i| format!("x{}", i))
            .collect();
        let names = input_features.unwrap_or(&default_names);

        self.powers_.iter()
            .map(|powers| {
                let terms: Vec<String> = powers.iter()
                    .enumerate()
                    .filter(|(_, &p)| p > 0)
                    .map(|(i, &p)| {
                        if p == 1 {
                            names[i].clone()
                        } else {
                            format!("{}^{}", names[i], p)
                        }
                    })
                    .collect();
                
                if terms.is_empty() {
                    "1".to_string()
                } else {
                    terms.join(" ")
                }
            })
            .collect()
    }
}

/// Spline Transformer - B-spline basis functions
pub struct SplineTransformer {
    pub n_knots: usize,
    pub degree: usize,
    pub knots: KnotPositions,
    pub include_bias: bool,
    knots_: Option<Vec<Vec<f32>>>,
    n_features_: usize,
    n_splines_per_feature_: usize,
}

#[derive(Clone)]
pub enum KnotPositions {
    Uniform,
    Quantile,
}

impl SplineTransformer {
    pub fn new(n_knots: usize) -> Self {
        SplineTransformer {
            n_knots,
            degree: 3,
            knots: KnotPositions::Uniform,
            include_bias: true,
            knots_: None,
            n_features_: 0,
            n_splines_per_feature_: 0,
        }
    }

    pub fn degree(mut self, d: usize) -> Self {
        self.degree = d;
        self
    }

    fn compute_knots(&self, x: &[f32], n_samples: usize) -> Vec<f32> {
        let mut sorted: Vec<f32> = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let x_min = sorted[0];
        let x_max = sorted[n_samples - 1];

        match self.knots {
            KnotPositions::Uniform => {
                let step = (x_max - x_min) / (self.n_knots - 1) as f32;
                (0..self.n_knots).map(|i| x_min + i as f32 * step).collect()
            }
            KnotPositions::Quantile => {
                (0..self.n_knots)
                    .map(|i| {
                        let q = i as f32 / (self.n_knots - 1) as f32;
                        let idx = ((n_samples - 1) as f32 * q) as usize;
                        sorted[idx]
                    })
                    .collect()
            }
        }
    }

    fn bspline_basis(&self, x: f32, knots: &[f32], i: usize, k: usize) -> f32 {
        if k == 0 {
            if i < knots.len() - 1 && x >= knots[i] && x < knots[i + 1] {
                return 1.0;
            }
            if i == knots.len() - 2 && x == knots[i + 1] {
                return 1.0;
            }
            return 0.0;
        }

        let mut result = 0.0f32;

        if i + k < knots.len() {
            let denom1 = knots[i + k] - knots[i];
            if denom1.abs() > 1e-10 {
                result += (x - knots[i]) / denom1 * self.bspline_basis(x, knots, i, k - 1);
            }
        }

        if i + k + 1 < knots.len() {
            let denom2 = knots[i + k + 1] - knots[i + 1];
            if denom2.abs() > 1e-10 {
                result += (knots[i + k + 1] - x) / denom2 * self.bspline_basis(x, knots, i + 1, k - 1);
            }
        }

        result
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_features_ = n_features;
        self.n_splines_per_feature_ = self.n_knots + self.degree - 1;

        // Compute knots for each feature
        let mut all_knots = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let feature_values: Vec<f32> = (0..n_samples)
                .map(|i| x_data[i * n_features + j])
                .collect();
            
            let mut knots = self.compute_knots(&feature_values, n_samples);
            
            // Add boundary knots for B-spline
            let x_min = knots[0];
            let x_max = knots[knots.len() - 1];
            let step = (x_max - x_min) / (self.n_knots - 1) as f32;
            
            for i in 0..self.degree {
                knots.insert(0, x_min - (i + 1) as f32 * step);
                knots.push(x_max + (i + 1) as f32 * step);
            }
            
            all_knots.push(knots);
        }

        self.knots_ = Some(all_knots);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let knots = self.knots_.as_ref().expect("Not fitted");
        
        let n_output = if self.include_bias {
            n_features * self.n_splines_per_feature_
        } else {
            n_features * (self.n_splines_per_feature_ - 1)
        };

        let mut result = vec![0.0f32; n_samples * n_output];

        for i in 0..n_samples {
            let mut col = 0;
            for j in 0..n_features {
                let x_val = x_data[i * n_features + j];
                let feature_knots = &knots[j];
                
                let start_basis = if self.include_bias { 0 } else { 1 };
                for b in start_basis..self.n_splines_per_feature_ {
                    result[i * n_output + col] = self.bspline_basis(x_val, feature_knots, b, self.degree);
                    col += 1;
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_output]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

/// Power Transformer - Yeo-Johnson and Box-Cox transformations
pub struct PowerTransformer {
    pub method: PowerMethod,
    pub standardize: bool,
    lambdas_: Option<Vec<f32>>,
    mean_: Option<Vec<f32>>,
    std_: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
pub enum PowerMethod {
    YeoJohnson,
    BoxCox,
}

impl PowerTransformer {
    pub fn new(method: PowerMethod) -> Self {
        PowerTransformer {
            method,
            standardize: true,
            lambdas_: None,
            mean_: None,
            std_: None,
        }
    }

    fn yeo_johnson_transform(&self, x: f32, lambda: f32) -> f32 {
        if x >= 0.0 {
            if (lambda - 0.0).abs() < 1e-10 {
                (x + 1.0).ln()
            } else {
                ((x + 1.0).powf(lambda) - 1.0) / lambda
            }
        } else {
            if (lambda - 2.0).abs() < 1e-10 {
                -((-x + 1.0).ln())
            } else {
                -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
            }
        }
    }

    fn box_cox_transform(&self, x: f32, lambda: f32) -> f32 {
        if x <= 0.0 {
            return f32::NAN;
        }
        if (lambda - 0.0).abs() < 1e-10 {
            x.ln()
        } else {
            (x.powf(lambda) - 1.0) / lambda
        }
    }

    fn optimize_lambda(&self, x: &[f32]) -> f32 {
        // Grid search for optimal lambda
        let mut best_lambda = 1.0f32;
        let mut best_score = f32::NEG_INFINITY;

        for lambda_int in -20..=20 {
            let lambda = lambda_int as f32 * 0.1;
            
            let transformed: Vec<f32> = x.iter()
                .map(|&xi| match self.method {
                    PowerMethod::YeoJohnson => self.yeo_johnson_transform(xi, lambda),
                    PowerMethod::BoxCox => self.box_cox_transform(xi, lambda),
                })
                .collect();

            if transformed.iter().any(|&t| t.is_nan() || t.is_infinite()) {
                continue;
            }

            // Score based on normality (simplified: use negative variance of transformed data)
            let mean: f32 = transformed.iter().sum::<f32>() / transformed.len() as f32;
            let var: f32 = transformed.iter().map(|&t| (t - mean).powi(2)).sum::<f32>() / transformed.len() as f32;
            
            // Prefer lambda that gives variance close to 1
            let score = -(var - 1.0).abs();
            
            if score > best_score {
                best_score = score;
                best_lambda = lambda;
            }
        }

        best_lambda
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut lambdas = Vec::with_capacity(n_features);
        
        for j in 0..n_features {
            let feature_values: Vec<f32> = (0..n_samples)
                .map(|i| x_data[i * n_features + j])
                .collect();
            
            let lambda = self.optimize_lambda(&feature_values);
            lambdas.push(lambda);
        }

        self.lambdas_ = Some(lambdas);

        // Compute mean and std for standardization
        if self.standardize {
            let transformed = self.transform_internal(x);
            let t_data = transformed.data_f32();

            let mean: Vec<f32> = (0..n_features)
                .map(|j| (0..n_samples).map(|i| t_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
                .collect();

            let std: Vec<f32> = (0..n_features)
                .map(|j| {
                    let m = mean[j];
                    ((0..n_samples).map(|i| (t_data[i * n_features + j] - m).powi(2)).sum::<f32>() 
                        / n_samples as f32).sqrt().max(1e-10)
                })
                .collect();

            self.mean_ = Some(mean);
            self.std_ = Some(std);
        }
    }

    fn transform_internal(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let lambdas = self.lambdas_.as_ref().expect("Not fitted");

        let result: Vec<f32> = (0..n_samples)
            .flat_map(|i| {
                (0..n_features).map(|j| {
                    let xi = x_data[i * n_features + j];
                    match self.method {
                        PowerMethod::YeoJohnson => self.yeo_johnson_transform(xi, lambdas[j]),
                        PowerMethod::BoxCox => self.box_cox_transform(xi, lambdas[j]),
                    }
                }).collect::<Vec<_>>()
            })
            .collect();

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let transformed = self.transform_internal(x);
        
        if self.standardize {
            let t_data = transformed.data_f32();
            let n_samples = x.dims()[0];
            let n_features = x.dims()[1];
            let mean = self.mean_.as_ref().unwrap();
            let std = self.std_.as_ref().unwrap();

            let result: Vec<f32> = (0..n_samples)
                .flat_map(|i| {
                    (0..n_features).map(|j| {
                        (t_data[i * n_features + j] - mean[j]) / std[j]
                    }).collect::<Vec<_>>()
                })
                .collect();

            Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
        } else {
            transformed
        }
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

/// Quantile Transformer - transform features to follow a uniform or normal distribution
pub struct QuantileTransformer {
    pub n_quantiles: usize,
    pub output_distribution: OutputDistribution,
    quantiles_: Option<Vec<Vec<f32>>>,
    references_: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
pub enum OutputDistribution {
    Uniform,
    Normal,
}

impl QuantileTransformer {
    pub fn new() -> Self {
        QuantileTransformer {
            n_quantiles: 1000,
            output_distribution: OutputDistribution::Uniform,
            quantiles_: None,
            references_: None,
        }
    }

    pub fn n_quantiles(mut self, n: usize) -> Self {
        self.n_quantiles = n;
        self
    }

    pub fn output_distribution(mut self, od: OutputDistribution) -> Self {
        self.output_distribution = od;
        self
    }

    fn normal_ppf(&self, p: f32) -> f32 {
        // Approximation of inverse normal CDF
        if p <= 0.0 { return f32::NEG_INFINITY; }
        if p >= 1.0 { return f32::INFINITY; }
        
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let n_quantiles = self.n_quantiles.min(n_samples);

        // Compute quantiles for each feature
        let quantiles: Vec<Vec<f32>> = (0..n_features)
            .map(|j| {
                let mut values: Vec<f32> = (0..n_samples)
                    .map(|i| x_data[i * n_features + j])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                (0..n_quantiles)
                    .map(|q| {
                        let idx = (q as f32 / (n_quantiles - 1) as f32 * (n_samples - 1) as f32) as usize;
                        values[idx]
                    })
                    .collect()
            })
            .collect();

        // Reference quantiles for output distribution
        let references: Vec<f32> = (0..n_quantiles)
            .map(|q| {
                let p = q as f32 / (n_quantiles - 1) as f32;
                match self.output_distribution {
                    OutputDistribution::Uniform => p,
                    OutputDistribution::Normal => self.normal_ppf(p.clamp(0.001, 0.999)),
                }
            })
            .collect();

        self.quantiles_ = Some(quantiles);
        self.references_ = Some(references);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let quantiles = self.quantiles_.as_ref().expect("Not fitted");
        let references = self.references_.as_ref().unwrap();
        let n_quantiles = references.len();

        let result: Vec<f32> = (0..n_samples)
            .flat_map(|i| {
                (0..n_features).map(|j| {
                    let xi = x_data[i * n_features + j];
                    let q = &quantiles[j];

                    // Find position in quantiles
                    let pos = q.iter().position(|&qv| qv >= xi).unwrap_or(n_quantiles - 1);
                    
                    if pos == 0 {
                        references[0]
                    } else if pos >= n_quantiles {
                        references[n_quantiles - 1]
                    } else {
                        // Linear interpolation
                        let lower = q[pos - 1];
                        let upper = q[pos];
                        let t = if (upper - lower).abs() > 1e-10 {
                            (xi - lower) / (upper - lower)
                        } else {
                            0.5
                        };
                        references[pos - 1] + t * (references[pos] - references[pos - 1])
                    }
                }).collect::<Vec<_>>()
            })
            .collect();

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

impl Default for QuantileTransformer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_features() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut poly = PolynomialFeatures::new(2);
        let result = poly.fit_transform(&x);
        assert!(result.dims()[1] > 2);
    }

    #[test]
    fn test_power_transformer() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let mut pt = PowerTransformer::new(PowerMethod::YeoJohnson);
        let result = pt.fit_transform(&x);
        assert_eq!(result.dims(), &[3, 2]);
    }

    #[test]
    fn test_quantile_transformer() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let mut qt = QuantileTransformer::new();
        let result = qt.fit_transform(&x);
        assert_eq!(result.dims(), &[3, 2]);
    }
}



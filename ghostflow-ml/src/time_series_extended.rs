//! Extended Time Series - SARIMA, STL Decomposition, ACF/PACF

use ghostflow_core::Tensor;

/// SARIMA - Seasonal ARIMA
/// SARIMA(p, d, q)(P, D, Q, s)
pub struct SARIMA {
    pub p: usize,  // AR order
    pub d: usize,  // Differencing order
    pub q: usize,  // MA order
    pub seasonal_p: usize,  // Seasonal AR order
    pub seasonal_d: usize,  // Seasonal differencing order
    pub seasonal_q: usize,  // Seasonal MA order
    pub seasonal_period: usize,  // Seasonal period (s)
    pub max_iter: usize,
    ar_coef_: Option<Vec<f32>>,
    ma_coef_: Option<Vec<f32>>,
    seasonal_ar_coef_: Option<Vec<f32>>,
    seasonal_ma_coef_: Option<Vec<f32>>,
    intercept_: f32,
    diff_values_: Option<Vec<f32>>,
}

impl SARIMA {
    pub fn new(p: usize, d: usize, q: usize, seasonal_p: usize, seasonal_d: usize, 
               seasonal_q: usize, seasonal_period: usize) -> Self {
        SARIMA {
            p, d, q,
            seasonal_p, seasonal_d, seasonal_q, seasonal_period,
            max_iter: 100,
            ar_coef_: None,
            ma_coef_: None,
            seasonal_ar_coef_: None,
            seasonal_ma_coef_: None,
            intercept_: 0.0,
            diff_values_: None,
        }
    }

    fn difference(y: &[f32], d: usize) -> Vec<f32> {
        let mut result = y.to_vec();
        for _ in 0..d {
            let diff: Vec<f32> = result.windows(2).map(|w| w[1] - w[0]).collect();
            result = diff;
        }
        result
    }

    fn seasonal_difference(y: &[f32], d: usize, period: usize) -> Vec<f32> {
        let mut result = y.to_vec();
        for _ in 0..d {
            if result.len() <= period {
                break;
            }
            let diff: Vec<f32> = (period..result.len())
                .map(|i| result[i] - result[i - period])
                .collect();
            result = diff;
        }
        result
    }

    fn compute_autocorrelation(y: &[f32], lag: usize) -> f32 {
        let n = y.len();
        if lag >= n { return 0.0; }
        
        let mean = y.iter().sum::<f32>() / n as f32;
        let var: f32 = y.iter().map(|&v| (v - mean).powi(2)).sum();
        
        if var < 1e-10 { return 0.0; }
        
        let cov: f32 = (0..n - lag)
            .map(|i| (y[i] - mean) * (y[i + lag] - mean))
            .sum();
        
        cov / var
    }

    pub fn fit(&mut self, y: &Tensor) {
        let y_data = y.data_f32();
        self.diff_values_ = Some(y_data.clone());

        // Apply regular differencing
        let mut y_diff = Self::difference(&y_data, self.d);
        
        // Apply seasonal differencing
        y_diff = Self::seasonal_difference(&y_diff, self.seasonal_d, self.seasonal_period);

        if y_diff.len() <= self.p + self.seasonal_p * self.seasonal_period {
            panic!("Not enough data after differencing");
        }

        let n = y_diff.len();
        let mean = y_diff.iter().sum::<f32>() / n as f32;
        let y_centered: Vec<f32> = y_diff.iter().map(|&v| v - mean).collect();

        // Fit AR coefficients using Yule-Walker
        if self.p > 0 {
            let mut r = vec![0.0f32; self.p + 1];
            for k in 0..=self.p {
                r[k] = Self::compute_autocorrelation(&y_centered, k);
            }

            let mut toeplitz = vec![0.0f32; self.p * self.p];
            for i in 0..self.p {
                for j in 0..self.p {
                    toeplitz[i * self.p + j] = r[(i as i32 - j as i32).unsigned_abs() as usize];
                }
            }

            let rhs: Vec<f32> = r[1..=self.p].to_vec();
            self.ar_coef_ = Some(solve_linear(&toeplitz, &rhs, self.p));
        } else {
            self.ar_coef_ = Some(vec![]);
        }

        // Fit seasonal AR coefficients
        if self.seasonal_p > 0 {
            let mut r = vec![0.0f32; self.seasonal_p + 1];
            for k in 0..=self.seasonal_p {
                r[k] = Self::compute_autocorrelation(&y_centered, k * self.seasonal_period);
            }

            let mut toeplitz = vec![0.0f32; self.seasonal_p * self.seasonal_p];
            for i in 0..self.seasonal_p {
                for j in 0..self.seasonal_p {
                    toeplitz[i * self.seasonal_p + j] = r[(i as i32 - j as i32).unsigned_abs() as usize];
                }
            }

            let rhs: Vec<f32> = r[1..=self.seasonal_p].to_vec();
            self.seasonal_ar_coef_ = Some(solve_linear(&toeplitz, &rhs, self.seasonal_p));
        } else {
            self.seasonal_ar_coef_ = Some(vec![]);
        }

        // Initialize MA coefficients (simplified)
        self.ma_coef_ = Some(vec![0.0; self.q]);
        self.seasonal_ma_coef_ = Some(vec![0.0; self.seasonal_q]);
        self.intercept_ = mean;
    }

    pub fn predict(&self, steps: usize) -> Tensor {
        let ar_coef = self.ar_coef_.as_ref().expect("Model not fitted");
        let seasonal_ar_coef = self.seasonal_ar_coef_.as_ref().unwrap();
        let diff_values = self.diff_values_.as_ref().unwrap();

        // Get differenced series
        let mut y_diff = Self::difference(diff_values, self.d);
        y_diff = Self::seasonal_difference(&y_diff, self.seasonal_d, self.seasonal_period);

        let _n = y_diff.len();
        let mut history = y_diff.clone();
        let mut predictions = Vec::with_capacity(steps);

        for _ in 0..steps {
            let mut pred = self.intercept_;

            // AR component
            for (i, &coef) in ar_coef.iter().enumerate() {
                if i < history.len() {
                    pred += coef * history[history.len() - 1 - i];
                }
            }

            // Seasonal AR component
            for (i, &coef) in seasonal_ar_coef.iter().enumerate() {
                let lag = (i + 1) * self.seasonal_period;
                if lag <= history.len() {
                    pred += coef * history[history.len() - lag];
                }
            }

            predictions.push(pred);
            history.push(pred);
        }

        // Inverse differencing
        let final_pred = self.inverse_difference(&predictions, diff_values);

        Tensor::from_slice(&final_pred, &[steps]).unwrap()
    }

    fn inverse_difference(&self, predictions: &[f32], original: &[f32]) -> Vec<f32> {
        let mut result = predictions.to_vec();

        // Inverse seasonal differencing
        if self.seasonal_d > 0 {
            let n_orig = original.len();
            for _ in 0..self.seasonal_d {
                let mut undiff = Vec::new();
                for (i, &pred) in result.iter().enumerate() {
                    let base_idx = n_orig - self.seasonal_period + i % self.seasonal_period;
                    if base_idx < n_orig {
                        undiff.push(pred + original[base_idx]);
                    } else {
                        undiff.push(pred);
                    }
                }
                result = undiff;
            }
        }

        // Inverse regular differencing
        if self.d > 0 {
            let last_val = original[original.len() - 1];
            let mut undiff = vec![last_val];
            for &pred in &result {
                undiff.push(undiff.last().unwrap() + pred);
            }
            result = undiff[1..].to_vec();
        }

        result
    }
}

/// STL Decomposition - Seasonal and Trend decomposition using Loess
pub struct STLDecomposition {
    pub period: usize,
    pub seasonal_deg: usize,
    pub trend_deg: usize,
    pub robust: bool,
    pub n_iter: usize,
    trend_: Option<Vec<f32>>,
    seasonal_: Option<Vec<f32>>,
    residual_: Option<Vec<f32>>,
}

impl STLDecomposition {
    pub fn new(period: usize) -> Self {
        STLDecomposition {
            period,
            seasonal_deg: 1,
            trend_deg: 1,
            robust: false,
            n_iter: 2,
            trend_: None,
            seasonal_: None,
            residual_: None,
        }
    }

    pub fn robust(mut self, r: bool) -> Self { self.robust = r; self }

    fn loess_smooth(&self, y: &[f32], weights: &[f32], bandwidth: usize) -> Vec<f32> {
        let n = y.len();
        let mut smoothed = vec![0.0f32; n];

        for i in 0..n {
            let start = i.saturating_sub(bandwidth / 2);
            let end = (i + bandwidth / 2 + 1).min(n);

            let mut sum_w = 0.0f32;
            let mut sum_wy = 0.0f32;

            for j in start..end {
                let dist = (i as f32 - j as f32).abs() / (bandwidth as f32 / 2.0);
                let tricube = if dist < 1.0 { (1.0 - dist.powi(3)).powi(3) } else { 0.0 };
                let w = tricube * weights[j];
                sum_w += w;
                sum_wy += w * y[j];
            }

            smoothed[i] = if sum_w > 1e-10 { sum_wy / sum_w } else { y[i] };
        }

        smoothed
    }

    pub fn fit(&mut self, y: &Tensor) {
        let y_data = y.data_f32();
        let n = y_data.len();

        let mut trend = vec![0.0f32; n];
        let mut seasonal = vec![0.0f32; n];
        let mut weights = vec![1.0f32; n];

        for _ in 0..self.n_iter {
            // Step 1: Detrend
            let detrended: Vec<f32> = y_data.iter().zip(trend.iter())
                .map(|(&y, &t)| y - t)
                .collect();

            // Step 2: Compute seasonal component
            // Average by season
            let mut seasonal_means = vec![0.0f32; self.period];
            let mut seasonal_counts = vec![0usize; self.period];

            for (i, &val) in detrended.iter().enumerate() {
                let season = i % self.period;
                seasonal_means[season] += val;
                seasonal_counts[season] += 1;
            }

            for i in 0..self.period {
                if seasonal_counts[i] > 0 {
                    seasonal_means[i] /= seasonal_counts[i] as f32;
                }
            }

            // Center seasonal component
            let seasonal_mean: f32 = seasonal_means.iter().sum::<f32>() / self.period as f32;
            for s in &mut seasonal_means {
                *s -= seasonal_mean;
            }

            // Extend seasonal to full length
            for i in 0..n {
                seasonal[i] = seasonal_means[i % self.period];
            }

            // Step 3: Deseasonalize and compute trend
            let deseasonalized: Vec<f32> = y_data.iter().zip(seasonal.iter())
                .map(|(&y, &s)| y - s)
                .collect();

            // Smooth to get trend
            let trend_bandwidth = (n / 2).max(3) | 1; // Ensure odd
            trend = self.loess_smooth(&deseasonalized, &weights, trend_bandwidth);

            // Step 4: Update weights for robust fitting
            if self.robust {
                let residual: Vec<f32> = y_data.iter()
                    .zip(trend.iter())
                    .zip(seasonal.iter())
                    .map(|((&y, &t), &s)| y - t - s)
                    .collect();

                let mut abs_residual: Vec<f32> = residual.iter().map(|&r| r.abs()).collect();
                abs_residual.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let h = 6.0 * abs_residual[n / 2]; // 6 * MAD

                for (i, &r) in residual.iter().enumerate() {
                    let u = r.abs() / h.max(1e-10);
                    weights[i] = if u < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
                }
            }
        }

        // Compute residual
        let residual: Vec<f32> = y_data.iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((&y, &t), &s)| y - t - s)
            .collect();

        self.trend_ = Some(trend);
        self.seasonal_ = Some(seasonal);
        self.residual_ = Some(residual);
    }

    pub fn trend(&self) -> Option<&Vec<f32>> { self.trend_.as_ref() }
    pub fn seasonal(&self) -> Option<&Vec<f32>> { self.seasonal_.as_ref() }
    pub fn residual(&self) -> Option<&Vec<f32>> { self.residual_.as_ref() }
}

/// Autocorrelation Function
pub fn acf(y: &Tensor, n_lags: usize) -> Vec<f32> {
    let y_data = y.data_f32();
    let n = y_data.len();
    let mean = y_data.iter().sum::<f32>() / n as f32;
    let var: f32 = y_data.iter().map(|&v| (v - mean).powi(2)).sum();

    if var < 1e-10 {
        return vec![1.0; n_lags + 1];
    }

    (0..=n_lags)
        .map(|lag| {
            if lag >= n { return 0.0; }
            let cov: f32 = (0..n - lag)
                .map(|i| (y_data[i] - mean) * (y_data[i + lag] - mean))
                .sum();
            cov / var
        })
        .collect()
}

/// Partial Autocorrelation Function (using Durbin-Levinson)
pub fn pacf(y: &Tensor, n_lags: usize) -> Vec<f32> {
    let acf_values = acf(y, n_lags);
    let mut pacf_values = vec![0.0f32; n_lags + 1];
    pacf_values[0] = 1.0;

    if n_lags == 0 { return pacf_values; }

    pacf_values[1] = acf_values[1];

    let mut phi = vec![vec![0.0f32; n_lags + 1]; n_lags + 1];
    phi[1][1] = acf_values[1];

    for k in 2..=n_lags {
        let mut num = acf_values[k];
        let mut den = 1.0f32;

        for j in 1..k {
            num -= phi[k - 1][j] * acf_values[k - j];
            den -= phi[k - 1][j] * acf_values[j];
        }

        phi[k][k] = if den.abs() > 1e-10 { num / den } else { 0.0 };
        pacf_values[k] = phi[k][k];

        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
    }

    pacf_values
}

/// Augmented Dickey-Fuller Test for stationarity
pub fn adf_test(y: &Tensor, max_lag: Option<usize>) -> (f32, f32) {
    let y_data = y.data_f32();
    let n = y_data.len();

    // Compute first difference
    let dy: Vec<f32> = y_data.windows(2).map(|w| w[1] - w[0]).collect();
    let n_diff = dy.len();

    // Determine lag order
    let lag = max_lag.unwrap_or(((n as f32).powf(1.0 / 3.0) * 2.0) as usize);
    let lag = lag.min(n_diff / 2);

    // Build regression: dy_t = alpha + beta * y_{t-1} + sum(gamma_i * dy_{t-i}) + e_t
    // We're interested in testing beta = 0

    // Simplified: just compute the t-statistic for the coefficient on y_{t-1}
    let start = lag + 1;
    if start >= n_diff { return (0.0, 1.0); }

    let n_obs = n_diff - start;
    
    // Compute OLS estimate of beta
    let mut sum_xy = 0.0f32;
    let mut sum_xx = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_x = 0.0f32;

    for i in start..n_diff {
        let x = y_data[i]; // y_{t-1}
        let y = dy[i];
        sum_xy += x * y;
        sum_xx += x * x;
        sum_y += y;
        sum_x += x;
    }

    let mean_x = sum_x / n_obs as f32;
    let mean_y = sum_y / n_obs as f32;
    
    let beta = (sum_xy - n_obs as f32 * mean_x * mean_y) / 
               (sum_xx - n_obs as f32 * mean_x * mean_x).max(1e-10);

    // Compute residual variance
    let mut sse = 0.0f32;
    for i in start..n_diff {
        let x = y_data[i];
        let y = dy[i];
        let pred = mean_y + beta * (x - mean_x);
        sse += (y - pred).powi(2);
    }
    let mse = sse / (n_obs - 2).max(1) as f32;

    // Standard error of beta
    let se_beta = (mse / (sum_xx - n_obs as f32 * mean_x * mean_x).max(1e-10)).sqrt();

    // t-statistic
    let t_stat = beta / se_beta.max(1e-10);

    // Critical values (approximate for 5% significance)
    // ADF critical value at 5% is approximately -2.86 for n > 100
    let critical_value = -2.86f32;
    let p_value = if t_stat < critical_value { 0.01 } else { 0.10 };

    (t_stat, p_value)
}

fn solve_linear(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    if n == 0 { return vec![]; }
    
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
            let tmp = aug[i * (n + 1) + j];
            aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j];
            aug[max_row * (n + 1) + j] = tmp;
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
    fn test_sarima() {
        let y = Tensor::from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        ], &[24]).unwrap();

        let mut sarima = SARIMA::new(1, 0, 0, 1, 0, 0, 12);
        sarima.fit(&y);
        let pred = sarima.predict(3);
        assert_eq!(pred.dims(), &[3]);
    }

    #[test]
    fn test_stl() {
        let y = Tensor::from_slice(&[
            1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
        ], &[12]).unwrap();

        let mut stl = STLDecomposition::new(4);
        stl.fit(&y);
        
        assert!(stl.trend().is_some());
        assert!(stl.seasonal().is_some());
        assert!(stl.residual().is_some());
    }

    #[test]
    fn test_acf_pacf() {
        let y = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0], &[10]).unwrap();
        
        let acf_vals = acf(&y, 5);
        assert_eq!(acf_vals.len(), 6);
        assert!((acf_vals[0] - 1.0).abs() < 1e-5);

        let pacf_vals = pacf(&y, 5);
        assert_eq!(pacf_vals.len(), 6);
    }
}



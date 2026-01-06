//! Time Series - ARIMA basics, Exponential Smoothing

use ghostflow_core::Tensor;

/// Simple Exponential Smoothing
pub struct SimpleExponentialSmoothing {
    pub alpha: f32,
    level_: Option<f32>,
    fitted_: Option<Vec<f32>>,
}

impl SimpleExponentialSmoothing {
    pub fn new(alpha: f32) -> Self {
        SimpleExponentialSmoothing {
            alpha: alpha.clamp(0.0, 1.0),
            level_: None,
            fitted_: None,
        }
    }

    pub fn fit(&mut self, y: &Tensor) {
        let y_data = y.data_f32();
        let n = y_data.len();

        let mut level = y_data[0];
        let mut fitted = vec![level];

        for i in 1..n {
            level = self.alpha * y_data[i] + (1.0 - self.alpha) * level;
            fitted.push(level);
        }

        self.level_ = Some(level);
        self.fitted_ = Some(fitted);
    }

    pub fn predict(&self, steps: usize) -> Tensor {
        let level = self.level_.expect("Model not fitted");
        let predictions = vec![level; steps];
        Tensor::from_slice(&predictions, &[steps]).unwrap()
    }

    pub fn fitted_values(&self) -> Option<&Vec<f32>> {
        self.fitted_.as_ref()
    }
}

/// Holt's Linear Trend (Double Exponential Smoothing)
pub struct HoltLinear {
    pub alpha: f32,
    pub beta: f32,
    level_: Option<f32>,
    trend_: Option<f32>,
    fitted_: Option<Vec<f32>>,
}

impl HoltLinear {
    pub fn new(alpha: f32, beta: f32) -> Self {
        HoltLinear {
            alpha: alpha.clamp(0.0, 1.0),
            beta: beta.clamp(0.0, 1.0),
            level_: None,
            trend_: None,
            fitted_: None,
        }
    }

    pub fn fit(&mut self, y: &Tensor) {
        let y_data = y.data_f32();
        let n = y_data.len();

        let mut level = y_data[0];
        let mut trend = if n > 1 { y_data[1] - y_data[0] } else { 0.0 };
        let mut fitted = vec![level];

        for i in 1..n {
            let prev_level = level;
            level = self.alpha * y_data[i] + (1.0 - self.alpha) * (level + trend);
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
            fitted.push(level + trend);
        }

        self.level_ = Some(level);
        self.trend_ = Some(trend);
        self.fitted_ = Some(fitted);
    }

    pub fn predict(&self, steps: usize) -> Tensor {
        let level = self.level_.expect("Model not fitted");
        let trend = self.trend_.unwrap();

        let predictions: Vec<f32> = (1..=steps)
            .map(|h| level + h as f32 * trend)
            .collect();

        Tensor::from_slice(&predictions, &[steps]).unwrap()
    }
}

/// Holt-Winters (Triple Exponential Smoothing)
pub struct HoltWinters {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub seasonal_periods: usize,
    pub seasonal: SeasonalType,
    level_: Option<f32>,
    trend_: Option<f32>,
    seasonal_: Option<Vec<f32>>,
    fitted_: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
pub enum SeasonalType {
    Additive,
    Multiplicative,
}

impl HoltWinters {
    pub fn new(seasonal_periods: usize) -> Self {
        HoltWinters {
            alpha: 0.2,
            beta: 0.1,
            gamma: 0.1,
            seasonal_periods,
            seasonal: SeasonalType::Additive,
            level_: None,
            trend_: None,
            seasonal_: None,
            fitted_: None,
        }
    }

    pub fn alpha(mut self, a: f32) -> Self { self.alpha = a.clamp(0.0, 1.0); self }
    pub fn beta(mut self, b: f32) -> Self { self.beta = b.clamp(0.0, 1.0); self }
    pub fn gamma(mut self, g: f32) -> Self { self.gamma = g.clamp(0.0, 1.0); self }

    pub fn fit(&mut self, y: &Tensor) {
        let y_data = y.data_f32();
        let n = y_data.len();
        let m = self.seasonal_periods;

        if n < 2 * m {
            panic!("Need at least 2 seasonal periods of data");
        }

        // Initialize level and trend
        let mut level: f32 = y_data[..m].iter().sum::<f32>() / m as f32;
        let mut trend = (y_data[m..2*m].iter().sum::<f32>() - y_data[..m].iter().sum::<f32>()) 
            / (m * m) as f32;

        // Initialize seasonal components
        let mut seasonal: Vec<f32> = match self.seasonal {
            SeasonalType::Additive => {
                (0..m).map(|i| y_data[i] - level).collect()
            }
            SeasonalType::Multiplicative => {
                (0..m).map(|i| y_data[i] / level.max(1e-10)).collect()
            }
        };

        let mut fitted = Vec::with_capacity(n);

        for i in 0..n {
            let s_idx = i % m;
            
            let forecast = match self.seasonal {
                SeasonalType::Additive => level + trend + seasonal[s_idx],
                SeasonalType::Multiplicative => (level + trend) * seasonal[s_idx],
            };
            fitted.push(forecast);

            if i >= m {
                let prev_level = level;
                
                match self.seasonal {
                    SeasonalType::Additive => {
                        level = self.alpha * (y_data[i] - seasonal[s_idx]) 
                            + (1.0 - self.alpha) * (level + trend);
                        trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
                        seasonal[s_idx] = self.gamma * (y_data[i] - level) 
                            + (1.0 - self.gamma) * seasonal[s_idx];
                    }
                    SeasonalType::Multiplicative => {
                        level = self.alpha * (y_data[i] / seasonal[s_idx].max(1e-10)) 
                            + (1.0 - self.alpha) * (level + trend);
                        trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
                        seasonal[s_idx] = self.gamma * (y_data[i] / level.max(1e-10)) 
                            + (1.0 - self.gamma) * seasonal[s_idx];
                    }
                }
            }
        }

        self.level_ = Some(level);
        self.trend_ = Some(trend);
        self.seasonal_ = Some(seasonal);
        self.fitted_ = Some(fitted);
    }

    pub fn predict(&self, steps: usize) -> Tensor {
        let level = self.level_.expect("Model not fitted");
        let trend = self.trend_.unwrap();
        let seasonal = self.seasonal_.as_ref().unwrap();
        let m = self.seasonal_periods;

        let predictions: Vec<f32> = (1..=steps)
            .map(|h| {
                let s_idx = (h - 1) % m;
                match self.seasonal {
                    SeasonalType::Additive => level + h as f32 * trend + seasonal[s_idx],
                    SeasonalType::Multiplicative => (level + h as f32 * trend) * seasonal[s_idx],
                }
            })
            .collect();

        Tensor::from_slice(&predictions, &[steps]).unwrap()
    }
}

/// Simple ARIMA (AutoRegressive Integrated Moving Average)
/// Simplified implementation for AR(p) model
pub struct ARIMA {
    pub p: usize, // AR order
    pub d: usize, // Differencing order
    pub q: usize, // MA order (simplified, not fully implemented)
    pub max_iter: usize,
    ar_coef_: Option<Vec<f32>>,
    intercept_: f32,
    diff_values_: Option<Vec<f32>>,
}

impl ARIMA {
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        ARIMA {
            p,
            d,
            q,
            max_iter: 100,
            ar_coef_: None,
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

    fn undifference(diff: &[f32], original: &[f32], d: usize) -> Vec<f32> {
        if d == 0 {
            return diff.to_vec();
        }

        let mut result = diff.to_vec();
        for i in (0..d).rev() {
            let last_val = original[original.len() - d + i];
            let mut undiff = vec![last_val];
            for &v in &result {
                undiff.push(undiff.last().unwrap() + v);
            }
            result = undiff[1..].to_vec();
        }
        result
    }

    pub fn fit(&mut self, y: &Tensor) {
        let y_data = y.data_f32();
        
        // Apply differencing
        let y_diff = Self::difference(&y_data, self.d);
        self.diff_values_ = Some(y_data.clone());

        if y_diff.len() <= self.p {
            panic!("Not enough data after differencing");
        }

        // Fit AR model using Yule-Walker equations
        let n = y_diff.len();
        let mean = y_diff.iter().sum::<f32>() / n as f32;
        let y_centered: Vec<f32> = y_diff.iter().map(|&v| v - mean).collect();

        // Compute autocorrelations
        let mut r = vec![0.0f32; self.p + 1];
        for k in 0..=self.p {
            for i in k..n {
                r[k] += y_centered[i] * y_centered[i - k];
            }
            r[k] /= n as f32;
        }

        // Solve Yule-Walker equations
        if self.p > 0 {
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

        self.intercept_ = mean;
    }

    pub fn predict(&self, steps: usize) -> Tensor {
        let ar_coef = self.ar_coef_.as_ref().expect("Model not fitted");
        let diff_values = self.diff_values_.as_ref().unwrap();
        
        let y_diff = Self::difference(diff_values, self.d);
        let n = y_diff.len();

        let mut predictions = Vec::with_capacity(steps);
        let mut history: Vec<f32> = y_diff[n.saturating_sub(self.p)..].to_vec();

        for _ in 0..steps {
            let mut pred = self.intercept_;
            for (i, &coef) in ar_coef.iter().enumerate() {
                if i < history.len() {
                    pred += coef * history[history.len() - 1 - i];
                }
            }
            predictions.push(pred);
            history.push(pred);
        }

        // Undifference
        let final_pred = Self::undifference(&predictions, diff_values, self.d);

        Tensor::from_slice(&final_pred, &[steps]).unwrap()
    }
}

/// Moving Average
pub struct MovingAverage {
    pub window: usize,
}

impl MovingAverage {
    pub fn new(window: usize) -> Self {
        MovingAverage { window }
    }

    pub fn transform(&self, y: &Tensor) -> Tensor {
        let y_data = y.data_f32();
        let n = y_data.len();

        let mut result = Vec::with_capacity(n);
        
        for i in 0..n {
            let start = i.saturating_sub(self.window - 1);
            let sum: f32 = y_data[start..=i].iter().sum();
            let count = i - start + 1;
            result.push(sum / count as f32);
        }

        Tensor::from_slice(&result, &[n]).unwrap()
    }
}

/// Exponentially Weighted Moving Average
pub struct EWMA {
    pub span: usize,
}

impl EWMA {
    pub fn new(span: usize) -> Self {
        EWMA { span }
    }

    pub fn transform(&self, y: &Tensor) -> Tensor {
        let y_data = y.data_f32();
        let n = y_data.len();
        let alpha = 2.0 / (self.span as f32 + 1.0);

        let mut result = vec![y_data[0]];
        
        for i in 1..n {
            let ewma = alpha * y_data[i] + (1.0 - alpha) * result[i - 1];
            result.push(ewma);
        }

        Tensor::from_slice(&result, &[n]).unwrap()
    }
}

fn solve_linear(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut aug = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Gaussian elimination
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
    fn test_simple_exp_smoothing() {
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let mut ses = SimpleExponentialSmoothing::new(0.3);
        ses.fit(&y);
        let pred = ses.predict(3);
        assert_eq!(pred.dims(), &[3]);
    }

    #[test]
    fn test_holt_linear() {
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let mut holt = HoltLinear::new(0.3, 0.1);
        holt.fit(&y);
        let pred = holt.predict(3);
        assert_eq!(pred.dims(), &[3]);
    }

    #[test]
    fn test_moving_average() {
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let ma = MovingAverage::new(3);
        let result = ma.transform(&y);
        assert_eq!(result.dims(), &[5]);
    }
}



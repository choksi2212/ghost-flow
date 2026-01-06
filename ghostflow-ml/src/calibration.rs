//! Probability Calibration - Isotonic Regression, Platt Scaling

use ghostflow_core::Tensor;

/// Isotonic Regression - monotonic regression
#[derive(Clone)]
pub struct IsotonicRegression {
    pub increasing: bool,
    pub out_of_bounds: OutOfBounds,
    x_thresholds_: Option<Vec<f32>>,
    y_thresholds_: Option<Vec<f32>>,
    x_min_: f32,
    x_max_: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum OutOfBounds {
    Nan,
    Clip,
    Raise,
}

impl IsotonicRegression {
    pub fn new() -> Self {
        IsotonicRegression {
            increasing: true,
            out_of_bounds: OutOfBounds::Clip,
            x_thresholds_: None,
            y_thresholds_: None,
            x_min_: 0.0,
            x_max_: 1.0,
        }
    }

    pub fn increasing(mut self, inc: bool) -> Self {
        self.increasing = inc;
        self
    }

    /// Pool Adjacent Violators Algorithm (PAVA)
    fn pava(&self, y: &[f32], weights: &[f32]) -> Vec<f32> {
        let n = y.len();
        if n == 0 {
            return vec![];
        }

        let mut result = y.to_vec();
        let mut w = weights.to_vec();

        // Pool adjacent violators
        loop {
            let mut changed = false;
            let mut i = 0;

            while i < result.len() - 1 {
                let violates = if self.increasing {
                    result[i] > result[i + 1]
                } else {
                    result[i] < result[i + 1]
                };

                if violates {
                    // Pool blocks
                    let new_val = (w[i] * result[i] + w[i + 1] * result[i + 1]) / (w[i] + w[i + 1]);
                    let new_weight = w[i] + w[i + 1];

                    result[i] = new_val;
                    w[i] = new_weight;
                    result.remove(i + 1);
                    w.remove(i + 1);

                    changed = true;
                } else {
                    i += 1;
                }
            }

            if !changed {
                break;
            }
        }

        result
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x_data.len();

        // Sort by x
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&a, &b| x_data[a].partial_cmp(&x_data[b]).unwrap());

        let x_sorted: Vec<f32> = indices.iter().map(|&i| x_data[i]).collect();
        let y_sorted: Vec<f32> = indices.iter().map(|&i| y_data[i]).collect();
        let weights = vec![1.0f32; n_samples];

        self.x_min_ = x_sorted[0];
        self.x_max_ = x_sorted[n_samples - 1];

        // Apply PAVA
        let y_isotonic = self.pava(&y_sorted, &weights);

        // Build piecewise constant function
        // Group consecutive x values with same y
        let mut x_thresholds = Vec::new();
        let mut y_thresholds = Vec::new();

        let mut i = 0;
        let mut iso_idx = 0;
        
        while i < n_samples && iso_idx < y_isotonic.len() {
            x_thresholds.push(x_sorted[i]);
            y_thresholds.push(y_isotonic[iso_idx]);

            // Find how many original points map to this isotonic value
            let mut count = 1;
            while i + count < n_samples && iso_idx < y_isotonic.len() {
                if count >= y_isotonic.len() - iso_idx {
                    break;
                }
                count += 1;
            }
            
            i += 1;
            if i < n_samples {
                iso_idx = (iso_idx + 1).min(y_isotonic.len() - 1);
            }
        }

        // Simplify: keep unique x values
        let mut final_x = vec![x_thresholds[0]];
        let mut final_y = vec![y_thresholds[0]];

        for i in 1..x_thresholds.len() {
            if (x_thresholds[i] - final_x.last().unwrap()).abs() > 1e-10 {
                final_x.push(x_thresholds[i]);
                final_y.push(y_thresholds[i]);
            }
        }

        self.x_thresholds_ = Some(final_x);
        self.y_thresholds_ = Some(final_y);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x_data.len();

        let x_thresh = self.x_thresholds_.as_ref().expect("Model not fitted");
        let y_thresh = self.y_thresholds_.as_ref().expect("Model not fitted");

        let predictions: Vec<f32> = x_data.iter()
            .map(|&xi| {
                // Handle out of bounds
                if xi < self.x_min_ {
                    match self.out_of_bounds {
                        OutOfBounds::Clip => return y_thresh[0],
                        OutOfBounds::Nan => return f32::NAN,
                        OutOfBounds::Raise => return y_thresh[0],
                    }
                }
                if xi > self.x_max_ {
                    match self.out_of_bounds {
                        OutOfBounds::Clip => return *y_thresh.last().unwrap(),
                        OutOfBounds::Nan => return f32::NAN,
                        OutOfBounds::Raise => return *y_thresh.last().unwrap(),
                    }
                }

                // Binary search for interval
                let mut lo = 0;
                let mut hi = x_thresh.len() - 1;

                while lo < hi {
                    let mid = (lo + hi + 1) / 2;
                    if x_thresh[mid] <= xi {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }

                // Linear interpolation
                if lo < x_thresh.len() - 1 {
                    let x0 = x_thresh[lo];
                    let x1 = x_thresh[lo + 1];
                    let y0 = y_thresh[lo];
                    let y1 = y_thresh[lo + 1];

                    if (x1 - x0).abs() > 1e-10 {
                        y0 + (y1 - y0) * (xi - x0) / (x1 - x0)
                    } else {
                        y0
                    }
                } else {
                    y_thresh[lo]
                }
            })
            .collect();

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor, y: &Tensor) -> Tensor {
        self.fit(x, y);
        self.transform(x)
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        self.transform(x)
    }
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Platt Scaling for probability calibration
#[derive(Clone)]
pub struct PlattScaling {
    pub max_iter: usize,
    pub tol: f32,
    a_: Option<f32>,
    b_: Option<f32>,
}

impl PlattScaling {
    pub fn new() -> Self {
        PlattScaling {
            max_iter: 100,
            tol: 1e-5,
            a_: None,
            b_: None,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }

    pub fn fit(&mut self, scores: &Tensor, y: &Tensor) {
        let scores_data = scores.data_f32();
        let y_data = y.data_f32();
        let n_samples = scores_data.len();

        // Target probabilities with Laplace smoothing
        let n_pos = y_data.iter().filter(|&&yi| yi > 0.5).count();
        let n_neg = n_samples - n_pos;

        let t_pos = (n_pos as f32 + 1.0) / (n_pos as f32 + 2.0);
        let t_neg = 1.0 / (n_neg as f32 + 2.0);

        let targets: Vec<f32> = y_data.iter()
            .map(|&yi| if yi > 0.5 { t_pos } else { t_neg })
            .collect();

        // Initialize A and B
        let mut a = 0.0f32;
        let mut b = (((n_neg + 1) as f32) / ((n_pos + 1) as f32)).ln();

        // Newton's method
        let min_step = 1e-10;
        let sigma = 1e-12;

        for _ in 0..self.max_iter {
            // Compute probabilities
            let probs: Vec<f32> = scores_data.iter()
                .map(|&s| Self::sigmoid(a * s + b))
                .collect();

            // Compute current loss
            let mut loss = 0.0f32;
            for i in 0..n_samples {
                let p = probs[i];
                loss -= targets[i] * p.max(1e-10).ln() + (1.0 - targets[i]) * (1.0 - p).max(1e-10).ln();
            }

            // Compute gradient and Hessian
            let mut d1a = 0.0f32;
            let mut d1b = 0.0f32;
            let mut d2a = 0.0f32;
            let mut d2b = 0.0f32;
            let mut d2ab = 0.0f32;

            for i in 0..n_samples {
                let p = probs[i];
                let t = targets[i];
                let d1 = p - t;
                let d2 = p * (1.0 - p);

                d1a += scores_data[i] * d1;
                d1b += d1;
                d2a += scores_data[i] * scores_data[i] * d2;
                d2b += d2;
                d2ab += scores_data[i] * d2;
            }

            // Add regularization
            d2a += sigma;
            d2b += sigma;

            // Solve 2x2 system
            let det = d2a * d2b - d2ab * d2ab;
            if det.abs() < 1e-10 {
                break;
            }

            let da = -(d2b * d1a - d2ab * d1b) / det;
            let db = -(-d2ab * d1a + d2a * d1b) / det;

            // Line search with backtracking
            let mut step = 1.0f32;
            let mut accepted = false;
            while step > min_step {
                let new_a = a + step * da;
                let new_b = b + step * db;

                // Compute new loss
                let mut new_loss = 0.0f32;
                for i in 0..n_samples {
                    let p = Self::sigmoid(new_a * scores_data[i] + new_b);
                    new_loss -= targets[i] * p.max(1e-10).ln() + (1.0 - targets[i]) * (1.0 - p).max(1e-10).ln();
                }

                // Accept if loss decreased
                if new_loss < loss {
                    a = new_a;
                    b = new_b;
                    accepted = true;
                    break;
                }
                
                // Reduce step size
                step *= 0.5;
            }
            
            // If no step accepted, use the update anyway (gradient descent step)
            if !accepted {
                a += 0.01 * da;
                b += 0.01 * db;
            }

            // Check convergence
            if da.abs() < self.tol && db.abs() < self.tol {
                break;
            }
        }

        self.a_ = Some(a);
        self.b_ = Some(b);
    }

    pub fn transform(&self, scores: &Tensor) -> Tensor {
        let scores_data = scores.data_f32();
        let n_samples = scores_data.len();

        let a = self.a_.expect("Model not fitted");
        let b = self.b_.expect("Model not fitted");

        let probs: Vec<f32> = scores_data.iter()
            .map(|&s| Self::sigmoid(a * s + b))
            .collect();

        Tensor::from_slice(&probs, &[n_samples]).unwrap()
    }

    pub fn fit_transform(&mut self, scores: &Tensor, y: &Tensor) -> Tensor {
        self.fit(scores, y);
        self.transform(scores)
    }
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibrated Classifier wrapper
pub struct CalibratedClassifier {
    pub method: CalibrationMethod,
    pub cv: usize,
    #[allow(dead_code)]
    calibrators_: Vec<CalibrationMethod>,
}

#[derive(Clone)]
pub enum CalibrationMethod {
    Sigmoid(PlattScaling),
    Isotonic(IsotonicRegression),
}

impl CalibratedClassifier {
    pub fn new(method: CalibrationMethod) -> Self {
        CalibratedClassifier {
            method,
            cv: 5,
            calibrators_: Vec::new(),
        }
    }

    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isotonic_regression() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let y = Tensor::from_slice(&[1.0f32, 3.0, 2.0, 4.0, 5.0], &[5]).unwrap();

        let mut ir = IsotonicRegression::new();
        let transformed = ir.fit_transform(&x, &y);
        
        assert_eq!(transformed.dims(), &[5]);
    }

    #[test]
    fn test_platt_scaling() {
        let scores = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0, 1.0], &[5]).unwrap();

        let mut ps = PlattScaling::new();
        let probs = ps.fit_transform(&scores, &y);
        
        let probs_data = probs.storage().as_slice::<f32>().to_vec();
        // Probabilities should be between 0 and 1
        assert!(probs_data.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }
}



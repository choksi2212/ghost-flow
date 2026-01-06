//! Hyperparameter Optimization
//!
//! Advanced algorithms for finding optimal hyperparameters.

use rand::prelude::*;
use std::collections::HashMap;

/// Parameter space definition
#[derive(Clone, Debug)]
pub enum ParameterSpace {
    Continuous { min: f32, max: f32, log_scale: bool },
    Integer { min: i32, max: i32 },
    Categorical { choices: Vec<String> },
}

/// Hyperparameter configuration
pub type Configuration = HashMap<String, ParameterValue>;

#[derive(Clone, Debug)]
pub enum ParameterValue {
    Float(f32),
    Int(i32),
    String(String),
}

/// Bayesian Optimization using Gaussian Process
/// 
/// Efficiently searches hyperparameter space by building a probabilistic model
/// of the objective function.
pub struct BayesianOptimization {
    pub n_iterations: usize,
    pub n_initial_points: usize,
    pub acquisition_function: AcquisitionFunction,
    pub xi: f32,  // Exploration-exploitation trade-off
    pub kappa: f32,  // For UCB acquisition
    
    parameter_space: HashMap<String, ParameterSpace>,
    observations: Vec<(Configuration, f32)>,
}

#[derive(Clone, Copy)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
}

impl BayesianOptimization {
    pub fn new(parameter_space: HashMap<String, ParameterSpace>) -> Self {
        Self {
            n_iterations: 50,
            n_initial_points: 10,
            acquisition_function: AcquisitionFunction::ExpectedImprovement,
            xi: 0.01,
            kappa: 2.576,
            parameter_space,
            observations: Vec::new(),
        }
    }

    pub fn n_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    pub fn n_initial_points(mut self, n: usize) -> Self {
        self.n_initial_points = n;
        self
    }

    /// Optimize a black-box function
    pub fn optimize<F>(&mut self, objective: F) -> (Configuration, f32)
    where
        F: Fn(&Configuration) -> f32,
    {
        let mut rng = thread_rng();

        // Initial random sampling
        for _ in 0..self.n_initial_points {
            let config = self.sample_random(&mut rng);
            let score = objective(&config);
            self.observations.push((config, score));
        }

        // Bayesian optimization loop
        for _ in 0..self.n_iterations {
            let next_config = self.suggest_next();
            let score = objective(&next_config);
            self.observations.push((next_config, score));
        }

        // Return best configuration
        self.observations
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone()
    }

    fn sample_random(&self, rng: &mut ThreadRng) -> Configuration {
        let mut config = HashMap::new();

        for (name, space) in &self.parameter_space {
            let value = match space {
                ParameterSpace::Continuous { min, max, log_scale } => {
                    let val = if *log_scale {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        (rng.gen::<f32>() * (log_max - log_min) + log_min).exp()
                    } else {
                        rng.gen::<f32>() * (max - min) + min
                    };
                    ParameterValue::Float(val)
                }
                ParameterSpace::Integer { min, max } => {
                    let val = rng.gen_range(*min..=*max);
                    ParameterValue::Int(val)
                }
                ParameterSpace::Categorical { choices } => {
                    let idx = rng.gen_range(0..choices.len());
                    ParameterValue::String(choices[idx].clone())
                }
            };
            config.insert(name.clone(), value);
        }

        config
    }

    fn suggest_next(&self) -> Configuration {
        let mut rng = thread_rng();
        let mut best_config = self.sample_random(&mut rng);
        let mut best_acquisition = f32::NEG_INFINITY;

        // Sample candidates and evaluate acquisition function
        for _ in 0..100 {
            let config = self.sample_random(&mut rng);
            let acquisition = self.evaluate_acquisition(&config);

            if acquisition > best_acquisition {
                best_acquisition = acquisition;
                best_config = config;
            }
        }

        best_config
    }

    fn evaluate_acquisition(&self, config: &Configuration) -> f32 {
        // Simplified acquisition function (in practice, would use GP)
        let (mean, std) = self.predict_gp(config);
        
        match self.acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                let best_y = self.observations.iter()
                    .map(|(_, y)| *y)
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                
                let z = (mean - best_y - self.xi) / (std + 1e-9);
                let ei = (mean - best_y - self.xi) * self.normal_cdf(z) + std * self.normal_pdf(z);
                ei
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                let best_y = self.observations.iter()
                    .map(|(_, y)| *y)
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                
                let z = (mean - best_y - self.xi) / (std + 1e-9);
                self.normal_cdf(z)
            }
            AcquisitionFunction::UpperConfidenceBound => {
                mean + self.kappa * std
            }
        }
    }

    fn predict_gp(&self, _config: &Configuration) -> (f32, f32) {
        // Simplified GP prediction (in practice, would use proper GP)
        // Returns (mean, std)
        
        if self.observations.is_empty() {
            return (0.0, 1.0);
        }

        // Simple average as mean, std based on variance
        let mean: f32 = self.observations.iter().map(|(_, y)| y).sum::<f32>() / self.observations.len() as f32;
        let variance: f32 = self.observations.iter()
            .map(|(_, y)| (y - mean).powi(2))
            .sum::<f32>() / self.observations.len() as f32;
        let std = variance.sqrt();

        (mean, std.max(0.1))
    }

    fn normal_cdf(&self, x: f32) -> f32 {
        0.5 * (1.0 + self.erf(x / 2.0_f32.sqrt()))
    }

    fn normal_pdf(&self, x: f32) -> f32 {
        (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt()
    }

    fn erf(&self, x: f32) -> f32 {
        // Approximation of error function
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

/// Random Search
/// 
/// Simple but effective baseline for hyperparameter optimization.
pub struct RandomSearch {
    pub n_iterations: usize,
    parameter_space: HashMap<String, ParameterSpace>,
}

impl RandomSearch {
    pub fn new(parameter_space: HashMap<String, ParameterSpace>) -> Self {
        Self {
            n_iterations: 100,
            parameter_space,
        }
    }

    pub fn n_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    pub fn optimize<F>(&self, objective: F) -> (Configuration, f32)
    where
        F: Fn(&Configuration) -> f32,
    {
        let mut rng = thread_rng();
        let mut best_config = self.sample_random(&mut rng);
        let mut best_score = objective(&best_config);

        for _ in 1..self.n_iterations {
            let config = self.sample_random(&mut rng);
            let score = objective(&config);

            if score > best_score {
                best_score = score;
                best_config = config;
            }
        }

        (best_config, best_score)
    }

    fn sample_random(&self, rng: &mut ThreadRng) -> Configuration {
        let mut config = HashMap::new();

        for (name, space) in &self.parameter_space {
            let value = match space {
                ParameterSpace::Continuous { min, max, log_scale } => {
                    let val = if *log_scale {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        (rng.gen::<f32>() * (log_max - log_min) + log_min).exp()
                    } else {
                        rng.gen::<f32>() * (max - min) + min
                    };
                    ParameterValue::Float(val)
                }
                ParameterSpace::Integer { min, max } => {
                    let val = rng.gen_range(*min..=*max);
                    ParameterValue::Int(val)
                }
                ParameterSpace::Categorical { choices } => {
                    let idx = rng.gen_range(0..choices.len());
                    ParameterValue::String(choices[idx].clone())
                }
            };
            config.insert(name.clone(), value);
        }

        config
    }
}

/// Grid Search
/// 
/// Exhaustive search over specified parameter values.
pub struct GridSearch {
    parameter_grid: HashMap<String, Vec<ParameterValue>>,
}

impl GridSearch {
    pub fn new(parameter_grid: HashMap<String, Vec<ParameterValue>>) -> Self {
        Self { parameter_grid }
    }

    pub fn optimize<F>(&self, objective: F) -> (Configuration, f32)
    where
        F: Fn(&Configuration) -> f32,
    {
        let configurations = self.generate_configurations();
        
        let mut best_config = configurations[0].clone();
        let mut best_score = objective(&best_config);

        for config in configurations.iter().skip(1) {
            let score = objective(config);
            if score > best_score {
                best_score = score;
                best_config = config.clone();
            }
        }

        (best_config, best_score)
    }

    fn generate_configurations(&self) -> Vec<Configuration> {
        let mut configurations = vec![HashMap::new()];

        for (name, values) in &self.parameter_grid {
            let mut new_configurations = Vec::new();

            for config in &configurations {
                for value in values {
                    let mut new_config = config.clone();
                    new_config.insert(name.clone(), value.clone());
                    new_configurations.push(new_config);
                }
            }

            configurations = new_configurations;
        }

        configurations
    }
}

/// Hyperband
/// 
/// Adaptive resource allocation and early-stopping algorithm.
/// Efficiently allocates resources to promising configurations.
pub struct Hyperband {
    pub max_iter: usize,
    pub eta: usize,
    parameter_space: HashMap<String, ParameterSpace>,
}

impl Hyperband {
    pub fn new(parameter_space: HashMap<String, ParameterSpace>) -> Self {
        Self {
            max_iter: 81,  // Maximum iterations per configuration
            eta: 3,        // Downsampling rate
            parameter_space,
        }
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn eta(mut self, eta: usize) -> Self {
        self.eta = eta;
        self
    }

    /// Optimize with early stopping
    /// 
    /// The objective function receives (config, budget) and returns score
    pub fn optimize<F>(&self, objective: F) -> (Configuration, f32)
    where
        F: Fn(&Configuration, usize) -> f32,
    {
        let mut rng = thread_rng();
        let s_max = (self.max_iter as f32).log(self.eta as f32).floor() as usize;
        let b = (s_max + 1) * self.max_iter;

        let mut best_config = None;
        let mut best_score = f32::NEG_INFINITY;

        // Successive halving with different resource allocations
        for s in (0..=s_max).rev() {
            let n = ((b as f32 / self.max_iter as f32 / (s + 1) as f32) * (self.eta as f32).powi(s as i32)).ceil() as usize;
            let r = self.max_iter * (self.eta as f32).powi(-(s as i32)) as usize;

            // Generate n random configurations
            let mut configs: Vec<(Configuration, f32)> = (0..n)
                .map(|_| {
                    let config = self.sample_random(&mut rng);
                    let score = objective(&config, r);
                    (config, score)
                })
                .collect();

            // Successive halving
            for i in 0..=s {
                let n_i = (n as f32 * (self.eta as f32).powi(-(i as i32))).floor() as usize;
                let r_i = r * (self.eta as f32).powi(i as i32) as usize;

                // Evaluate all configurations with budget r_i
                for (config, score) in configs.iter_mut() {
                    *score = objective(config, r_i);
                }

                // Sort by score and keep top n_i / eta
                configs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let keep = (n_i as f32 / self.eta as f32).ceil() as usize;
                configs.truncate(keep.min(configs.len()));
            }

            // Update best configuration
            if let Some((config, score)) = configs.first() {
                if *score > best_score {
                    best_score = *score;
                    best_config = Some(config.clone());
                }
            }
        }

        (best_config.unwrap(), best_score)
    }

    fn sample_random(&self, rng: &mut ThreadRng) -> Configuration {
        let mut config = HashMap::new();

        for (name, space) in &self.parameter_space {
            let value = match space {
                ParameterSpace::Continuous { min, max, log_scale } => {
                    let val = if *log_scale {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        (rng.gen::<f32>() * (log_max - log_min) + log_min).exp()
                    } else {
                        rng.gen::<f32>() * (max - min) + min
                    };
                    ParameterValue::Float(val)
                }
                ParameterSpace::Integer { min, max } => {
                    let val = rng.gen_range(*min..=*max);
                    ParameterValue::Int(val)
                }
                ParameterSpace::Categorical { choices } => {
                    let idx = rng.gen_range(0..choices.len());
                    ParameterValue::String(choices[idx].clone())
                }
            };
            config.insert(name.clone(), value);
        }

        config
    }
}

/// BOHB (Bayesian Optimization and HyperBand)
/// 
/// Combines Bayesian optimization with Hyperband's adaptive resource allocation.
/// Uses a tree-structured Parzen estimator (TPE) for configuration selection.
pub struct BOHB {
    pub max_iter: usize,
    pub eta: usize,
    pub min_points_in_model: usize,
    pub top_n_percent: usize,
    pub bandwidth_factor: f32,
    parameter_space: HashMap<String, ParameterSpace>,
    observations: Vec<(Configuration, usize, f32)>,  // (config, budget, score)
}

impl BOHB {
    pub fn new(parameter_space: HashMap<String, ParameterSpace>) -> Self {
        Self {
            max_iter: 81,
            eta: 3,
            min_points_in_model: 10,
            top_n_percent: 15,
            bandwidth_factor: 3.0,
            parameter_space,
            observations: Vec::new(),
        }
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn eta(mut self, eta: usize) -> Self {
        self.eta = eta;
        self
    }

    /// Optimize using BOHB
    pub fn optimize<F>(&mut self, objective: F) -> (Configuration, f32)
    where
        F: Fn(&Configuration, usize) -> f32,
    {
        let mut rng = thread_rng();
        let s_max = (self.max_iter as f32).log(self.eta as f32).floor() as usize;
        let b = (s_max + 1) * self.max_iter;

        let mut best_config = None;
        let mut best_score = f32::NEG_INFINITY;

        for s in (0..=s_max).rev() {
            let n = ((b as f32 / self.max_iter as f32 / (s + 1) as f32) * (self.eta as f32).powi(s as i32)).ceil() as usize;
            let r = self.max_iter * (self.eta as f32).powi(-(s as i32)) as usize;

            // Generate configurations using TPE or random sampling
            let mut configs: Vec<(Configuration, f32)> = (0..n)
                .map(|_| {
                    let config = if self.observations.len() >= self.min_points_in_model {
                        self.sample_tpe(&mut rng)
                    } else {
                        self.sample_random(&mut rng)
                    };
                    let score = objective(&config, r);
                    self.observations.push((config.clone(), r, score));
                    (config, score)
                })
                .collect();

            // Successive halving
            for i in 0..=s {
                let n_i = (n as f32 * (self.eta as f32).powi(-(i as i32))).floor() as usize;
                let r_i = r * (self.eta as f32).powi(i as i32) as usize;

                for (config, score) in configs.iter_mut() {
                    *score = objective(config, r_i);
                    self.observations.push((config.clone(), r_i, *score));
                }

                configs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let keep = (n_i as f32 / self.eta as f32).ceil() as usize;
                configs.truncate(keep.min(configs.len()));
            }

            if let Some((config, score)) = configs.first() {
                if *score > best_score {
                    best_score = *score;
                    best_config = Some(config.clone());
                }
            }
        }

        (best_config.unwrap(), best_score)
    }

    fn sample_tpe(&self, rng: &mut ThreadRng) -> Configuration {
        // Tree-structured Parzen Estimator sampling
        // Split observations into good and bad based on top_n_percent
        let mut sorted_obs = self.observations.clone();
        sorted_obs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let split_idx = (sorted_obs.len() * self.top_n_percent / 100).max(1);
        let good_obs: Vec<_> = sorted_obs.iter().take(split_idx).collect();
        let bad_obs: Vec<_> = sorted_obs.iter().skip(split_idx).collect();

        // Sample from good distribution
        let mut config = HashMap::new();

        for (name, space) in &self.parameter_space {
            let value = match space {
                ParameterSpace::Continuous { min, max, log_scale } => {
                    // Build KDE from good observations
                    let good_values: Vec<f32> = good_obs
                        .iter()
                        .filter_map(|(c, _, _)| {
                            if let Some(ParameterValue::Float(v)) = c.get(name) {
                                Some(*v)
                            } else {
                                None
                            }
                        })
                        .collect();

                    let val = if !good_values.is_empty() {
                        // Sample from KDE
                        let idx = rng.gen_range(0..good_values.len());
                        let base = good_values[idx];
                        let bandwidth = (max - min) / self.bandwidth_factor;
                        let noise = rng.gen::<f32>() * bandwidth - bandwidth / 2.0;
                        (base + noise).clamp(*min, *max)
                    } else {
                        // Fallback to random
                        if *log_scale {
                            let log_min = min.ln();
                            let log_max = max.ln();
                            (rng.gen::<f32>() * (log_max - log_min) + log_min).exp()
                        } else {
                            rng.gen::<f32>() * (max - min) + min
                        }
                    };
                    ParameterValue::Float(val)
                }
                ParameterSpace::Integer { min, max } => {
                    let good_values: Vec<i32> = good_obs
                        .iter()
                        .filter_map(|(c, _, _)| {
                            if let Some(ParameterValue::Int(v)) = c.get(name) {
                                Some(*v)
                            } else {
                                None
                            }
                        })
                        .collect();

                    let val = if !good_values.is_empty() {
                        let idx = rng.gen_range(0..good_values.len());
                        good_values[idx]
                    } else {
                        rng.gen_range(*min..=*max)
                    };
                    ParameterValue::Int(val)
                }
                ParameterSpace::Categorical { choices } => {
                    let good_values: Vec<String> = good_obs
                        .iter()
                        .filter_map(|(c, _, _)| {
                            if let Some(ParameterValue::String(v)) = c.get(name) {
                                Some(v.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    let val = if !good_values.is_empty() {
                        let idx = rng.gen_range(0..good_values.len());
                        good_values[idx].clone()
                    } else {
                        let idx = rng.gen_range(0..choices.len());
                        choices[idx].clone()
                    };
                    ParameterValue::String(val)
                }
            };
            config.insert(name.clone(), value);
        }

        config
    }

    fn sample_random(&self, rng: &mut ThreadRng) -> Configuration {
        let mut config = HashMap::new();

        for (name, space) in &self.parameter_space {
            let value = match space {
                ParameterSpace::Continuous { min, max, log_scale } => {
                    let val = if *log_scale {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        (rng.gen::<f32>() * (log_max - log_min) + log_min).exp()
                    } else {
                        rng.gen::<f32>() * (max - min) + min
                    };
                    ParameterValue::Float(val)
                }
                ParameterSpace::Integer { min, max } => {
                    let val = rng.gen_range(*min..=*max);
                    ParameterValue::Int(val)
                }
                ParameterSpace::Categorical { choices } => {
                    let idx = rng.gen_range(0..choices.len());
                    ParameterValue::String(choices[idx].clone())
                }
            };
            config.insert(name.clone(), value);
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_search() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "learning_rate".to_string(),
            ParameterSpace::Continuous { min: 0.001, max: 0.1, log_scale: true },
        );
        param_space.insert(
            "n_estimators".to_string(),
            ParameterSpace::Integer { min: 10, max: 100 },
        );

        let rs = RandomSearch::new(param_space).n_iterations(10);

        let (best_config, best_score) = rs.optimize(|config| {
            // Dummy objective function
            match config.get("learning_rate") {
                Some(ParameterValue::Float(lr)) => *lr * 10.0,
                _ => 0.0,
            }
        });

        assert!(best_score > 0.0);
        assert!(best_config.contains_key("learning_rate"));
    }

    #[test]
    fn test_grid_search() {
        let mut param_grid = HashMap::new();
        param_grid.insert(
            "param1".to_string(),
            vec![ParameterValue::Float(0.1), ParameterValue::Float(0.2)],
        );
        param_grid.insert(
            "param2".to_string(),
            vec![ParameterValue::Int(10), ParameterValue::Int(20)],
        );

        let gs = GridSearch::new(param_grid);

        let (best_config, _) = gs.optimize(|config| {
            match (config.get("param1"), config.get("param2")) {
                (Some(ParameterValue::Float(p1)), Some(ParameterValue::Int(p2))) => {
                    p1 * (*p2 as f32)
                }
                _ => 0.0,
            }
        });

        assert!(best_config.contains_key("param1"));
        assert!(best_config.contains_key("param2"));
    }

    #[test]
    fn test_hyperband() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "learning_rate".to_string(),
            ParameterSpace::Continuous { min: 0.001, max: 0.1, log_scale: true },
        );
        param_space.insert(
            "n_layers".to_string(),
            ParameterSpace::Integer { min: 1, max: 5 },
        );

        let hb = Hyperband::new(param_space)
            .max_iter(27)
            .eta(3);

        let (best_config, best_score) = hb.optimize(|config, budget| {
            // Simulate training with budget (number of iterations)
            let lr = match config.get("learning_rate") {
                Some(ParameterValue::Float(v)) => *v,
                _ => 0.01,
            };
            let n_layers = match config.get("n_layers") {
                Some(ParameterValue::Int(v)) => *v,
                _ => 2,
            };

            // Score improves with budget and depends on hyperparameters
            let base_score = lr * 10.0 + n_layers as f32;
            base_score * (budget as f32).sqrt() / 10.0
        });

        assert!(best_score > 0.0);
        assert!(best_config.contains_key("learning_rate"));
        assert!(best_config.contains_key("n_layers"));
    }

    #[test]
    fn test_bohb() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "learning_rate".to_string(),
            ParameterSpace::Continuous { min: 0.001, max: 0.1, log_scale: true },
        );
        param_space.insert(
            "batch_size".to_string(),
            ParameterSpace::Integer { min: 16, max: 128 },
        );

        let mut bohb = BOHB::new(param_space)
            .max_iter(27)
            .eta(3);

        let (best_config, best_score) = bohb.optimize(|config, budget| {
            let lr = match config.get("learning_rate") {
                Some(ParameterValue::Float(v)) => *v,
                _ => 0.01,
            };
            let batch_size = match config.get("batch_size") {
                Some(ParameterValue::Int(v)) => *v,
                _ => 32,
            };

            // Simulate validation score
            let base_score = (lr * 100.0).ln() + (batch_size as f32 / 32.0);
            base_score * (budget as f32).sqrt() / 5.0
        });

        assert!(best_score > 0.0);
        assert!(best_config.contains_key("learning_rate"));
        assert!(best_config.contains_key("batch_size"));
    }

    #[test]
    fn test_bohb_tpe_sampling() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "x".to_string(),
            ParameterSpace::Continuous { min: -5.0, max: 5.0, log_scale: false },
        );

        let mut bohb = BOHB::new(param_space)
            .max_iter(9)
            .eta(3);

        // Optimize a simple quadratic function
        let (best_config, best_score) = bohb.optimize(|config, _budget| {
            let x = match config.get("x") {
                Some(ParameterValue::Float(v)) => *v,
                _ => 0.0,
            };
            // Maximize -(x-2)^2, optimum at x=2
            -(x - 2.0).powi(2)
        });

        // Should find value close to 2
        if let Some(ParameterValue::Float(x)) = best_config.get("x") {
            assert!((x - 2.0).abs() < 1.0, "Expected x close to 2, got {}", x);
        }
        assert!(best_score > -2.0);
    }
}


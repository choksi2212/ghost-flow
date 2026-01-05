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
}

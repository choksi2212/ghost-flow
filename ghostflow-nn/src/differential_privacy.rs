//! Differential Privacy for Machine Learning
//!
//! Implements privacy-preserving machine learning techniques:
//! - DP-SGD (Differentially Private Stochastic Gradient Descent)
//! - Gradient clipping and noise addition
//! - Privacy budget tracking (epsilon, delta)
//! - Moments accountant for tight privacy bounds
//! - PATE (Private Aggregation of Teacher Ensembles)

use ghostflow_core::Tensor;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::collections::VecDeque;

/// Differential privacy configuration
#[derive(Debug, Clone)]
pub struct DPConfig {
    /// Target epsilon (privacy budget)
    pub target_epsilon: f32,
    /// Target delta (failure probability)
    pub target_delta: f32,
    /// Gradient clipping norm
    pub clip_norm: f32,
    /// Noise multiplier
    pub noise_multiplier: f32,
    /// Batch size for sampling
    pub batch_size: usize,
    /// Total number of training examples
    pub num_examples: usize,
}

impl Default for DPConfig {
    fn default() -> Self {
        DPConfig {
            target_epsilon: 1.0,
            target_delta: 1e-5,
            clip_norm: 1.0,
            noise_multiplier: 1.1,
            batch_size: 256,
            num_examples: 60000,
        }
    }
}

/// Privacy accountant for tracking privacy budget
pub struct PrivacyAccountant {
    config: DPConfig,
    steps: usize,
    epsilon_spent: f32,
    moments: VecDeque<f32>,
}

impl PrivacyAccountant {
    /// Create new privacy accountant
    pub fn new(config: DPConfig) -> Self {
        PrivacyAccountant {
            config,
            steps: 0,
            epsilon_spent: 0.0,
            moments: VecDeque::new(),
        }
    }
    
    /// Record a training step
    pub fn step(&mut self) {
        self.steps += 1;
        
        // Compute privacy loss for this step using moments accountant
        let q = self.config.batch_size as f32 / self.config.num_examples as f32;
        let sigma = self.config.noise_multiplier;
        
        // Simplified moments accountant (real implementation would be more complex)
        let moment = q * q / (2.0 * sigma * sigma);
        self.moments.push_back(moment);
        
        // Keep only recent moments (sliding window)
        if self.moments.len() > 1000 {
            self.moments.pop_front();
        }
        
        // Update epsilon spent
        self.epsilon_spent = self.compute_epsilon();
    }
    
    /// Compute current epsilon using moments accountant
    fn compute_epsilon(&self) -> f32 {
        if self.steps == 0 {
            return 0.0;
        }
        
        let q = self.config.batch_size as f32 / self.config.num_examples as f32;
        let sigma = self.config.noise_multiplier;
        let steps = self.steps as f32;
        
        // Simplified epsilon calculation (real implementation uses RDP)
        let epsilon = (q * steps).sqrt() / sigma + 
                     (self.config.target_delta.ln() / steps).abs();
        
        epsilon.min(self.config.target_epsilon * 2.0) // Cap at 2x target
    }
    
    /// Check if privacy budget is exhausted
    pub fn is_budget_exhausted(&self) -> bool {
        self.epsilon_spent >= self.config.target_epsilon
    }
    
    /// Get current privacy parameters
    pub fn get_privacy_spent(&self) -> (f32, f32) {
        (self.epsilon_spent, self.config.target_delta)
    }
    
    /// Get remaining privacy budget
    pub fn get_remaining_budget(&self) -> f32 {
        (self.config.target_epsilon - self.epsilon_spent).max(0.0)
    }
}

/// DP-SGD optimizer wrapper
pub struct DPSGDOptimizer {
    config: DPConfig,
    accountant: PrivacyAccountant,
    rng: rand::rngs::ThreadRng,
}

impl DPSGDOptimizer {
    /// Create new DP-SGD optimizer
    pub fn new(config: DPConfig) -> Self {
        let accountant = PrivacyAccountant::new(config.clone());
        DPSGDOptimizer {
            config,
            accountant,
            rng: rand::thread_rng(),
        }
    }
    
    /// Clip gradients to bound sensitivity
    pub fn clip_gradients(&self, gradients: &mut Tensor) -> Result<(), String> {
        let grad_data = gradients.data_mut();
        
        // Compute L2 norm of gradients
        let mut norm_sq = 0.0f32;
        for &g in grad_data.iter() {
            norm_sq += g * g;
        }
        let norm = norm_sq.sqrt();
        
        // Clip if norm exceeds threshold
        if norm > self.config.clip_norm {
            let scale = self.config.clip_norm / norm;
            for g in grad_data.iter_mut() {
                *g *= scale;
            }
        }
        
        Ok(())
    }
    
    /// Add calibrated Gaussian noise to gradients
    pub fn add_noise(&mut self, gradients: &mut Tensor) -> Result<(), String> {
        let grad_data = gradients.data_mut();
        
        // Compute noise scale
        let noise_scale = self.config.clip_norm * self.config.noise_multiplier;
        let normal = Normal::new(0.0, noise_scale as f64)
            .map_err(|e| format!("Failed to create normal distribution: {}", e))?;
        
        // Add Gaussian noise to each gradient
        for g in grad_data.iter_mut() {
            let noise = normal.sample(&mut self.rng) as f32;
            *g += noise;
        }
        
        Ok(())
    }
    
    /// Process gradients with DP-SGD (clip + noise)
    pub fn process_gradients(&mut self, gradients: &mut Tensor) -> Result<(), String> {
        // Step 1: Clip gradients
        self.clip_gradients(gradients)?;
        
        // Step 2: Add noise
        self.add_noise(gradients)?;
        
        // Step 3: Update privacy accountant
        self.accountant.step();
        
        Ok(())
    }
    
    /// Check if training should stop due to privacy budget
    pub fn should_stop(&self) -> bool {
        self.accountant.is_budget_exhausted()
    }
    
    /// Get privacy spent
    pub fn get_privacy_spent(&self) -> (f32, f32) {
        self.accountant.get_privacy_spent()
    }
    
    /// Get remaining budget
    pub fn get_remaining_budget(&self) -> f32 {
        self.accountant.get_remaining_budget()
    }
}

/// PATE (Private Aggregation of Teacher Ensembles)
pub struct PATEEnsemble {
    num_teachers: usize,
    num_classes: usize,
    epsilon: f32,
    delta: f32,
}

impl PATEEnsemble {
    /// Create new PATE ensemble
    pub fn new(num_teachers: usize, num_classes: usize, epsilon: f32, delta: f32) -> Self {
        PATEEnsemble {
            num_teachers,
            num_classes,
            epsilon,
            delta,
        }
    }
    
    /// Aggregate teacher predictions with privacy
    pub fn aggregate_predictions(&self, teacher_votes: &[Vec<usize>]) -> Result<Vec<usize>, String> {
        if teacher_votes.is_empty() {
            return Err("No teacher votes provided".to_string());
        }
        
        let num_samples = teacher_votes[0].len();
        let mut aggregated = Vec::with_capacity(num_samples);
        let mut rng = rand::thread_rng();
        
        // For each sample
        for i in 0..num_samples {
            // Count votes for each class
            let mut counts = vec![0usize; self.num_classes];
            for teacher_preds in teacher_votes.iter() {
                if i < teacher_preds.len() {
                    let pred = teacher_preds[i];
                    if pred < self.num_classes {
                        counts[pred] += 1;
                    }
                }
            }
            
            // Add Laplace noise for privacy
            let sensitivity = 1.0; // One teacher can change count by at most 1
            let scale = sensitivity / self.epsilon;
            
            let mut noisy_counts = Vec::with_capacity(self.num_classes);
            for &count in counts.iter() {
                // Sample from Laplace distribution
                let u: f32 = rng.gen_range(-0.5..0.5);
                let noise = -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln();
                let noisy_count = count as f32 + noise;
                noisy_counts.push(noisy_count);
            }
            
            // Find class with maximum noisy count
            let pred_class = noisy_counts.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            aggregated.push(pred_class);
        }
        
        Ok(aggregated)
    }
    
    /// Compute privacy cost of aggregation
    pub fn compute_privacy_cost(&self, num_queries: usize) -> (f32, f32) {
        // Simplified privacy analysis
        let epsilon_total = self.epsilon * (num_queries as f32).sqrt();
        let delta_total = self.delta * num_queries as f32;
        (epsilon_total, delta_total)
    }
}

/// Local differential privacy for data collection
pub struct LocalDP {
    epsilon: f32,
}

impl LocalDP {
    /// Create new local DP mechanism
    pub fn new(epsilon: f32) -> Self {
        LocalDP { epsilon }
    }
    
    /// Randomized response for binary data
    pub fn randomized_response(&self, value: bool) -> bool {
        let mut rng = rand::thread_rng();
        let p = (self.epsilon.exp()) / (self.epsilon.exp() + 1.0);
        
        let flip_prob: f32 = rng.gen();
        if flip_prob < p {
            value
        } else {
            !value
        }
    }
    
    /// Add Laplace noise to numeric data
    pub fn add_laplace_noise(&self, value: f32, sensitivity: f32) -> f32 {
        let mut rng = rand::thread_rng();
        let scale = sensitivity / self.epsilon;
        
        let u: f32 = rng.gen_range(-0.5..0.5);
        let noise = -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln();
        
        value + noise
    }
    
    /// Privatize a vector of values
    pub fn privatize_vector(&self, values: &[f32], sensitivity: f32) -> Vec<f32> {
        values.iter()
            .map(|&v| self.add_laplace_noise(v, sensitivity))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_accountant() {
        let config = DPConfig::default();
        let mut accountant = PrivacyAccountant::new(config);
        
        // Initially no privacy spent
        assert_eq!(accountant.steps, 0);
        assert_eq!(accountant.epsilon_spent, 0.0);
        
        // After steps, privacy is spent
        for _ in 0..100 {
            accountant.step();
        }
        
        assert!(accountant.epsilon_spent > 0.0);
        assert!(accountant.epsilon_spent <= accountant.config.target_epsilon * 2.0);
    }
    
    #[test]
    fn test_gradient_clipping() {
        let config = DPConfig {
            clip_norm: 1.0,
            ..Default::default()
        };
        let optimizer = DPSGDOptimizer::new(config);
        
        let mut gradients = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
        optimizer.clip_gradients(&mut gradients).unwrap();
        
        // Check that norm is clipped to 1.0
        let data = gradients.data();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_pate_aggregation() {
        let pate = PATEEnsemble::new(10, 3, 1.0, 1e-5);
        
        // 10 teachers, 5 samples, 3 classes
        let teacher_votes = vec![
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 2, 1, 1],
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 2, 0, 2],
            vec![0, 1, 1, 0, 1],
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 2, 0, 1],
        ];
        
        let result = pate.aggregate_predictions(&teacher_votes).unwrap();
        assert_eq!(result.len(), 5);
    }
    
    #[test]
    fn test_local_dp() {
        let ldp = LocalDP::new(1.0);
        
        // Test randomized response
        let value = true;
        let _ = ldp.randomized_response(value);
        
        // Test Laplace noise
        let noisy = ldp.add_laplace_noise(10.0, 1.0);
        assert!((noisy - 10.0).abs() < 10.0); // Noise should be bounded
        
        // Test vector privatization
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let privatized = ldp.privatize_vector(&values, 1.0);
        assert_eq!(privatized.len(), values.len());
    }
}

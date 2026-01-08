//! Adversarial Training and Robustness
//!
//! Implements adversarial attack and defense methods:
//! - FGSM (Fast Gradient Sign Method)
//! - PGD (Projected Gradient Descent)
//! - C&W (Carlini & Wagner) attack
//! - Adversarial training
//! - Certified defenses

use ghostflow_core::Tensor;
use rand::Rng;

/// Adversarial attack configuration
#[derive(Debug, Clone)]
pub struct AttackConfig {
    /// Attack type
    pub attack_type: AttackType,
    /// Perturbation budget (epsilon)
    pub epsilon: f32,
    /// Number of attack iterations
    pub num_iterations: usize,
    /// Step size for iterative attacks
    pub step_size: f32,
    /// Random initialization
    pub random_init: bool,
}

impl Default for AttackConfig {
    fn default() -> Self {
        AttackConfig {
            attack_type: AttackType::PGD,
            epsilon: 0.3,
            num_iterations: 40,
            step_size: 0.01,
            random_init: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttackType {
    /// Fast Gradient Sign Method
    FGSM,
    /// Projected Gradient Descent
    PGD,
    /// Carlini & Wagner
    CW,
    /// DeepFool
    DeepFool,
}

/// Adversarial attack generator
pub struct AdversarialAttack {
    config: AttackConfig,
}

impl AdversarialAttack {
    /// Create new adversarial attack
    pub fn new(config: AttackConfig) -> Self {
        AdversarialAttack { config }
    }
    
    /// Generate adversarial example using FGSM
    pub fn fgsm(&self, input: &Tensor, gradient: &Tensor) -> Result<Tensor, String> {
        let input_data = input.data();
        let grad_data = gradient.data();
        
        if input_data.len() != grad_data.len() {
            return Err("Input and gradient dimensions must match".to_string());
        }
        
        // Compute perturbation: epsilon * sign(gradient)
        let mut perturbed = Vec::with_capacity(input_data.len());
        for (&x, &g) in input_data.iter().zip(grad_data.iter()) {
            let sign = if g > 0.0 { 1.0 } else if g < 0.0 { -1.0 } else { 0.0 };
            let perturbed_x = x + self.config.epsilon * sign;
            // Clip to valid range [0, 1]
            perturbed.push(perturbed_x.max(0.0).min(1.0));
        }
        
        Tensor::from_vec(perturbed, input.shape().to_vec())
    }
    
    /// Generate adversarial example using PGD
    pub fn pgd(&self, input: &Tensor, compute_gradient: impl Fn(&Tensor) -> Result<Tensor, String>) 
        -> Result<Tensor, String> {
        let mut perturbed = input.clone();
        let input_data = input.data();
        
        // Random initialization within epsilon ball
        if self.config.random_init {
            let mut rng = rand::thread_rng();
            let perturbed_data = perturbed.data_mut();
            for (p, &x) in perturbed_data.iter_mut().zip(input_data.iter()) {
                let noise: f32 = rng.gen_range(-self.config.epsilon..self.config.epsilon);
                *p = (x + noise).max(0.0).min(1.0);
            }
        }
        
        // Iterative FGSM
        for _ in 0..self.config.num_iterations {
            // Compute gradient
            let gradient = compute_gradient(&perturbed)?;
            let grad_data = gradient.data();
            let perturbed_data = perturbed.data_mut();
            
            // Update perturbation
            for (p, &g) in perturbed_data.iter_mut().zip(grad_data.iter()) {
                let sign = if g > 0.0 { 1.0 } else if g < 0.0 { -1.0 } else { 0.0 };
                *p += self.config.step_size * sign;
            }
            
            // Project back to epsilon ball around original input
            let perturbed_data = perturbed.data_mut();
            for (p, &x) in perturbed_data.iter_mut().zip(input_data.iter()) {
                // Clip perturbation to epsilon ball
                let delta = (*p - x).max(-self.config.epsilon).min(self.config.epsilon);
                *p = (x + delta).max(0.0).min(1.0);
            }
        }
        
        Ok(perturbed)
    }
    
    /// Generate adversarial example using C&W attack
    pub fn cw(&self, input: &Tensor, target_class: usize, 
              compute_logits: impl Fn(&Tensor) -> Result<Tensor, String>) 
        -> Result<Tensor, String> {
        let mut perturbed = input.clone();
        let input_data = input.data();
        let c = 1.0; // Confidence parameter
        
        for _ in 0..self.config.num_iterations {
            // Compute logits
            let logits = compute_logits(&perturbed)?;
            let logits_data = logits.data();
            
            if logits_data.len() <= target_class {
                return Err("Invalid target class".to_string());
            }
            
            // Compute C&W loss: max(max(Z_i) - Z_target, -kappa)
            let target_logit = logits_data[target_class];
            let max_other = logits_data.iter()
                .enumerate()
                .filter(|(i, _)| *i != target_class)
                .map(|(_, &l)| l)
                .fold(f32::NEG_INFINITY, f32::max);
            
            let loss = (max_other - target_logit + c).max(0.0);
            
            // Gradient approximation (simplified)
            let perturbed_data = perturbed.data_mut();
            for (p, &x) in perturbed_data.iter_mut().zip(input_data.iter()) {
                let grad_sign = if loss > 0.0 { 1.0 } else { -1.0 };
                *p -= self.config.step_size * grad_sign;
                
                // Project to epsilon ball
                let delta = (*p - x).max(-self.config.epsilon).min(self.config.epsilon);
                *p = (x + delta).max(0.0).min(1.0);
            }
        }
        
        Ok(perturbed)
    }
    
    /// Generate adversarial example using DeepFool
    pub fn deepfool(&self, input: &Tensor, 
                    compute_gradient: impl Fn(&Tensor, usize) -> Result<Tensor, String>,
                    num_classes: usize) 
        -> Result<Tensor, String> {
        let mut perturbed = input.clone();
        let input_data = input.data();
        
        for _ in 0..self.config.num_iterations {
            let mut min_distance = f32::INFINITY;
            let mut best_perturbation = vec![0.0; input_data.len()];
            
            // Find minimal perturbation to cross decision boundary
            for class in 0..num_classes {
                let gradient = compute_gradient(&perturbed, class)?;
                let grad_data = gradient.data();
                
                // Compute distance to decision boundary
                let grad_norm: f32 = grad_data.iter().map(|g| g * g).sum::<f32>().sqrt();
                if grad_norm > 1e-8 {
                    let distance = 1.0 / grad_norm;
                    
                    if distance < min_distance {
                        min_distance = distance;
                        best_perturbation = grad_data.iter()
                            .map(|&g| (distance * g / grad_norm).min(self.config.epsilon))
                            .collect();
                    }
                }
            }
            
            // Apply perturbation
            let perturbed_data = perturbed.data_mut();
            for (p, (&x, &delta)) in perturbed_data.iter_mut()
                .zip(input_data.iter().zip(best_perturbation.iter())) {
                *p = (x + delta).max(0.0).min(1.0);
            }
        }
        
        Ok(perturbed)
    }
}

/// Adversarial training configuration
#[derive(Debug, Clone)]
pub struct AdversarialTrainingConfig {
    /// Fraction of adversarial examples in each batch
    pub adversarial_ratio: f32,
    /// Attack configuration for training
    pub attack_config: AttackConfig,
    /// Use label smoothing
    pub label_smoothing: f32,
}

impl Default for AdversarialTrainingConfig {
    fn default() -> Self {
        AdversarialTrainingConfig {
            adversarial_ratio: 0.5,
            attack_config: AttackConfig::default(),
            label_smoothing: 0.1,
        }
    }
}

/// Adversarial training wrapper
pub struct AdversarialTrainer {
    config: AdversarialTrainingConfig,
    attack: AdversarialAttack,
}

impl AdversarialTrainer {
    /// Create new adversarial trainer
    pub fn new(config: AdversarialTrainingConfig) -> Self {
        let attack = AdversarialAttack::new(config.attack_config.clone());
        AdversarialTrainer { config, attack }
    }
    
    /// Generate mixed batch of clean and adversarial examples
    pub fn generate_training_batch(&self, 
                                   clean_inputs: &[Tensor],
                                   compute_gradient: impl Fn(&Tensor) -> Result<Tensor, String>)
        -> Result<Vec<Tensor>, String> {
        let num_adversarial = (clean_inputs.len() as f32 * self.config.adversarial_ratio) as usize;
        let mut batch = Vec::with_capacity(clean_inputs.len());
        
        // Add clean examples
        for input in clean_inputs.iter().skip(num_adversarial) {
            batch.push(input.clone());
        }
        
        // Add adversarial examples
        for input in clean_inputs.iter().take(num_adversarial) {
            let adv_example = self.attack.pgd(input, &compute_gradient)?;
            batch.push(adv_example);
        }
        
        Ok(batch)
    }
    
    /// Apply label smoothing for robustness
    pub fn smooth_labels(&self, labels: &Tensor, num_classes: usize) -> Result<Tensor, String> {
        let label_data = labels.data();
        let smoothing = self.config.label_smoothing;
        let mut smoothed = Vec::with_capacity(label_data.len() * num_classes);
        
        for &label in label_data.iter() {
            let label_idx = label as usize;
            if label_idx >= num_classes {
                return Err("Label index out of bounds".to_string());
            }
            
            for i in 0..num_classes {
                if i == label_idx {
                    smoothed.push(1.0 - smoothing);
                } else {
                    smoothed.push(smoothing / (num_classes - 1) as f32);
                }
            }
        }
        
        Tensor::from_vec(smoothed, vec![label_data.len(), num_classes])
    }
}

/// Certified defense using randomized smoothing
pub struct RandomizedSmoothing {
    /// Noise standard deviation
    pub sigma: f32,
    /// Number of samples for certification
    pub num_samples: usize,
    /// Confidence level
    pub alpha: f32,
}

impl RandomizedSmoothing {
    /// Create new randomized smoothing defense
    pub fn new(sigma: f32, num_samples: usize, alpha: f32) -> Self {
        RandomizedSmoothing {
            sigma,
            num_samples,
            alpha,
        }
    }
    
    /// Predict with randomized smoothing
    pub fn predict(&self, input: &Tensor, 
                   model_predict: impl Fn(&Tensor) -> Result<usize, String>)
        -> Result<usize, String> {
        let mut rng = rand::thread_rng();
        let input_data = input.data();
        let mut class_counts = std::collections::HashMap::new();
        
        // Sample predictions with Gaussian noise
        for _ in 0..self.num_samples {
            let mut noisy_input = Vec::with_capacity(input_data.len());
            for &x in input_data.iter() {
                let noise: f32 = rng.gen::<f32>() * self.sigma;
                noisy_input.push((x + noise).max(0.0).min(1.0));
            }
            
            let noisy_tensor = Tensor::from_vec(noisy_input, input.shape().to_vec())?;
            let pred = model_predict(&noisy_tensor)?;
            *class_counts.entry(pred).or_insert(0) += 1;
        }
        
        // Return most common prediction
        class_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
            .ok_or_else(|| "No predictions generated".to_string())
    }
    
    /// Compute certified radius
    pub fn certify(&self, input: &Tensor,
                   model_predict: impl Fn(&Tensor) -> Result<usize, String>)
        -> Result<(usize, f32), String> {
        let predicted_class = self.predict(input, &model_predict)?;
        
        // Compute lower bound on probability of predicted class
        let mut rng = rand::thread_rng();
        let input_data = input.data();
        let mut correct_count = 0;
        
        for _ in 0..self.num_samples {
            let mut noisy_input = Vec::with_capacity(input_data.len());
            for &x in input_data.iter() {
                let noise: f32 = rng.gen::<f32>() * self.sigma;
                noisy_input.push((x + noise).max(0.0).min(1.0));
            }
            
            let noisy_tensor = Tensor::from_vec(noisy_input, input.shape().to_vec())?;
            let pred = model_predict(&noisy_tensor)?;
            if pred == predicted_class {
                correct_count += 1;
            }
        }
        
        let p_lower = (correct_count as f32 / self.num_samples as f32) - self.alpha;
        
        // Compute certified radius using Neyman-Pearson lemma
        let radius = if p_lower > 0.5 {
            self.sigma * (2.0 * p_lower - 1.0).sqrt()
        } else {
            0.0
        };
        
        Ok((predicted_class, radius))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fgsm_attack() {
        let config = AttackConfig {
            attack_type: AttackType::FGSM,
            epsilon: 0.1,
            ..Default::default()
        };
        let attack = AdversarialAttack::new(config);
        
        let input = Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], vec![4]).unwrap();
        let gradient = Tensor::from_vec(vec![1.0, -1.0, 0.5, -0.5], vec![4]).unwrap();
        
        let adv = attack.fgsm(&input, &gradient).unwrap();
        let adv_data = adv.data();
        
        // Check perturbation is applied correctly
        assert!((adv_data[0] - 0.6).abs() < 1e-5); // 0.5 + 0.1 * sign(1.0)
        assert!((adv_data[1] - 0.4).abs() < 1e-5); // 0.5 + 0.1 * sign(-1.0)
    }
    
    #[test]
    fn test_label_smoothing() {
        let config = AdversarialTrainingConfig {
            label_smoothing: 0.1,
            ..Default::default()
        };
        let trainer = AdversarialTrainer::new(config);
        
        let labels = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let smoothed = trainer.smooth_labels(&labels, 3).unwrap();
        
        assert_eq!(smoothed.shape(), &[3, 3]);
        let data = smoothed.data();
        
        // Check first label (class 0)
        assert!((data[0] - 0.9).abs() < 1e-5); // 1.0 - 0.1
        assert!((data[1] - 0.05).abs() < 1e-5); // 0.1 / 2
        assert!((data[2] - 0.05).abs() < 1e-5); // 0.1 / 2
    }
    
    #[test]
    fn test_randomized_smoothing() {
        let smoothing = RandomizedSmoothing::new(0.1, 100, 0.05);
        
        let input = Tensor::from_vec(vec![0.5; 10], vec![10]).unwrap();
        
        // Mock model that always predicts class 0
        let model = |_: &Tensor| Ok(0);
        
        let pred = smoothing.predict(&input, model).unwrap();
        assert_eq!(pred, 0);
    }
}

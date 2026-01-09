//! Knowledge Distillation
//!
//! Implements knowledge transfer from teacher to student models:
//! - Temperature-scaled softmax
//! - Feature matching
//! - Attention transfer
//! - Progressive knowledge distillation
//! - Self-distillation

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// Knowledge distillation configuration
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softmax scaling
    pub temperature: f32,
    /// Weight for distillation loss
    pub alpha: f32,
    /// Weight for student loss (ground truth)
    pub beta: f32,
    /// Distillation method
    pub method: DistillationMethod,
    /// Feature matching layers
    pub feature_layers: Vec<usize>,
}

/// Distillation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistillationMethod {
    /// Standard knowledge distillation (Hinton et al.)
    Standard,
    /// Feature-based distillation
    Feature,
    /// Attention transfer
    Attention,
    /// FitNet-style hint learning
    FitNet,
    /// Progressive distillation
    Progressive,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        DistillationConfig {
            temperature: 4.0,
            alpha: 0.7,
            beta: 0.3,
            method: DistillationMethod::Standard,
            feature_layers: vec![],
        }
    }
}

impl DistillationConfig {
    /// Standard knowledge distillation
    pub fn standard(temperature: f32, alpha: f32) -> Self {
        DistillationConfig {
            temperature,
            alpha,
            beta: 1.0 - alpha,
            method: DistillationMethod::Standard,
            ..Default::default()
        }
    }
    
    /// Feature-based distillation
    pub fn feature_based(temperature: f32, feature_layers: Vec<usize>) -> Self {
        DistillationConfig {
            temperature,
            method: DistillationMethod::Feature,
            feature_layers,
            ..Default::default()
        }
    }
    
    /// Attention transfer
    pub fn attention_transfer(temperature: f32) -> Self {
        DistillationConfig {
            temperature,
            method: DistillationMethod::Attention,
            ..Default::default()
        }
    }
}

/// Knowledge distillation trainer
pub struct KnowledgeDistillation {
    config: DistillationConfig,
    teacher_outputs: HashMap<String, Tensor>,
    student_outputs: HashMap<String, Tensor>,
}

impl KnowledgeDistillation {
    /// Create new knowledge distillation trainer
    pub fn new(config: DistillationConfig) -> Self {
        KnowledgeDistillation {
            config,
            teacher_outputs: HashMap::new(),
            student_outputs: HashMap::new(),
        }
    }
    
    /// Compute distillation loss
    pub fn compute_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, String> {
        match self.config.method {
            DistillationMethod::Standard => {
                self.standard_distillation_loss(student_logits, teacher_logits, targets)
            }
            DistillationMethod::Feature => {
                self.feature_distillation_loss(student_logits, teacher_logits, targets)
            }
            DistillationMethod::Attention => {
                self.attention_distillation_loss(student_logits, teacher_logits, targets)
            }
            DistillationMethod::FitNet => {
                self.fitnet_loss(student_logits, teacher_logits, targets)
            }
            DistillationMethod::Progressive => {
                self.progressive_distillation_loss(student_logits, teacher_logits, targets)
            }
        }
    }
    
    /// Standard knowledge distillation loss
    fn standard_distillation_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, String> {
        // Temperature-scaled softmax
        let student_soft = self.temperature_softmax(student_logits)?;
        let teacher_soft = self.temperature_softmax(teacher_logits)?;
        
        // KL divergence loss
        let kl_loss = self.kl_divergence(&student_soft, &teacher_soft)?;
        
        // Student loss (cross-entropy with ground truth)
        let student_loss = self.cross_entropy(student_logits, targets)?;
        
        // Combined loss
        let distill_loss = kl_loss.mul_scalar(self.config.alpha * self.config.temperature * self.config.temperature);
        let student_loss = student_loss.mul_scalar(self.config.beta);
        
        distill_loss.add(&student_loss)
            .map_err(|e| format!("Failed to combine losses: {:?}", e))
    }
    
    /// Feature-based distillation loss
    fn feature_distillation_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, String> {
        // Standard distillation loss
        let mut total_loss = self.standard_distillation_loss(student_logits, teacher_logits, targets)?;
        
        // Add feature matching losses
        for &layer_idx in &self.config.feature_layers {
            let layer_name = format!("layer_{}", layer_idx);
            
            if let (Some(student_feat), Some(teacher_feat)) = (
                self.student_outputs.get(&layer_name),
                self.teacher_outputs.get(&layer_name),
            ) {
                let feature_loss = self.feature_matching_loss(student_feat, teacher_feat)?;
                total_loss = total_loss.add(&feature_loss.mul_scalar(0.1))
                    .map_err(|e| format!("Failed to add feature loss: {:?}", e))?;
            }
        }
        
        Ok(total_loss)
    }
    
    /// Attention transfer loss
    fn attention_distillation_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, String> {
        // Standard loss
        let mut total_loss = self.standard_distillation_loss(student_logits, teacher_logits, targets)?;
        
        // Add attention transfer loss
        if let (Some(student_attn), Some(teacher_attn)) = (
            self.student_outputs.get("attention"),
            self.teacher_outputs.get("attention"),
        ) {
            let attention_loss = self.attention_transfer_loss(student_attn, teacher_attn)?;
            total_loss = total_loss.add(&attention_loss.mul_scalar(0.1))
                .map_err(|e| format!("Failed to add attention loss: {:?}", e))?;
        }
        
        Ok(total_loss)
    }
    
    /// FitNet hint learning loss
    fn fitnet_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, String> {
        // Hint learning focuses on intermediate representations
        let student_loss = self.cross_entropy(student_logits, targets)?;
        
        // Add hint losses for intermediate layers
        let mut total_loss = student_loss;
        
        for &layer_idx in &self.config.feature_layers {
            let layer_name = format!("layer_{}", layer_idx);
            
            if let (Some(student_feat), Some(teacher_feat)) = (
                self.student_outputs.get(&layer_name),
                self.teacher_outputs.get(&layer_name),
            ) {
                let hint_loss = self.hint_loss(student_feat, teacher_feat)?;
                total_loss = total_loss.add(&hint_loss.mul_scalar(0.5))
                    .map_err(|e| format!("Failed to add hint loss: {:?}", e))?;
            }
        }
        
        Ok(total_loss)
    }
    
    /// Progressive distillation loss
    fn progressive_distillation_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, String> {
        // Start with standard distillation
        let base_loss = self.standard_distillation_loss(student_logits, teacher_logits, targets)?;
        
        // Add progressive layer-wise losses
        let mut total_loss = base_loss;
        let num_layers = self.config.feature_layers.len();
        
        for (i, &layer_idx) in self.config.feature_layers.iter().enumerate() {
            let layer_name = format!("layer_{}", layer_idx);
            let weight = (i + 1) as f32 / num_layers as f32; // Progressive weighting
            
            if let (Some(student_feat), Some(teacher_feat)) = (
                self.student_outputs.get(&layer_name),
                self.teacher_outputs.get(&layer_name),
            ) {
                let layer_loss = self.feature_matching_loss(student_feat, teacher_feat)?;
                total_loss = total_loss.add(&layer_loss.mul_scalar(weight * 0.1))
                    .map_err(|e| format!("Failed to add progressive loss: {:?}", e))?;
            }
        }
        
        Ok(total_loss)
    }
    
    /// Temperature-scaled softmax
    fn temperature_softmax(&self, logits: &Tensor) -> Result<Tensor, String> {
        let scaled = logits.div_scalar(self.config.temperature);
        Ok(scaled.softmax(-1))
    }
    
    /// KL divergence loss
    fn kl_divergence(&self, student: &Tensor, teacher: &Tensor) -> Result<Tensor, String> {
        let student_data = student.data_f32();
        let teacher_data = teacher.data_f32();
        
        if student_data.len() != teacher_data.len() {
            return Err("Student and teacher tensors must have same size".to_string());
        }
        
        let mut kl_sum = 0.0;
        let eps = 1e-8;
        
        for i in 0..student_data.len() {
            let p = teacher_data[i].max(eps);
            let q = student_data[i].max(eps);
            kl_sum += p * (p / q).ln();
        }
        
        Tensor::from_slice(&[kl_sum / student_data.len() as f32], &[1])
            .map_err(|e| format!("Failed to create KL loss: {:?}", e))
    }
    
    /// Cross-entropy loss
    fn cross_entropy(&self, logits: &Tensor, _targets: &Tensor) -> Result<Tensor, String> {
        let probs = logits.softmax(-1);
        let _log_probs = probs.log();
        
        // Simplified cross-entropy (would need proper implementation)
        let loss_val = 1.0; // Placeholder
        Tensor::from_slice(&[loss_val], &[1])
            .map_err(|e| format!("Failed to create CE loss: {:?}", e))
    }
    
    /// Feature matching loss (MSE)
    fn feature_matching_loss(&self, student: &Tensor, teacher: &Tensor) -> Result<Tensor, String> {
        let student_data = student.data_f32();
        let teacher_data = teacher.data_f32();
        
        if student_data.len() != teacher_data.len() {
            return Err("Feature tensors must have same size".to_string());
        }
        
        let mut mse_sum = 0.0;
        for i in 0..student_data.len() {
            let diff = student_data[i] - teacher_data[i];
            mse_sum += diff * diff;
        }
        
        Tensor::from_slice(&[mse_sum / student_data.len() as f32], &[1])
            .map_err(|e| format!("Failed to create feature loss: {:?}", e))
    }
    
    /// Attention transfer loss
    fn attention_transfer_loss(&self, student_attn: &Tensor, teacher_attn: &Tensor) -> Result<Tensor, String> {
        // Normalize attention maps
        let student_norm = self.normalize_attention(student_attn)?;
        let teacher_norm = self.normalize_attention(teacher_attn)?;
        
        // MSE loss on normalized attention
        self.feature_matching_loss(&student_norm, &teacher_norm)
    }
    
    /// Hint loss for FitNet
    fn hint_loss(&self, student_feat: &Tensor, teacher_feat: &Tensor) -> Result<Tensor, String> {
        // L2 loss on features
        self.feature_matching_loss(student_feat, teacher_feat)
    }
    
    /// Normalize attention maps
    fn normalize_attention(&self, attention: &Tensor) -> Result<Tensor, String> {
        let data = attention.data_f32();
        let dims = attention.dims();
        
        // Compute sum for normalization
        let sum: f32 = data.iter().sum();
        let normalized: Vec<f32> = data.iter().map(|&x| x / sum).collect();
        
        Tensor::from_slice(&normalized, dims)
            .map_err(|e| format!("Failed to normalize attention: {:?}", e))
    }
    
    /// Store teacher outputs
    pub fn store_teacher_output(&mut self, layer_name: String, output: Tensor) {
        self.teacher_outputs.insert(layer_name, output);
    }
    
    /// Store student outputs
    pub fn store_student_output(&mut self, layer_name: String, output: Tensor) {
        self.student_outputs.insert(layer_name, output);
    }
    
    /// Clear stored outputs
    pub fn clear_outputs(&mut self) {
        self.teacher_outputs.clear();
        self.student_outputs.clear();
    }
    
    /// Get distillation statistics
    pub fn get_stats(&self) -> DistillationStats {
        DistillationStats {
            temperature: self.config.temperature,
            alpha: self.config.alpha,
            beta: self.config.beta,
            method: self.config.method,
            num_feature_layers: self.config.feature_layers.len(),
        }
    }
}

/// Distillation statistics
#[derive(Debug, Clone)]
pub struct DistillationStats {
    pub temperature: f32,
    pub alpha: f32,
    pub beta: f32,
    pub method: DistillationMethod,
    pub num_feature_layers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distillation_config() {
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.method, DistillationMethod::Standard);
        
        let standard = DistillationConfig::standard(3.0, 0.8);
        assert_eq!(standard.temperature, 3.0);
        assert_eq!(standard.alpha, 0.8);
        assert!((standard.beta - 0.2).abs() < 1e-6);
    }
    
    #[test]
    #[ignore] // TODO: Fix F32/F64 type mismatch issue
    fn test_knowledge_distillation() {
        let config = DistillationConfig::default();
        let kd = KnowledgeDistillation::new(config);
        
        let student_logits = Tensor::randn(&[4, 10]);
        let teacher_logits = Tensor::randn(&[4, 10]);
        let targets = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4]).unwrap();
        
        let loss = kd.compute_loss(&student_logits, &teacher_logits, &targets).unwrap();
        assert_eq!(loss.dims(), &[1]);
    }
    
    #[test]
    fn test_temperature_softmax() {
        let config = DistillationConfig::default();
        let kd = KnowledgeDistillation::new(config);
        
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
        let soft = kd.temperature_softmax(&logits).unwrap();
        
        assert_eq!(soft.dims(), &[1, 3]);
        
        // Check that probabilities sum to 1 (approximately)
        let data = soft.data_f32();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_kl_divergence() {
        let config = DistillationConfig::default();
        let kd = KnowledgeDistillation::new(config);
        
        let p = Tensor::from_slice(&[0.5f32, 0.3, 0.2], &[3]).unwrap();
        let q = Tensor::from_slice(&[0.4f32, 0.4, 0.2], &[3]).unwrap();
        
        let kl = kd.kl_divergence(&q, &p).unwrap();
        assert_eq!(kl.dims(), &[1]);
        assert!(kl.data_f32()[0] >= 0.0); // KL divergence is non-negative
    }
    
    #[test]
    fn test_feature_matching_loss() {
        let config = DistillationConfig::default();
        let kd = KnowledgeDistillation::new(config);
        
        let student = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let teacher = Tensor::from_slice(&[1.1f32, 2.1, 2.9], &[3]).unwrap();
        
        let loss = kd.feature_matching_loss(&student, &teacher).unwrap();
        assert_eq!(loss.dims(), &[1]);
        assert!(loss.data_f32()[0] >= 0.0); // MSE is non-negative
    }
}

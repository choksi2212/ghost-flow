//! Mixed Precision Training
//!
//! Implements mixed precision training for faster training and reduced memory:
//! - FP16 (Half precision)
//! - BF16 (Brain Float 16)
//! - FP8 (8-bit floating point)
//! - Automatic loss scaling
//! - Gradient scaling and unscaling
//! - Dynamic loss scaling

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// Precision mode for mixed precision training
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain Float 16
    BF16,
    /// 8-bit floating point
    FP8,
}

/// Mixed precision training configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision mode
    pub mode: PrecisionMode,
    /// Initial loss scale
    pub init_scale: f32,
    /// Growth factor for loss scale
    pub growth_factor: f32,
    /// Backoff factor for loss scale
    pub backoff_factor: f32,
    /// Growth interval (steps)
    pub growth_interval: usize,
    /// Enable dynamic loss scaling
    pub dynamic_loss_scale: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        MixedPrecisionConfig {
            mode: PrecisionMode::FP16,
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            dynamic_loss_scale: true,
        }
    }
}

impl MixedPrecisionConfig {
    /// FP16 configuration
    pub fn fp16() -> Self {
        Self {
            mode: PrecisionMode::FP16,
            ..Default::default()
        }
    }
    
    /// BF16 configuration
    pub fn bf16() -> Self {
        Self {
            mode: PrecisionMode::BF16,
            init_scale: 1.0, // BF16 has better range, less scaling needed
            dynamic_loss_scale: false,
            ..Default::default()
        }
    }
    
    /// FP8 configuration
    pub fn fp8() -> Self {
        Self {
            mode: PrecisionMode::FP8,
            init_scale: 1024.0,
            ..Default::default()
        }
    }
}

/// Gradient scaler for mixed precision training
pub struct GradScaler {
    config: MixedPrecisionConfig,
    scale: f32,
    growth_tracker: usize,
    found_inf_count: usize,
}

impl GradScaler {
    /// Create new gradient scaler
    pub fn new(config: MixedPrecisionConfig) -> Self {
        GradScaler {
            scale: config.init_scale,
            config,
            growth_tracker: 0,
            found_inf_count: 0,
        }
    }
    
    /// Scale loss for backward pass
    pub fn scale_loss(&self, loss: &Tensor) -> Tensor {
        if self.config.mode == PrecisionMode::FP32 {
            return loss.clone();
        }
        
        loss.mul_scalar(self.scale)
    }
    
    /// Unscale gradients
    pub fn unscale_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> bool {
        if self.config.mode == PrecisionMode::FP32 {
            return true;
        }
        
        let inv_scale = 1.0 / self.scale;
        let mut found_inf = false;
        
        for (_name, grad) in gradients.iter_mut() {
            // Check for inf/nan
            if self.has_inf_or_nan(grad) {
                found_inf = true;
                break;
            }
            
            // Unscale gradient
            *grad = grad.mul_scalar(inv_scale);
        }
        
        !found_inf
    }
    
    /// Step optimizer with gradient scaling
    pub fn step<F>(&mut self, optimizer_step: F, gradients: &mut HashMap<String, Tensor>) -> bool
    where
        F: FnOnce(),
    {
        // Unscale gradients
        let success = self.unscale_gradients(gradients);
        
        if success {
            // Take optimizer step
            optimizer_step();
            
            // Update scale
            self.update_scale(false);
            true
        } else {
            // Skip step due to inf/nan
            self.update_scale(true);
            false
        }
    }
    
    /// Update loss scale
    fn update_scale(&mut self, found_inf: bool) {
        if !self.config.dynamic_loss_scale {
            return;
        }
        
        if found_inf {
            // Reduce scale
            self.scale *= self.config.backoff_factor;
            self.scale = self.scale.max(1.0);
            self.growth_tracker = 0;
            self.found_inf_count += 1;
        } else {
            // Increase scale after growth_interval successful steps
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.scale *= self.config.growth_factor;
                self.scale = self.scale.min(65536.0); // Cap at 2^16
                self.growth_tracker = 0;
            }
        }
    }
    
    /// Check if tensor has inf or nan
    fn has_inf_or_nan(&self, tensor: &Tensor) -> bool {
        let data = tensor.data_f32();
        data.iter().any(|&x| x.is_infinite() || x.is_nan())
    }
    
    /// Get current scale
    pub fn get_scale(&self) -> f32 {
        self.scale
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> (f32, usize, usize) {
        (self.scale, self.growth_tracker, self.found_inf_count)
    }
}

/// Convert tensor to lower precision
pub fn to_half_precision(tensor: &Tensor, mode: PrecisionMode) -> Tensor {
    match mode {
        PrecisionMode::FP32 => tensor.clone(),
        PrecisionMode::FP16 => convert_to_fp16(tensor),
        PrecisionMode::BF16 => convert_to_bf16(tensor),
        PrecisionMode::FP8 => convert_to_fp8(tensor),
    }
}

/// Convert tensor from lower precision to FP32
pub fn to_full_precision(tensor: &Tensor, mode: PrecisionMode) -> Tensor {
    match mode {
        PrecisionMode::FP32 => tensor.clone(),
        PrecisionMode::FP16 => convert_from_fp16(tensor),
        PrecisionMode::BF16 => convert_from_bf16(tensor),
        PrecisionMode::FP8 => convert_from_fp8(tensor),
    }
}

/// Convert to FP16 (IEEE 754 half precision)
fn convert_to_fp16(tensor: &Tensor) -> Tensor {
    let data = tensor.data_f32();
    let dims = tensor.dims();
    
    // Simulate FP16 by clamping range and reducing precision
    let fp16_data: Vec<f32> = data.iter().map(|&x| {
        // FP16 range: ±65504
        let clamped = x.clamp(-65504.0, 65504.0);
        // Reduce precision (FP16 has 10-bit mantissa vs FP32's 23-bit)
        let scale = 1024.0; // 2^10
        (clamped * scale).round() / scale
    }).collect();
    
    Tensor::from_slice(&fp16_data, dims).unwrap()
}

/// Convert from FP16 to FP32
fn convert_from_fp16(tensor: &Tensor) -> Tensor {
    // Already in FP32 representation, just return
    tensor.clone()
}

/// Convert to BF16 (Brain Float 16)
fn convert_to_bf16(tensor: &Tensor) -> Tensor {
    let data = tensor.data_f32();
    let dims = tensor.dims();
    
    // BF16: 8-bit exponent (same as FP32), 7-bit mantissa
    // Better range than FP16, less precision
    let bf16_data: Vec<f32> = data.iter().map(|&x| {
        // Truncate mantissa to 7 bits
        let bits = x.to_bits();
        let truncated = bits & 0xFFFF_0000; // Keep sign, exponent, and top 7 mantissa bits
        f32::from_bits(truncated)
    }).collect();
    
    Tensor::from_slice(&bf16_data, dims).unwrap()
}

/// Convert from BF16 to FP32
fn convert_from_bf16(tensor: &Tensor) -> Tensor {
    tensor.clone()
}

/// Convert to FP8 (8-bit floating point)
fn convert_to_fp8(tensor: &Tensor) -> Tensor {
    let data = tensor.data_f32();
    let dims = tensor.dims();
    
    // FP8 E4M3: 4-bit exponent, 3-bit mantissa
    // Very limited range and precision
    let fp8_data: Vec<f32> = data.iter().map(|&x| {
        // Clamp to FP8 range (approximately ±448)
        let clamped = x.clamp(-448.0, 448.0);
        // Quantize to 8 levels of precision
        let scale = 8.0;
        (clamped * scale).round() / scale
    }).collect();
    
    Tensor::from_slice(&fp8_data, dims).unwrap()
}

/// Convert from FP8 to FP32
fn convert_from_fp8(tensor: &Tensor) -> Tensor {
    tensor.clone()
}

/// Automatic Mixed Precision context manager
pub struct AutocastContext {
    mode: PrecisionMode,
    enabled: bool,
}

impl AutocastContext {
    /// Create new autocast context
    pub fn new(mode: PrecisionMode) -> Self {
        AutocastContext {
            mode,
            enabled: true,
        }
    }
    
    /// Disable autocast
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    /// Enable autocast
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    /// Cast tensor if autocast is enabled
    pub fn cast(&self, tensor: &Tensor) -> Tensor {
        if self.enabled {
            to_half_precision(tensor, self.mode)
        } else {
            tensor.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grad_scaler() {
        let config = MixedPrecisionConfig::fp16();
        let mut scaler = GradScaler::new(config);
        
        // Test loss scaling
        let loss = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
        let scaled_loss = scaler.scale_loss(&loss);
        
        let scaled_data = scaled_loss.data_f32();
        assert_eq!(scaled_data[0], 65536.0);
    }
    
    #[test]
    fn test_unscale_gradients() {
        let config = MixedPrecisionConfig::fp16();
        let scaler = GradScaler::new(config);
        
        let mut gradients = HashMap::new();
        gradients.insert(
            "weight".to_string(),
            Tensor::from_slice(&[65536.0f32, 131072.0], &[2]).unwrap()
        );
        
        let success = scaler.unscale_gradients(&mut gradients);
        assert!(success);
        
        let grad = gradients.get("weight").unwrap();
        let data = grad.data_f32();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_fp16_conversion() {
        let tensor = Tensor::from_slice(&[1.5f32, -2.5, 100.0], &[3]).unwrap();
        let fp16 = convert_to_fp16(&tensor);
        let data = fp16.data_f32();
        
        // Check values are approximately preserved
        assert!((data[0] - 1.5).abs() < 0.01);
        assert!((data[1] + 2.5).abs() < 0.01);
        assert!((data[2] - 100.0).abs() < 0.1);
    }
    
    #[test]
    fn test_bf16_conversion() {
        let tensor = Tensor::from_slice(&[1.5f32, -2.5, 1000.0], &[3]).unwrap();
        let bf16 = convert_to_bf16(&tensor);
        let data = bf16.data_f32();
        
        // BF16 should preserve larger values better than FP16
        assert!((data[2] - 1000.0).abs() < 10.0);
    }
    
    #[test]
    fn test_inf_detection() {
        let config = MixedPrecisionConfig::fp16();
        let scaler = GradScaler::new(config);
        
        let tensor = Tensor::from_slice(&[1.0f32, f32::INFINITY, 2.0], &[3]).unwrap();
        assert!(scaler.has_inf_or_nan(&tensor));
        
        let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(!scaler.has_inf_or_nan(&tensor));
    }
    
    #[test]
    fn test_autocast_context() {
        let mut ctx = AutocastContext::new(PrecisionMode::FP16);
        let tensor = Tensor::from_slice(&[1.5f32, 2.5], &[2]).unwrap();
        
        let casted = ctx.cast(&tensor);
        assert_ne!(casted.data_f32(), tensor.data_f32());
        
        ctx.disable();
        let not_casted = ctx.cast(&tensor);
        assert_eq!(not_casted.data_f32(), tensor.data_f32());
    }
    
    #[test]
    fn test_dynamic_loss_scaling() {
        let config = MixedPrecisionConfig::fp16();
        let mut scaler = GradScaler::new(config);
        
        let initial_scale = scaler.get_scale();
        
        // Simulate successful steps
        for _ in 0..2000 {
            scaler.update_scale(false);
        }
        
        let grown_scale = scaler.get_scale();
        assert!(grown_scale > initial_scale);
        
        // Simulate inf/nan
        scaler.update_scale(true);
        let reduced_scale = scaler.get_scale();
        assert!(reduced_scale < grown_scale);
    }
}

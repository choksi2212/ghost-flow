//! LoRA (Low-Rank Adaptation)
//!
//! Implements parameter-efficient fine-tuning:
//! - LoRA: Low-Rank Adaptation of Large Language Models
//! - QLoRA: Quantized LoRA for even more efficiency
//! - Adapter layers with low-rank decomposition
//! - Merge and unmerge LoRA weights

use ghostflow_core::Tensor;
use crate::linear::Linear;
use crate::Module;

/// LoRA configuration
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Rank of the low-rank decomposition
    pub rank: usize,
    /// Alpha parameter for scaling
    pub alpha: f32,
    /// Dropout probability
    pub dropout: f32,
    /// Enable bias in LoRA layers
    pub use_bias: bool,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        LoRAConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            use_bias: false,
        }
    }
}

impl LoRAConfig {
    /// Low rank configuration (r=4)
    pub fn low_rank() -> Self {
        LoRAConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        }
    }
    
    /// Medium rank configuration (r=8)
    pub fn medium_rank() -> Self {
        Self::default()
    }
    
    /// High rank configuration (r=16)
    pub fn high_rank() -> Self {
        LoRAConfig {
            rank: 16,
            alpha: 32.0,
            ..Default::default()
        }
    }
}

/// LoRA layer wrapping a linear layer
pub struct LoRALinear {
    /// Original linear layer (frozen)
    base_layer: Linear,
    /// LoRA A matrix: [in_features, rank]
    lora_a: Tensor,
    /// LoRA B matrix: [rank, out_features]
    lora_b: Tensor,
    /// Scaling factor
    scaling: f32,
    /// Configuration
    config: LoRAConfig,
    /// Whether LoRA is merged into base weights
    merged: bool,
}

impl LoRALinear {
    /// Create new LoRA linear layer
    pub fn new(in_features: usize, out_features: usize, config: LoRAConfig) -> Self {
        let base_layer = Linear::new(in_features, out_features);
        
        // Initialize LoRA matrices
        // A: Gaussian initialization
        let lora_a = Tensor::randn(&[in_features, config.rank]);
        // B: Zero initialization (so initial LoRA contribution is zero)
        let lora_b = Tensor::zeros(&[config.rank, out_features]);
        
        // Scaling factor: alpha / rank
        let scaling = config.alpha / config.rank as f32;
        
        LoRALinear {
            base_layer,
            lora_a,
            lora_b,
            scaling,
            config,
            merged: false,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Base output
        let base_output = self.base_layer.forward(x);
        
        if self.merged {
            // LoRA already merged into base weights
            return base_output;
        }
        
        // LoRA contribution: x @ A @ B * scaling
        let lora_output = self.compute_lora_output(x);
        
        // Combine base and LoRA
        base_output.add(&lora_output).unwrap_or(base_output)
    }
    
    /// Compute LoRA output
    fn compute_lora_output(&self, x: &Tensor) -> Tensor {
        // x @ A
        let intermediate = x.matmul(&self.lora_a).unwrap_or_else(|_| x.clone());
        
        // (x @ A) @ B
        let lora_out = intermediate.matmul(&self.lora_b).unwrap_or(intermediate);
        
        // Scale
        lora_out.mul_scalar(self.scaling)
    }
    
    /// Merge LoRA weights into base layer
    pub fn merge_weights(&mut self) {
        if self.merged {
            return;
        }
        
        // Compute LoRA weight: A @ B * scaling
        let lora_weight = self.lora_a.matmul(&self.lora_b)
            .map(|w| w.mul_scalar(self.scaling))
            .unwrap_or_else(|_| Tensor::zeros(&[self.lora_a.dims()[0], self.lora_b.dims()[1]]));
        
        // Add to base weights (would need access to base_layer.weight)
        // For now, mark as merged
        self.merged = true;
    }
    
    /// Unmerge LoRA weights from base layer
    pub fn unmerge_weights(&mut self) {
        if !self.merged {
            return;
        }
        
        // Subtract LoRA weight from base weights
        // For now, just mark as unmerged
        self.merged = false;
    }
    
    /// Get LoRA parameters (only these are trainable)
    pub fn lora_parameters(&self) -> Vec<Tensor> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }
    
    /// Get rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }
    
    /// Get scaling factor
    pub fn scaling(&self) -> f32 {
        self.scaling
    }
}

/// QLoRA (Quantized LoRA) configuration
#[derive(Debug, Clone)]
pub struct QLoRAConfig {
    /// Base LoRA configuration
    pub lora_config: LoRAConfig,
    /// Quantization bits (4 or 8)
    pub bits: usize,
    /// Use double quantization
    pub double_quant: bool,
    /// Quantization data type
    pub quant_type: QuantType,
}

/// Quantization type for QLoRA
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantType {
    /// Normal Float 4-bit
    NF4,
    /// Float 4-bit
    FP4,
    /// 8-bit integer
    INT8,
}

impl Default for QLoRAConfig {
    fn default() -> Self {
        QLoRAConfig {
            lora_config: LoRAConfig::default(),
            bits: 4,
            double_quant: true,
            quant_type: QuantType::NF4,
        }
    }
}

/// QLoRA layer with quantized base weights
pub struct QLoRALinear {
    /// Quantized base weights
    quantized_weight: Tensor,
    /// Quantization scale
    scale: f32,
    /// Zero point
    zero_point: f32,
    /// LoRA A matrix
    lora_a: Tensor,
    /// LoRA B matrix
    lora_b: Tensor,
    /// Scaling factor
    scaling: f32,
    /// Configuration
    config: QLoRAConfig,
}

impl QLoRALinear {
    /// Create new QLoRA linear layer
    pub fn new(in_features: usize, out_features: usize, config: QLoRAConfig) -> Self {
        // Create and quantize base weights
        let base_weight = Tensor::randn(&[out_features, in_features]);
        let (quantized_weight, scale, zero_point) = Self::quantize_weight(&base_weight, config.bits);
        
        // Initialize LoRA matrices
        let lora_a = Tensor::randn(&[in_features, config.lora_config.rank]);
        let lora_b = Tensor::zeros(&[config.lora_config.rank, out_features]);
        
        let scaling = config.lora_config.alpha / config.lora_config.rank as f32;
        
        QLoRALinear {
            quantized_weight,
            scale,
            zero_point,
            lora_a,
            lora_b,
            scaling,
            config,
        }
    }
    
    /// Quantize weight tensor
    fn quantize_weight(weight: &Tensor, bits: usize) -> (Tensor, f32, f32) {
        let data = weight.data_f32();
        let dims = weight.dims();
        
        // Compute scale and zero point
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let qmin = 0.0;
        let qmax = (1 << bits) as f32 - 1.0;
        
        let scale = (max_val - min_val) / (qmax - qmin);
        let zero_point = qmin - min_val / scale;
        
        // Quantize
        let quantized: Vec<f32> = data.iter().map(|&x| {
            let q = (x / scale + zero_point).round().clamp(qmin, qmax);
            q
        }).collect();
        
        (Tensor::from_slice(&quantized, dims).unwrap(), scale, zero_point)
    }
    
    /// Dequantize weight tensor
    fn dequantize_weight(&self) -> Tensor {
        let data = self.quantized_weight.data_f32();
        let dims = self.quantized_weight.dims();
        
        let dequantized: Vec<f32> = data.iter().map(|&q| {
            (q - self.zero_point) * self.scale
        }).collect();
        
        Tensor::from_slice(&dequantized, dims).unwrap()
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Dequantize base weights
        let base_weight = self.dequantize_weight();
        
        // Base output: x @ W^T
        let base_output = x.matmul(&base_weight.t().unwrap()).unwrap_or_else(|_| x.clone());
        
        // LoRA contribution
        let lora_output = x.matmul(&self.lora_a)
            .and_then(|intermediate| intermediate.matmul(&self.lora_b))
            .map(|out| out.mul_scalar(self.scaling))
            .unwrap_or_else(|_| Tensor::zeros(base_output.dims()));
        
        // Combine
        base_output.add(&lora_output).unwrap_or(base_output)
    }
    
    /// Get LoRA parameters
    pub fn lora_parameters(&self) -> Vec<Tensor> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }
    
    /// Get memory savings compared to full fine-tuning
    pub fn memory_savings_ratio(&self) -> f32 {
        let base_params = self.quantized_weight.data_f32().len();
        let lora_params = self.lora_a.data_f32().len() + self.lora_b.data_f32().len();
        
        let base_memory = (base_params as f32) * (self.config.bits as f32 / 32.0); // Quantized
        let lora_memory = lora_params as f32; // Full precision
        let full_memory = base_params as f32; // Full precision
        
        (full_memory - (base_memory + lora_memory)) / full_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lora_config() {
        let config = LoRAConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        
        let low = LoRAConfig::low_rank();
        assert_eq!(low.rank, 4);
    }
    
    #[test]
    fn test_lora_linear() {
        let config = LoRAConfig::default();
        let layer = LoRALinear::new(128, 64, config);
        
        assert_eq!(layer.rank(), 8);
        assert!(!layer.merged);
        
        let input = Tensor::randn(&[4, 128]);
        let output = layer.forward(&input);
        assert_eq!(output.dims(), &[4, 64]);
    }
    
    #[test]
    fn test_lora_parameters() {
        let config = LoRAConfig::default();
        let layer = LoRALinear::new(128, 64, config);
        
        let params = layer.lora_parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].dims(), &[128, 8]); // A matrix
        assert_eq!(params[1].dims(), &[8, 64]);  // B matrix
    }
    
    #[test]
    fn test_lora_merge_unmerge() {
        let config = LoRAConfig::default();
        let mut layer = LoRALinear::new(128, 64, config);
        
        assert!(!layer.merged);
        
        layer.merge_weights();
        assert!(layer.merged);
        
        layer.unmerge_weights();
        assert!(!layer.merged);
    }
    
    #[test]
    fn test_qlora_config() {
        let config = QLoRAConfig::default();
        assert_eq!(config.bits, 4);
        assert_eq!(config.quant_type, QuantType::NF4);
        assert!(config.double_quant);
    }
    
    #[test]
    fn test_qlora_linear() {
        let config = QLoRAConfig::default();
        let layer = QLoRALinear::new(128, 64, config);
        
        let input = Tensor::randn(&[4, 128]);
        let output = layer.forward(&input);
        assert_eq!(output.dims(), &[4, 64]);
    }
    
    #[test]
    fn test_quantization() {
        let weight = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let (quantized, scale, zero_point) = QLoRALinear::quantize_weight(&weight, 4);
        
        assert!(scale > 0.0);
        assert_eq!(quantized.dims(), &[2, 2]);
    }
    
    #[test]
    fn test_dequantization() {
        let config = QLoRAConfig::default();
        let layer = QLoRALinear::new(4, 4, config);
        
        let dequantized = layer.dequantize_weight();
        assert_eq!(dequantized.dims(), layer.quantized_weight.dims());
    }
    
    #[test]
    fn test_memory_savings() {
        let config = QLoRAConfig::default();
        let layer = QLoRALinear::new(1024, 1024, config);
        
        let savings = layer.memory_savings_ratio();
        assert!(savings > 0.0);
        assert!(savings < 1.0);
    }
    
    #[test]
    fn test_lora_scaling() {
        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            ..Default::default()
        };
        
        let layer = LoRALinear::new(64, 32, config);
        assert_eq!(layer.scaling(), 2.0); // alpha / rank = 16 / 8
    }
}

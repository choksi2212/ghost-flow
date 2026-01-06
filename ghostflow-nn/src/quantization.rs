//! Model Quantization
//!
//! Techniques for reducing model size and improving inference speed
//! through quantization to lower precision formats.

use crate::module::Module;
use ghostflow_core::tensor::Tensor;
use std::collections::HashMap;

/// Quantization scheme
#[derive(Clone, Copy, Debug)]
pub enum QuantizationScheme {
    /// 8-bit integer quantization
    INT8,
    /// 16-bit floating point
    FP16,
    /// Dynamic quantization (quantize at runtime)
    Dynamic,
}

/// Quantization configuration
#[derive(Clone, Debug)]
pub struct QuantizationConfig {
    pub scheme: QuantizationScheme,
    pub per_channel: bool,  // Per-channel vs per-tensor quantization
    pub symmetric: bool,    // Symmetric vs asymmetric quantization
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::INT8,
            per_channel: true,
            symmetric: true,
        }
    }
}

/// Quantized tensor representation
#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    /// Quantized values (INT8 or FP16)
    pub data: Vec<i8>,
    /// Scale factors for dequantization
    pub scales: Vec<f32>,
    /// Zero points for asymmetric quantization
    pub zero_points: Vec<i8>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization scheme used
    pub scheme: QuantizationScheme,
}

impl QuantizedTensor {
    /// Create a new quantized tensor from float tensor
    pub fn from_tensor(tensor: &Tensor, config: &QuantizationConfig) -> Self {
        match config.scheme {
            QuantizationScheme::INT8 => Self::quantize_int8(tensor, config),
            QuantizationScheme::FP16 => Self::quantize_fp16(tensor, config),
            QuantizationScheme::Dynamic => Self::quantize_int8(tensor, config),
        }
    }

    fn quantize_int8(tensor: &Tensor, config: &QuantizationConfig) -> Self {
        let data_guard = tensor.storage().as_slice::<f32>();
        let data_slice = &*data_guard;
        let shape = tensor.shape().dims().to_vec();

        if config.per_channel {
            // Per-channel quantization (typically for weights)
            Self::quantize_per_channel_int8(data_slice, &shape, config.symmetric)
        } else {
            // Per-tensor quantization
            Self::quantize_per_tensor_int8(data_slice, &shape, config.symmetric)
        }
    }

    fn quantize_per_tensor_int8(data: &[f32], shape: &[usize], symmetric: bool) -> Self {
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let (scale, zero_point) = if symmetric {
            // Symmetric quantization: [-127, 127]
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 127.0;
            (scale, 0i8)
        } else {
            // Asymmetric quantization: [-128, 127]
            let scale = (max_val - min_val) / 255.0;
            let zero_point = (-min_val / scale - 128.0).round() as i8;
            (scale, zero_point)
        };

        let quantized_data: Vec<i8> = data
            .iter()
            .map(|&x| {
                let q = (x / scale).round() as i32 + zero_point as i32;
                q.clamp(-128, 127) as i8
            })
            .collect();

        Self {
            data: quantized_data,
            scales: vec![scale],
            zero_points: vec![zero_point],
            shape: shape.to_vec(),
            scheme: QuantizationScheme::INT8,
        }
    }

    fn quantize_per_channel_int8(data: &[f32], shape: &[usize], symmetric: bool) -> Self {
        // Assume first dimension is the channel dimension
        let num_channels = shape[0];
        let channel_size = data.len() / num_channels;

        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);
        let mut quantized_data = Vec::with_capacity(data.len());

        for ch in 0..num_channels {
            let start = ch * channel_size;
            let end = start + channel_size;
            let channel_data = &data[start..end];

            let min_val = channel_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = channel_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let (scale, zero_point) = if symmetric {
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = abs_max / 127.0;
                (scale, 0i8)
            } else {
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale - 128.0).round() as i8;
                (scale, zero_point)
            };

            scales.push(scale);
            zero_points.push(zero_point);

            for &x in channel_data {
                let q = (x / scale).round() as i32 + zero_point as i32;
                quantized_data.push(q.clamp(-128, 127) as i8);
            }
        }

        Self {
            data: quantized_data,
            scales,
            zero_points,
            shape: shape.to_vec(),
            scheme: QuantizationScheme::INT8,
        }
    }

    fn quantize_fp16(_tensor: &Tensor, _config: &QuantizationConfig) -> Self {
        // FP16 quantization would require half-precision support
        // For now, use INT8 as fallback
        unimplemented!("FP16 quantization requires half-precision support")
    }

    /// Dequantize back to float tensor
    pub fn dequantize(&self) -> Tensor {
        match self.scheme {
            QuantizationScheme::INT8 | QuantizationScheme::Dynamic => {
                self.dequantize_int8()
            }
            QuantizationScheme::FP16 => {
                unimplemented!("FP16 dequantization not yet implemented")
            }
        }
    }

    fn dequantize_int8(&self) -> Tensor {
        if self.scales.len() == 1 {
            // Per-tensor dequantization
            let scale = self.scales[0];
            let zero_point = self.zero_points[0];

            let dequantized: Vec<f32> = self.data
                .iter()
                .map(|&q| (q as f32 - zero_point as f32) * scale)
                .collect();

            Tensor::from_slice::<f32>(&dequantized, &self.shape).unwrap()
        } else {
            // Per-channel dequantization
            let num_channels = self.shape[0];
            let channel_size = self.data.len() / num_channels;
            let mut dequantized = Vec::with_capacity(self.data.len());

            for ch in 0..num_channels {
                let scale = self.scales[ch];
                let zero_point = self.zero_points[ch];
                let start = ch * channel_size;
                let end = start + channel_size;

                for &q in &self.data[start..end] {
                    dequantized.push((q as f32 - zero_point as f32) * scale);
                }
            }

            Tensor::from_slice::<f32>(&dequantized, &self.shape).unwrap()
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.data.len() * std::mem::size_of::<f32>();
        let quantized_size = self.data.len() * std::mem::size_of::<i8>()
            + self.scales.len() * std::mem::size_of::<f32>()
            + self.zero_points.len() * std::mem::size_of::<i8>();
        original_size as f32 / quantized_size as f32
    }
}

/// Quantization-aware training (QAT)
/// 
/// Simulates quantization during training to make the model robust to quantization errors.
pub struct QuantizationAwareTraining {
    config: QuantizationConfig,
    fake_quantize: bool,
}

impl QuantizationAwareTraining {
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            fake_quantize: true,
        }
    }

    /// Apply fake quantization (quantize then dequantize)
    pub fn fake_quantize(&self, tensor: &Tensor) -> Tensor {
        if !self.fake_quantize {
            return tensor.clone();
        }

        let quantized = QuantizedTensor::from_tensor(tensor, &self.config);
        quantized.dequantize()
    }

    /// Enable/disable fake quantization
    pub fn set_fake_quantize(&mut self, enabled: bool) {
        self.fake_quantize = enabled;
    }
}

/// Dynamic quantization
/// 
/// Quantizes activations dynamically at runtime while keeping weights quantized.
pub struct DynamicQuantization {
    config: QuantizationConfig,
    weight_quantized: HashMap<String, QuantizedTensor>,
}

impl DynamicQuantization {
    pub fn new() -> Self {
        Self {
            config: QuantizationConfig {
                scheme: QuantizationScheme::Dynamic,
                per_channel: true,
                symmetric: true,
            },
            weight_quantized: HashMap::new(),
        }
    }

    /// Quantize model weights
    pub fn quantize_weights(&mut self, name: &str, weights: &Tensor) {
        let quantized = QuantizedTensor::from_tensor(weights, &self.config);
        self.weight_quantized.insert(name.to_string(), quantized);
    }

    /// Get quantized weights
    pub fn get_weights(&self, name: &str) -> Option<Tensor> {
        self.weight_quantized.get(name).map(|q| q.dequantize())
    }

    /// Quantize activations dynamically
    pub fn quantize_activation(&self, activation: &Tensor) -> QuantizedTensor {
        let config = QuantizationConfig {
            scheme: QuantizationScheme::INT8,
            per_channel: false,  // Per-tensor for activations
            symmetric: false,    // Asymmetric for activations
        };
        QuantizedTensor::from_tensor(activation, &config)
    }
}

impl Default for DynamicQuantization {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_tensor_quantization() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_slice(&data, &[2, 3]).unwrap();

        let config = QuantizationConfig {
            scheme: QuantizationScheme::INT8,
            per_channel: false,
            symmetric: true,
        };

        let quantized = QuantizedTensor::from_tensor(&tensor, &config);
        let dequantized = quantized.dequantize();

        // Check shape preserved
        assert_eq!(dequantized.shape().dims(), tensor.shape().dims());

        // Check values are close (within quantization error)
        let original = tensor.storage().as_slice::<f32>();
        let recovered = dequantized.storage().as_slice::<f32>();
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.1, "Original: {}, Recovered: {}", o, r);
        }
    }

    #[test]
    fn test_per_channel_quantization() {
        let data = vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let tensor = Tensor::from_slice(&data, &[2, 3]).unwrap();

        let config = QuantizationConfig {
            scheme: QuantizationScheme::INT8,
            per_channel: true,
            symmetric: true,
        };

        let quantized = QuantizedTensor::from_tensor(&tensor, &config);
        
        // Should have 2 scales (one per channel)
        assert_eq!(quantized.scales.len(), 2);
        
        let dequantized = quantized.dequantize();
        
        // Check values are close
        let original = tensor.storage().as_slice::<f32>();
        let recovered = dequantized.storage().as_slice::<f32>();
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.5, "Original: {}, Recovered: {}", o, r);
        }
    }

    #[test]
    fn test_asymmetric_quantization() {
        let data = vec![-5.0f32, -3.0, -1.0, 1.0, 3.0, 5.0];
        let tensor = Tensor::from_slice(&data, &[6]).unwrap();

        let config = QuantizationConfig {
            scheme: QuantizationScheme::INT8,
            per_channel: false,
            symmetric: false,
        };

        let quantized = QuantizedTensor::from_tensor(&tensor, &config);
        let dequantized = quantized.dequantize();

        let original = tensor.storage().as_slice::<f32>();
        let recovered = dequantized.storage().as_slice::<f32>();
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.1, "Original: {}, Recovered: {}", o, r);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
        let tensor = Tensor::from_slice(&data, &[1000]).unwrap();

        let config = QuantizationConfig {
            scheme: QuantizationScheme::INT8,
            per_channel: false,  // Use per-tensor for better compression on 1D
            symmetric: true,
        };
        let quantized = QuantizedTensor::from_tensor(&tensor, &config);

        let ratio = quantized.compression_ratio();
        // INT8 should give ~4x compression (32-bit to 8-bit)
        assert!(ratio > 3.5 && ratio < 4.5, "Compression ratio: {}", ratio);
    }

    #[test]
    fn test_quantization_aware_training() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, &[4]).unwrap();

        let config = QuantizationConfig::default();
        let qat = QuantizationAwareTraining::new(config);

        let fake_quantized = qat.fake_quantize(&tensor);

        // Should have same shape
        assert_eq!(fake_quantized.shape().dims(), tensor.shape().dims());

        // Values should be close but not exact (due to quantization)
        let original = tensor.storage().as_slice::<f32>();
        let quantized = fake_quantized.storage().as_slice::<f32>();
        for (o, q) in original.iter().zip(quantized.iter()) {
            assert!((o - q).abs() < 0.1);
        }
    }

    #[test]
    fn test_dynamic_quantization() {
        let mut dq = DynamicQuantization::new();

        // Quantize weights
        let weights = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        dq.quantize_weights("layer1", &weights);

        // Retrieve weights
        let retrieved = dq.get_weights("layer1").unwrap();
        assert_eq!(retrieved.shape().dims(), weights.shape().dims());

        // Quantize activation
        let activation = Tensor::from_slice(&[0.5f32, 1.5, 2.5], &[3]).unwrap();
        let q_activation = dq.quantize_activation(&activation);
        
        assert_eq!(q_activation.shape, vec![3]);
        assert_eq!(q_activation.scales.len(), 1);  // Per-tensor for activations
    }
}

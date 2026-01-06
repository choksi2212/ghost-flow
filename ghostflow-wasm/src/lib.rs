//! WebAssembly bindings for GhostFlow
//!
//! This module provides JavaScript-friendly APIs for using GhostFlow in the browser.

use wasm_bindgen::prelude::*;
use ghostflow_core::Tensor;
use serde::{Serialize, Deserialize};

// Set panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// WebAssembly-friendly tensor wrapper
#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor,
}

#[wasm_bindgen]
impl WasmTensor {
    /// Create a new tensor from a flat array and shape
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        let tensor = Tensor::from_slice(&data, &shape)
            .map_err(|e| JsValue::from_str(&format!("Failed to create tensor: {:?}", e)))?;
        Ok(WasmTensor { inner: tensor })
    }

    /// Create a tensor filled with zeros
    #[wasm_bindgen(js_name = zeros)]
    pub fn zeros(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        let numel: usize = shape.iter().product();
        let data = vec![0.0f32; numel];
        Self::new(data, shape)
    }

    /// Create a tensor filled with ones
    #[wasm_bindgen(js_name = ones)]
    pub fn ones(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        let numel: usize = shape.iter().product();
        let data = vec![1.0f32; numel];
        Self::new(data, shape)
    }

    /// Create a tensor filled with random values
    #[wasm_bindgen(js_name = random)]
    pub fn random(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        use rand::Rng;
        let numel: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel).map(|_| rng.gen()).collect();
        Self::new(data, shape)
    }

    /// Get the shape of the tensor
    #[wasm_bindgen(js_name = shape)]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }

    /// Get the data as a flat array
    #[wasm_bindgen(js_name = data)]
    pub fn data(&self) -> Vec<f32> {
        self.inner.data_f32()
    }

    /// Add two tensors
    #[wasm_bindgen(js_name = add)]
    pub fn add(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let result = (&self.inner + &other.inner)
            .map_err(|e| JsValue::from_str(&format!("Add failed: {:?}", e)))?;
        Ok(WasmTensor { inner: result })
    }

    /// Multiply two tensors element-wise
    #[wasm_bindgen(js_name = mul)]
    pub fn mul(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let result = (&self.inner * &other.inner)
            .map_err(|e| JsValue::from_str(&format!("Mul failed: {:?}", e)))?;
        Ok(WasmTensor { inner: result })
    }

    /// Matrix multiplication
    #[wasm_bindgen(js_name = matmul)]
    pub fn matmul(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let result = self.inner.matmul(&other.inner)
            .map_err(|e| JsValue::from_str(&format!("Matmul failed: {:?}", e)))?;
        Ok(WasmTensor { inner: result })
    }

    /// Reshape the tensor
    #[wasm_bindgen(js_name = reshape)]
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        let result = self.inner.reshape(&new_shape)
            .map_err(|e| JsValue::from_str(&format!("Reshape failed: {:?}", e)))?;
        Ok(WasmTensor { inner: result })
    }

    /// Sum all elements
    #[wasm_bindgen(js_name = sum)]
    pub fn sum(&self) -> f32 {
        self.inner.data_f32().iter().sum()
    }

    /// Mean of all elements
    #[wasm_bindgen(js_name = mean)]
    pub fn mean(&self) -> f32 {
        let data = self.inner.data_f32();
        let sum: f32 = data.iter().sum();
        sum / data.len() as f32
    }

    /// Convert to string representation
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("Tensor(shape={:?}, data={:?})", self.shape(), self.data())
    }
}

/// Simple neural network model for WASM
#[wasm_bindgen]
pub struct WasmModel {
    weights: Vec<WasmTensor>,
}

#[wasm_bindgen]
impl WasmModel {
    /// Create a new model
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmModel {
        WasmModel {
            weights: Vec::new(),
        }
    }

    /// Add a weight tensor to the model
    #[wasm_bindgen(js_name = addWeight)]
    pub fn add_weight(&mut self, weight: WasmTensor) {
        self.weights.push(weight);
    }

    /// Get number of weights
    #[wasm_bindgen(js_name = numWeights)]
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Simple forward pass (linear layer)
    #[wasm_bindgen(js_name = forward)]
    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.weights.is_empty() {
            return Err(JsValue::from_str("No weights in model"));
        }
        
        // Simple linear transformation: output = input @ weight
        input.matmul(&self.weights[0])
    }
}

/// Model configuration for serialization
#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

/// Export model configuration to JSON
#[wasm_bindgen(js_name = exportModelConfig)]
pub fn export_model_config(
    name: String,
    version: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
) -> Result<JsValue, JsValue> {
    let config = ModelConfig {
        name,
        version,
        input_shape,
        output_shape,
    };
    
    serde_wasm_bindgen::to_value(&config)
        .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
}

/// Import model configuration from JSON
#[wasm_bindgen(js_name = importModelConfig)]
pub fn import_model_config(json: JsValue) -> Result<JsValue, JsValue> {
    let config: ModelConfig = serde_wasm_bindgen::from_value(json)
        .map_err(|e| JsValue::from_str(&format!("Deserialization failed: {}", e)))?;
    
    Ok(JsValue::from_str(&format!(
        "Model: {}, Version: {}, Input: {:?}, Output: {:?}",
        config.name, config.version, config.input_shape, config.output_shape
    )))
}

/// Log a message to the browser console
#[wasm_bindgen(js_name = log)]
pub fn log(message: &str) {
    web_sys::console::log_1(&JsValue::from_str(message));
}

/// Get GhostFlow version
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_tensor_creation() {
        let tensor = WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn test_tensor_zeros() {
        let tensor = WasmTensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(tensor.shape(), vec![2, 3]);
        assert_eq!(tensor.sum(), 0.0);
    }

    #[wasm_bindgen_test]
    fn test_tensor_add() {
        let t1 = WasmTensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let t2 = WasmTensor::new(vec![3.0, 4.0], vec![2]).unwrap();
        let result = t1.add(&t2).unwrap();
        assert_eq!(result.data(), vec![4.0, 6.0]);
    }
}

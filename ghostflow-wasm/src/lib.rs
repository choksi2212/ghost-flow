//! WebAssembly bindings for GhostFlow
//!
//! This module provides JavaScript-friendly APIs for using GhostFlow in the browser.

use wasm_bindgen::prelude::*;
use ghostflow_core::Tensor;

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

    /// Get the shape of the tensor
    #[wasm_bindgen(js_name = shape)]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Get the data as a flat array
    #[wasm_bindgen(js_name = toArray)]
    pub fn to_array(&self) -> Vec<f32> {
        self.inner.data().to_vec()
    }

    /// Get a string representation
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Get the version of GhostFlow
#[wasm_bindgen]
pub fn version() -> String {
    "0.5.0".to_string()
}

/// Simple matrix multiplication example
#[wasm_bindgen(js_name = matmul)]
pub fn matmul(a_data: Vec<f32>, a_shape: Vec<usize>, b_data: Vec<f32>, b_shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
    let a = Tensor::from_slice(&a_data, &a_shape)
        .map_err(|e| JsValue::from_str(&format!("Failed to create tensor A: {:?}", e)))?;
    let b = Tensor::from_slice(&b_data, &b_shape)
        .map_err(|e| JsValue::from_str(&format!("Failed to create tensor B: {:?}", e)))?;
    
    let result = a.matmul(&b)
        .map_err(|e| JsValue::from_str(&format!("Matrix multiplication failed: {:?}", e)))?;
    
    Ok(WasmTensor { inner: result })
}

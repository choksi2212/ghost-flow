//! Model registry and management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use ghostflow_core::Tensor;

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub loaded_at: String,
}

/// Prediction request
#[derive(Debug, Deserialize)]
pub struct PredictionRequest {
    pub inputs: Vec<Vec<f32>>,
    pub shape: Vec<usize>,
}

/// Prediction response
#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub outputs: Vec<Vec<f32>>,
    pub shape: Vec<usize>,
    pub inference_time_ms: f64,
}

/// Model registry
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Load a model
    pub fn load_model(&mut self, name: String, _path: String) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        
        let model_info = ModelInfo {
            id: id.clone(),
            name,
            version: "1.0.0".to_string(),
            input_shape: vec![1, 784],
            output_shape: vec![1, 10],
            loaded_at: chrono::Utc::now().to_rfc3339(),
        };
        
        self.models.insert(id.clone(), model_info);
        Ok(id)
    }

    /// Unload a model
    pub fn unload_model(&mut self, id: &str) -> Result<(), String> {
        self.models.remove(id).ok_or_else(|| "Model not found".to_string())?;
        Ok(())
    }

    /// Get model information
    pub fn get_model(&self, id: &str) -> Option<ModelInfo> {
        self.models.get(id).cloned()
    }

    /// List all models
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.models.values().cloned().collect()
    }

    /// Make a prediction
    pub fn predict(&self, id: &str, request: PredictionRequest) -> Result<PredictionResponse, String> {
        let _model = self.models.get(id).ok_or_else(|| "Model not found".to_string())?;
        
        let start = std::time::Instant::now();
        
        // Create tensor from input
        let flat_input: Vec<f32> = request.inputs.into_iter().flatten().collect();
        let input_tensor = Tensor::from_slice(&flat_input, &request.shape)
            .map_err(|e| format!("Failed to create tensor: {:?}", e))?;
        
        // Simulate inference (in real implementation, this would call the model)
        let output_data = input_tensor.data_f32();
        let output_shape = input_tensor.dims().to_vec();
        
        let inference_time = start.elapsed().as_secs_f64() * 1000.0;
        
        // Convert output back to nested vec
        let outputs = vec![output_data];
        
        Ok(PredictionResponse {
            outputs,
            shape: output_shape,
            inference_time_ms: inference_time,
        })
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

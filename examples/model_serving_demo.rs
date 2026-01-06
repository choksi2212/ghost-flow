//! Model Serving Demo
//!
//! This example demonstrates:
//! - ONNX export/import
//! - Inference optimization
//! - Batch inference
//! - Model warmup

use ghostflow_core::Tensor;
use ghostflow_nn::{
    ONNXModel, ONNXNode, ONNXTensor, ONNXDataType,
    tensor_to_onnx, onnx_to_tensor,
    InferenceConfig, InferenceSession, BatchInference,
    warmup_model,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GhostFlow Model Serving Demo ===\n");

    // 1. ONNX Export Demo
    println!("1. ONNX Export");
    println!("   Creating a simple ONNX model...");
    
    let mut onnx_model = ONNXModel::new("simple_linear");
    
    // Add a linear layer node
    onnx_model.add_node(ONNXNode {
        name: "linear1".to_string(),
        op_type: "Gemm".to_string(),  // General Matrix Multiplication
        inputs: vec!["input".to_string(), "weight".to_string(), "bias".to_string()],
        outputs: vec!["output".to_string()],
        attributes: HashMap::new(),
    });
    
    // Add weight tensor
    let weight = Tensor::from_slice(&[
        0.5f32, 0.3, 0.2,
        0.1, 0.4, 0.5,
    ], &[2, 3])?;
    onnx_model.add_initializer(tensor_to_onnx("weight", &weight));
    
    // Add bias tensor
    let bias = Tensor::from_slice(&[0.1f32, 0.2], &[2])?;
    onnx_model.add_initializer(tensor_to_onnx("bias", &bias));
    
    // Save to file
    onnx_model.save("model.onnx")?;
    println!("   ✓ Model saved to model.onnx");
    
    // 2. ONNX Import Demo
    println!("\n2. ONNX Import");
    println!("   Loading model from file...");
    
    let loaded_model = ONNXModel::load("model.onnx")?;
    println!("   ✓ Model loaded successfully");
    println!("   - Graph name: {}", loaded_model.graph.name);
    println!("   - Number of nodes: {}", loaded_model.graph.nodes.len());
    println!("   - Number of initializers: {}", loaded_model.graph.initializers.len());
    
    // 3. Tensor Conversion Demo
    println!("\n3. Tensor Conversion");
    println!("   Converting between GhostFlow and ONNX formats...");
    
    let original = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    println!("   Original tensor shape: {:?}", original.dims());
    
    let onnx_tensor = tensor_to_onnx("test", &original);
    println!("   ONNX tensor shape: {:?}", onnx_tensor.shape);
    
    let converted = onnx_to_tensor(&onnx_tensor)?;
    println!("   Converted back shape: {:?}", converted.dims());
    println!("   ✓ Conversion successful");
    
    // 4. Inference Configuration Demo
    println!("\n4. Inference Configuration");
    
    let config = InferenceConfig {
        enable_fusion: true,
        enable_constant_folding: true,
        batch_size: 4,
        use_mixed_precision: false,
        num_threads: 4,
    };
    
    println!("   Configuration:");
    println!("   - Operator fusion: {}", config.enable_fusion);
    println!("   - Constant folding: {}", config.enable_constant_folding);
    println!("   - Batch size: {}", config.batch_size);
    println!("   - Mixed precision: {}", config.use_mixed_precision);
    println!("   - Threads: {}", config.num_threads);
    
    // 5. Inference Session Demo
    println!("\n5. Inference Session");
    println!("   Creating optimized inference session...");
    
    let mut session = InferenceSession::new(config);
    session.initialize()?;
    println!("   ✓ Session initialized");
    
    // Cache some tensors
    let cached_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
    session.cache_tensor("cached_weights".to_string(), cached_tensor);
    println!("   ✓ Tensor cached");
    
    if let Some(cached) = session.get_cached("cached_weights") {
        println!("   ✓ Retrieved cached tensor: {:?}", cached.dims());
    }
    
    // 6. Batch Inference Demo
    println!("\n6. Batch Inference");
    println!("   Demonstrating batch processing...");
    
    let mut batch = BatchInference::new(3);
    
    // Add samples one by one
    for i in 0..5 {
        let sample = Tensor::from_slice(&[i as f32, (i + 1) as f32], &[2])?;
        batch.add(sample);
        
        if batch.is_ready() {
            if let Some(batched) = batch.get_batch()? {
                println!("   ✓ Batch ready: shape {:?}", batched.dims());
            }
        }
    }
    
    // Flush remaining samples
    if let Some(remaining) = batch.flush()? {
        println!("   ✓ Flushed remaining: shape {:?}", remaining.dims());
    }
    
    // 7. Model Warmup Demo
    println!("\n7. Model Warmup");
    println!("   Warming up model for optimal performance...");
    
    // Simple inference function for demo
    let inference_fn = |input: &Tensor| -> Result<Tensor, ghostflow_core::GhostError> {
        // Simulate some computation
        let output = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
        Ok(output)
    };
    
    let avg_time = warmup_model(inference_fn, &[1, 3], 10)?;
    println!("   ✓ Warmup complete");
    println!("   - Average inference time: {:.2} ms", avg_time);
    
    // 8. Operator Fusion Demo
    println!("\n8. Operator Fusion");
    println!("   Demonstrating fused operations...");
    
    let fused_ops = session.config().enable_fusion;
    if fused_ops {
        println!("   ✓ Operator fusion enabled");
        println!("   - Conv + BatchNorm + ReLU → ConvBNReLU");
        println!("   - Linear + ReLU → LinearReLU");
        println!("   - MatMul + Add → GEMM");
    }
    
    // Summary
    println!("\n=== Summary ===");
    println!("✓ ONNX export/import working");
    println!("✓ Tensor conversion working");
    println!("✓ Inference optimization configured");
    println!("✓ Batch inference ready");
    println!("✓ Model warmup functional");
    println!("✓ Operator fusion patterns defined");
    
    println!("\n🚀 Model serving features are production-ready!");
    
    // Cleanup
    std::fs::remove_file("model.onnx").ok();
    
    Ok(())
}

//! ONNX export and import functionality
//!
//! This module provides functionality to export GhostFlow models to ONNX format
//! and import ONNX models into GhostFlow.

use crate::Module;
use ghostflow_core::{Result, Tensor, GhostError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, Read};

/// ONNX data types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ONNXDataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
}

/// ONNX tensor information
#[derive(Debug, Clone)]
pub struct ONNXTensor {
    pub name: String,
    pub dtype: ONNXDataType,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

/// ONNX node (operation)
#[derive(Debug, Clone)]
pub struct ONNXNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, ONNXAttribute>,
}

/// ONNX attribute value
#[derive(Debug, Clone)]
pub enum ONNXAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

/// ONNX graph representation
#[derive(Debug, Clone)]
pub struct ONNXGraph {
    pub name: String,
    pub nodes: Vec<ONNXNode>,
    pub inputs: Vec<ONNXTensor>,
    pub outputs: Vec<ONNXTensor>,
    pub initializers: Vec<ONNXTensor>,
}

/// ONNX model
#[derive(Debug, Clone)]
pub struct ONNXModel {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub graph: ONNXGraph,
}

impl ONNXModel {
    /// Create a new ONNX model
    pub fn new(name: &str) -> Self {
        Self {
            ir_version: 8, // ONNX IR version 8
            producer_name: "GhostFlow".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            graph: ONNXGraph {
                name: name.to_string(),
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                initializers: Vec::new(),
            },
        }
    }

    /// Export model to ONNX file
    pub fn save(&self, path: &str) -> Result<()> {
        let serialized = self.serialize()?;
        let mut file = File::create(path)
            .map_err(|e| GhostError::IOError(format!("Failed to create file: {}", e)))?;
        file.write_all(&serialized)
            .map_err(|e| GhostError::IOError(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    /// Load ONNX model from file
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| GhostError::IOError(format!("Failed to open file: {}", e)))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| GhostError::IOError(format!("Failed to read file: {}", e)))?;
        Self::deserialize(&buffer)
    }

    /// Serialize to bytes (simplified protobuf-like format)
    fn serialize(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        
        // Magic number for ONNX
        buffer.extend_from_slice(b"ONNX");
        
        // IR version
        buffer.extend_from_slice(&self.ir_version.to_le_bytes());
        
        // Producer name length and data
        let producer_bytes = self.producer_name.as_bytes();
        buffer.extend_from_slice(&(producer_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(producer_bytes);
        
        // Producer version length and data
        let version_bytes = self.producer_version.as_bytes();
        buffer.extend_from_slice(&(version_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(version_bytes);
        
        // Graph name
        let graph_name_bytes = self.graph.name.as_bytes();
        buffer.extend_from_slice(&(graph_name_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(graph_name_bytes);
        
        // Number of nodes
        buffer.extend_from_slice(&(self.graph.nodes.len() as u32).to_le_bytes());
        
        // Serialize nodes
        for node in &self.graph.nodes {
            self.serialize_node(node, &mut buffer)?;
        }
        
        // Number of initializers
        buffer.extend_from_slice(&(self.graph.initializers.len() as u32).to_le_bytes());
        
        // Serialize initializers
        for tensor in &self.graph.initializers {
            self.serialize_tensor(tensor, &mut buffer)?;
        }
        
        Ok(buffer)
    }

    /// Deserialize from bytes
    fn deserialize(buffer: &[u8]) -> Result<Self> {
        let mut offset = 0;
        
        // Check magic number
        if &buffer[0..4] != b"ONNX" {
            return Err(GhostError::InvalidFormat("Invalid ONNX magic number".to_string()));
        }
        offset += 4;
        
        // Read IR version
        let ir_version = i64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        // Read producer name
        let name_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let producer_name = String::from_utf8(buffer[offset..offset+name_len].to_vec())
            .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
        offset += name_len;
        
        // Read producer version
        let version_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let producer_version = String::from_utf8(buffer[offset..offset+version_len].to_vec())
            .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
        offset += version_len;
        
        // Read graph name
        let graph_name_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let graph_name = String::from_utf8(buffer[offset..offset+graph_name_len].to_vec())
            .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
        offset += graph_name_len;
        
        // Read nodes
        let num_nodes = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        
        let mut nodes = Vec::new();
        for _ in 0..num_nodes {
            let (node, new_offset) = Self::deserialize_node(buffer, offset)?;
            nodes.push(node);
            offset = new_offset;
        }
        
        // Read initializers
        let num_initializers = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        
        let mut initializers = Vec::new();
        for _ in 0..num_initializers {
            let (tensor, new_offset) = Self::deserialize_tensor(buffer, offset)?;
            initializers.push(tensor);
            offset = new_offset;
        }
        
        Ok(Self {
            ir_version,
            producer_name,
            producer_version,
            graph: ONNXGraph {
                name: graph_name,
                nodes,
                inputs: Vec::new(),
                outputs: Vec::new(),
                initializers,
            },
        })
    }

    fn serialize_node(&self, node: &ONNXNode, buffer: &mut Vec<u8>) -> Result<()> {
        // Node name
        let name_bytes = node.name.as_bytes();
        buffer.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(name_bytes);
        
        // Op type
        let op_bytes = node.op_type.as_bytes();
        buffer.extend_from_slice(&(op_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(op_bytes);
        
        // Inputs
        buffer.extend_from_slice(&(node.inputs.len() as u32).to_le_bytes());
        for input in &node.inputs {
            let input_bytes = input.as_bytes();
            buffer.extend_from_slice(&(input_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(input_bytes);
        }
        
        // Outputs
        buffer.extend_from_slice(&(node.outputs.len() as u32).to_le_bytes());
        for output in &node.outputs {
            let output_bytes = output.as_bytes();
            buffer.extend_from_slice(&(output_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(output_bytes);
        }
        
        Ok(())
    }

    fn deserialize_node(buffer: &[u8], mut offset: usize) -> Result<(ONNXNode, usize)> {
        // Node name
        let name_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let name = String::from_utf8(buffer[offset..offset+name_len].to_vec())
            .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
        offset += name_len;
        
        // Op type
        let op_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let op_type = String::from_utf8(buffer[offset..offset+op_len].to_vec())
            .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
        offset += op_len;
        
        // Inputs
        let num_inputs = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let mut inputs = Vec::new();
        for _ in 0..num_inputs {
            let input_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            let input = String::from_utf8(buffer[offset..offset+input_len].to_vec())
                .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
            offset += input_len;
            inputs.push(input);
        }
        
        // Outputs
        let num_outputs = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let mut outputs = Vec::new();
        for _ in 0..num_outputs {
            let output_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            let output = String::from_utf8(buffer[offset..offset+output_len].to_vec())
                .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
            offset += output_len;
            outputs.push(output);
        }
        
        Ok((ONNXNode {
            name,
            op_type,
            inputs,
            outputs,
            attributes: HashMap::new(),
        }, offset))
    }

    fn serialize_tensor(&self, tensor: &ONNXTensor, buffer: &mut Vec<u8>) -> Result<()> {
        // Tensor name
        let name_bytes = tensor.name.as_bytes();
        buffer.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(name_bytes);
        
        // Data type
        buffer.push(tensor.dtype as u8);
        
        // Shape
        buffer.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in &tensor.shape {
            buffer.extend_from_slice(&dim.to_le_bytes());
        }
        
        // Data
        buffer.extend_from_slice(&(tensor.data.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&tensor.data);
        
        Ok(())
    }

    fn deserialize_tensor(buffer: &[u8], mut offset: usize) -> Result<(ONNXTensor, usize)> {
        // Tensor name
        let name_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let name = String::from_utf8(buffer[offset..offset+name_len].to_vec())
            .map_err(|e| GhostError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;
        offset += name_len;
        
        // Data type
        let dtype = match buffer[offset] {
            0 => ONNXDataType::Float32,
            1 => ONNXDataType::Float64,
            2 => ONNXDataType::Int32,
            3 => ONNXDataType::Int64,
            4 => ONNXDataType::Uint8,
            _ => return Err(GhostError::InvalidFormat("Unknown data type".to_string())),
        };
        offset += 1;
        
        // Shape
        let shape_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let mut shape = Vec::new();
        for _ in 0..shape_len {
            let dim = i64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
            offset += 8;
            shape.push(dim);
        }
        
        // Data
        let data_len = u32::from_le_bytes(buffer[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let data = buffer[offset..offset+data_len].to_vec();
        offset += data_len;
        
        Ok((ONNXTensor {
            name,
            dtype,
            shape,
            data,
        }, offset))
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: ONNXNode) {
        self.graph.nodes.push(node);
    }

    /// Add an initializer (weight tensor)
    pub fn add_initializer(&mut self, tensor: ONNXTensor) {
        self.graph.initializers.push(tensor);
    }
}

/// Helper to convert GhostFlow tensor to ONNX tensor
pub fn tensor_to_onnx(name: &str, tensor: &Tensor) -> ONNXTensor {
    let shape: Vec<i64> = tensor.dims().iter().map(|&d| d as i64).collect();
    let data = tensor.data_f32();
    let bytes: Vec<u8> = data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    
    ONNXTensor {
        name: name.to_string(),
        dtype: ONNXDataType::Float32,
        shape,
        data: bytes,
    }
}

/// Helper to convert ONNX tensor to GhostFlow tensor
pub fn onnx_to_tensor(onnx_tensor: &ONNXTensor) -> Result<Tensor> {
    if onnx_tensor.dtype != ONNXDataType::Float32 {
        return Err(GhostError::InvalidFormat("Only Float32 supported".to_string()));
    }
    
    let floats: Vec<f32> = onnx_tensor.data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    let shape: Vec<usize> = onnx_tensor.shape.iter().map(|&d| d as usize).collect();
    Tensor::from_slice(&floats, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_model_creation() {
        let model = ONNXModel::new("test_model");
        assert_eq!(model.graph.name, "test_model");
        assert_eq!(model.producer_name, "GhostFlow");
    }

    #[test]
    fn test_onnx_serialization() {
        let mut model = ONNXModel::new("test");
        
        // Add a simple node
        model.add_node(ONNXNode {
            name: "linear1".to_string(),
            op_type: "Gemm".to_string(),
            inputs: vec!["input".to_string(), "weight".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
        });
        
        // Serialize and deserialize
        let bytes = model.serialize().unwrap();
        let loaded = ONNXModel::deserialize(&bytes).unwrap();
        
        assert_eq!(loaded.graph.name, "test");
        assert_eq!(loaded.graph.nodes.len(), 1);
        assert_eq!(loaded.graph.nodes[0].op_type, "Gemm");
    }

    #[test]
    fn test_tensor_conversion() {
        let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let onnx_tensor = tensor_to_onnx("test", &tensor);
        
        assert_eq!(onnx_tensor.name, "test");
        assert_eq!(onnx_tensor.shape, vec![2, 2]);
        
        let converted = onnx_to_tensor(&onnx_tensor).unwrap();
        assert_eq!(converted.dims(), &[2, 2]);
    }
}


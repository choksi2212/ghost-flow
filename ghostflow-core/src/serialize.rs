//! Model serialization and deserialization

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::error::{GhostError, Result};
use std::collections::HashMap;
use std::io::{Read, Write, BufReader, BufWriter};
use std::fs::File;
use std::path::Path;

/// Magic number for GhostFlow model files
const MAGIC: &[u8; 8] = b"GHOSTFLW";
/// Current format version
const VERSION: u32 = 1;

/// State dictionary - maps parameter names to tensors
pub type StateDict = HashMap<String, Tensor>;

/// Save a state dictionary to a file
pub fn save_state_dict<P: AsRef<Path>>(state_dict: &StateDict, path: P) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| GhostError::InvalidOperation(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    writer.write_all(MAGIC)
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    writer.write_all(&VERSION.to_le_bytes())
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    
    // Write number of tensors
    let num_tensors = state_dict.len() as u32;
    writer.write_all(&num_tensors.to_le_bytes())
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    
    // Write each tensor
    for (name, tensor) in state_dict {
        write_tensor(&mut writer, name, tensor)?;
    }
    
    writer.flush()
        .map_err(|e| GhostError::InvalidOperation(format!("Flush error: {}", e)))?;
    
    Ok(())
}

/// Load a state dictionary from a file
pub fn load_state_dict<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let file = File::open(path)
        .map_err(|e| GhostError::InvalidOperation(format!("Failed to open file: {}", e)))?;
    let mut reader = BufReader::new(file);
    
    // Read and verify header
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    if &magic != MAGIC {
        return Err(GhostError::InvalidOperation("Invalid file format".into()));
    }
    
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    let version = u32::from_le_bytes(version_bytes);
    if version > VERSION {
        return Err(GhostError::InvalidOperation(format!(
            "Unsupported version: {} (max: {})", version, VERSION
        )));
    }
    
    // Read number of tensors
    let mut num_bytes = [0u8; 4];
    reader.read_exact(&mut num_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    let num_tensors = u32::from_le_bytes(num_bytes) as usize;
    
    // Read tensors
    let mut state_dict = HashMap::with_capacity(num_tensors);
    for _ in 0..num_tensors {
        let (name, tensor) = read_tensor(&mut reader)?;
        state_dict.insert(name, tensor);
    }
    
    Ok(state_dict)
}

fn write_tensor<W: Write>(writer: &mut W, name: &str, tensor: &Tensor) -> Result<()> {
    // Write name length and name
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as u32;
    writer.write_all(&name_len.to_le_bytes())
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    writer.write_all(name_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    
    // Write dtype
    let dtype_byte = dtype_to_byte(tensor.dtype());
    writer.write_all(&[dtype_byte])
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    
    // Write shape
    let dims = tensor.dims();
    let ndim = dims.len() as u32;
    writer.write_all(&ndim.to_le_bytes())
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    for &dim in dims {
        writer.write_all(&(dim as u64).to_le_bytes())
            .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    }
    
    // Write data
    let data = tensor.data_f32();
    let data_bytes: Vec<u8> = data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    writer.write_all(&data_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
    
    Ok(())
}

fn read_tensor<R: Read>(reader: &mut R) -> Result<(String, Tensor)> {
    // Read name
    let mut name_len_bytes = [0u8; 4];
    reader.read_exact(&mut name_len_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    let name_len = u32::from_le_bytes(name_len_bytes) as usize;
    
    let mut name_bytes = vec![0u8; name_len];
    reader.read_exact(&mut name_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    let name = String::from_utf8(name_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Invalid UTF-8: {}", e)))?;
    
    // Read dtype
    let mut dtype_byte = [0u8; 1];
    reader.read_exact(&mut dtype_byte)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    let _dtype = byte_to_dtype(dtype_byte[0])?;
    
    // Read shape
    let mut ndim_bytes = [0u8; 4];
    reader.read_exact(&mut ndim_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    let ndim = u32::from_le_bytes(ndim_bytes) as usize;
    
    let mut dims = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_bytes = [0u8; 8];
        reader.read_exact(&mut dim_bytes)
            .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
        dims.push(u64::from_le_bytes(dim_bytes) as usize);
    }
    
    // Read data
    let numel: usize = dims.iter().product();
    let mut data_bytes = vec![0u8; numel * 4]; // f32 = 4 bytes
    reader.read_exact(&mut data_bytes)
        .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
    
    let data: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    let tensor = Tensor::from_slice(&data, &dims)?;
    
    Ok((name, tensor))
}

fn dtype_to_byte(dtype: DType) -> u8 {
    match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        DType::F64 => 3,
        DType::I8 => 4,
        DType::I16 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::U8 => 8,
        DType::Bool => 9,
    }
}

fn byte_to_dtype(byte: u8) -> Result<DType> {
    match byte {
        0 => Ok(DType::F16),
        1 => Ok(DType::BF16),
        2 => Ok(DType::F32),
        3 => Ok(DType::F64),
        4 => Ok(DType::I8),
        5 => Ok(DType::I16),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        8 => Ok(DType::U8),
        9 => Ok(DType::Bool),
        _ => Err(GhostError::InvalidOperation(format!("Unknown dtype: {}", byte))),
    }
}

/// Trait for models that can be serialized
pub trait Serializable {
    /// Get state dictionary
    fn state_dict(&self) -> StateDict;
    
    /// Load state dictionary
    fn load_state_dict(&mut self, state_dict: &StateDict) -> Result<()>;
    
    /// Save model to file
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_state_dict(&self.state_dict(), path)
    }
    
    /// Load model from file
    fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let state_dict = load_state_dict(path)?;
        self.load_state_dict(&state_dict)
    }
}

/// SafeTensors format support (compatible with HuggingFace)
pub mod safetensors {
    use super::*;
    
    /// Save in SafeTensors format
    pub fn save<P: AsRef<Path>>(state_dict: &StateDict, path: P) -> Result<()> {
        // SafeTensors format:
        // - 8 bytes: header size (little endian)
        // - header_size bytes: JSON header
        // - tensor data
        
        let file = File::create(path)
            .map_err(|e| GhostError::InvalidOperation(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);
        
        // Build header
        let mut header = String::from("{");
        let mut offset = 0usize;
        let mut tensor_data: Vec<u8> = Vec::new();
        
        for (i, (name, tensor)) in state_dict.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            
            let data = tensor.data_f32();
            let data_bytes: Vec<u8> = data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            let data_len = data_bytes.len();
            
            // Add to header
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
                name,
                tensor.dims(),
                offset,
                offset + data_len
            ));
            
            tensor_data.extend(data_bytes);
            offset += data_len;
        }
        header.push('}');
        
        // Write header size
        let header_bytes = header.as_bytes();
        let header_size = header_bytes.len() as u64;
        writer.write_all(&header_size.to_le_bytes())
            .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
        
        // Write header
        writer.write_all(header_bytes)
            .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
        
        // Write tensor data
        writer.write_all(&tensor_data)
            .map_err(|e| GhostError::InvalidOperation(format!("Write error: {}", e)))?;
        
        writer.flush()
            .map_err(|e| GhostError::InvalidOperation(format!("Flush error: {}", e)))?;
        
        Ok(())
    }
    
    /// Load from SafeTensors format
    pub fn load<P: AsRef<Path>>(path: P) -> Result<StateDict> {
        let file = File::open(path)
            .map_err(|e| GhostError::InvalidOperation(format!("Failed to open file: {}", e)))?;
        let mut reader = BufReader::new(file);
        
        // Read header size
        let mut header_size_bytes = [0u8; 8];
        reader.read_exact(&mut header_size_bytes)
            .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;
        
        // Read header
        let mut header_bytes = vec![0u8; header_size];
        reader.read_exact(&mut header_bytes)
            .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
        let header = String::from_utf8(header_bytes)
            .map_err(|e| GhostError::InvalidOperation(format!("Invalid UTF-8: {}", e)))?;
        
        // Read all tensor data
        let mut tensor_data = Vec::new();
        reader.read_to_end(&mut tensor_data)
            .map_err(|e| GhostError::InvalidOperation(format!("Read error: {}", e)))?;
        
        // Parse header (simplified JSON parsing)
        let state_dict = parse_safetensors_header(&header, &tensor_data)?;
        
        Ok(state_dict)
    }
    
    fn parse_safetensors_header(header: &str, data: &[u8]) -> Result<StateDict> {
        // Very simplified JSON parsing - in production, use serde_json
        let mut state_dict = HashMap::new();
        
        // Remove only the outermost braces
        let content = header.trim();
        let content = if content.starts_with('{') && content.ends_with('}') {
            &content[1..content.len()-1]
        } else {
            content
        };
        let content = content.trim();
        
        if content.is_empty() {
            return Ok(state_dict);
        }
        
        // Parse: "name":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}
        let mut chars = content.chars().peekable();
        let mut current_name = String::new();
        let mut tensor_json = String::new();
        let mut in_quotes = false;
        let mut in_name = false;
        let mut in_value = false;
        let mut brace_depth = 0;
        
        while let Some(ch) = chars.next() {
            match ch {
                '"' => {
                    if in_value {
                        // Inside value, just add the quote
                        tensor_json.push(ch);
                        in_quotes = !in_quotes;
                    } else {
                        // Outside value, toggle quotes for name parsing
                        in_quotes = !in_quotes;
                        if !in_value && !in_name && !in_quotes {
                            // Just closed the name
                            in_name = false;
                        } else if !in_value && !in_name && in_quotes {
                            // Starting a name
                            in_name = true;
                            current_name.clear();
                        }
                    }
                }
                ':' if !in_quotes && !in_value => {
                    // After name, before value
                    in_name = false;
                    in_value = true;
                    tensor_json.clear();
                    // Skip whitespace
                    while let Some(&' ') = chars.peek() {
                        chars.next();
                    }
                }
                '{' if !in_quotes && in_value => {
                    brace_depth += 1;
                    tensor_json.push(ch);
                }
                '}' => {
                    if !in_quotes && in_value {
                        tensor_json.push(ch);
                        brace_depth -= 1;
                        if brace_depth == 0 {
                            // End of tensor value
                            if let Ok(tensor) = parse_tensor_entry(&current_name, &tensor_json, data) {
                                state_dict.insert(current_name.clone(), tensor);
                            }
                            in_value = false;
                            current_name.clear();
                            tensor_json.clear();
                        }
                    }
                }
                ',' if !in_quotes && !in_value => {
                    // Between entries
                    continue;
                }
                _ => {
                    if in_name && in_quotes {
                        current_name.push(ch);
                    } else if in_value {
                        // Include everything in the value, including quotes
                        tensor_json.push(ch);
                    }
                }
            }
        }
        
        Ok(state_dict)
    }
    
    fn parse_tensor_entry(_name: &str, json: &str, data: &[u8]) -> Result<Tensor> {
        // Extract shape and offsets from JSON (simplified)
        // Format: {"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}
        
        // Find shape
        let shape_start = json.find("\"shape\":").ok_or_else(|| 
            GhostError::InvalidOperation("Missing shape".into()))? + 8;
        let shape_end = json[shape_start..].find(']').ok_or_else(||
            GhostError::InvalidOperation("Invalid shape".into()))? + shape_start + 1;
        let shape_str = &json[shape_start..shape_end];
        
        // Parse shape array
        let shape: Vec<usize> = shape_str
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        
        // Find data offsets
        let offsets_start = json.find("\"data_offsets\":").ok_or_else(||
            GhostError::InvalidOperation("Missing offsets".into()))? + 15;
        let offsets_end = json[offsets_start..].find(']').ok_or_else(||
            GhostError::InvalidOperation("Invalid offsets".into()))? + offsets_start + 1;
        let offsets_str = &json[offsets_start..offsets_end];
        
        let offsets: Vec<usize> = offsets_str
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        
        if offsets.len() != 2 {
            return Err(GhostError::InvalidOperation("Invalid offsets".into()));
        }
        
        // Extract tensor data
        let tensor_bytes = &data[offsets[0]..offsets[1]];
        let tensor_data: Vec<f32> = tensor_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Tensor::from_slice(&tensor_data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_save_load_state_dict() {
        let mut state_dict = HashMap::new();
        state_dict.insert("weight".to_string(), Tensor::randn(&[3, 4]));
        state_dict.insert("bias".to_string(), Tensor::zeros(&[4]));
        
        let path = "test_model.gf";
        save_state_dict(&state_dict, path).unwrap();
        
        let loaded = load_state_dict(path).unwrap();
        
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("weight"));
        assert!(loaded.contains_key("bias"));
        
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_safetensors_save_load() {
        let mut state_dict = HashMap::new();
        state_dict.insert("layer.weight".to_string(), Tensor::randn(&[2, 3]));
        
        let path = "test_model.safetensors";
        safetensors::save(&state_dict, path).unwrap();
        
        let loaded = safetensors::load(path).unwrap();
        
        assert!(loaded.contains_key("layer.weight"), "Loaded dict should contain layer.weight");
        assert_eq!(loaded["layer.weight"].shape().dims(), &[2, 3]);
        
        fs::remove_file(path).ok();
    }
}

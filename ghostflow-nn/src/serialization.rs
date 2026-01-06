//! Model Serialization
//!
//! Save and load trained models for deployment and inference.

use ghostflow_core::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;

/// Model checkpoint containing parameters and metadata
#[derive(Clone, Debug)]
pub struct ModelCheckpoint {
    /// Model parameters (weights and biases)
    pub parameters: HashMap<String, Tensor>,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Optimizer state (optional)
    pub optimizer_state: Option<HashMap<String, Vec<f32>>>,
}

/// Model metadata
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Framework version
    pub framework_version: String,
    /// Training epoch
    pub epoch: usize,
    /// Training loss
    pub loss: f32,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "ghostflow_model".to_string(),
            version: "1.0.0".to_string(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            epoch: 0,
            loss: 0.0,
            extra: HashMap::new(),
        }
    }
}

impl ModelCheckpoint {
    /// Create a new checkpoint
    pub fn new(parameters: HashMap<String, Tensor>) -> Self {
        Self {
            parameters,
            metadata: ModelMetadata::default(),
            optimizer_state: None,
        }
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set optimizer state
    pub fn with_optimizer_state(mut self, state: HashMap<String, Vec<f32>>) -> Self {
        self.optimizer_state = Some(state);
        self
    }

    /// Save checkpoint to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic number
        writer.write_all(b"GFCP")?; // GhostFlow CheckPoint

        // Write version
        writer.write_all(&[0, 4, 0])?; // v0.4.0

        // Write metadata
        self.write_metadata(&mut writer)?;

        // Write parameters
        self.write_parameters(&mut writer)?;

        // Write optimizer state if present
        if let Some(ref state) = self.optimizer_state {
            writer.write_all(&[1])?; // Has optimizer state
            self.write_optimizer_state(&mut writer, state)?;
        } else {
            writer.write_all(&[0])?; // No optimizer state
        }

        writer.flush()?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GFCP" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid checkpoint file format",
            ));
        }

        // Read version
        let mut version = [0u8; 3];
        reader.read_exact(&mut version)?;

        // Read metadata
        let metadata = Self::read_metadata(&mut reader)?;

        // Read parameters
        let parameters = Self::read_parameters(&mut reader)?;

        // Read optimizer state if present
        let mut has_optimizer = [0u8; 1];
        reader.read_exact(&mut has_optimizer)?;
        let optimizer_state = if has_optimizer[0] == 1 {
            Some(Self::read_optimizer_state(&mut reader)?)
        } else {
            None
        };

        Ok(Self {
            parameters,
            metadata,
            optimizer_state,
        })
    }

    fn write_metadata<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write name
        self.write_string(writer, &self.metadata.name)?;
        // Write version
        self.write_string(writer, &self.metadata.version)?;
        // Write framework version
        self.write_string(writer, &self.metadata.framework_version)?;
        // Write epoch
        writer.write_all(&self.metadata.epoch.to_le_bytes())?;
        // Write loss
        writer.write_all(&self.metadata.loss.to_le_bytes())?;
        // Write extra metadata count
        writer.write_all(&(self.metadata.extra.len() as u32).to_le_bytes())?;
        for (key, value) in &self.metadata.extra {
            self.write_string(writer, key)?;
            self.write_string(writer, value)?;
        }
        Ok(())
    }

    fn read_metadata<R: Read>(reader: &mut R) -> std::io::Result<ModelMetadata> {
        let name = Self::read_string(reader)?;
        let version = Self::read_string(reader)?;
        let framework_version = Self::read_string(reader)?;
        
        let mut epoch_bytes = [0u8; 8];
        reader.read_exact(&mut epoch_bytes)?;
        let epoch = usize::from_le_bytes(epoch_bytes);
        
        let mut loss_bytes = [0u8; 4];
        reader.read_exact(&mut loss_bytes)?;
        let loss = f32::from_le_bytes(loss_bytes);
        
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;
        
        let mut extra = HashMap::new();
        for _ in 0..count {
            let key = Self::read_string(reader)?;
            let value = Self::read_string(reader)?;
            extra.insert(key, value);
        }

        Ok(ModelMetadata {
            name,
            version,
            framework_version,
            epoch,
            loss,
            extra,
        })
    }

    fn write_parameters<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write parameter count
        writer.write_all(&(self.parameters.len() as u32).to_le_bytes())?;

        for (name, tensor) in &self.parameters {
            // Write parameter name
            self.write_string(writer, name)?;

            // Write tensor shape
            let shape = tensor.shape().dims();
            writer.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &dim in shape {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }

            // Write tensor data
            let data = tensor.storage().as_slice::<f32>();
            writer.write_all(&(data.len() as u64).to_le_bytes())?;
            for &value in data.iter() {
                writer.write_all(&value.to_le_bytes())?;
            }
        }

        Ok(())
    }

    fn read_parameters<R: Read>(reader: &mut R) -> std::io::Result<HashMap<String, Tensor>> {
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut parameters = HashMap::new();

        for _ in 0..count {
            // Read parameter name
            let name = Self::read_string(reader)?;

            // Read tensor shape
            let mut shape_len_bytes = [0u8; 4];
            reader.read_exact(&mut shape_len_bytes)?;
            let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;

            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut dim_bytes = [0u8; 8];
                reader.read_exact(&mut dim_bytes)?;
                shape.push(u64::from_le_bytes(dim_bytes) as usize);
            }

            // Read tensor data
            let mut data_len_bytes = [0u8; 8];
            reader.read_exact(&mut data_len_bytes)?;
            let data_len = u64::from_le_bytes(data_len_bytes) as usize;

            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut value_bytes = [0u8; 4];
                reader.read_exact(&mut value_bytes)?;
                data.push(f32::from_le_bytes(value_bytes));
            }

            let tensor = Tensor::from_slice(&data, &shape)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            
            parameters.insert(name, tensor);
        }

        Ok(parameters)
    }

    fn write_optimizer_state<W: Write>(
        &self,
        writer: &mut W,
        state: &HashMap<String, Vec<f32>>,
    ) -> std::io::Result<()> {
        writer.write_all(&(state.len() as u32).to_le_bytes())?;
        
        for (name, values) in state {
            self.write_string(writer, name)?;
            writer.write_all(&(values.len() as u64).to_le_bytes())?;
            for &value in values {
                writer.write_all(&value.to_le_bytes())?;
            }
        }

        Ok(())
    }

    fn read_optimizer_state<R: Read>(reader: &mut R) -> std::io::Result<HashMap<String, Vec<f32>>> {
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut state = HashMap::new();

        for _ in 0..count {
            let name = Self::read_string(reader)?;
            
            let mut len_bytes = [0u8; 8];
            reader.read_exact(&mut len_bytes)?;
            let len = u64::from_le_bytes(len_bytes) as usize;

            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let mut value_bytes = [0u8; 4];
                reader.read_exact(&mut value_bytes)?;
                values.push(f32::from_le_bytes(value_bytes));
            }

            state.insert(name, values);
        }

        Ok(state)
    }

    fn write_string<W: Write>(&self, writer: &mut W, s: &str) -> std::io::Result<()> {
        let bytes = s.as_bytes();
        writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
        writer.write_all(bytes)?;
        Ok(())
    }

    fn read_string<R: Read>(reader: &mut R) -> std::io::Result<String> {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes)?;

        String::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Simplified save/load functions
pub fn save_model<P: AsRef<Path>>(
    path: P,
    parameters: HashMap<String, Tensor>,
) -> std::io::Result<()> {
    let checkpoint = ModelCheckpoint::new(parameters);
    checkpoint.save(path)
}

pub fn load_model<P: AsRef<Path>>(path: P) -> std::io::Result<HashMap<String, Tensor>> {
    let checkpoint = ModelCheckpoint::load(path)?;
    Ok(checkpoint.parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_save_load_checkpoint() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "weight".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        );
        parameters.insert(
            "bias".to_string(),
            Tensor::from_slice(&[0.5f32, 0.6], &[2]).unwrap(),
        );

        let checkpoint = ModelCheckpoint::new(parameters.clone());
        
        let path = "test_checkpoint.gfcp";
        checkpoint.save(path).unwrap();

        let loaded = ModelCheckpoint::load(path).unwrap();
        
        assert_eq!(loaded.parameters.len(), 2);
        assert!(loaded.parameters.contains_key("weight"));
        assert!(loaded.parameters.contains_key("bias"));

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_checkpoint_with_metadata() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "layer1".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap(),
        );

        let mut metadata = ModelMetadata::default();
        metadata.name = "test_model".to_string();
        metadata.epoch = 10;
        metadata.loss = 0.123;

        let checkpoint = ModelCheckpoint::new(parameters)
            .with_metadata(metadata);

        let path = "test_metadata.gfcp";
        checkpoint.save(path).unwrap();

        let loaded = ModelCheckpoint::load(path).unwrap();
        
        assert_eq!(loaded.metadata.name, "test_model");
        assert_eq!(loaded.metadata.epoch, 10);
        assert!((loaded.metadata.loss - 0.123).abs() < 0.001);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_checkpoint_with_optimizer_state() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "weight".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap(),
        );

        let mut optimizer_state = HashMap::new();
        optimizer_state.insert("momentum".to_string(), vec![0.1f32, 0.2]);

        let checkpoint = ModelCheckpoint::new(parameters)
            .with_optimizer_state(optimizer_state);

        let path = "test_optimizer.gfcp";
        checkpoint.save(path).unwrap();

        let loaded = ModelCheckpoint::load(path).unwrap();
        
        assert!(loaded.optimizer_state.is_some());
        let state = loaded.optimizer_state.unwrap();
        assert!(state.contains_key("momentum"));

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_simple_save_load() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "test".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap(),
        );

        let path = "test_simple.gfcp";
        save_model(path, parameters.clone()).unwrap();

        let loaded = load_model(path).unwrap();
        
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("test"));

        // Cleanup
        fs::remove_file(path).ok();
    }
}

//! Common Dataset Loaders
//!
//! Pre-built loaders for popular ML datasets.

use ghostflow_core::tensor::Tensor;
use std::path::Path;
use std::fs::File;
use std::io::{Read, BufReader};

/// MNIST dataset loader
/// 
/// Loads the MNIST handwritten digits dataset.
pub struct MNIST {
    train_images: Vec<Vec<f32>>,
    train_labels: Vec<u8>,
    test_images: Vec<Vec<f32>>,
    test_labels: Vec<u8>,
    image_size: (usize, usize),
}

impl MNIST {
    /// Load MNIST from directory containing the 4 files
    pub fn load<P: AsRef<Path>>(data_dir: P) -> std::io::Result<Self> {
        let data_dir = data_dir.as_ref();

        let train_images = Self::load_images(
            data_dir.join("train-images-idx3-ubyte"),
        )?;
        let train_labels = Self::load_labels(
            data_dir.join("train-labels-idx1-ubyte"),
        )?;
        let test_images = Self::load_images(
            data_dir.join("t10k-images-idx3-ubyte"),
        )?;
        let test_labels = Self::load_labels(
            data_dir.join("t10k-labels-idx1-ubyte"),
        )?;

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            image_size: (28, 28),
        })
    }

    fn load_images<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Vec<f32>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let magic_num = u32::from_be_bytes(magic);
        
        if magic_num != 2051 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid MNIST image file",
            ));
        }

        // Read dimensions
        let mut dims = [0u8; 12];
        reader.read_exact(&mut dims)?;
        let num_images = u32::from_be_bytes([dims[0], dims[1], dims[2], dims[3]]) as usize;
        let rows = u32::from_be_bytes([dims[4], dims[5], dims[6], dims[7]]) as usize;
        let cols = u32::from_be_bytes([dims[8], dims[9], dims[10], dims[11]]) as usize;

        // Read images
        let mut images = Vec::with_capacity(num_images);
        let image_size = rows * cols;

        for _ in 0..num_images {
            let mut image_bytes = vec![0u8; image_size];
            reader.read_exact(&mut image_bytes)?;
            
            // Normalize to [0, 1]
            let image: Vec<f32> = image_bytes
                .iter()
                .map(|&b| b as f32 / 255.0)
                .collect();
            
            images.push(image);
        }

        Ok(images)
    }

    fn load_labels<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let magic_num = u32::from_be_bytes(magic);
        
        if magic_num != 2049 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid MNIST label file",
            ));
        }

        // Read number of labels
        let mut num_bytes = [0u8; 4];
        reader.read_exact(&mut num_bytes)?;
        let num_labels = u32::from_be_bytes(num_bytes) as usize;

        // Read labels
        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)?;

        Ok(labels)
    }

    /// Get training data
    pub fn train_data(&self) -> (&[Vec<f32>], &[u8]) {
        (&self.train_images, &self.train_labels)
    }

    /// Get test data
    pub fn test_data(&self) -> (&[Vec<f32>], &[u8]) {
        (&self.test_images, &self.test_labels)
    }

    /// Get a batch of training data as tensors
    pub fn train_batch(&self, start: usize, batch_size: usize) -> (Tensor, Tensor) {
        let end = (start + batch_size).min(self.train_images.len());
        let batch_images: Vec<f32> = self.train_images[start..end]
            .iter()
            .flat_map(|img| img.iter().copied())
            .collect();
        
        let batch_labels: Vec<f32> = self.train_labels[start..end]
            .iter()
            .map(|&label| label as f32)
            .collect();

        let images_tensor = Tensor::from_slice(
            &batch_images,
            &[end - start, 1, 28, 28],
        ).unwrap();

        let labels_tensor = Tensor::from_slice(
            &batch_labels,
            &[end - start],
        ).unwrap();

        (images_tensor, labels_tensor)
    }

    /// Get number of training samples
    pub fn train_size(&self) -> usize {
        self.train_images.len()
    }

    /// Get number of test samples
    pub fn test_size(&self) -> usize {
        self.test_images.len()
    }
}

/// CIFAR-10 dataset loader
pub struct CIFAR10 {
    train_images: Vec<Vec<f32>>,
    train_labels: Vec<u8>,
    test_images: Vec<Vec<f32>>,
    test_labels: Vec<u8>,
}

impl CIFAR10 {
    /// Load CIFAR-10 from directory
    pub fn load<P: AsRef<Path>>(data_dir: P) -> std::io::Result<Self> {
        let data_dir = data_dir.as_ref();

        let mut train_images = Vec::new();
        let mut train_labels = Vec::new();

        // Load training batches
        for i in 1..=5 {
            let batch_file = data_dir.join(format!("data_batch_{}.bin", i));
            let (images, labels) = Self::load_batch(&batch_file)?;
            train_images.extend(images);
            train_labels.extend(labels);
        }

        // Load test batch
        let test_file = data_dir.join("test_batch.bin");
        let (test_images, test_labels) = Self::load_batch(&test_file)?;

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    fn load_batch<P: AsRef<Path>>(path: P) -> std::io::Result<(Vec<Vec<f32>>, Vec<u8>)> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut images = Vec::new();
        let mut labels = Vec::new();

        // Each record is 3073 bytes: 1 label + 3072 pixels (32x32x3)
        loop {
            let mut label = [0u8; 1];
            match reader.read_exact(&mut label) {
                Ok(_) => {},
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let mut image_bytes = [0u8; 3072];
            reader.read_exact(&mut image_bytes)?;

            // Normalize to [0, 1]
            let image: Vec<f32> = image_bytes
                .iter()
                .map(|&b| b as f32 / 255.0)
                .collect();

            images.push(image);
            labels.push(label[0]);
        }

        Ok((images, labels))
    }

    /// Get training data
    pub fn train_data(&self) -> (&[Vec<f32>], &[u8]) {
        (&self.train_images, &self.train_labels)
    }

    /// Get test data
    pub fn test_data(&self) -> (&[Vec<f32>], &[u8]) {
        (&self.test_images, &self.test_labels)
    }

    /// Get a batch of training data as tensors
    pub fn train_batch(&self, start: usize, batch_size: usize) -> (Tensor, Tensor) {
        let end = (start + batch_size).min(self.train_images.len());
        let batch_images: Vec<f32> = self.train_images[start..end]
            .iter()
            .flat_map(|img| img.iter().copied())
            .collect();
        
        let batch_labels: Vec<f32> = self.train_labels[start..end]
            .iter()
            .map(|&label| label as f32)
            .collect();

        let images_tensor = Tensor::from_slice(
            &batch_images,
            &[end - start, 3, 32, 32],
        ).unwrap();

        let labels_tensor = Tensor::from_slice(
            &batch_labels,
            &[end - start],
        ).unwrap();

        (images_tensor, labels_tensor)
    }

    /// Get number of training samples
    pub fn train_size(&self) -> usize {
        self.train_images.len()
    }

    /// Get number of test samples
    pub fn test_size(&self) -> usize {
        self.test_images.len()
    }
}

/// Generic dataset trait
pub trait Dataset {
    /// Get a single sample
    fn get(&self, index: usize) -> (Tensor, Tensor);
    
    /// Get dataset size
    fn len(&self) -> usize;
    
    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory dataset
pub struct InMemoryDataset {
    data: Vec<(Tensor, Tensor)>,
}

impl InMemoryDataset {
    pub fn new(data: Vec<(Tensor, Tensor)>) -> Self {
        Self { data }
    }
}

impl Dataset for InMemoryDataset {
    fn get(&self, index: usize) -> (Tensor, Tensor) {
        self.data[index].clone()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_dataset() {
        let data = vec![
            (
                Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap(),
                Tensor::from_slice(&[0.0f32], &[1]).unwrap(),
            ),
            (
                Tensor::from_slice(&[3.0f32, 4.0], &[2]).unwrap(),
                Tensor::from_slice(&[1.0f32], &[1]).unwrap(),
            ),
        ];

        let dataset = InMemoryDataset::new(data);
        
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());

        let (x, y) = dataset.get(0);
        assert_eq!(x.shape().dims(), &[2]);
        assert_eq!(y.shape().dims(), &[1]);
    }
}

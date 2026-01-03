//! DataLoader for batching and iterating over datasets

use ghostflow_core::Tensor;
use crate::dataset::Dataset;
use crate::sampler::{SequentialSampler, RandomSampler};
use rayon::prelude::*;

/// DataLoader for efficient batch loading
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        DataLoader {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            num_workers: 0,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Get number of batches
    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create an iterator over batches
    pub fn iter(&self) -> DataLoaderIter<'_, D> {
        let indices: Vec<usize> = if self.shuffle {
            RandomSampler::new(self.dataset.len()).collect()
        } else {
            SequentialSampler::new(self.dataset.len()).collect()
        };

        DataLoaderIter {
            loader: self,
            indices,
            current_batch: 0,
        }
    }
}

/// Iterator over DataLoader batches
pub struct DataLoaderIter<'a, D: Dataset> {
    loader: &'a DataLoader<D>,
    indices: Vec<usize>,
    current_batch: usize,
}

impl<'a, D: Dataset> Iterator for DataLoaderIter<'a, D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_batch * self.loader.batch_size;
        
        if start >= self.indices.len() {
            return None;
        }

        let end = (start + self.loader.batch_size).min(self.indices.len());
        
        if self.loader.drop_last && end - start < self.loader.batch_size {
            return None;
        }

        let batch_indices = &self.indices[start..end];
        self.current_batch += 1;

        // Collect batch samples
        let samples: Vec<(Tensor, Tensor)> = if self.loader.num_workers > 0 {
            batch_indices
                .par_iter()
                .map(|&idx| self.loader.dataset.get(idx))
                .collect()
        } else {
            batch_indices
                .iter()
                .map(|&idx| self.loader.dataset.get(idx))
                .collect()
        };

        // Stack into batch tensors
        Some(collate_batch(samples))
    }
}

/// Collate samples into a batch
fn collate_batch(samples: Vec<(Tensor, Tensor)>) -> (Tensor, Tensor) {
    let batch_size = samples.len();
    
    if batch_size == 0 {
        return (Tensor::zeros(&[0]), Tensor::zeros(&[0]));
    }

    // Get shapes from first sample
    let data_shape = samples[0].0.dims().to_vec();
    let target_shape = samples[0].1.dims().to_vec();
    let first_data_numel = samples[0].0.numel();
    let first_target_numel = samples[0].1.numel();

    // Collect all data
    let mut data_vec: Vec<f32> = Vec::with_capacity(batch_size * first_data_numel);
    let mut target_vec: Vec<f32> = Vec::with_capacity(batch_size * first_target_numel);

    for (data, target) in samples {
        data_vec.extend(data.data_f32());
        target_vec.extend(target.data_f32());
    }

    // Create batch shapes
    let mut batch_data_shape = vec![batch_size];
    batch_data_shape.extend(&data_shape);

    let mut batch_target_shape = vec![batch_size];
    batch_target_shape.extend(&target_shape);

    (
        Tensor::from_slice(&data_vec, &batch_data_shape).unwrap(),
        Tensor::from_slice(&target_vec, &batch_target_shape).unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;

    #[test]
    fn test_dataloader() {
        let data = Tensor::randn(&[100, 10]);
        let targets = Tensor::randn(&[100, 1]);
        let dataset = TensorDataset::new(data, targets);
        
        let loader = DataLoader::new(dataset, 16);
        
        let mut count = 0;
        for (batch_data, _batch_target) in loader.iter() {
            assert!(batch_data.dims()[0] <= 16);
            count += 1;
        }
        
        assert_eq!(count, 7); // ceil(100/16) = 7
    }

    #[test]
    fn test_dataloader_shuffle() {
        let data = Tensor::arange(0.0, 10.0, 1.0).reshape(&[10, 1]).unwrap();
        let targets = Tensor::zeros(&[10, 1]);
        let dataset = TensorDataset::new(data, targets);
        
        let loader = DataLoader::new(dataset, 5).shuffle(true);
        
        // Just verify it runs without error
        for _ in loader.iter() {}
    }
}

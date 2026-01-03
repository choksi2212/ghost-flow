//! Dataset trait and implementations

use ghostflow_core::Tensor;

/// Base trait for datasets
pub trait Dataset: Send + Sync {
    /// Get a single item by index
    fn get(&self, index: usize) -> (Tensor, Tensor);
    
    /// Get the total number of items
    fn len(&self) -> usize;
    
    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory tensor dataset
pub struct TensorDataset {
    data: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    pub fn new(data: Tensor, targets: Tensor) -> Self {
        assert_eq!(data.dims()[0], targets.dims()[0], 
            "Data and targets must have same number of samples");
        TensorDataset { data, targets }
    }
}

impl Dataset for TensorDataset {
    fn get(&self, index: usize) -> (Tensor, Tensor) {
        // Get single sample
        let data_dims = self.data.dims();
        let target_dims = self.targets.dims();
        
        let data_slice_size: usize = data_dims[1..].iter().product();
        let target_slice_size: usize = if target_dims.len() > 1 {
            target_dims[1..].iter().product()
        } else {
            1
        };
        
        let data_vec = self.data.data_f32();
        let target_vec = self.targets.data_f32();
        
        let data_start = index * data_slice_size;
        let data_end = data_start + data_slice_size;
        let sample_data = &data_vec[data_start..data_end];
        
        let target_start = index * target_slice_size;
        let target_end = target_start + target_slice_size;
        let sample_target = &target_vec[target_start..target_end];
        
        let data_shape: Vec<usize> = data_dims[1..].to_vec();
        let target_shape: Vec<usize> = if target_dims.len() > 1 {
            target_dims[1..].to_vec()
        } else {
            vec![1]
        };
        
        (
            Tensor::from_slice(sample_data, &data_shape).unwrap(),
            Tensor::from_slice(sample_target, &target_shape).unwrap(),
        )
    }

    fn len(&self) -> usize {
        self.data.dims()[0]
    }
}

/// Subset of a dataset
pub struct Subset<D: Dataset> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Subset { dataset, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    fn get(&self, index: usize) -> (Tensor, Tensor) {
        self.dataset.get(self.indices[index])
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Concatenation of multiple datasets
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
}

impl<D: Dataset> ConcatDataset<D> {
    pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;
        
        for ds in &datasets {
            total += ds.len();
            cumulative_sizes.push(total);
        }
        
        ConcatDataset {
            datasets,
            cumulative_sizes,
        }
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D> {
    fn get(&self, index: usize) -> (Tensor, Tensor) {
        let mut dataset_idx = 0;
        let mut sample_idx = index;
        
        for (i, &size) in self.cumulative_sizes.iter().enumerate() {
            if index < size {
                dataset_idx = i;
                if i > 0 {
                    sample_idx = index - self.cumulative_sizes[i - 1];
                }
                break;
            }
        }
        
        self.datasets[dataset_idx].get(sample_idx)
    }

    fn len(&self) -> usize {
        *self.cumulative_sizes.last().unwrap_or(&0)
    }
}

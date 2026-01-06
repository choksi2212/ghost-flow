//! Distributed Training
//!
//! Support for multi-GPU training with data and model parallelism.

use ghostflow_core::tensor::Tensor;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Distributed training backend
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistributedBackend {
    /// NVIDIA NCCL for GPU communication
    NCCL,
    /// Gloo for CPU/GPU communication
    Gloo,
    /// MPI for HPC environments
    MPI,
}

/// Distributed training configuration
#[derive(Clone, Debug)]
pub struct DistributedConfig {
    /// Backend to use
    pub backend: DistributedBackend,
    /// World size (total number of processes)
    pub world_size: usize,
    /// Rank of this process
    pub rank: usize,
    /// Master address for coordination
    pub master_addr: String,
    /// Master port
    pub master_port: u16,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            backend: DistributedBackend::NCCL,
            world_size: 1,
            rank: 0,
            master_addr: "localhost".to_string(),
            master_port: 29500,
        }
    }
}

/// Data parallel training
/// 
/// Replicates the model across GPUs and splits data batches.
/// Gradients are averaged across all GPUs after backward pass.
pub struct DataParallel {
    config: DistributedConfig,
    device_ids: Vec<usize>,
    gradient_buckets: Arc<Mutex<HashMap<String, Vec<Tensor>>>>,
}

impl DataParallel {
    pub fn new(config: DistributedConfig, device_ids: Vec<usize>) -> Self {
        Self {
            config,
            device_ids,
            gradient_buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Split batch across devices
    pub fn split_batch(&self, batch: &Tensor) -> Vec<Tensor> {
        let batch_size = batch.shape().dims()[0];
        let per_device = batch_size / self.device_ids.len();
        
        let mut splits = Vec::new();
        for i in 0..self.device_ids.len() {
            let start = i * per_device;
            let end = if i == self.device_ids.len() - 1 {
                batch_size
            } else {
                (i + 1) * per_device
            };
            
            // Create a view of the batch for this device
            // In practice, this would also move to the specific GPU
            let split = self.slice_batch(batch, start, end);
            splits.push(split);
        }
        
        splits
    }

    fn slice_batch(&self, batch: &Tensor, start: usize, end: usize) -> Tensor {
        // Simplified batch slicing
        // In practice, would use proper tensor slicing
        let batch_data = batch.storage().as_slice::<f32>();
        let dims = batch.shape().dims();
        let row_size = dims[1..].iter().product::<usize>();
        
        let slice_data: Vec<f32> = batch_data[start * row_size..end * row_size].to_vec();
        let mut new_dims = dims.to_vec();
        new_dims[0] = end - start;
        
        Tensor::from_slice(&slice_data, &new_dims).unwrap()
    }

    /// All-reduce gradients across devices
    pub fn all_reduce_gradients(&self, gradients: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let mut averaged = HashMap::new();
        
        for (name, grad) in gradients {
            // Simulate all-reduce by averaging
            // In practice, this would use NCCL/Gloo for actual GPU communication
            let averaged_grad = self.average_gradient(grad);
            averaged.insert(name.clone(), averaged_grad);
        }
        
        averaged
    }

    fn average_gradient(&self, grad: &Tensor) -> Tensor {
        // Simulate averaging across devices
        // In practice, would use collective communication
        let scale = 1.0 / self.config.world_size as f32;
        
        let grad_data = grad.storage().as_slice::<f32>();
        let scaled_data: Vec<f32> = grad_data.iter().map(|&x| x * scale).collect();
        
        Tensor::from_slice(&scaled_data, grad.shape().dims()).unwrap()
    }

    /// Broadcast model parameters from rank 0 to all ranks
    pub fn broadcast_parameters(&self, parameters: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        if self.config.rank == 0 {
            // Rank 0 keeps its parameters
            parameters.clone()
        } else {
            // Other ranks would receive from rank 0
            // In practice, this would use actual broadcast communication
            parameters.clone()
        }
    }
}

/// Model parallel training
/// 
/// Splits the model across multiple GPUs.
/// Different layers run on different devices.
pub struct ModelParallel {
    config: DistributedConfig,
    layer_placement: HashMap<String, usize>,
}

impl ModelParallel {
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            config,
            layer_placement: HashMap::new(),
        }
    }

    /// Assign a layer to a specific device
    pub fn place_layer(&mut self, layer_name: &str, device_id: usize) {
        self.layer_placement.insert(layer_name.to_string(), device_id);
    }

    /// Get device for a layer
    pub fn get_device(&self, layer_name: &str) -> Option<usize> {
        self.layer_placement.get(layer_name).copied()
    }

    /// Automatic layer placement using a simple strategy
    pub fn auto_place_layers(&mut self, layer_names: &[String], num_devices: usize) {
        let layers_per_device = (layer_names.len() + num_devices - 1) / num_devices;
        
        for (i, name) in layer_names.iter().enumerate() {
            let device = i / layers_per_device;
            self.place_layer(name, device.min(num_devices - 1));
        }
    }

    /// Transfer tensor between devices
    pub fn transfer(&self, tensor: &Tensor, _from_device: usize, _to_device: usize) -> Tensor {
        // In practice, this would perform actual GPU-to-GPU transfer
        // For now, just clone the tensor
        tensor.clone()
    }
}

/// Gradient accumulation
/// 
/// Accumulates gradients over multiple micro-batches before updating.
/// Useful for training with large effective batch sizes on limited memory.
pub struct GradientAccumulator {
    accumulation_steps: usize,
    current_step: usize,
    accumulated_gradients: HashMap<String, Tensor>,
}

impl GradientAccumulator {
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulation_steps,
            current_step: 0,
            accumulated_gradients: HashMap::new(),
        }
    }

    /// Accumulate gradients from a micro-batch
    pub fn accumulate(&mut self, gradients: &HashMap<String, Tensor>) {
        for (name, grad) in gradients {
            let should_add = self.accumulated_gradients.contains_key(name);
            
            if should_add {
                // Add to existing accumulated gradient
                let accumulated = self.accumulated_gradients.get(name).unwrap();
                let sum = self.add_tensors(accumulated, grad);
                self.accumulated_gradients.insert(name.clone(), sum);
            } else {
                // First accumulation
                self.accumulated_gradients.insert(name.clone(), grad.clone());
            }
        }
        
        self.current_step += 1;
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.storage().as_slice::<f32>();
        let b_data = b.storage().as_slice::<f32>();
        
        let sum: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();
        
        Tensor::from_slice(&sum, a.shape().dims()).unwrap()
    }

    /// Check if ready to update (accumulated enough steps)
    pub fn should_update(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get accumulated gradients and reset
    pub fn get_and_reset(&mut self) -> HashMap<String, Tensor> {
        let gradients = self.accumulated_gradients.clone();
        self.accumulated_gradients.clear();
        self.current_step = 0;
        
        // Scale by accumulation steps
        let scale = 1.0 / self.accumulation_steps as f32;
        gradients.into_iter()
            .map(|(name, grad)| {
                let scaled = self.scale_tensor(&grad, scale);
                (name, scaled)
            })
            .collect()
    }

    fn scale_tensor(&self, tensor: &Tensor, scale: f32) -> Tensor {
        let data = tensor.storage().as_slice::<f32>();
        let scaled: Vec<f32> = data.iter().map(|&x| x * scale).collect();
        Tensor::from_slice(&scaled, tensor.shape().dims()).unwrap()
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.accumulated_gradients.clear();
        self.current_step = 0;
    }
}

/// Distributed Data Parallel (DDP)
/// 
/// Combines data parallelism with efficient gradient synchronization.
/// Overlaps communication with computation for better performance.
pub struct DistributedDataParallel {
    data_parallel: DataParallel,
    gradient_accumulator: Option<GradientAccumulator>,
    find_unused_parameters: bool,
}

impl DistributedDataParallel {
    pub fn new(
        config: DistributedConfig,
        device_ids: Vec<usize>,
        gradient_accumulation_steps: Option<usize>,
    ) -> Self {
        let gradient_accumulator = gradient_accumulation_steps
            .map(GradientAccumulator::new);
        
        Self {
            data_parallel: DataParallel::new(config, device_ids),
            gradient_accumulator,
            find_unused_parameters: false,
        }
    }

    /// Enable finding unused parameters
    pub fn find_unused_parameters(mut self, enabled: bool) -> Self {
        self.find_unused_parameters = enabled;
        self
    }

    /// Forward pass with data parallelism
    pub fn forward(&self, batch: &Tensor) -> Vec<Tensor> {
        self.data_parallel.split_batch(batch)
    }

    /// Backward pass with gradient synchronization
    pub fn backward(&mut self, gradients: &HashMap<String, Tensor>) -> Option<HashMap<String, Tensor>> {
        // All-reduce gradients across devices
        let reduced_gradients = self.data_parallel.all_reduce_gradients(gradients);
        
        // Handle gradient accumulation if enabled
        if let Some(ref mut accumulator) = self.gradient_accumulator {
            accumulator.accumulate(&reduced_gradients);
            
            if accumulator.should_update() {
                Some(accumulator.get_and_reset())
            } else {
                None
            }
        } else {
            Some(reduced_gradients)
        }
    }

    /// Synchronize parameters across all processes
    pub fn sync_parameters(&self, parameters: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        self.data_parallel.broadcast_parameters(parameters)
    }
}

/// Pipeline parallelism
/// 
/// Splits model into stages and processes micro-batches in a pipeline.
pub struct PipelineParallel {
    num_stages: usize,
    num_micro_batches: usize,
    current_stage: usize,
}

impl PipelineParallel {
    pub fn new(num_stages: usize, num_micro_batches: usize) -> Self {
        Self {
            num_stages,
            num_micro_batches,
            current_stage: 0,
        }
    }

    /// Split batch into micro-batches
    pub fn create_micro_batches(&self, batch: &Tensor) -> Vec<Tensor> {
        let batch_size = batch.shape().dims()[0];
        let micro_batch_size = batch_size / self.num_micro_batches;
        
        let mut micro_batches = Vec::new();
        for i in 0..self.num_micro_batches {
            let start = i * micro_batch_size;
            let end = if i == self.num_micro_batches - 1 {
                batch_size
            } else {
                (i + 1) * micro_batch_size
            };
            
            let micro_batch = self.slice_batch(batch, start, end);
            micro_batches.push(micro_batch);
        }
        
        micro_batches
    }

    fn slice_batch(&self, batch: &Tensor, start: usize, end: usize) -> Tensor {
        let batch_data = batch.storage().as_slice::<f32>();
        let dims = batch.shape().dims();
        let row_size = dims[1..].iter().product::<usize>();
        
        let slice_data: Vec<f32> = batch_data[start * row_size..end * row_size].to_vec();
        let mut new_dims = dims.to_vec();
        new_dims[0] = end - start;
        
        Tensor::from_slice(&slice_data, &new_dims).unwrap()
    }

    /// Get current pipeline stage
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Advance to next stage
    pub fn next_stage(&mut self) {
        self.current_stage = (self.current_stage + 1) % self.num_stages;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_parallel_split_batch() {
        let config = DistributedConfig {
            world_size: 2,
            rank: 0,
            ..Default::default()
        };
        let dp = DataParallel::new(config, vec![0, 1]);

        let batch = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let splits = dp.split_batch(&batch);

        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].shape().dims()[0], 2);
        assert_eq!(splits[1].shape().dims()[0], 2);
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut accumulator = GradientAccumulator::new(4);

        let grad1 = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad2 = Tensor::from_slice(&[2.0f32, 3.0, 4.0], &[3]).unwrap();

        let mut grads = HashMap::new();
        grads.insert("layer1".to_string(), grad1);

        accumulator.accumulate(&grads);
        assert!(!accumulator.should_update());

        accumulator.accumulate(&grads);
        accumulator.accumulate(&grads);
        accumulator.accumulate(&grads);
        assert!(accumulator.should_update());

        let final_grads = accumulator.get_and_reset();
        assert!(final_grads.contains_key("layer1"));
        assert_eq!(accumulator.current_step, 0);
    }

    #[test]
    fn test_model_parallel_placement() {
        let config = DistributedConfig::default();
        let mut mp = ModelParallel::new(config);

        mp.place_layer("layer1", 0);
        mp.place_layer("layer2", 1);
        mp.place_layer("layer3", 0);

        assert_eq!(mp.get_device("layer1"), Some(0));
        assert_eq!(mp.get_device("layer2"), Some(1));
        assert_eq!(mp.get_device("layer3"), Some(0));
    }

    #[test]
    fn test_auto_layer_placement() {
        let config = DistributedConfig::default();
        let mut mp = ModelParallel::new(config);

        let layers = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
            "layer4".to_string(),
        ];

        mp.auto_place_layers(&layers, 2);

        assert_eq!(mp.get_device("layer1"), Some(0));
        assert_eq!(mp.get_device("layer2"), Some(0));
        assert_eq!(mp.get_device("layer3"), Some(1));
        assert_eq!(mp.get_device("layer4"), Some(1));
    }

    #[test]
    fn test_ddp_forward_backward() {
        let config = DistributedConfig {
            world_size: 2,
            rank: 0,
            ..Default::default()
        };
        let mut ddp = DistributedDataParallel::new(config, vec![0, 1], Some(2));

        let batch = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let splits = ddp.forward(&batch);
        assert_eq!(splits.len(), 2);

        let mut gradients = HashMap::new();
        gradients.insert("layer1".to_string(), Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap());

        // First backward - should accumulate
        let result = ddp.backward(&gradients);
        assert!(result.is_none());

        // Second backward - should return accumulated gradients
        let result = ddp.backward(&gradients);
        assert!(result.is_some());
    }

    #[test]
    fn test_pipeline_parallel() {
        let pp = PipelineParallel::new(4, 8);

        let batch = Tensor::from_slice(&(0..32).map(|x| x as f32).collect::<Vec<_>>(), &[8, 4]).unwrap();
        let micro_batches = pp.create_micro_batches(&batch);

        assert_eq!(micro_batches.len(), 8);
        assert_eq!(micro_batches[0].shape().dims()[0], 1);
    }

    #[test]
    fn test_all_reduce_gradients() {
        let config = DistributedConfig {
            world_size: 4,
            rank: 0,
            ..Default::default()
        };
        let dp = DataParallel::new(config, vec![0, 1, 2, 3]);

        let mut gradients = HashMap::new();
        gradients.insert(
            "layer1".to_string(),
            Tensor::from_slice(&[4.0f32, 8.0, 12.0], &[3]).unwrap()
        );

        let reduced = dp.all_reduce_gradients(&gradients);
        let grad_data = reduced.get("layer1").unwrap().storage().as_slice::<f32>();
        
        // Should be averaged across 4 devices
        assert!((grad_data[0] - 1.0).abs() < 0.01);
        assert!((grad_data[1] - 2.0).abs() < 0.01);
        assert!((grad_data[2] - 3.0).abs() < 0.01);
    }
}

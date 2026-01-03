//! Distributed Training - Data Parallelism and Model Parallelism
//!
//! This module provides utilities for distributed training across multiple nodes/processes.

use std::sync::{Arc, Mutex};
use ghostflow_core::Tensor;

/// Distributed training strategy
#[derive(Clone, Copy, Debug)]
pub enum DistributedStrategy {
    /// Data parallelism - split data across workers
    DataParallel,
    /// Model parallelism - split model across workers
    ModelParallel,
    /// Hybrid approach
    Hybrid,
}

/// Communication backend for distributed training
#[derive(Clone, Copy, Debug)]
pub enum CommunicationBackend {
    /// Thread-based (single machine)
    Threads,
    /// Process-based (single or multiple machines)
    Processes,
    /// MPI-based (multiple machines)
    MPI,
}

/// Gradient aggregation method
#[derive(Clone, Copy, Debug)]
pub enum GradientAggregation {
    /// Average gradients across workers
    Average,
    /// Sum gradients across workers
    Sum,
    /// Weighted average
    WeightedAverage,
}

/// Distributed trainer configuration
pub struct DistributedConfig {
    pub strategy: DistributedStrategy,
    pub backend: CommunicationBackend,
    pub world_size: usize,
    pub rank: usize,
    pub gradient_aggregation: GradientAggregation,
    pub sync_frequency: usize,
}

impl DistributedConfig {
    pub fn new(world_size: usize, rank: usize) -> Self {
        DistributedConfig {
            strategy: DistributedStrategy::DataParallel,
            backend: CommunicationBackend::Threads,
            world_size,
            rank,
            gradient_aggregation: GradientAggregation::Average,
            sync_frequency: 1,
        }
    }

    pub fn strategy(mut self, strategy: DistributedStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn backend(mut self, backend: CommunicationBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn gradient_aggregation(mut self, agg: GradientAggregation) -> Self {
        self.gradient_aggregation = agg;
        self
    }

    pub fn sync_frequency(mut self, freq: usize) -> Self {
        self.sync_frequency = freq;
        self
    }
}

/// Data parallel trainer
pub struct DataParallelTrainer {
    config: DistributedConfig,
    local_gradients: Arc<Mutex<Vec<Vec<f32>>>>,
    global_gradients: Arc<Mutex<Vec<Vec<f32>>>>,
    iteration: usize,
}

impl DataParallelTrainer {
    pub fn new(config: DistributedConfig) -> Self {
        DataParallelTrainer {
            config,
            local_gradients: Arc::new(Mutex::new(Vec::new())),
            global_gradients: Arc::new(Mutex::new(Vec::new())),
            iteration: 0,
        }
    }

    /// Split data across workers
    pub fn split_data(&self, data: &Tensor, labels: &Tensor) -> (Tensor, Tensor) {
        let n_samples = data.dims()[0];
        let n_features = data.dims()[1];
        let samples_per_worker = n_samples / self.config.world_size;
        
        let start_idx = self.config.rank * samples_per_worker;
        let end_idx = if self.config.rank == self.config.world_size - 1 {
            n_samples
        } else {
            (self.config.rank + 1) * samples_per_worker
        };

        let data_slice = &data.data_f32()[start_idx * n_features..end_idx * n_features];
        let labels_slice = &labels.data_f32()[start_idx..end_idx];

        let local_data = Tensor::from_slice(data_slice, &[end_idx - start_idx, n_features]).unwrap();
        let local_labels = Tensor::from_slice(labels_slice, &[end_idx - start_idx]).unwrap();

        (local_data, local_labels)
    }

    /// Accumulate local gradients
    pub fn accumulate_gradients(&mut self, gradients: Vec<Vec<f32>>) {
        let mut local_grads = self.local_gradients.lock().unwrap();
        *local_grads = gradients;
    }

    /// Synchronize gradients across workers
    pub fn sync_gradients(&mut self) -> Vec<Vec<f32>> {
        self.iteration += 1;

        // Only sync at specified frequency
        if self.iteration % self.config.sync_frequency != 0 {
            return self.local_gradients.lock().unwrap().clone();
        }

        match self.config.backend {
            CommunicationBackend::Threads => self.sync_gradients_threads(),
            CommunicationBackend::Processes => self.sync_gradients_processes(),
            CommunicationBackend::MPI => self.sync_gradients_mpi(),
        }
    }

    fn sync_gradients_threads(&self) -> Vec<Vec<f32>> {
        // Simplified thread-based synchronization
        let local_grads = self.local_gradients.lock().unwrap();
        let mut global_grads = self.global_gradients.lock().unwrap();

        if global_grads.is_empty() {
            *global_grads = local_grads.clone();
        } else {
            // Aggregate gradients
            for (global_layer, local_layer) in global_grads.iter_mut().zip(local_grads.iter()) {
                for (g, l) in global_layer.iter_mut().zip(local_layer.iter()) {
                    match self.config.gradient_aggregation {
                        GradientAggregation::Average => {
                            *g = (*g * (self.config.world_size - 1) as f32 + l) / self.config.world_size as f32;
                        }
                        GradientAggregation::Sum => {
                            *g += l;
                        }
                        GradientAggregation::WeightedAverage => {
                            *g = (*g + l) / 2.0;
                        }
                    }
                }
            }
        }

        global_grads.clone()
    }

    fn sync_gradients_processes(&self) -> Vec<Vec<f32>> {
        // Placeholder for process-based synchronization
        // In a real implementation, this would use IPC mechanisms
        self.local_gradients.lock().unwrap().clone()
    }

    fn sync_gradients_mpi(&self) -> Vec<Vec<f32>> {
        // Placeholder for MPI-based synchronization
        // In a real implementation, this would use MPI libraries
        self.local_gradients.lock().unwrap().clone()
    }

    /// All-reduce operation for gradients
    pub fn all_reduce(&self, gradients: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simplified all-reduce
        let mut reduced = gradients.to_vec();

        match self.config.gradient_aggregation {
            GradientAggregation::Average => {
                for layer in &mut reduced {
                    for grad in layer {
                        *grad /= self.config.world_size as f32;
                    }
                }
            }
            GradientAggregation::Sum => {
                // Already summed
            }
            GradientAggregation::WeightedAverage => {
                for layer in &mut reduced {
                    for grad in layer {
                        *grad /= self.config.world_size as f32;
                    }
                }
            }
        }

        reduced
    }

    /// Broadcast parameters from rank 0 to all workers
    pub fn broadcast_parameters(&self, parameters: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if self.config.rank == 0 {
            // Master broadcasts
            parameters.to_vec()
        } else {
            // Workers receive
            // In a real implementation, this would receive from master
            parameters.to_vec()
        }
    }

    pub fn is_master(&self) -> bool {
        self.config.rank == 0
    }
}

/// Distributed data loader
pub struct DistributedDataLoader {
    pub batch_size: usize,
    pub world_size: usize,
    pub rank: usize,
    pub shuffle: bool,
    pub drop_last: bool,
}

impl DistributedDataLoader {
    pub fn new(batch_size: usize, world_size: usize, rank: usize) -> Self {
        DistributedDataLoader {
            batch_size,
            world_size,
            rank,
            shuffle: true,
            drop_last: false,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn drop_last(mut self, drop: bool) -> Self {
        self.drop_last = drop;
        self
    }

    /// Get batches for this worker
    pub fn get_batches(&self, data: &Tensor, labels: &Tensor) -> Vec<(Tensor, Tensor)> {
        let n_samples = data.dims()[0];
        let n_features = data.dims()[1];
        
        // Calculate samples per worker
        let samples_per_worker = n_samples / self.world_size;
        let start_idx = self.rank * samples_per_worker;
        let end_idx = if self.rank == self.world_size - 1 {
            n_samples
        } else {
            (self.rank + 1) * samples_per_worker
        };

        let worker_samples = end_idx - start_idx;
        let n_batches = if self.drop_last {
            worker_samples / self.batch_size
        } else {
            (worker_samples + self.batch_size - 1) / self.batch_size
        };

        let mut batches = Vec::new();
        let data_slice = data.data_f32();
        let labels_slice = labels.data_f32();

        for batch_idx in 0..n_batches {
            let batch_start = start_idx + batch_idx * self.batch_size;
            let batch_end = (batch_start + self.batch_size).min(end_idx);
            let batch_size = batch_end - batch_start;

            let batch_data: Vec<f32> = (batch_start..batch_end)
                .flat_map(|i| data_slice[i * n_features..(i + 1) * n_features].to_vec())
                .collect();
            let batch_labels: Vec<f32> = (batch_start..batch_end)
                .map(|i| labels_slice[i])
                .collect();

            let data_tensor = Tensor::from_slice(&batch_data, &[batch_size, n_features]).unwrap();
            let labels_tensor = Tensor::from_slice(&batch_labels, &[batch_size]).unwrap();

            batches.push((data_tensor, labels_tensor));
        }

        batches
    }
}

/// Gradient compression for communication efficiency
pub struct GradientCompression {
    pub method: CompressionMethod,
    pub compression_ratio: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Top-K sparsification
    TopK,
    /// Random sparsification
    Random,
    /// Quantization
    Quantization,
}

impl GradientCompression {
    pub fn new(method: CompressionMethod) -> Self {
        GradientCompression {
            method,
            compression_ratio: 0.1,
        }
    }

    pub fn compression_ratio(mut self, ratio: f32) -> Self {
        self.compression_ratio = ratio;
        self
    }

    pub fn compress(&self, gradients: &[f32]) -> (Vec<usize>, Vec<f32>) {
        match self.method {
            CompressionMethod::None => {
                let indices: Vec<usize> = (0..gradients.len()).collect();
                (indices, gradients.to_vec())
            }
            CompressionMethod::TopK => self.compress_topk(gradients),
            CompressionMethod::Random => self.compress_random(gradients),
            CompressionMethod::Quantization => self.compress_quantize(gradients),
        }
    }

    fn compress_topk(&self, gradients: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let k = (gradients.len() as f32 * self.compression_ratio) as usize;
        let mut indexed: Vec<(usize, f32)> = gradients.iter()
            .enumerate()
            .map(|(i, &g)| (i, g.abs()))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let values: Vec<f32> = indexed.iter().map(|(i, _)| gradients[*i]).collect();

        (indices, values)
    }

    fn compress_random(&self, gradients: &[f32]) -> (Vec<usize>, Vec<f32>) {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let k = (gradients.len() as f32 * self.compression_ratio) as usize;

        let mut indices: Vec<usize> = (0..gradients.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(k);

        let values: Vec<f32> = indices.iter().map(|&i| gradients[i]).collect();

        (indices, values)
    }

    fn compress_quantize(&self, gradients: &[f32]) -> (Vec<usize>, Vec<f32>) {
        // Simple 8-bit quantization
        let max_abs = gradients.iter().map(|g| g.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;

        let quantized: Vec<f32> = gradients.iter()
            .map(|&g| (g / scale).round() * scale)
            .collect();

        let indices: Vec<usize> = (0..gradients.len()).collect();
        (indices, quantized)
    }

    pub fn decompress(&self, indices: &[usize], values: &[f32], size: usize) -> Vec<f32> {
        let mut decompressed = vec![0.0f32; size];
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if idx < size {
                decompressed[idx] = val;
            }
        }
        decompressed
    }
}

/// Ring All-Reduce implementation
pub struct RingAllReduce {
    pub world_size: usize,
    pub rank: usize,
}

impl RingAllReduce {
    pub fn new(world_size: usize, rank: usize) -> Self {
        RingAllReduce { world_size, rank }
    }

    /// Perform ring all-reduce on gradients
    pub fn all_reduce(&self, gradients: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simplified ring all-reduce
        // In a real implementation, this would communicate in a ring topology
        let mut result = gradients.to_vec();

        // Simulate ring communication
        for layer in &mut result {
            let sum: f32 = layer.iter().sum();
            let avg = sum / self.world_size as f32;
            for grad in layer {
                *grad = avg;
            }
        }

        result
    }

    #[allow(dead_code)]
    fn get_next_rank(&self) -> usize {
        (self.rank + 1) % self.world_size
    }

    #[allow(dead_code)]
    fn get_prev_rank(&self) -> usize {
        (self.rank + self.world_size - 1) % self.world_size
    }
}
/// Distributed optimizer wrapper
pub struct DistributedOptimizer<O> {
    #[allow(dead_code)]
    optimizer: O,
    trainer: DataParallelTrainer,
    compression: Option<GradientCompression>,
}

impl<O> DistributedOptimizer<O> {
    pub fn new(optimizer: O, config: DistributedConfig) -> Self {
        DistributedOptimizer {
            optimizer,
            trainer: DataParallelTrainer::new(config),
            compression: None,
        }
    }

    pub fn with_compression(mut self, compression: GradientCompression) -> Self {
        self.compression = Some(compression);
        self
    }

    pub fn step(&mut self, params: &mut [f32], local_grads: &[f32]) {
        // Compress gradients if enabled
        let grads_to_sync = if let Some(ref compression) = self.compression {
            let (indices, values) = compression.compress(local_grads);
            compression.decompress(&indices, &values, local_grads.len())
        } else {
            local_grads.to_vec()
        };

        // Synchronize gradients
        let grad_vec = vec![grads_to_sync];
        self.trainer.accumulate_gradients(grad_vec);
        let synced_grads = self.trainer.sync_gradients();

        // Apply optimizer step with synchronized gradients
        if !synced_grads.is_empty() && !synced_grads[0].is_empty() {
            // In a real implementation, this would call the optimizer's step method
            for (p, g) in params.iter_mut().zip(synced_grads[0].iter()) {
                *p -= 0.01 * g; // Simplified update
            }
        }
    }

    pub fn is_master(&self) -> bool {
        self.trainer.is_master()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_parallel_trainer() {
        let config = DistributedConfig::new(2, 0);
        let trainer = DataParallelTrainer::new(config);

        let data = Tensor::from_slice(&vec![1.0f32; 100], &[10, 10]).unwrap();
        let labels = Tensor::from_slice(&vec![0.0f32; 10], &[10]).unwrap();

        let (local_data, _local_labels) = trainer.split_data(&data, &labels);
        assert_eq!(local_data.dims()[0], 5); // Half the data
    }

    #[test]
    fn test_distributed_data_loader() {
        let loader = DistributedDataLoader::new(2, 2, 0);
        
        let data = Tensor::from_slice(&vec![1.0f32; 100], &[10, 10]).unwrap();
        let labels = Tensor::from_slice(&vec![0.0f32; 10], &[10]).unwrap();

        let batches = loader.get_batches(&data, &labels);
        assert!(batches.len() > 0);
    }

    #[test]
    fn test_gradient_compression() {
        let compression = GradientCompression::new(CompressionMethod::TopK)
            .compression_ratio(0.5);

        let gradients = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (indices, values) = compression.compress(&gradients);

        assert!(indices.len() <= (gradients.len() as f32 * 0.5) as usize + 1);
        
        let decompressed = compression.decompress(&indices, &values, gradients.len());
        assert_eq!(decompressed.len(), gradients.len());
    }

    #[test]
    fn test_ring_all_reduce() {
        let ring = RingAllReduce::new(4, 0);
        let gradients = vec![vec![1.0, 2.0, 3.0]];
        
        let result = ring.all_reduce(&gradients);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }
}

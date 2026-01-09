//! ZeRO Optimizer (Zero Redundancy Optimizer)
//!
//! Implements memory-efficient distributed training:
//! - ZeRO Stage 1: Optimizer state partitioning
//! - ZeRO Stage 2: Gradient partitioning
//! - ZeRO Stage 3: Parameter partitioning
//! - ZeRO-Offload: CPU/NVMe offloading
//! - Communication optimization

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// ZeRO stage configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZeRoStage {
    /// Stage 1: Optimizer state partitioning
    Stage1,
    /// Stage 2: Gradient partitioning
    Stage2,
    /// Stage 3: Parameter partitioning
    Stage3,
}

/// ZeRO configuration
#[derive(Debug, Clone)]
pub struct ZeRoConfig {
    /// ZeRO stage
    pub stage: ZeRoStage,
    /// Number of processes/GPUs
    pub world_size: usize,
    /// Current process rank
    pub rank: usize,
    /// Enable CPU offloading
    pub cpu_offload: bool,
    /// Enable NVMe offloading
    pub nvme_offload: bool,
    /// Overlap communication with computation
    pub overlap_comm: bool,
    /// Bucket size for gradient accumulation
    pub bucket_size: usize,
}

impl Default for ZeRoConfig {
    fn default() -> Self {
        ZeRoConfig {
            stage: ZeRoStage::Stage2,
            world_size: 1,
            rank: 0,
            cpu_offload: false,
            nvme_offload: false,
            overlap_comm: true,
            bucket_size: 25_000_000, // 25M parameters
        }
    }
}

impl ZeRoConfig {
    /// Stage 1 configuration (optimizer state partitioning)
    pub fn stage1(world_size: usize, rank: usize) -> Self {
        ZeRoConfig {
            stage: ZeRoStage::Stage1,
            world_size,
            rank,
            ..Default::default()
        }
    }
    
    /// Stage 2 configuration (gradient partitioning)
    pub fn stage2(world_size: usize, rank: usize) -> Self {
        ZeRoConfig {
            stage: ZeRoStage::Stage2,
            world_size,
            rank,
            ..Default::default()
        }
    }
    
    /// Stage 3 configuration (parameter partitioning)
    pub fn stage3(world_size: usize, rank: usize) -> Self {
        ZeRoConfig {
            stage: ZeRoStage::Stage3,
            world_size,
            rank,
            ..Default::default()
        }
    }
    
    /// ZeRO-Offload configuration
    pub fn with_offload(mut self, cpu: bool, nvme: bool) -> Self {
        self.cpu_offload = cpu;
        self.nvme_offload = nvme;
        self
    }
}

/// Parameter partition information
#[derive(Debug, Clone)]
pub struct ParameterPartition {
    /// Parameter name
    pub name: String,
    /// Owner rank
    pub owner_rank: usize,
    /// Start index in flattened parameters
    pub start_idx: usize,
    /// End index in flattened parameters
    pub end_idx: usize,
    /// Original shape
    pub shape: Vec<usize>,
}

/// ZeRO optimizer state
pub struct ZeRoOptimizer {
    config: ZeRoConfig,
    /// Partitioned parameters (only owned parameters)
    partitioned_params: HashMap<String, Tensor>,
    /// Partitioned gradients
    partitioned_grads: HashMap<String, Tensor>,
    /// Partitioned optimizer states (momentum, variance, etc.)
    partitioned_states: HashMap<String, HashMap<String, Tensor>>,
    /// Parameter partition map
    param_partitions: Vec<ParameterPartition>,
    /// Gradient buckets for communication
    gradient_buckets: Vec<Vec<String>>,
    /// Communication buffer
    comm_buffer: Vec<f32>,
    /// CPU offload buffer
    cpu_buffer: HashMap<String, Vec<f32>>,
    /// Learning rate
    learning_rate: f32,
}

impl ZeRoOptimizer {
    /// Create new ZeRO optimizer
    pub fn new(config: ZeRoConfig, learning_rate: f32) -> Self {
        ZeRoOptimizer {
            config,
            partitioned_params: HashMap::new(),
            partitioned_grads: HashMap::new(),
            partitioned_states: HashMap::new(),
            param_partitions: Vec::new(),
            gradient_buckets: Vec::new(),
            comm_buffer: Vec::new(),
            cpu_buffer: HashMap::new(),
            learning_rate,
        }
    }
    
    /// Partition parameters across ranks
    pub fn partition_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<(), String> {
        let total_params: usize = params.values()
            .map(|t| t.data_f32().len())
            .sum();
        
        let params_per_rank = (total_params + self.config.world_size - 1) / self.config.world_size;
        
        let mut current_idx = 0;
        
        for (name, tensor) in params {
            let param_size = tensor.data_f32().len();
            let start_idx = current_idx;
            let end_idx = current_idx + param_size;
            
            // Determine owner rank
            let owner_rank = current_idx / params_per_rank;
            
            let partition = ParameterPartition {
                name: name.clone(),
                owner_rank,
                start_idx,
                end_idx,
                shape: tensor.dims().to_vec(),
            };
            
            self.param_partitions.push(partition);
            
            // Store parameter if owned by this rank
            if owner_rank == self.config.rank {
                self.partitioned_params.insert(name.clone(), tensor.clone());
                
                // Initialize optimizer states
                let dims = tensor.dims();
                let size = tensor.data_f32().len();
                let zeros_data = vec![0.0f32; size];
                
                let mut states = HashMap::new();
                states.insert("momentum".to_string(), Tensor::from_slice(&zeros_data, dims).unwrap());
                states.insert("variance".to_string(), Tensor::from_slice(&zeros_data, dims).unwrap());
                self.partitioned_states.insert(name.clone(), states);
            }
            
            current_idx = end_idx;
        }
        
        Ok(())
    }
    
    /// Partition gradients (Stage 2+)
    pub fn partition_gradients(&mut self, grads: &HashMap<String, Tensor>) -> Result<(), String> {
        if self.config.stage == ZeRoStage::Stage1 {
            // Stage 1: Keep all gradients, only partition optimizer states
            for (name, grad) in grads {
                self.partitioned_grads.insert(name.clone(), grad.clone());
            }
            return Ok(());
        }
        
        // Stage 2+: Partition gradients
        for partition in &self.param_partitions {
            if partition.owner_rank == self.config.rank {
                if let Some(grad) = grads.get(&partition.name) {
                    self.partitioned_grads.insert(partition.name.clone(), grad.clone());
                }
            }
        }
        
        Ok(())
    }
    
    /// Reduce-scatter gradients across ranks
    pub fn reduce_scatter_gradients(&mut self) -> Result<(), String> {
        // Simulate reduce-scatter operation
        // In real implementation, this would use NCCL or similar
        
        let grad_names: Vec<String> = self.partitioned_grads.keys().cloned().collect();
        
        for name in grad_names {
            if let Some(grad) = self.partitioned_grads.get(&name) {
                // Simulate averaging gradients across ranks
                let data = grad.data_f32();
                let averaged: Vec<f32> = data.iter()
                    .map(|&x| x / self.config.world_size as f32)
                    .collect();
                
                let averaged_grad = Tensor::from_slice(&averaged, grad.dims())
                    .map_err(|e| format!("Failed to create averaged gradient: {:?}", e))?;
                
                self.partitioned_grads.insert(name, averaged_grad);
            }
        }
        
        Ok(())
    }
    
    /// All-gather parameters (Stage 3)
    pub fn all_gather_parameters(&self) -> Result<HashMap<String, Tensor>, String> {
        if self.config.stage != ZeRoStage::Stage3 {
            return Ok(self.partitioned_params.clone());
        }
        
        // Simulate all-gather operation
        // In real implementation, this would gather parameters from all ranks
        let mut all_params = HashMap::new();
        
        for (name, param) in &self.partitioned_params {
            all_params.insert(name.clone(), param.clone());
        }
        
        Ok(all_params)
    }
    
    /// Optimizer step with ZeRO
    pub fn step(&mut self) -> Result<(), String> {
        // Update only owned parameters
        for (name, param) in &mut self.partitioned_params {
            if let Some(grad) = self.partitioned_grads.get(name) {
                if let Some(states) = self.partitioned_states.get_mut(name) {
                    // Adam-style update
                    let beta1 = 0.9;
                    let beta2 = 0.999;
                    let eps = 1e-8;
                    
                    // Get data from states
                    let m_data = states.get("momentum").unwrap().data_f32();
                    let v_data = states.get("variance").unwrap().data_f32();
                    let g_data = grad.data_f32();
                    let p_data = param.data_f32();
                    
                    let mut new_m = Vec::with_capacity(m_data.len());
                    let mut new_v = Vec::with_capacity(v_data.len());
                    let mut new_p = Vec::with_capacity(p_data.len());
                    
                    for i in 0..m_data.len() {
                        let m = beta1 * m_data[i] + (1.0 - beta1) * g_data[i];
                        let v = beta2 * v_data[i] + (1.0 - beta2) * g_data[i] * g_data[i];
                        let p = p_data[i] - self.learning_rate * m / (v.sqrt() + eps);
                        
                        new_m.push(m);
                        new_v.push(v);
                        new_p.push(p);
                    }
                    
                    // Get dims before updating
                    let m_dims = states.get("momentum").unwrap().dims().to_vec();
                    let v_dims = states.get("variance").unwrap().dims().to_vec();
                    let p_dims = param.dims().to_vec();
                    
                    // Update states
                    states.insert("momentum".to_string(), Tensor::from_slice(&new_m, &m_dims)
                        .map_err(|e| format!("Failed to create momentum tensor: {:?}", e))?);
                    states.insert("variance".to_string(), Tensor::from_slice(&new_v, &v_dims)
                        .map_err(|e| format!("Failed to create variance tensor: {:?}", e))?);
                    *param = Tensor::from_slice(&new_p, &p_dims)
                        .map_err(|e| format!("Failed to create param tensor: {:?}", e))?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Offload to CPU
    pub fn offload_to_cpu(&mut self, name: &str) -> Result<(), String> {
        if !self.config.cpu_offload {
            return Ok(());
        }
        
        if let Some(param) = self.partitioned_params.get(name) {
            let data = param.data_f32().to_vec();
            self.cpu_buffer.insert(name.to_string(), data);
            // In real implementation, would remove from GPU memory
        }
        
        Ok(())
    }
    
    /// Load from CPU
    pub fn load_from_cpu(&mut self, name: &str) -> Result<(), String> {
        if !self.config.cpu_offload {
            return Ok(());
        }
        
        if let Some(data) = self.cpu_buffer.get(name) {
            if let Some(partition) = self.param_partitions.iter().find(|p| p.name == name) {
                let tensor = Tensor::from_slice(data, &partition.shape)
                    .map_err(|e| format!("Failed to load from CPU: {:?}", e))?;
                self.partitioned_params.insert(name.to_string(), tensor);
            }
        }
        
        Ok(())
    }
    
    /// Get memory savings ratio
    pub fn memory_savings_ratio(&self) -> f32 {
        match self.config.stage {
            ZeRoStage::Stage1 => {
                // Stage 1: Only optimizer states partitioned (saves ~4x memory for Adam)
                // Memory = params + grads + (optimizer_states / N)
                // Savings = (4 - 4/N) / 4 = 1 - 1/N
                let n = self.config.world_size as f32;
                (n - 1.0) / n * 0.5  // ~50% of total for optimizer states
            }
            ZeRoStage::Stage2 => {
                // Stage 2: Optimizer states + gradients partitioned
                // Memory = params + (grads + optimizer_states) / N
                // Savings = (4 - 1 - 3/N) / 4
                let n = self.config.world_size as f32;
                (n - 1.0) / n * 0.75  // ~75% of total for grads + optimizer states
            }
            ZeRoStage::Stage3 => {
                // Stage 3: Everything partitioned
                // Memory = (params + grads + optimizer_states) / N
                // Savings = (4 - 4/N) / 4 = (N-1)/N
                let n = self.config.world_size as f32;
                (n - 1.0) / n
            }
        }
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> ZeRoStats {
        let total_params: usize = self.partitioned_params.values()
            .map(|t| t.data_f32().len())
            .sum();
        
        let total_grads: usize = self.partitioned_grads.values()
            .map(|t| t.data_f32().len())
            .sum();
        
        ZeRoStats {
            stage: self.config.stage,
            world_size: self.config.world_size,
            rank: self.config.rank,
            num_partitioned_params: self.partitioned_params.len(),
            total_param_elements: total_params,
            total_grad_elements: total_grads,
            memory_savings: self.memory_savings_ratio(),
            cpu_offload_enabled: self.config.cpu_offload,
        }
    }
}

/// ZeRO statistics
#[derive(Debug, Clone)]
pub struct ZeRoStats {
    pub stage: ZeRoStage,
    pub world_size: usize,
    pub rank: usize,
    pub num_partitioned_params: usize,
    pub total_param_elements: usize,
    pub total_grad_elements: usize,
    pub memory_savings: f32,
    pub cpu_offload_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_config() {
        let config = ZeRoConfig::default();
        assert_eq!(config.stage, ZeRoStage::Stage2);
        assert_eq!(config.world_size, 1);
        
        let stage3 = ZeRoConfig::stage3(4, 0);
        assert_eq!(stage3.stage, ZeRoStage::Stage3);
        assert_eq!(stage3.world_size, 4);
    }
    
    #[test]
    fn test_zero_optimizer_creation() {
        let config = ZeRoConfig::stage2(4, 0);
        let optimizer = ZeRoOptimizer::new(config, 0.001);
        
        let stats = optimizer.get_stats();
        assert_eq!(stats.world_size, 4);
        assert_eq!(stats.rank, 0);
    }
    
    #[test]
    fn test_partition_parameters() {
        let config = ZeRoConfig::stage2(2, 0);
        let mut optimizer = ZeRoOptimizer::new(config, 0.001);
        
        let mut params = HashMap::new();
        params.insert("layer1".to_string(), Tensor::randn(&[10, 10]));
        params.insert("layer2".to_string(), Tensor::randn(&[20, 20]));
        
        optimizer.partition_parameters(&params).unwrap();
        assert!(optimizer.param_partitions.len() > 0);
    }
    
    #[test]
    fn test_memory_savings_ratio() {
        let config1 = ZeRoConfig::stage1(4, 0);
        let optimizer1 = ZeRoOptimizer::new(config1, 0.001);
        let savings1 = optimizer1.memory_savings_ratio();
        
        let config2 = ZeRoConfig::stage2(4, 0);
        let optimizer2 = ZeRoOptimizer::new(config2, 0.001);
        let savings2 = optimizer2.memory_savings_ratio();
        
        let config3 = ZeRoConfig::stage3(4, 0);
        let optimizer3 = ZeRoOptimizer::new(config3, 0.001);
        let savings3 = optimizer3.memory_savings_ratio();
        
        // Stage 3 should have highest savings
        assert!(savings3 > savings2);
        assert!(savings2 > savings1);
    }
    
    #[test]
    fn test_offload_config() {
        let config = ZeRoConfig::stage3(4, 0)
            .with_offload(true, false);
        
        assert!(config.cpu_offload);
        assert!(!config.nvme_offload);
    }
    
    #[test]
    fn test_cpu_offload() {
        let config = ZeRoConfig::stage2(2, 0).with_offload(true, false);
        let mut optimizer = ZeRoOptimizer::new(config, 0.001);
        
        let mut params = HashMap::new();
        params.insert("layer1".to_string(), Tensor::randn(&[5, 5]));
        
        optimizer.partition_parameters(&params).unwrap();
        optimizer.offload_to_cpu("layer1").unwrap();
        
        assert!(optimizer.cpu_buffer.contains_key("layer1"));
    }
}

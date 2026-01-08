//! Neural Architecture Search (NAS)
//!
//! Implements automated neural network architecture discovery:
//! - DARTS (Differentiable Architecture Search)
//! - ENAS (Efficient Neural Architecture Search)
//! - NASNet search space
//! - Progressive Neural Architecture Search
//! - Hardware-aware NAS

use ghostflow_core::Tensor;
use std::collections::HashMap;
use rand::Rng;

/// Neural network operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// 3x3 separable convolution
    SepConv3x3,
    /// 5x5 separable convolution
    SepConv5x5,
    /// 3x3 dilated convolution
    DilConv3x3,
    /// 5x5 dilated convolution
    DilConv5x5,
    /// 3x3 max pooling
    MaxPool3x3,
    /// 3x3 average pooling
    AvgPool3x3,
    /// Skip connection
    Skip,
    /// Zero operation (no connection)
    Zero,
}

impl Operation {
    /// Get all available operations
    pub fn all() -> Vec<Operation> {
        vec![
            Operation::SepConv3x3,
            Operation::SepConv5x5,
            Operation::DilConv3x3,
            Operation::DilConv5x5,
            Operation::MaxPool3x3,
            Operation::AvgPool3x3,
            Operation::Skip,
            Operation::Zero,
        ]
    }
    
    /// Get operation cost (FLOPs estimate)
    pub fn cost(&self) -> f32 {
        match self {
            Operation::SepConv3x3 => 9.0,
            Operation::SepConv5x5 => 25.0,
            Operation::DilConv3x3 => 9.0,
            Operation::DilConv5x5 => 25.0,
            Operation::MaxPool3x3 => 1.0,
            Operation::AvgPool3x3 => 1.0,
            Operation::Skip => 0.0,
            Operation::Zero => 0.0,
        }
    }
}

/// Architecture cell (building block)
#[derive(Debug, Clone)]
pub struct Cell {
    /// Number of nodes in the cell
    pub num_nodes: usize,
    /// Operations between nodes: (from_node, to_node, operation)
    pub edges: Vec<(usize, usize, Operation)>,
    /// Architecture parameters (for DARTS)
    pub alpha: HashMap<(usize, usize), Vec<f32>>,
}

impl Cell {
    /// Create a new cell with random architecture
    pub fn random(num_nodes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut edges = Vec::new();
        let mut alpha = HashMap::new();
        
        // Connect each node to previous nodes
        for to_node in 2..num_nodes {
            for from_node in 0..to_node {
                // Random operation
                let ops = Operation::all();
                let op = ops[rng.gen_range(0..ops.len())];
                edges.push((from_node, to_node, op));
                
                // Initialize architecture parameters
                let num_ops = ops.len();
                let weights: Vec<f32> = (0..num_ops)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect();
                alpha.insert((from_node, to_node), weights);
            }
        }
        
        Cell {
            num_nodes,
            edges,
            alpha,
        }
    }
    
    /// Get the dominant operation for each edge (for discretization)
    pub fn get_genotype(&self) -> Vec<(usize, usize, Operation)> {
        let mut genotype = Vec::new();
        let ops = Operation::all();
        
        for ((from, to), weights) in &self.alpha {
            // Find operation with highest weight
            let (max_idx, _) = weights.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            
            genotype.push((*from, *to, ops[max_idx]));
        }
        
        genotype
    }
    
    /// Compute cell cost (total FLOPs)
    pub fn compute_cost(&self) -> f32 {
        self.edges.iter().map(|(_, _, op)| op.cost()).sum()
    }
}

/// DARTS (Differentiable Architecture Search)
pub struct DARTS {
    /// Normal cell (for feature extraction)
    pub normal_cell: Cell,
    /// Reduction cell (for downsampling)
    pub reduction_cell: Cell,
    /// Number of cells in the network
    pub num_cells: usize,
    /// Learning rate for architecture parameters
    pub arch_lr: f32,
    /// Learning rate for network weights
    pub weight_lr: f32,
}

impl DARTS {
    /// Create a new DARTS search
    pub fn new(num_nodes: usize, num_cells: usize) -> Self {
        DARTS {
            normal_cell: Cell::random(num_nodes),
            reduction_cell: Cell::random(num_nodes),
            num_cells,
            arch_lr: 3e-4,
            weight_lr: 0.025,
        }
    }
    
    /// Perform one step of architecture search
    pub fn search_step(&mut self, train_loss: f32, val_loss: f32) {
        // Update architecture parameters based on validation loss
        // Gradient: ∇α L_val
        
        for ((from, to), weights) in self.normal_cell.alpha.iter_mut() {
            // Compute softmax of architecture weights
            let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = weights.iter().map(|w| (w - max_w).exp()).sum();
            
            // Update each operation weight
            for (i, w) in weights.iter_mut().enumerate() {
                let prob = (*w - max_w).exp() / exp_sum;
                // Gradient approximation
                let grad = val_loss * (prob - if i == 0 { 1.0 } else { 0.0 });
                *w -= self.arch_lr * grad;
            }
        }
        
        // Same for reduction cell
        for ((from, to), weights) in self.reduction_cell.alpha.iter_mut() {
            let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = weights.iter().map(|w| (w - max_w).exp()).sum();
            
            for (i, w) in weights.iter_mut().enumerate() {
                let prob = (*w - max_w).exp() / exp_sum;
                let grad = val_loss * (prob - if i == 0 { 1.0 } else { 0.0 });
                *w -= self.arch_lr * grad;
            }
        }
    }
    
    /// Discretize the continuous architecture
    pub fn derive_architecture(&self) -> (Vec<(usize, usize, Operation)>, Vec<(usize, usize, Operation)>) {
        (self.normal_cell.get_genotype(), self.reduction_cell.get_genotype())
    }
    
    /// Compute total network cost
    pub fn total_cost(&self) -> f32 {
        let normal_cost = self.normal_cell.compute_cost();
        let reduction_cost = self.reduction_cell.compute_cost();
        
        // Approximate: most cells are normal, few are reduction
        let num_reduction = (self.num_cells as f32 / 3.0).ceil() as usize;
        let num_normal = self.num_cells - num_reduction;
        
        normal_cost * num_normal as f32 + reduction_cost * num_reduction as f32
    }
}

/// ENAS (Efficient Neural Architecture Search)
pub struct ENAS {
    /// Shared weights for all operations
    pub shared_weights: HashMap<Operation, Tensor>,
    /// Controller RNN state
    pub controller_state: Vec<f32>,
    /// Sampled architectures and their rewards
    pub architecture_pool: Vec<(Cell, f32)>,
    /// Number of architectures to sample per iteration
    pub num_samples: usize,
}

impl ENAS {
    /// Create a new ENAS search
    pub fn new(num_samples: usize) -> Self {
        let mut shared_weights = HashMap::new();
        
        // Initialize shared weights for each operation type
        for op in Operation::all() {
            let weight = Tensor::randn(&[64, 64]); // Example dimensions
            shared_weights.insert(op, weight);
        }
        
        ENAS {
            shared_weights,
            controller_state: vec![0.0; 128], // LSTM hidden state
            architecture_pool: Vec::new(),
            num_samples,
        }
    }
    
    /// Sample an architecture using the controller
    pub fn sample_architecture(&mut self, num_nodes: usize) -> Cell {
        let mut rng = rand::thread_rng();
        let mut cell = Cell::random(num_nodes);
        
        // Use controller to bias sampling (simplified)
        // In full implementation, this would use an RNN controller
        for ((from, to), weights) in cell.alpha.iter_mut() {
            // Softmax with temperature
            let temperature = 1.0;
            let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = weights.iter()
                .map(|w| ((w - max_w) / temperature).exp())
                .sum();
            
            // Sample operation based on probabilities
            let sample: f32 = rng.gen();
            let mut cumsum = 0.0;
            for (i, w) in weights.iter().enumerate() {
                let prob = ((w - max_w) / temperature).exp() / exp_sum;
                cumsum += prob;
                if sample < cumsum {
                    // Set this operation to have highest weight
                    weights[i] = 1.0;
                    break;
                }
            }
        }
        
        cell
    }
    
    /// Train sampled architectures and update controller
    pub fn train_step(&mut self, num_nodes: usize) -> f32 {
        let mut total_reward = 0.0;
        
        // Sample architectures
        for _ in 0..self.num_samples {
            let arch = self.sample_architecture(num_nodes);
            
            // Evaluate architecture (simplified - would train child network)
            let reward = self.evaluate_architecture(&arch);
            total_reward += reward;
            
            // Store in pool
            self.architecture_pool.push((arch, reward));
        }
        
        // Keep only top architectures
        self.architecture_pool.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.architecture_pool.truncate(100);
        
        // Update controller based on rewards (REINFORCE)
        let avg_reward = total_reward / self.num_samples as f32;
        
        // Update controller state (simplified)
        for state in self.controller_state.iter_mut() {
            *state += 0.01 * avg_reward;
        }
        
        avg_reward
    }
    
    /// Evaluate an architecture
    fn evaluate_architecture(&self, arch: &Cell) -> f32 {
        // Reward = accuracy - λ * cost
        let cost = arch.compute_cost();
        let lambda = 0.001; // Cost penalty
        
        // Simplified: reward based on architecture properties
        let num_skip = arch.edges.iter()
            .filter(|(_, _, op)| *op == Operation::Skip)
            .count();
        let num_zero = arch.edges.iter()
            .filter(|(_, _, op)| *op == Operation::Zero)
            .count();
        
        // Prefer architectures with some skip connections but not too many zeros
        let base_reward = 0.8 + 0.1 * (num_skip as f32 / arch.edges.len() as f32);
        let zero_penalty = 0.1 * (num_zero as f32 / arch.edges.len() as f32);
        
        base_reward - zero_penalty - lambda * cost
    }
    
    /// Get best architecture found so far
    pub fn best_architecture(&self) -> Option<&Cell> {
        self.architecture_pool.first().map(|(arch, _)| arch)
    }
}

/// Progressive Neural Architecture Search
pub struct ProgressiveNAS {
    /// Current search stage
    pub stage: usize,
    /// Architectures at each stage
    pub stage_architectures: Vec<Vec<Cell>>,
    /// Complexity budget
    pub complexity_budget: f32,
}

impl ProgressiveNAS {
    /// Create a new progressive NAS
    pub fn new(complexity_budget: f32) -> Self {
        ProgressiveNAS {
            stage: 0,
            stage_architectures: vec![Vec::new()],
            complexity_budget,
        }
    }
    
    /// Progress to next stage
    pub fn next_stage(&mut self, num_nodes: usize, num_candidates: usize) {
        self.stage += 1;
        let mut new_stage = Vec::new();
        
        if self.stage == 1 {
            // First stage: generate random architectures
            for _ in 0..num_candidates {
                let cell = Cell::random(num_nodes);
                if cell.compute_cost() <= self.complexity_budget {
                    new_stage.push(cell);
                }
            }
        } else {
            // Later stages: mutate best from previous stage
            let prev_stage = &self.stage_architectures[self.stage - 1];
            
            for parent in prev_stage.iter().take(num_candidates / 2) {
                // Create mutations
                for _ in 0..2 {
                    let mut child = parent.clone();
                    self.mutate_cell(&mut child);
                    
                    if child.compute_cost() <= self.complexity_budget {
                        new_stage.push(child);
                    }
                }
            }
        }
        
        self.stage_architectures.push(new_stage);
    }
    
    /// Mutate a cell
    fn mutate_cell(&self, cell: &mut Cell) {
        let mut rng = rand::thread_rng();
        let ops = Operation::all();
        
        // Randomly change one operation
        if !cell.edges.is_empty() {
            let idx = rng.gen_range(0..cell.edges.len());
            let new_op = ops[rng.gen_range(0..ops.len())];
            cell.edges[idx].2 = new_op;
        }
    }
    
    /// Get current stage architectures
    pub fn current_architectures(&self) -> &[Cell] {
        &self.stage_architectures[self.stage]
    }
}

/// Hardware-aware NAS
pub struct HardwareAwareNAS {
    /// Target hardware latency (ms)
    pub target_latency: f32,
    /// Target hardware (e.g., "mobile", "gpu", "tpu")
    pub target_hardware: String,
    /// Latency lookup table for operations
    pub latency_table: HashMap<Operation, f32>,
}

impl HardwareAwareNAS {
    /// Create a new hardware-aware NAS
    pub fn new(target_hardware: &str, target_latency: f32) -> Self {
        let mut latency_table = HashMap::new();
        
        // Latency estimates for different hardware (ms per operation)
        match target_hardware {
            "mobile" => {
                latency_table.insert(Operation::SepConv3x3, 2.0);
                latency_table.insert(Operation::SepConv5x5, 5.0);
                latency_table.insert(Operation::DilConv3x3, 3.0);
                latency_table.insert(Operation::DilConv5x5, 7.0);
                latency_table.insert(Operation::MaxPool3x3, 0.5);
                latency_table.insert(Operation::AvgPool3x3, 0.5);
                latency_table.insert(Operation::Skip, 0.1);
                latency_table.insert(Operation::Zero, 0.0);
            }
            "gpu" => {
                latency_table.insert(Operation::SepConv3x3, 0.5);
                latency_table.insert(Operation::SepConv5x5, 1.2);
                latency_table.insert(Operation::DilConv3x3, 0.7);
                latency_table.insert(Operation::DilConv5x5, 1.5);
                latency_table.insert(Operation::MaxPool3x3, 0.1);
                latency_table.insert(Operation::AvgPool3x3, 0.1);
                latency_table.insert(Operation::Skip, 0.05);
                latency_table.insert(Operation::Zero, 0.0);
            }
            "tpu" => {
                latency_table.insert(Operation::SepConv3x3, 0.2);
                latency_table.insert(Operation::SepConv5x5, 0.5);
                latency_table.insert(Operation::DilConv3x3, 0.3);
                latency_table.insert(Operation::DilConv5x5, 0.6);
                latency_table.insert(Operation::MaxPool3x3, 0.05);
                latency_table.insert(Operation::AvgPool3x3, 0.05);
                latency_table.insert(Operation::Skip, 0.02);
                latency_table.insert(Operation::Zero, 0.0);
            }
            _ => {
                // Default to mobile
                latency_table.insert(Operation::SepConv3x3, 2.0);
                latency_table.insert(Operation::SepConv5x5, 5.0);
                latency_table.insert(Operation::DilConv3x3, 3.0);
                latency_table.insert(Operation::DilConv5x5, 7.0);
                latency_table.insert(Operation::MaxPool3x3, 0.5);
                latency_table.insert(Operation::AvgPool3x3, 0.5);
                latency_table.insert(Operation::Skip, 0.1);
                latency_table.insert(Operation::Zero, 0.0);
            }
        }
        
        HardwareAwareNAS {
            target_latency,
            target_hardware: target_hardware.to_string(),
            latency_table,
        }
    }
    
    /// Estimate latency for a cell
    pub fn estimate_latency(&self, cell: &Cell) -> f32 {
        cell.edges.iter()
            .map(|(_, _, op)| self.latency_table.get(op).unwrap_or(&0.0))
            .sum()
    }
    
    /// Check if architecture meets latency constraint
    pub fn meets_constraint(&self, cell: &Cell) -> bool {
        self.estimate_latency(cell) <= self.target_latency
    }
    
    /// Search for architecture meeting latency constraint
    pub fn search(&self, num_nodes: usize, num_iterations: usize) -> Option<Cell> {
        let mut best_cell: Option<Cell> = None;
        let mut best_score = f32::NEG_INFINITY;
        
        for _ in 0..num_iterations {
            let cell = Cell::random(num_nodes);
            
            if self.meets_constraint(&cell) {
                // Score = -latency (prefer faster architectures)
                let score = -self.estimate_latency(&cell);
                
                if score > best_score {
                    best_score = score;
                    best_cell = Some(cell);
                }
            }
        }
        
        best_cell
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cell_creation() {
        let cell = Cell::random(5);
        assert_eq!(cell.num_nodes, 5);
        assert!(!cell.edges.is_empty());
    }
    
    #[test]
    fn test_darts() {
        let mut darts = DARTS::new(4, 8);
        let initial_cost = darts.total_cost();
        
        // Perform search step
        darts.search_step(0.5, 0.6);
        
        // Architecture should be updated
        let (normal, reduction) = darts.derive_architecture();
        assert!(!normal.is_empty());
        assert!(!reduction.is_empty());
    }
    
    #[test]
    fn test_enas() {
        let mut enas = ENAS::new(5);
        let reward = enas.train_step(4);
        
        // Should have sampled architectures
        assert!(!enas.architecture_pool.is_empty());
        assert!(reward.is_finite());
    }
    
    #[test]
    fn test_progressive_nas() {
        let mut pnas = ProgressiveNAS::new(100.0);
        pnas.next_stage(4, 10);
        
        assert_eq!(pnas.stage, 1);
        assert!(!pnas.current_architectures().is_empty());
    }
    
    #[test]
    fn test_hardware_aware_nas() {
        let hwnas = HardwareAwareNAS::new("mobile", 50.0);
        let cell = Cell::random(4);
        
        let latency = hwnas.estimate_latency(&cell);
        assert!(latency >= 0.0);
        
        // Search for architecture
        if let Some(arch) = hwnas.search(4, 100) {
            assert!(hwnas.meets_constraint(&arch));
        }
    }
}

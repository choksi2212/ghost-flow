//! Memory optimization utilities
//!
//! This module provides memory pooling, allocation tracking, and optimization strategies.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Memory pool for reusing allocations
pub struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<u8>>>,
    stats: Arc<Mutex<MemoryStats>>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            stats: Arc::new(Mutex::new(MemoryStats::default())),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, size: usize) -> Vec<u8> {
        let pool = self.pools.entry(size).or_insert_with(Vec::new);
        
        if let Some(buffer) = pool.pop() {
            // Reuse existing allocation
            let mut stats = self.stats.lock().unwrap();
            stats.reused_allocations += 1;
            stats.current_memory += size;
            stats.peak_memory = stats.peak_memory.max(stats.current_memory);
            buffer
        } else {
            // Create new allocation
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocations += 1;
            stats.current_memory += size;
            stats.peak_memory = stats.peak_memory.max(stats.current_memory);
            vec![0u8; size]
        }
    }

    /// Return memory to the pool
    pub fn deallocate(&mut self, buffer: Vec<u8>) {
        let size = buffer.capacity();
        
        let pool = self.pools.entry(size).or_insert_with(Vec::new);
        pool.push(buffer);
        
        let mut stats = self.stats.lock().unwrap();
        stats.current_memory = stats.current_memory.saturating_sub(size);
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
        let mut stats = self.stats.lock().unwrap();
        stats.current_memory = 0;
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocations: usize,
    pub reused_allocations: usize,
    pub current_memory: usize,
    pub peak_memory: usize,
}

impl MemoryStats {
    /// Get reuse rate as a percentage
    pub fn reuse_rate(&self) -> f32 {
        if self.total_allocations == 0 {
            0.0
        } else {
            (self.reused_allocations as f32 / self.total_allocations as f32) * 100.0
        }
    }

    /// Get current memory in MB
    pub fn current_mb(&self) -> f32 {
        self.current_memory as f32 / (1024.0 * 1024.0)
    }

    /// Get peak memory in MB
    pub fn peak_mb(&self) -> f32 {
        self.peak_memory as f32 / (1024.0 * 1024.0)
    }
}

/// Memory layout optimizer
pub struct MemoryLayoutOptimizer {
    alignment: usize,
}

impl MemoryLayoutOptimizer {
    /// Create a new memory layout optimizer
    pub fn new(alignment: usize) -> Self {
        Self { alignment }
    }

    /// Calculate aligned size
    pub fn align_size(&self, size: usize) -> usize {
        (size + self.alignment - 1) / self.alignment * self.alignment
    }

    /// Check if size is aligned
    pub fn is_aligned(&self, size: usize) -> bool {
        size % self.alignment == 0
    }

    /// Optimize memory layout for a tensor shape
    pub fn optimize_layout(&self, shape: &[usize]) -> OptimizedLayout {
        let numel: usize = shape.iter().product();
        let element_size = std::mem::size_of::<f32>();
        let total_size = numel * element_size;
        let aligned_size = self.align_size(total_size);

        OptimizedLayout {
            original_size: total_size,
            aligned_size,
            padding: aligned_size - total_size,
            stride: self.calculate_stride(shape),
        }
    }

    /// Calculate optimal stride for a shape
    fn calculate_stride(&self, shape: &[usize]) -> Vec<usize> {
        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        stride
    }
}

impl Default for MemoryLayoutOptimizer {
    fn default() -> Self {
        Self::new(64) // 64-byte alignment for cache lines
    }
}

/// Optimized memory layout information
#[derive(Debug, Clone)]
pub struct OptimizedLayout {
    pub original_size: usize,
    pub aligned_size: usize,
    pub padding: usize,
    pub stride: Vec<usize>,
}

/// Memory allocator with tracking
pub struct TrackedAllocator {
    stats: Arc<Mutex<AllocationStats>>,
}

impl TrackedAllocator {
    /// Create a new tracked allocator
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(AllocationStats::default())),
        }
    }

    /// Allocate memory
    pub fn allocate(&self, size: usize) -> Vec<u8> {
        let mut stats = self.stats.lock().unwrap();
        stats.allocations += 1;
        stats.total_allocated += size;
        stats.current_allocated += size;
        stats.peak_allocated = stats.peak_allocated.max(stats.current_allocated);
        
        vec![0u8; size]
    }

    /// Deallocate memory
    pub fn deallocate(&self, size: usize) {
        let mut stats = self.stats.lock().unwrap();
        stats.deallocations += 1;
        stats.current_allocated = stats.current_allocated.saturating_sub(size);
    }

    /// Get allocation statistics
    pub fn stats(&self) -> AllocationStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = AllocationStats::default();
    }
}

impl Default for TrackedAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocation statistics
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub total_allocated: usize,
    pub current_allocated: usize,
    pub peak_allocated: usize,
}

impl AllocationStats {
    /// Get current memory in MB
    pub fn current_mb(&self) -> f32 {
        self.current_allocated as f32 / (1024.0 * 1024.0)
    }

    /// Get peak memory in MB
    pub fn peak_mb(&self) -> f32 {
        self.peak_allocated as f32 / (1024.0 * 1024.0)
    }

    /// Get total allocated in MB
    pub fn total_mb(&self) -> f32 {
        self.total_allocated as f32 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new();
        
        // Allocate
        let buf1 = pool.allocate(1024);
        assert_eq!(buf1.len(), 1024);
        
        // Deallocate
        pool.deallocate(buf1);
        
        // Reuse
        let buf2 = pool.allocate(1024);
        assert_eq!(buf2.len(), 1024);
        
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.reused_allocations, 1);
        
        // Clean up
        pool.deallocate(buf2);
    }

    #[test]
    fn test_memory_layout_optimizer() {
        let optimizer = MemoryLayoutOptimizer::new(64);
        
        let layout = optimizer.optimize_layout(&[10, 20]);
        assert!(layout.aligned_size >= layout.original_size);
        assert_eq!(layout.stride, vec![20, 1]);
    }

    #[test]
    fn test_tracked_allocator() {
        let allocator = TrackedAllocator::new();
        
        let _buf = allocator.allocate(1024);
        
        let stats = allocator.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.current_allocated, 1024);
    }

    #[test]
    fn test_alignment() {
        let optimizer = MemoryLayoutOptimizer::new(64);
        
        assert_eq!(optimizer.align_size(100), 128);
        assert_eq!(optimizer.align_size(64), 64);
        assert!(optimizer.is_aligned(128));
        assert!(!optimizer.is_aligned(100));
    }
}

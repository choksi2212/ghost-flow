//! GPU memory management
//!
//! Efficient memory allocation and pooling for GPU tensors

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// GPU memory pool for efficient allocation
#[derive(Debug)]
pub struct GpuMemoryPool {
    /// Free memory blocks by size
    free_blocks: Arc<Mutex<HashMap<usize, Vec<*mut u8>>>>,
    /// Allocated blocks
    allocated: Arc<Mutex<HashMap<*mut u8, usize>>>,
    /// Total allocated memory
    total_allocated: Arc<Mutex<usize>>,
    /// Peak memory usage
    peak_usage: Arc<Mutex<usize>>,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    pub fn new() -> Self {
        Self {
            free_blocks: Arc::new(Mutex::new(HashMap::new())),
            allocated: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
        }
    }

    /// Allocate GPU memory (with pooling)
    pub fn allocate(&self, size: usize) -> Result<*mut u8, String> {
        // Round up to nearest power of 2 for better pooling
        let rounded_size = size.next_power_of_two();
        
        // Try to reuse from pool
        let mut free_blocks = self.free_blocks.lock().unwrap();
        if let Some(blocks) = free_blocks.get_mut(&rounded_size) {
            if let Some(ptr) = blocks.pop() {
                // Reuse existing block
                let mut allocated = self.allocated.lock().unwrap();
                allocated.insert(ptr, rounded_size);
                return Ok(ptr);
            }
        }
        drop(free_blocks);
        
        // Allocate new block
        #[cfg(feature = "cuda")]
        let ptr = unsafe {
            let mut ptr: *mut u8 = std::ptr::null_mut();
            use crate::ffi;
            let result = ffi::cudaMalloc(
                &mut ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
                rounded_size,
            );
            if result != 0 {
                return Err(format!("CUDA malloc failed with error code: {}", result));
            }
            ptr
        };
        
        #[cfg(not(feature = "cuda"))]
        let ptr = {
            // CPU fallback - just allocate regular memory
            vec![0u8; rounded_size].leak().as_mut_ptr()
        };
        
        // Track allocation
        let mut allocated = self.allocated.lock().unwrap();
        allocated.insert(ptr, rounded_size);
        
        let mut total = self.total_allocated.lock().unwrap();
        *total += rounded_size;
        
        let mut peak = self.peak_usage.lock().unwrap();
        if *total > *peak {
            *peak = *total;
        }
        
        Ok(ptr)
    }

    /// Free GPU memory (return to pool)
    pub fn free(&self, ptr: *mut u8) -> Result<(), String> {
        let mut allocated = self.allocated.lock().unwrap();
        
        if let Some(size) = allocated.remove(&ptr) {
            // Return to pool instead of freeing
            let mut free_blocks = self.free_blocks.lock().unwrap();
            free_blocks.entry(size).or_insert_with(Vec::new).push(ptr);
            
            let mut total = self.total_allocated.lock().unwrap();
            *total -= size;
            
            Ok(())
        } else {
            Err("Attempted to free unallocated pointer".to_string())
        }
    }

    /// Clear the memory pool (actually free memory)
    pub fn clear(&self) -> Result<(), String> {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        
        for (_, blocks) in free_blocks.iter() {
            for &ptr in blocks {
                #[cfg(feature = "cuda")]
                unsafe {
                    use crate::ffi;
                    ffi::cudaFree(ptr as *mut std::ffi::c_void);
                }
                
                #[cfg(not(feature = "cuda"))]
                unsafe {
                    // Free CPU memory
                    let _ = Vec::from_raw_parts(ptr, 0, 0);
                }
            }
        }
        
        free_blocks.clear();
        Ok(())
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        *self.total_allocated.lock().unwrap()
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        *self.peak_usage.lock().unwrap()
    }
}

impl Drop for GpuMemoryPool {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            let _ = self.clear();
        }
    }
}

/// GPU tensor wrapper
pub struct GpuTensor {
    /// Pointer to GPU memory
    ptr: *mut f32,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Memory pool reference
    #[allow(dead_code)]
    pool: Arc<GpuMemoryPool>,
}

impl GpuTensor {
    /// Create a new GPU tensor
    #[cfg(feature = "cuda")]
    pub fn new(shape: Vec<usize>, pool: Arc<GpuMemoryPool>) -> Result<Self, String> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * std::mem::size_of::<f32>();
        
        let ptr = pool.allocate(size_bytes)? as *mut f32;
        
        Ok(Self { ptr, shape, pool })
    }

    /// Copy data from CPU to GPU
    pub fn copy_from_cpu(&mut self, data: &[f32]) -> Result<(), String> {
        let numel: usize = self.shape.iter().product();
        if data.len() != numel {
            return Err("Data size mismatch".to_string());
        }
        
        #[cfg(feature = "cuda")]
        unsafe {
            use crate::ffi;
            let result = ffi::cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                numel * std::mem::size_of::<f32>(),
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if result != 0 {
                return Err(format!("CUDA memcpy H2D failed with error code: {}", result));
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        unsafe {
            // CPU fallback - just copy memory
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, numel);
        }
        
        Ok(())
    }

    /// Copy data from GPU to CPU
    pub fn copy_to_cpu(&self) -> Result<Vec<f32>, String> {
        let numel: usize = self.shape.iter().product();
        let mut data = vec![0.0f32; numel];
        
        #[cfg(feature = "cuda")]
        unsafe {
            use crate::ffi;
            let result = ffi::cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                numel * std::mem::size_of::<f32>(),
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if result != 0 {
                return Err(format!("CUDA memcpy D2H failed with error code: {}", result));
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        unsafe {
            // CPU fallback - just copy memory
            std::ptr::copy_nonoverlapping(self.ptr, data.as_mut_ptr(), numel);
        }
        
        Ok(data)
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Drop for GpuTensor {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            let _ = self.pool.free(self.ptr as *mut u8);
        }
    }
}

/// Global GPU memory pool
static mut GLOBAL_GPU_POOL: Option<Arc<GpuMemoryPool>> = None;

/// Get or create the global GPU memory pool
#[allow(static_mut_refs)]
pub fn get_global_gpu_pool() -> Arc<GpuMemoryPool> {
    unsafe {
        if GLOBAL_GPU_POOL.is_none() {
            GLOBAL_GPU_POOL = Some(Arc::new(GpuMemoryPool::new()));
        }
        GLOBAL_GPU_POOL.as_ref().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = GpuMemoryPool::new();
        assert_eq!(pool.current_usage(), 0);
    }
}

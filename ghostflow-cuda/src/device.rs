//! CUDA device management - Real Implementation

use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;
use crate::memory::GpuMemoryPool;
use crate::ffi;
use std::sync::atomic::{AtomicBool, Ordering};
use parking_lot::Mutex;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// CUDA device handle with real device properties
#[derive(Debug)]
pub struct CudaDevice {
    /// Device ID
    pub id: i32,
    /// Device name
    pub name: String,
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    /// Total memory in bytes
    pub total_memory: usize,
    /// Free memory in bytes
    pub free_memory: usize,
    /// Number of multiprocessors
    pub multiprocessor_count: i32,
    /// Max threads per block
    pub max_threads_per_block: i32,
    /// Max threads per multiprocessor
    pub max_threads_per_mp: i32,
    /// Warp size
    pub warp_size: i32,
    /// Shared memory per block
    pub shared_mem_per_block: usize,
    /// Default stream
    pub default_stream: CudaStream,
    /// Memory pool
    pub memory_pool: Mutex<GpuMemoryPool>,
    /// cuBLAS handle
    #[cfg(feature = "cuda")]
    cublas_handle: ffi::cublasHandle_t,
}

impl CudaDevice {
    /// Initialize CUDA runtime
    pub fn init() -> CudaResult<()> {
        if INITIALIZED.swap(true, Ordering::SeqCst) {
            return Err(CudaError::AlreadyInitialized);
        }
        
        // Set device 0 as default
        #[cfg(feature = "cuda")]
        unsafe {
            let err = ffi::cudaSetDevice(0);
            if err != 0 {
                INITIALIZED.store(false, Ordering::SeqCst);
                return Err(CudaError::DriverError(err));
            }
        }
        
        Ok(())
    }

    /// Get number of CUDA devices
    pub fn count() -> CudaResult<i32> {
        #[cfg(feature = "cuda")]
        {
            let mut count: i32 = 0;
            unsafe {
                let err = ffi::cudaGetDeviceCount(&mut count);
                if err != 0 {
                    return Err(CudaError::DriverError(err));
                }
            }
            Ok(count)
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Ok(0)
        }
    }

    /// Create device handle for given device ID
    pub fn new(device_id: i32) -> CudaResult<Self> {
        let count = Self::count()?;
        if device_id >= count || device_id < 0 {
            return Err(CudaError::InvalidDevice(device_id));
        }

        #[cfg(feature = "cuda")]
        {
            // Set device
            unsafe {
                let err = ffi::cudaSetDevice(device_id);
                if err != 0 {
                    return Err(CudaError::DriverError(err));
                }
            }
            
            // Get device properties
            let mut props = cudaDeviceProp::default();
            unsafe {
                let err = ffi::cudaGetDeviceProperties(&mut props, device_id);
                if err != 0 {
                    return Err(CudaError::DriverError(err));
                }
            }
            
            // Extract name
            let name = unsafe {
                CStr::from_ptr(props.name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            
            // Get memory info
            let mut free_mem: usize = 0;
            let mut total_mem: usize = 0;
            unsafe {
                let err = ffi::cudaMemGetInfo(&mut free_mem, &mut total_mem);
                if err != 0 {
                    return Err(CudaError::DriverError(err));
                }
            }
            
            // Create cuBLAS handle
            let mut cublas_handle: ffi::cublasHandle_t = std::ptr::null_mut();
            unsafe {
                let status = ffi::cublasCreate_v2(&mut cublas_handle);
                if status != 0 {
                    return Err(CudaError::CublasError(status));
                }
            }
            
            Ok(CudaDevice {
                id: device_id,
                name,
                compute_capability: (props.major, props.minor),
                total_memory: props.totalGlobalMem,
                free_memory: free_mem,
                multiprocessor_count: props.multiProcessorCount,
                max_threads_per_block: props.maxThreadsPerBlock,
                max_threads_per_mp: props.maxThreadsPerMultiProcessor,
                warp_size: props.warpSize,
                shared_mem_per_block: props.sharedMemPerBlock,
                default_stream: CudaStream::default_stream(),
                memory_pool: Mutex::new(GpuMemoryPool::new(device_id, free_mem)),
                cublas_handle,
            })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::DeviceNotFound)
        }
    }

    /// Set current device
    pub fn set_current(&self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let err = ffi::cudaSetDevice(self.id);
            if err != 0 {
                return Err(CudaError::DriverError(err));
            }
        }
        Ok(())
    }

    /// Get current device ID
    pub fn current() -> CudaResult<i32> {
        let device: i32 = -1;
        
        #[cfg(feature = "cuda")]
        unsafe {
            let err = ffi::cudaGetDevice(&mut device);
            if err != 0 {
                return Err(CudaError::DriverError(err));
            }
        }
        
        Ok(device)
    }

    /// Synchronize device - wait for all operations to complete
    pub fn synchronize() -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let err = ffi::cudaDeviceSynchronize();
            if err != 0 {
                return Err(CudaError::SyncError);
            }
        }
        Ok(())
    }

    /// Get current free memory
    pub fn get_free_memory(&self) -> CudaResult<usize> {
        #[cfg(feature = "cuda")]
        {
            let mut free_mem: usize = 0;
            let mut total_mem: usize = 0;
            unsafe {
                let err = ffi::cudaMemGetInfo(&mut free_mem, &mut total_mem);
                if err != 0 {
                    return Err(CudaError::DriverError(err));
                }
            }
            Ok(free_mem)
        }
        
        #[cfg(not(feature = "cuda"))]
        Ok(0)
    }

    /// Get cuBLAS handle
    #[cfg(feature = "cuda")]
    pub fn cublas_handle(&self) -> ffi::cublasHandle_t {
        self.cublas_handle
    }

    /// Get device properties as string
    pub fn properties_string(&self) -> String {
        format!(
            "Device {}: {}\n\
             Compute Capability: {}.{}\n\
             Total Memory: {:.2} GB\n\
             Free Memory: {:.2} GB\n\
             Multiprocessors: {}\n\
             Max Threads/Block: {}\n\
             Warp Size: {}\n\
             Shared Mem/Block: {} KB",
            self.id,
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            self.free_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            self.multiprocessor_count,
            self.max_threads_per_block,
            self.warp_size,
            self.shared_mem_per_block / 1024,
        )
    }
    
    /// Check if device supports tensor cores
    pub fn has_tensor_cores(&self) -> bool {
        // Tensor cores available on Volta (7.0) and later
        self.compute_capability.0 >= 7
    }
    
    /// Check if device supports FP16
    pub fn supports_fp16(&self) -> bool {
        // FP16 available on Pascal (6.0) and later
        self.compute_capability.0 >= 6
    }
    
    /// Check if device supports BF16
    pub fn supports_bf16(&self) -> bool {
        // BF16 available on Ampere (8.0) and later
        self.compute_capability.0 >= 8
    }
    
    /// Get optimal block size for a kernel
    pub fn optimal_block_size(&self, shared_mem_per_thread: usize) -> usize {
        let max_threads = self.max_threads_per_block as usize;
        let shared_limit = self.shared_mem_per_block / shared_mem_per_thread.max(1);
        
        // Round down to multiple of warp size
        let optimal = max_threads.min(shared_limit);
        (optimal / self.warp_size as usize) * self.warp_size as usize
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if !self.cublas_handle.is_null() {
                ffi::cublasDestroy_v2(self.cublas_handle);
            }
        }
    }
}

/// Device guard for automatic device switching
pub struct DeviceGuard {
    #[allow(dead_code)]
    previous_device: i32,
}

impl DeviceGuard {
    pub fn new(_device_id: i32) -> CudaResult<Self> {
        let previous = CudaDevice::current()?;
        
        #[cfg(feature = "cuda")]
        unsafe {
            let err = ffi::cudaSetDevice(_device_id);
            if err != 0 {
                return Err(CudaError::DriverError(err));
            }
        }
        
        Ok(DeviceGuard { previous_device: previous })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            let _ = ffi::cudaSetDevice(self.previous_device);
        }
    }
}

/// Get all available CUDA devices
pub fn get_all_devices() -> CudaResult<Vec<CudaDevice>> {
    let count = CudaDevice::count()?;
    let mut devices = Vec::with_capacity(count as usize);
    
    for i in 0..count {
        devices.push(CudaDevice::new(i)?);
    }
    
    Ok(devices)
}

/// Select best device based on compute capability and memory
pub fn select_best_device() -> CudaResult<CudaDevice> {
    let devices = get_all_devices()?;
    
    if devices.is_empty() {
        return Err(CudaError::DeviceNotFound);
    }
    
    // Score devices by compute capability and memory
    let best = devices.into_iter()
        .max_by_key(|d| {
            let compute_score = d.compute_capability.0 * 10 + d.compute_capability.1;
            let memory_score = (d.total_memory / (1024 * 1024 * 1024)) as i32; // GB
            compute_score * 100 + memory_score
        })
        .unwrap();
    
    Ok(best)
}

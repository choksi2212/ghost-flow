//! CUDA tensor type - Real GPU tensor implementation

use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;
use crate::blas::CuBlas;
use ghostflow_core::{DType, Shape, Strides, Tensor};

/// Tensor stored on CUDA device
#[derive(Debug)]
pub struct CudaTensor {
    /// GPU memory pointer
    ptr: *mut f32,
    /// Shape
    shape: Shape,
    /// Strides
    strides: Strides,
    /// Data type
    dtype: DType,
    /// Device ID
    device_id: i32,
    /// Size in bytes
    size_bytes: usize,
}

impl CudaTensor {
    /// Create new CUDA tensor with given shape (uninitialized)
    pub fn new(shape: &[usize], dtype: DType, device_id: i32) -> CudaResult<Self> {
        let shape = Shape::new(shape);
        let strides = shape.default_strides();
        let size_bytes = shape.numel() * dtype.size_bytes();
        
        // Allocate GPU memory
        let pool = crate::memory::get_global_gpu_pool();
        let ptr = (*pool).allocate(size_bytes)
            .map_err(|_e| CudaError::OutOfMemory)? as *mut f32;
        
        Ok(CudaTensor {
            ptr,
            shape,
            strides,
            dtype,
            device_id,
            size_bytes,
        })
    }

    /// Create CUDA tensor from CPU tensor (copies data to GPU)
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn from_tensor(tensor: &Tensor, device_id: i32) -> CudaResult<Self> {
        let shape = tensor.shape().clone();
        let strides = shape.default_strides();
        let dtype = tensor.dtype();
        let size_bytes = shape.numel() * dtype.size_bytes();
        
        let data = tensor.data_f32();
        
        // Allocate GPU memory
        let pool = crate::memory::get_global_gpu_pool();
        let ptr = (*pool).allocate(size_bytes)
            .map_err(|_| CudaError::OutOfMemory)? as *mut f32;
        
        // Copy data to GPU
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpy(
                ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(CudaTensor {
            ptr,
            shape,
            strides,
            dtype,
            device_id,
            size_bytes,
        })
    }

    /// Create CUDA tensor from CPU tensor (async copy)
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn from_tensor_async(tensor: &Tensor, device_id: i32, stream: &CudaStream) -> CudaResult<Self> {
        let shape = tensor.shape().clone();
        let strides = shape.default_strides();
        let dtype = tensor.dtype();
        let size_bytes = shape.numel() * dtype.size_bytes();
        
        let data = tensor.data_f32();
        
        // Allocate GPU memory
        let pool = crate::memory::get_global_gpu_pool();
        let ptr = (*pool).allocate(size_bytes)
            .map_err(|_| CudaError::OutOfMemory)? as *mut f32;
        
        // Copy data to GPU asynchronously
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpyAsync(
                ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream.as_raw(),
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(CudaTensor {
            ptr,
            shape,
            strides,
            dtype,
            device_id,
            size_bytes,
        })
    }

    /// Copy back to CPU tensor
    pub fn to_tensor(&self) -> CudaResult<Tensor> {
        let data = vec![0.0f32; self.numel()];
        
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                data.len() * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Tensor::from_slice(&data, self.shape.dims())
            .map_err(|e| CudaError::InvalidValue(e.to_string()))
    }

    /// Copy back to CPU tensor (async)
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn to_tensor_async(&self, stream: &CudaStream) -> CudaResult<Vec<f32>> {
        let data = vec![0.0f32; self.numel()];
        
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpyAsync(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                data.len() * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream.as_raw(),
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(data)
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get dimensions
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get raw buffer pointer
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    /// Get mutable raw buffer pointer
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Create zeros tensor
    pub fn zeros(shape: &[usize], device_id: i32) -> CudaResult<Self> {
        let tensor = Self::new(shape, DType::F32, device_id)?;
        
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemset(
                tensor.ptr as *mut std::ffi::c_void,
                0,
                tensor.size_bytes,
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(tensor)
    }

    /// Create ones tensor
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn ones(shape: &[usize], device_id: i32) -> CudaResult<Self> {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();
        let data: Vec<f32> = vec![1.0; numel];
        let size_bytes = numel * std::mem::size_of::<f32>();
        
        // Allocate GPU memory
        let pool = crate::memory::get_global_gpu_pool();
        let ptr = (*pool).allocate(size_bytes)
            .map_err(|_| CudaError::OutOfMemory)? as *mut f32;
        
        // Copy data to GPU
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpy(
                ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                size_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(CudaTensor {
            ptr,
            shape: shape_obj.clone(),
            strides: shape_obj.default_strides(),
            dtype: DType::F32,
            device_id,
            size_bytes,
        })
    }

    /// Create tensor filled with value
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn full(shape: &[usize], value: f32, device_id: i32) -> CudaResult<Self> {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();
        let data: Vec<f32> = vec![value; numel];
        let size_bytes = numel * std::mem::size_of::<f32>();
        
        // Allocate GPU memory
        let pool = crate::memory::get_global_gpu_pool();
        let ptr = (*pool).allocate(size_bytes)
            .map_err(|_| CudaError::OutOfMemory)? as *mut f32;
        
        // Copy data to GPU
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpy(
                ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                size_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(CudaTensor {
            ptr,
            shape: shape_obj.clone(),
            strides: shape_obj.default_strides(),
            dtype: DType::F32,
            device_id,
            size_bytes,
        })
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: &[usize]) -> CudaResult<Self> {
        let new_shape = Shape::new(new_shape);
        if new_shape.numel() != self.shape.numel() {
            return Err(CudaError::InvalidValue(
                format!("Cannot reshape {} elements to {:?}", self.numel(), new_shape.dims())
            ));
        }
        
        // Create new tensor with same data
        let pool = crate::memory::get_global_gpu_pool();
        let ptr = (*pool).allocate(self.size_bytes)
            .map_err(|_| CudaError::OutOfMemory)? as *mut f32;
        
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpy(
                ptr as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                self.size_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(CudaTensor {
            ptr,
            shape: new_shape.clone(),
            strides: new_shape.default_strides(),
            dtype: self.dtype,
            device_id: self.device_id,
            size_bytes: self.size_bytes,
        })
    }

    /// Transpose dimensions
    pub fn transpose(&self, dim0: usize, dim1: usize) -> CudaResult<Self> {
        // For now, copy to CPU, transpose, copy back
        // TODO: Implement GPU transpose kernel
        let cpu = self.to_tensor()?;
        let transposed = cpu.transpose(dim0, dim1)
            .map_err(|e| CudaError::InvalidValue(e.to_string()))?;
        Self::from_tensor(&transposed, self.device_id)
    }

    /// Check if tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    /// Deep clone
    pub fn clone_tensor(&self) -> CudaResult<Self> {
        let pool = crate::memory::get_global_gpu_pool();
        let ptr = (*pool).allocate(self.size_bytes)
            .map_err(|_| CudaError::OutOfMemory)? as *mut f32;
        
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_runtime_sys::*;
            let result = cudaMemcpy(
                ptr as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                self.size_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            );
            if result != cudaError_t::cudaSuccess {
                return Err(CudaError::MemcpyError);
            }
        }
        
        Ok(CudaTensor {
            ptr,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            device_id: self.device_id,
            size_bytes: self.size_bytes,
        })
    }

    // ==================== Operations ====================

    /// Element-wise addition
    pub fn add(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        if self.shape.dims() != other.shape.dims() {
            return Err(CudaError::InvalidValue("Shape mismatch for add".into()));
        }
        
        // Copy self to result
        let mut result = self.clone_tensor()?;
        
        // Use cuBLAS SAXPY: y = 1.0 * x + y
        let cublas = CuBlas::new()?;
        cublas.saxpy(
            self.numel() as i32,
            1.0,
            other.as_ptr() as *const f32,
            1,
            result.as_mut_ptr() as *mut f32,
            1,
        )?;
        
        Ok(result)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        if self.shape.dims() != other.shape.dims() {
            return Err(CudaError::InvalidValue("Shape mismatch for sub".into()));
        }
        
        let mut result = self.clone_tensor()?;
        
        // y = -1.0 * x + y
        let cublas = CuBlas::new()?;
        cublas.saxpy(
            self.numel() as i32,
            -1.0,
            other.as_ptr() as *const f32,
            1,
            result.as_mut_ptr() as *mut f32,
            1,
        )?;
        
        Ok(result)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> CudaResult<CudaTensor> {
        let mut result = self.clone_tensor()?;
        
        let cublas = CuBlas::new()?;
        cublas.sscal(
            self.numel() as i32,
            scalar,
            result.as_mut_ptr() as *mut f32,
            1,
        )?;
        
        Ok(result)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        let cublas = CuBlas::new()?;
        cublas.matmul(self, other)
    }

    /// Dot product
    pub fn dot(&self, other: &CudaTensor) -> CudaResult<f32> {
        if self.numel() != other.numel() {
            return Err(CudaError::InvalidValue("Size mismatch for dot".into()));
        }
        
        let cublas = CuBlas::new()?;
        cublas.sdot(
            self.numel() as i32,
            self.as_ptr() as *const f32,
            1,
            other.as_ptr() as *const f32,
            1,
        )
    }

    /// L2 norm
    pub fn norm(&self) -> CudaResult<f32> {
        let cublas = CuBlas::new()?;
        cublas.snrm2(
            self.numel() as i32,
            self.as_ptr() as *const f32,
            1,
        )
    }

    /// Sum all elements
    pub fn sum(&self) -> CudaResult<f32> {
        // For now, copy to CPU and sum
        // TODO: Implement GPU reduction kernel
        let cpu = self.to_tensor()?;
        Ok(cpu.data_f32().iter().sum())
    }

    /// Mean of all elements
    pub fn mean(&self) -> CudaResult<f32> {
        let sum = self.sum()?;
        Ok(sum / self.numel() as f32)
    }

    /// Max element
    pub fn max(&self) -> CudaResult<f32> {
        let cpu = self.to_tensor()?;
        Ok(cpu.data_f32().iter().cloned().fold(f32::NEG_INFINITY, f32::max))
    }

    /// Min element
    pub fn min(&self) -> CudaResult<f32> {
        let cpu = self.to_tensor()?;
        Ok(cpu.data_f32().iter().cloned().fold(f32::INFINITY, f32::min))
    }

    /// ReLU activation (in-place capable)
    pub fn relu(&self) -> CudaResult<CudaTensor> {
        // TODO: Implement GPU kernel
        let cpu = self.to_tensor()?;
        let result = cpu.relu();
        Self::from_tensor(&result, self.device_id)
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> CudaResult<CudaTensor> {
        let cpu = self.to_tensor()?;
        let result = cpu.sigmoid();
        Self::from_tensor(&result, self.device_id)
    }

    /// GELU activation
    pub fn gelu(&self) -> CudaResult<CudaTensor> {
        let cpu = self.to_tensor()?;
        let result = cpu.gelu();
        Self::from_tensor(&result, self.device_id)
    }

    /// Softmax
    pub fn softmax(&self, dim: i32) -> CudaResult<CudaTensor> {
        let cpu = self.to_tensor()?;
        let result = cpu.softmax(dim);
        Self::from_tensor(&result, self.device_id)
    }

    /// Exponential
    pub fn exp(&self) -> CudaResult<CudaTensor> {
        let cpu = self.to_tensor()?;
        let result = cpu.exp();
        Self::from_tensor(&result, self.device_id)
    }

    /// Natural logarithm
    pub fn log(&self) -> CudaResult<CudaTensor> {
        let cpu = self.to_tensor()?;
        let result = cpu.log();
        Self::from_tensor(&result, self.device_id)
    }

    /// Square root
    pub fn sqrt(&self) -> CudaResult<CudaTensor> {
        let cpu = self.to_tensor()?;
        let result = cpu.sqrt();
        Self::from_tensor(&result, self.device_id)
    }

    /// Power
    pub fn pow(&self, exp: f32) -> CudaResult<CudaTensor> {
        let cpu = self.to_tensor()?;
        let result = cpu.pow(exp);
        Self::from_tensor(&result, self.device_id)
    }
}

impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        self.clone_tensor().expect("Failed to clone CudaTensor")
    }
}

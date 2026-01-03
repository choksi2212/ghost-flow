//! cuBLAS wrapper for linear algebra operations

use crate::error::{CudaError, CudaResult};
use crate::ffi::cublasHandle_t;
use crate::stream::CudaStream;
use crate::tensor::CudaTensor;

/// cuBLAS handle wrapper
pub struct CuBlas {
    #[allow(dead_code)]
    handle: cublasHandle_t,
}

impl CuBlas {
    /// Create new cuBLAS handle
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut handle: cublasHandle_t = std::ptr::null_mut();
            
            unsafe {
                let status = ffi::cublasCreate_v2(&mut handle);
                if status != 0 {
                    return Err(CudaError::CublasError(status));
                }
            }
            
            Ok(CuBlas { handle })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Ok(CuBlas {
                handle: std::ptr::null_mut(),
            })
        }
    }

    /// Set stream for cuBLAS operations
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn set_stream(&self, stream: &CudaStream) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let status = ffi::cublasSetStream_v2(self.handle, stream.handle());
            if status != 0 {
                return Err(CudaError::CublasError(status));
            }
        }
        Ok(())
    }

    /// SGEMM: C = alpha * op(A) * op(B) + beta * C
    /// 
    /// This is the core matrix multiplication operation.
    /// 
    /// # Arguments
    /// * `trans_a` - Whether to transpose A
    /// * `trans_b` - Whether to transpose B
    /// * `m` - Number of rows of op(A) and C
    /// * `n` - Number of columns of op(B) and C
    /// * `k` - Number of columns of op(A) and rows of op(B)
    /// * `alpha` - Scalar multiplier for A*B
    /// * `a` - Matrix A
    /// * `lda` - Leading dimension of A
    /// * `b` - Matrix B
    /// * `ldb` - Leading dimension of B
    /// * `beta` - Scalar multiplier for C
    /// * `c` - Matrix C (output)
    /// * `ldc` - Leading dimension of C
    pub fn sgemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
            let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
            
            let status = ffi::cublasSgemm_v2(
                self.handle,
                op_a,
                op_b,
                m, n, k,
                &alpha,
                a, lda,
                b, ldb,
                &beta,
                c, ldc,
            );
            
            if status != 0 {
                return Err(CudaError::CublasError(status));
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            // CPU fallback - naive implementation
            unsafe {
                for i in 0..m as usize {
                    for j in 0..n as usize {
                        let mut sum = if beta != 0.0 {
                            beta * *c.add(i + j * ldc as usize)
                        } else {
                            0.0
                        };
                        
                        for l in 0..k as usize {
                            let a_idx = if trans_a { l + i * lda as usize } else { i + l * lda as usize };
                            let b_idx = if trans_b { j + l * ldb as usize } else { l + j * ldb as usize };
                            sum += alpha * *a.add(a_idx) * *b.add(b_idx);
                        }
                        
                        *c.add(i + j * ldc as usize) = sum;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// SAXPY: y = alpha * x + y
    pub fn saxpy(
        &self,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let status = ffi::cublasSaxpy_v2(
                self.handle,
                n,
                &alpha,
                x, incx,
                y, incy,
            );
            
            if status != 0 {
                return Err(CudaError::CublasError(status));
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        unsafe {
            for i in 0..n as usize {
                let xi = *x.add(i * incx as usize);
                let yi = y.add(i * incy as usize);
                *yi = alpha * xi + *yi;
            }
        }
        
        Ok(())
    }

    /// SDOT: result = x . y
    pub fn sdot(
        &self,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> CudaResult<f32> {
        let mut result: f32 = 0.0;
        
        #[cfg(feature = "cuda")]
        unsafe {
            let status = ffi::cublasSdot_v2(
                self.handle,
                n,
                x, incx,
                y, incy,
                &mut result,
            );
            
            if status != 0 {
                return Err(CudaError::CublasError(status));
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        unsafe {
            for i in 0..n as usize {
                result += *x.add(i * incx as usize) * *y.add(i * incy as usize);
            }
        }
        
        Ok(result)
    }

    /// SNRM2: result = ||x||_2
    pub fn snrm2(
        &self,
        n: i32,
        x: *const f32,
        incx: i32,
    ) -> CudaResult<f32> {
        #[cfg(feature = "cuda")]
        {
            let mut result: f32 = 0.0;
            unsafe {
                let status = ffi::cublasSnrm2_v2(
                    self.handle,
                    n,
                    x, incx,
                    &mut result,
                );
                
                if status != 0 {
                    return Err(CudaError::CublasError(status));
                }
            }
            Ok(result)
        }
        
        #[cfg(not(feature = "cuda"))]
        unsafe {
            let mut sum = 0.0f32;
            for i in 0..n as usize {
                let xi = *x.add(i * incx as usize);
                sum += xi * xi;
            }
            Ok(sum.sqrt())
        }
    }

    /// SSCAL: x = alpha * x
    pub fn sscal(
        &self,
        n: i32,
        alpha: f32,
        x: *mut f32,
        incx: i32,
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let status = ffi::cublasSscal_v2(
                self.handle,
                n,
                &alpha,
                x, incx,
            );
            
            if status != 0 {
                return Err(CudaError::CublasError(status));
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        unsafe {
            for i in 0..n as usize {
                let xi = x.add(i * incx as usize);
                *xi = alpha * *xi;
            }
        }
        
        Ok(())
    }

    /// Matrix multiplication for CudaTensors
    pub fn matmul(&self, a: &CudaTensor, b: &CudaTensor) -> CudaResult<CudaTensor> {
        let a_dims = a.dims();
        let b_dims = b.dims();
        
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(CudaError::InvalidValue("matmul requires 2D tensors".into()));
        }
        
        let m = a_dims[0] as i32;
        let k = a_dims[1] as i32;
        let n = b_dims[1] as i32;
        
        if k != b_dims[0] as i32 {
            return Err(CudaError::InvalidValue(format!(
                "Matrix dimensions don't match: [{}, {}] x [{}, {}]",
                m, k, b_dims[0], n
            )));
        }
        
        // Create output tensor
        let mut c = CudaTensor::zeros(&[m as usize, n as usize], a.device_id())?;
        
        // cuBLAS uses column-major, so we compute C^T = B^T * A^T
        // which gives us C in row-major
        self.sgemm(
            false, false,
            n, m, k,
            1.0,
            b.as_ptr() as *const f32, n,
            a.as_ptr() as *const f32, k,
            0.0,
            c.as_mut_ptr() as *mut f32, n,
        )?;
        
        Ok(c)
    }
}

impl Drop for CuBlas {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if !self.handle.is_null() {
            unsafe {
                let _ = ffi::cublasDestroy_v2(self.handle);
            }
        }
    }
}

unsafe impl Send for CuBlas {}
unsafe impl Sync for CuBlas {}

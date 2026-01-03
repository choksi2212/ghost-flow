//! Matrix multiplication and linear algebra operations

use crate::tensor::Tensor;
use crate::error::{GhostError, Result};
use rayon::prelude::*;

impl Tensor {
    /// Matrix multiplication
    /// Supports:
    /// - 2D x 2D: standard matmul
    /// - Batched: broadcast batch dimensions
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a_dims = self.dims();
        let b_dims = other.dims();

        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(GhostError::InvalidOperation(
                "matmul requires at least 2D tensors".to_string()
            ));
        }

        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let k2 = b_dims[b_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];

        if k != k2 {
            return Err(GhostError::ShapeMismatch {
                expected: vec![m, k],
                got: vec![k2, n],
            });
        }

        // Handle batched matmul
        if a_dims.len() == 2 && b_dims.len() == 2 {
            return self.matmul_2d(other, m, k, n);
        }

        // Batched case
        self.batched_matmul(other)
    }

    /// 2D matrix multiplication (optimized)
    fn matmul_2d(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        // Use BLAS if available and matrix is large enough
        #[cfg(feature = "blas")]
        {
            const BLAS_THRESHOLD: usize = 64;
            if m >= BLAS_THRESHOLD && n >= BLAS_THRESHOLD && k >= BLAS_THRESHOLD {
                return self.matmul_blas(other, m, k, n);
            }
        }
        
        // Fallback to optimized blocked implementation
        self.matmul_blocked(other, m, k, n)
    }

    /// BLAS-accelerated matrix multiplication (10-50x faster for large matrices)
    #[cfg(feature = "blas")]
    fn matmul_blas(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        use cblas::*;
        
        let a = self.data_f32();
        let b = other.data_f32();
        let mut c = vec![0.0f32; m * n];
        
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                m as i32,
                n as i32,
                k as i32,
                1.0,           // alpha
                &a,
                k as i32,      // lda
                &b,
                n as i32,      // ldb
                0.0,           // beta
                &mut c,
                n as i32,      // ldc
            );
        }
        
        Tensor::from_slice(&c, &[m, n])
    }

    /// Blocked/tiled matrix multiplication (cache-optimized fallback)
    fn matmul_blocked(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let a = self.data_f32();
        let b = other.data_f32();
        
        // Use blocked/tiled multiplication for cache efficiency
        let mut c = vec![0.0f32; m * n];
        
        const BLOCK_SIZE: usize = 64;
        
        // Parallel over output rows
        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for jb in (0..n).step_by(BLOCK_SIZE) {
                let j_end = (jb + BLOCK_SIZE).min(n);
                
                for kb in (0..k).step_by(BLOCK_SIZE) {
                    let k_end = (kb + BLOCK_SIZE).min(k);
                    
                    for kk in kb..k_end {
                        let a_ik = a[i * k + kk];
                        for j in jb..j_end {
                            row[j] += a_ik * b[kk * n + j];
                        }
                    }
                }
            }
        });

        Tensor::from_slice(&c, &[m, n])
    }

    /// Batched matrix multiplication
    fn batched_matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a_dims = self.dims();
        let b_dims = other.dims();
        
        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let n = b_dims[b_dims.len() - 1];
        
        // Compute batch dimensions
        let a_batch: Vec<usize> = a_dims[..a_dims.len() - 2].to_vec();
        let b_batch: Vec<usize> = b_dims[..b_dims.len() - 2].to_vec();
        
        // Broadcast batch dimensions
        let batch_dims = broadcast_batch_dims(&a_batch, &b_batch)?;
        let batch_size: usize = batch_dims.iter().product();
        
        let a = self.data_f32();
        let b = other.data_f32();
        
        let a_batch_stride = m * k;
        let b_batch_stride = k * n;
        let c_batch_stride = m * n;
        
        let mut result = vec![0.0f32; batch_size * m * n];
        
        result.par_chunks_mut(c_batch_stride).enumerate().for_each(|(batch_idx, c_batch)| {
            let a_idx = batch_idx % (a_batch.iter().product::<usize>().max(1));
            let b_idx = batch_idx % (b_batch.iter().product::<usize>().max(1));
            
            let a_start = a_idx * a_batch_stride;
            let b_start = b_idx * b_batch_stride;
            
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[a_start + i * k + kk] * b[b_start + kk * n + j];
                    }
                    c_batch[i * n + j] = sum;
                }
            }
        });
        
        let mut out_shape = batch_dims;
        out_shape.push(m);
        out_shape.push(n);
        
        Tensor::from_slice(&result, &out_shape)
    }

    /// Vector dot product
    pub fn dot(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(GhostError::InvalidOperation(
                "dot requires 1D tensors".to_string()
            ));
        }
        
        if self.numel() != other.numel() {
            return Err(GhostError::ShapeMismatch {
                expected: self.dims().to_vec(),
                got: other.dims().to_vec(),
            });
        }
        
        let a = self.data_f32();
        let b = other.data_f32();
        
        let dot: f32 = a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| x * y)
            .sum();
        
        Tensor::from_slice(&[dot], &[])
    }

    /// Outer product of two vectors
    pub fn outer(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(GhostError::InvalidOperation(
                "outer requires 1D tensors".to_string()
            ));
        }
        
        let a = self.data_f32();
        let b = other.data_f32();
        let m = a.len();
        let n = b.len();
        
        let result: Vec<f32> = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                b.iter().map(|&bj| a[i] * bj).collect::<Vec<_>>()
            })
            .collect();
        
        Tensor::from_slice(&result, &[m, n])
    }

    /// Matrix-vector multiplication
    pub fn mv(&self, vec: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || vec.ndim() != 1 {
            return Err(GhostError::InvalidOperation(
                "mv requires 2D matrix and 1D vector".to_string()
            ));
        }
        
        let m = self.dims()[0];
        let n = self.dims()[1];
        
        if vec.numel() != n {
            return Err(GhostError::ShapeMismatch {
                expected: vec![n],
                got: vec.dims().to_vec(),
            });
        }
        
        let mat = self.data_f32();
        let v = vec.data_f32();
        
        let result: Vec<f32> = (0..m)
            .into_par_iter()
            .map(|i| {
                (0..n).map(|j| mat[i * n + j] * v[j]).sum()
            })
            .collect();
        
        Tensor::from_slice(&result, &[m])
    }

    /// Batch matrix-matrix multiplication (bmm)
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 3 || other.ndim() != 3 {
            return Err(GhostError::InvalidOperation(
                "bmm requires 3D tensors".to_string()
            ));
        }
        
        self.matmul(other)
    }

    /// Compute trace of a matrix
    pub fn trace(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(GhostError::InvalidOperation(
                "trace requires 2D tensor".to_string()
            ));
        }
        
        let dims = self.dims();
        let n = dims[0].min(dims[1]);
        let data = self.data_f32();
        let cols = dims[1];
        
        let trace: f32 = (0..n).map(|i| data[i * cols + i]).sum();
        
        Tensor::from_slice(&[trace], &[])
    }

    /// Compute diagonal of a matrix
    pub fn diag(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(GhostError::InvalidOperation(
                "diag requires 2D tensor".to_string()
            ));
        }
        
        let dims = self.dims();
        let n = dims[0].min(dims[1]);
        let data = self.data_f32();
        let cols = dims[1];
        
        let diag: Vec<f32> = (0..n).map(|i| data[i * cols + i]).collect();
        
        Tensor::from_slice(&diag, &[n])
    }
}

/// Broadcast batch dimensions
fn broadcast_batch_dims(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        
        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return Err(GhostError::BroadcastError {
                a: a.to_vec(),
                b: b.to_vec(),
            });
        }
    }
    
    result.reverse();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        // [2, 3] x [3, 2] = [2, 2]
        let a = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3]
        ).unwrap();
        let b = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2]
        ).unwrap();
        
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        
        // Expected: [[22, 28], [49, 64]]
        let data = c.data_f32();
        assert_eq!(data[0], 22.0);
        assert_eq!(data[1], 28.0);
        assert_eq!(data[2], 49.0);
        assert_eq!(data[3], 64.0);
    }

    #[test]
    fn test_dot() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3]).unwrap();
        
        let dot = a.dot(&b).unwrap();
        assert_eq!(dot.data_f32()[0], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_mv() {
        let mat = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let vec = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        
        let result = mat.mv(&vec).unwrap();
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.data_f32(), vec![5.0, 11.0]); // [1*1+2*2, 3*1+4*2]
    }

    #[test]
    fn test_trace() {
        let mat = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let trace = mat.trace().unwrap();
        assert_eq!(trace.data_f32()[0], 5.0); // 1 + 4
    }
}

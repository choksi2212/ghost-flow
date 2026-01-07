//! Sparse tensor operations
//!
//! Implements sparse tensor formats:
//! - COO (Coordinate format)
//! - CSR (Compressed Sparse Row)
//! - CSC (Compressed Sparse Column)

use crate::tensor::Tensor;
use crate::error::{GhostError, Result};
use std::collections::HashMap;

/// Sparse tensor in COO (Coordinate) format
#[derive(Debug, Clone)]
pub struct SparseTensorCOO {
    /// Non-zero values
    pub values: Vec<f32>,
    /// Row indices
    pub rows: Vec<usize>,
    /// Column indices
    pub cols: Vec<usize>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Number of non-zero elements
    pub nnz: usize,
}

impl SparseTensorCOO {
    /// Create a new sparse tensor in COO format
    pub fn new(values: Vec<f32>, rows: Vec<usize>, cols: Vec<usize>, shape: Vec<usize>) -> Result<Self> {
        if values.len() != rows.len() || values.len() != cols.len() {
            return Err(GhostError::InvalidShape(
                "Values, rows, and cols must have the same length".to_string()
            ));
        }
        
        let nnz = values.len();
        
        Ok(SparseTensorCOO {
            values,
            rows,
            cols,
            shape,
            nnz,
        })
    }
    
    /// Create sparse tensor from dense tensor (threshold-based)
    pub fn from_dense(tensor: &Tensor, threshold: f32) -> Self {
        let data = tensor.data_f32();
        let shape = tensor.dims().to_vec();
        
        let mut values = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        
        if shape.len() == 2 {
            let (nrows, ncols) = (shape[0], shape[1]);
            for i in 0..nrows {
                for j in 0..ncols {
                    let val = data[i * ncols + j];
                    if val.abs() > threshold {
                        values.push(val);
                        rows.push(i);
                        cols.push(j);
                    }
                }
            }
        }
        
        let nnz = values.len();
        SparseTensorCOO {
            values,
            rows,
            cols,
            shape,
            nnz,
        }
    }
    
    /// Convert to dense tensor
    pub fn to_dense(&self) -> Result<Tensor> {
        if self.shape.len() != 2 {
            return Err(GhostError::InvalidShape(
                "Only 2D sparse tensors supported".to_string()
            ));
        }
        
        let (nrows, ncols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0f32; nrows * ncols];
        
        for i in 0..self.nnz {
            let row = self.rows[i];
            let col = self.cols[i];
            let val = self.values[i];
            data[row * ncols + col] = val;
        }
        
        Tensor::from_slice(&data, &self.shape)
    }
    
    /// Sparse matrix-vector multiplication
    pub fn spmv(&self, vec: &Tensor) -> Result<Tensor> {
        if self.shape.len() != 2 {
            return Err(GhostError::InvalidShape(
                "SpMV requires 2D sparse matrix".to_string()
            ));
        }
        
        let vec_data = vec.data_f32();
        if vec_data.len() != self.shape[1] {
            return Err(GhostError::InvalidShape(
                format!("Vector length {} doesn't match matrix columns {}", vec_data.len(), self.shape[1])
            ));
        }
        
        let mut result = vec![0.0f32; self.shape[0]];
        
        for i in 0..self.nnz {
            let row = self.rows[i];
            let col = self.cols[i];
            let val = self.values[i];
            result[row] += val * vec_data[col];
        }
        
        Tensor::from_slice(&result, &[self.shape[0]])
    }
    
    /// Sparse matrix-matrix multiplication
    pub fn spmm(&self, other: &SparseTensorCOO) -> Result<SparseTensorCOO> {
        if self.shape[1] != other.shape[0] {
            return Err(GhostError::InvalidShape(
                format!("Matrix dimensions don't match: {:?} x {:?}", self.shape, other.shape)
            ));
        }
        
        // Build hash map for efficient lookup
        let mut other_map: HashMap<(usize, usize), f32> = HashMap::new();
        for i in 0..other.nnz {
            other_map.insert((other.rows[i], other.cols[i]), other.values[i]);
        }
        
        let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();
        
        // Compute C = A * B
        for i in 0..self.nnz {
            let row = self.rows[i];
            let k = self.cols[i];
            let a_val = self.values[i];
            
            // Find all B[k, col]
            for j in 0..other.nnz {
                if other.rows[j] == k {
                    let col = other.cols[j];
                    let b_val = other.values[j];
                    *result_map.entry((row, col)).or_insert(0.0) += a_val * b_val;
                }
            }
        }
        
        // Convert result map to COO format
        let mut values = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        
        for ((row, col), val) in result_map {
            if val.abs() > 1e-10 {
                values.push(val);
                rows.push(row);
                cols.push(col);
            }
        }
        
        SparseTensorCOO::new(values, rows, cols, vec![self.shape[0], other.shape[1]])
    }
    
    /// Element-wise addition with another sparse tensor
    pub fn add(&self, other: &SparseTensorCOO) -> Result<SparseTensorCOO> {
        if self.shape != other.shape {
            return Err(GhostError::InvalidShape(
                format!("Shape mismatch: {:?} vs {:?}", self.shape, other.shape)
            ));
        }
        
        let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();
        
        // Add values from self
        for i in 0..self.nnz {
            let key = (self.rows[i], self.cols[i]);
            *result_map.entry(key).or_insert(0.0) += self.values[i];
        }
        
        // Add values from other
        for i in 0..other.nnz {
            let key = (other.rows[i], other.cols[i]);
            *result_map.entry(key).or_insert(0.0) += other.values[i];
        }
        
        // Convert to COO
        let mut values = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        
        for ((row, col), val) in result_map {
            if val.abs() > 1e-10 {
                values.push(val);
                rows.push(row);
                cols.push(col);
            }
        }
        
        SparseTensorCOO::new(values, rows, cols, self.shape.clone())
    }
    
    /// Transpose the sparse matrix
    pub fn transpose(&self) -> Result<SparseTensorCOO> {
        if self.shape.len() != 2 {
            return Err(GhostError::InvalidShape(
                "Transpose only supported for 2D tensors".to_string()
            ));
        }
        
        SparseTensorCOO::new(
            self.values.clone(),
            self.cols.clone(),
            self.rows.clone(),
            vec![self.shape[1], self.shape[0]],
        )
    }
    
    /// Get sparsity ratio (fraction of zero elements)
    pub fn sparsity(&self) -> f32 {
        let total_elements: usize = self.shape.iter().product();
        1.0 - (self.nnz as f32 / total_elements as f32)
    }
}

/// Sparse tensor in CSR (Compressed Sparse Row) format
#[derive(Debug, Clone)]
pub struct SparseTensorCSR {
    /// Non-zero values
    pub values: Vec<f32>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Row pointers
    pub row_ptrs: Vec<usize>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Number of non-zero elements
    pub nnz: usize,
}

impl SparseTensorCSR {
    /// Create a new sparse tensor in CSR format
    pub fn new(values: Vec<f32>, col_indices: Vec<usize>, row_ptrs: Vec<usize>, shape: Vec<usize>) -> Result<Self> {
        if values.len() != col_indices.len() {
            return Err(GhostError::InvalidShape(
                "Values and col_indices must have the same length".to_string()
            ));
        }
        
        let nnz = values.len();
        
        Ok(SparseTensorCSR {
            values,
            col_indices,
            row_ptrs,
            shape,
            nnz,
        })
    }
    
    /// Convert from COO format
    pub fn from_coo(coo: &SparseTensorCOO) -> Result<Self> {
        if coo.shape.len() != 2 {
            return Err(GhostError::InvalidShape(
                "CSR only supports 2D tensors".to_string()
            ));
        }
        
        let nrows = coo.shape[0];
        
        // Sort by row, then column
        let mut indices: Vec<usize> = (0..coo.nnz).collect();
        indices.sort_by_key(|&i| (coo.rows[i], coo.cols[i]));
        
        let mut values = Vec::with_capacity(coo.nnz);
        let mut col_indices = Vec::with_capacity(coo.nnz);
        let mut row_ptrs = vec![0; nrows + 1];
        
        for &i in &indices {
            values.push(coo.values[i]);
            col_indices.push(coo.cols[i]);
        }
        
        // Build row pointers
        for &i in &indices {
            row_ptrs[coo.rows[i] + 1] += 1;
        }
        
        for i in 1..=nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }
        
        SparseTensorCSR::new(values, col_indices, row_ptrs, coo.shape.clone())
    }
    
    /// Convert to COO format
    pub fn to_coo(&self) -> Result<SparseTensorCOO> {
        let mut values = Vec::with_capacity(self.nnz);
        let mut rows = Vec::with_capacity(self.nnz);
        let mut cols = Vec::with_capacity(self.nnz);
        
        for row in 0..self.shape[0] {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            
            for i in start..end {
                values.push(self.values[i]);
                rows.push(row);
                cols.push(self.col_indices[i]);
            }
        }
        
        SparseTensorCOO::new(values, rows, cols, self.shape.clone())
    }
    
    /// Sparse matrix-vector multiplication (optimized for CSR)
    pub fn spmv(&self, vec: &Tensor) -> Result<Tensor> {
        let vec_data = vec.data_f32();
        if vec_data.len() != self.shape[1] {
            return Err(GhostError::InvalidShape(
                format!("Vector length {} doesn't match matrix columns {}", vec_data.len(), self.shape[1])
            ));
        }
        
        let mut result = vec![0.0f32; self.shape[0]];
        
        for row in 0..self.shape[0] {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            
            let mut sum = 0.0;
            for i in start..end {
                let col = self.col_indices[i];
                let val = self.values[i];
                sum += val * vec_data[col];
            }
            result[row] = sum;
        }
        
        Tensor::from_slice(&result, &[self.shape[0]])
    }
}

/// Sparse tensor in CSC (Compressed Sparse Column) format
#[derive(Debug, Clone)]
pub struct SparseTensorCSC {
    /// Non-zero values
    pub values: Vec<f32>,
    /// Row indices
    pub row_indices: Vec<usize>,
    /// Column pointers
    pub col_ptrs: Vec<usize>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Number of non-zero elements
    pub nnz: usize,
}

impl SparseTensorCSC {
    /// Create a new sparse tensor in CSC format
    pub fn new(values: Vec<f32>, row_indices: Vec<usize>, col_ptrs: Vec<usize>, shape: Vec<usize>) -> Result<Self> {
        if values.len() != row_indices.len() {
            return Err(GhostError::InvalidShape(
                "Values and row_indices must have the same length".to_string()
            ));
        }
        
        let nnz = values.len();
        
        Ok(SparseTensorCSC {
            values,
            row_indices,
            col_ptrs,
            shape,
            nnz,
        })
    }
    
    /// Convert from COO format
    pub fn from_coo(coo: &SparseTensorCOO) -> Result<Self> {
        if coo.shape.len() != 2 {
            return Err(GhostError::InvalidShape(
                "CSC only supports 2D tensors".to_string()
            ));
        }
        
        let ncols = coo.shape[1];
        
        // Sort by column, then row
        let mut indices: Vec<usize> = (0..coo.nnz).collect();
        indices.sort_by_key(|&i| (coo.cols[i], coo.rows[i]));
        
        let mut values = Vec::with_capacity(coo.nnz);
        let mut row_indices = Vec::with_capacity(coo.nnz);
        let mut col_ptrs = vec![0; ncols + 1];
        
        for &i in &indices {
            values.push(coo.values[i]);
            row_indices.push(coo.rows[i]);
        }
        
        // Build column pointers
        for &i in &indices {
            col_ptrs[coo.cols[i] + 1] += 1;
        }
        
        for i in 1..=ncols {
            col_ptrs[i] += col_ptrs[i - 1];
        }
        
        SparseTensorCSC::new(values, row_indices, col_ptrs, coo.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_coo_creation() {
        let values = vec![1.0, 2.0, 3.0];
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let shape = vec![3, 3];
        
        let sparse = SparseTensorCOO::new(values, rows, cols, shape).unwrap();
        assert_eq!(sparse.nnz, 3);
        assert_eq!(sparse.sparsity(), 2.0 / 3.0);
    }
    
    #[test]
    fn test_sparse_from_dense() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let dense = Tensor::from_slice(&data, &[3, 3]).unwrap();
        
        let sparse = SparseTensorCOO::from_dense(&dense, 0.5);
        assert_eq!(sparse.nnz, 3);
    }
    
    #[test]
    fn test_sparse_to_dense() {
        let values = vec![1.0, 2.0, 3.0];
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let shape = vec![3, 3];
        
        let sparse = SparseTensorCOO::new(values, rows, cols, shape).unwrap();
        let dense = sparse.to_dense().unwrap();
        
        assert_eq!(dense.dims(), &[3, 3]);
        assert_eq!(dense.data_f32()[0], 1.0);
        assert_eq!(dense.data_f32()[4], 2.0);
        assert_eq!(dense.data_f32()[8], 3.0);
    }
    
    #[test]
    fn test_sparse_spmv() {
        let values = vec![1.0, 2.0, 3.0];
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let shape = vec![3, 3];
        
        let sparse = SparseTensorCOO::new(values, rows, cols, shape).unwrap();
        let vec = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        
        let result = sparse.spmv(&vec).unwrap();
        assert_eq!(result.data_f32(), vec![1.0, 4.0, 9.0]);
    }
    
    #[test]
    fn test_csr_conversion() {
        let values = vec![1.0, 2.0, 3.0];
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let shape = vec![3, 3];
        
        let coo = SparseTensorCOO::new(values, rows, cols, shape).unwrap();
        let csr = SparseTensorCSR::from_coo(&coo).unwrap();
        
        assert_eq!(csr.nnz, 3);
        assert_eq!(csr.row_ptrs.len(), 4);
    }
}

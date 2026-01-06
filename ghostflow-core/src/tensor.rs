//! Core Tensor type - the foundation of GhostFlow

use std::sync::Arc;
use parking_lot::RwLock;
use rand_distr::{Distribution, Normal, Uniform};

use crate::dtype::{DType, TensorElement};
use crate::shape::{Shape, Strides};
use crate::storage::Storage;
use crate::error::{GhostError, Result};

/// The core Tensor type
/// 
/// Tensors are multi-dimensional arrays with:
/// - Shared storage (enables zero-copy views)
/// - Shape and strides (enables non-contiguous layouts)
/// - Optional gradient tracking for autograd
#[derive(Debug)]
pub struct Tensor {
    /// Underlying data storage (shared for views)
    storage: Storage,
    /// Shape of the tensor
    shape: Shape,
    /// Memory strides
    strides: Strides,
    /// Offset into storage (for views)
    offset: usize,
    /// Whether to track gradients
    requires_grad: bool,
    /// Accumulated gradient
    grad: Option<Arc<RwLock<Tensor>>>,
}

impl Tensor {
    // ==================== Creation ====================

    /// Create a new tensor from a flat slice and shape
    pub fn from_slice<T: TensorElement>(data: &[T], shape: &[usize]) -> Result<Self> {
        let shape = Shape::new(shape);
        if data.len() != shape.numel() {
            return Err(GhostError::InvalidShape(format!(
                "Data length {} doesn't match shape {:?} (numel={})",
                data.len(),
                shape.dims(),
                shape.numel()
            )));
        }

        let strides = shape.default_strides();
        let storage = Storage::from_slice(data);

        Ok(Tensor {
            storage,
            shape,
            strides,
            offset: 0,
            requires_grad: false,
            grad: None,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(shape, 0.0f32)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, 1.0f32)
    }

    /// Create a tensor filled with a constant value
    pub fn full<T: TensorElement>(shape: &[usize], value: T) -> Self {
        let shape = Shape::new(shape);
        let numel = shape.numel();
        let data: Vec<T> = vec![value; numel];
        let strides = shape.default_strides();
        let storage = Storage::from_slice(&data);

        Tensor {
            storage,
            shape,
            strides,
            offset: 0,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor with random values from uniform distribution [0, 1)
    pub fn rand(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0.0f32, 1.0);
        let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
        
        Tensor::from_slice(&data, shape).unwrap()
    }

    /// Create a tensor with random values from standard normal distribution
    pub fn randn(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let numel = shape_obj.numel();
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0f32, 1.0).unwrap();
        let data: Vec<f32> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
        
        Tensor::from_slice(&data, shape).unwrap()
    }

    /// Create an identity matrix
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_slice(&data, &[n, n]).unwrap()
    }

    /// Create a 1D tensor with evenly spaced values
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let mut val = start;
        while val < end {
            data.push(val);
            val += step;
        }
        let len = data.len();
        Tensor::from_slice(&data, &[len]).unwrap()
    }

    /// Create a 1D tensor with n evenly spaced values between start and end
    pub fn linspace(start: f32, end: f32, n: usize) -> Self {
        if n == 0 {
            return Tensor::from_slice::<f32>(&[], &[0]).unwrap();
        }
        if n == 1 {
            return Tensor::from_slice(&[start], &[1]).unwrap();
        }
        
        let step = (end - start) / (n - 1) as f32;
        let data: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        Tensor::from_slice(&data, &[n]).unwrap()
    }

    // ==================== Properties ====================

    /// Get the shape of the tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get dimensions as slice
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Get strides
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    /// Check if gradient tracking is enabled
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    // ==================== Gradient ====================

    /// Enable gradient tracking
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Get gradient if available
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.as_ref().map(|g| g.read().clone())
    }

    /// Get reference to underlying storage
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Set gradient
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(Arc::new(RwLock::new(grad)));
    }

    /// Zero out gradient
    pub fn zero_grad(&mut self) {
        if let Some(ref grad) = self.grad {
            let mut g = grad.write();
            let zeros = Tensor::zeros(g.dims());
            *g = zeros;
        }
    }

    // ==================== Data Access ====================

    /// Get data as f32 slice (for f32 tensors)
    pub fn data_f32(&self) -> Vec<f32> {
        let guard = self.storage.as_slice::<f32>();
        if self.is_contiguous() && self.offset == 0 {
            guard.to_vec()
        } else {
            // Handle non-contiguous case
            self.to_contiguous_data::<f32>()
        }
    }

    /// Convert to contiguous data (handles views and non-contiguous layouts)
    fn to_contiguous_data<T: TensorElement>(&self) -> Vec<T> {
        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);
        let guard = self.storage.as_slice::<T>();
        
        // Iterate through all indices
        self.for_each_index(|indices| {
            let offset = self.compute_offset(indices);
            result.push(guard[offset]);
        });
        
        result
    }

    /// Compute linear offset from indices
    fn compute_offset(&self, indices: &[usize]) -> usize {
        self.offset + self.strides.offset(indices)
    }

    /// Iterate through all valid indices
    fn for_each_index<F: FnMut(&[usize])>(&self, mut f: F) {
        let dims = self.dims();
        if dims.is_empty() {
            f(&[]);
            return;
        }

        let mut indices = vec![0usize; dims.len()];
        loop {
            f(&indices);
            
            // Increment indices
            let mut i = dims.len() - 1;
            loop {
                indices[i] += 1;
                if indices[i] < dims[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return;
                }
                i -= 1;
            }
        }
    }

    // ==================== Shape Operations ====================

    /// Reshape tensor to new shape (must have same numel)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let new_shape = Shape::new(new_shape);
        if new_shape.numel() != self.numel() {
            return Err(GhostError::InvalidShape(format!(
                "Cannot reshape tensor of {} elements to shape {:?}",
                self.numel(),
                new_shape.dims()
            )));
        }

        // If contiguous, can just change shape/strides
        if self.is_contiguous() {
            let new_strides = new_shape.default_strides();
            return Ok(Tensor {
                storage: self.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.offset,
                requires_grad: self.requires_grad,
                grad: None,
            });
        }

        // Non-contiguous: need to copy data
        let data = self.to_contiguous_data::<f32>();
        Tensor::from_slice(&data, new_shape.dims())
    }

    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Result<Tensor> {
        self.reshape(&[self.numel()])
    }

    /// Transpose dimensions
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(GhostError::DimOutOfBounds {
                dim: dim0.max(dim1),
                ndim: self.ndim(),
            });
        }

        let mut new_shape = self.shape.dims().to_vec();
        let mut new_strides = self.strides.as_slice().to_vec();
        
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Tensor {
            storage: self.storage.clone(),
            shape: Shape::from(new_shape),
            strides: Strides::from(new_strides.as_slice()),
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad: None,
        })
    }

    /// Transpose for 2D tensors (matrix transpose)
    pub fn t(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(GhostError::InvalidOperation(
                "t() only works on 2D tensors".to_string()
            ));
        }
        self.transpose(0, 1)
    }

    /// Squeeze: remove dimensions of size 1
    pub fn squeeze(&self) -> Tensor {
        let new_dims: Vec<usize> = self.dims().iter()
            .filter(|&&d| d != 1)
            .copied()
            .collect();
        
        if new_dims.is_empty() {
            // Scalar case
            let data = self.data_f32();
            Tensor::from_slice(&data, &[]).unwrap()
        } else {
            self.reshape(&new_dims).unwrap()
        }
    }

    /// Unsqueeze: add dimension of size 1 at position
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        if dim > self.ndim() {
            return Err(GhostError::DimOutOfBounds {
                dim,
                ndim: self.ndim() + 1,
            });
        }

        let mut new_dims = self.dims().to_vec();
        new_dims.insert(dim, 1);
        self.reshape(&new_dims)
    }

    // ==================== Clone ====================

    /// Deep clone (copies data)
    pub fn deep_clone(&self) -> Self {
        let data = self.data_f32();
        Tensor::from_slice(&data, self.dims()).unwrap()
    }
}

impl Clone for Tensor {
    /// Shallow clone (shares storage)
    fn clone(&self) -> Self {
        Tensor {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
        }
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={}, dtype={})", self.shape, self.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.dims(), &[2, 2]);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_zeros_ones() {
        let zeros = Tensor::zeros(&[3, 3]);
        let ones = Tensor::ones(&[3, 3]);
        
        assert!(zeros.data_f32().iter().all(|&x| x == 0.0));
        assert!(ones.data_f32().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::arange(0.0, 12.0, 1.0);
        let reshaped = t.reshape(&[3, 4]).unwrap();
        assert_eq!(reshaped.dims(), &[3, 4]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let transposed = t.t().unwrap();
        assert_eq!(transposed.dims(), &[3, 2]);
    }
}

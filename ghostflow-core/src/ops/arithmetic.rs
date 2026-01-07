//! Element-wise arithmetic operations

use crate::tensor::Tensor;
use crate::error::Result;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Macro to conditionally use parallel or sequential iteration
macro_rules! map_elements {
    ($data:expr, $op:expr) => {{
        #[cfg(feature = "rayon")]
        { $data.par_iter().map($op).collect() }
        #[cfg(not(feature = "rayon"))]
        { $data.iter().map($op).collect() }
    }};
}

impl Tensor {
    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data_f32();
        let b = other.data_f32();
        
        // Handle broadcasting
        let (result, shape) = broadcast_binary_op(&a, self.dims(), &b, other.dims(), |x, y| x + y)?;
        Tensor::from_slice(&result, &shape)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data_f32();
        let b = other.data_f32();
        
        let (result, shape) = broadcast_binary_op(&a, self.dims(), &b, other.dims(), |x, y| x - y)?;
        Tensor::from_slice(&result, &shape)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data_f32();
        let b = other.data_f32();
        
        let (result, shape) = broadcast_binary_op(&a, self.dims(), &b, other.dims(), |x, y| x * y)?;
        Tensor::from_slice(&result, &shape)
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data_f32();
        let b = other.data_f32();
        
        let (result, shape) = broadcast_binary_op(&a, self.dims(), &b, other.dims(), |x, y| x / y)?;
        Tensor::from_slice(&result, &shape)
    }

    /// Add scalar
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x + scalar);
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Subtract scalar
    pub fn sub_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x - scalar);
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Multiply by scalar
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x * scalar);
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Divide by scalar
    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x / scalar);
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Negation
    pub fn neg(&self) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| -x);
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Absolute value
    pub fn abs(&self) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x.abs());
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Power
    pub fn pow(&self, exp: f32) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x.powf(exp));
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Square root
    pub fn sqrt(&self) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x.sqrt());
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Exponential
    pub fn exp(&self) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x.exp());
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Natural logarithm
    pub fn log(&self) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x.ln());
        Tensor::from_slice(&data, self.dims()).unwrap()
    }

    /// Clamp values to range
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        let data: Vec<f32> = map_elements!(self.data_f32(), |&x| x.clamp(min, max));
        Tensor::from_slice(&data, self.dims()).unwrap()
    }
}

/// Broadcast and apply binary operation
fn broadcast_binary_op<F>(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    op: F,
) -> Result<(Vec<f32>, Vec<usize>)>
where
    F: Fn(f32, f32) -> f32 + Sync,
{
    use crate::shape::Shape;
    
    let shape_a = Shape::new(a_shape);
    let shape_b = Shape::new(b_shape);
    let result_shape = shape_a.broadcast_with(&shape_b)?;
    let result_dims = result_shape.dims().to_vec();
    let numel = result_shape.numel();

    // Fast path: same shape
    if a_shape == b_shape {
        #[cfg(feature = "rayon")]
        let result: Vec<f32> = a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let result: Vec<f32> = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        return Ok((result, result_dims));
    }

    // Broadcast path
    let a_strides = compute_broadcast_strides(a_shape, &result_dims);
    let b_strides = compute_broadcast_strides(b_shape, &result_dims);

    #[cfg(feature = "rayon")]
    let result: Vec<f32> = (0..numel)
        .into_par_iter()
        .map(|i| {
            let a_idx = compute_broadcast_index(i, &result_dims, &a_strides);
            let b_idx = compute_broadcast_index(i, &result_dims, &b_strides);
            op(a[a_idx], b[b_idx])
        })
        .collect();
    #[cfg(not(feature = "rayon"))]
    let result: Vec<f32> = (0..numel)
        .map(|i| {
            let a_idx = compute_broadcast_index(i, &result_dims, &a_strides);
            let b_idx = compute_broadcast_index(i, &result_dims, &b_strides);
            op(a[a_idx], b[b_idx])
        })
        .collect();

    Ok((result, result_dims))
}

/// Compute strides for broadcasting
fn compute_broadcast_strides(shape: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let ndim = target_shape.len();
    let offset = ndim - shape.len();
    
    let mut strides = vec![0usize; ndim];
    let mut stride = 1usize;
    
    for i in (0..shape.len()).rev() {
        if shape[i] == target_shape[i + offset] {
            strides[i + offset] = stride;
            stride *= shape[i];
        } else {
            // Broadcast dimension (size 1)
            strides[i + offset] = 0;
        }
    }
    
    strides
}

/// Compute source index for broadcast
fn compute_broadcast_index(flat_idx: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut idx = 0;
    let mut remaining = flat_idx;
    
    for i in (0..shape.len()).rev() {
        let coord = remaining % shape[i];
        remaining /= shape[i];
        idx += coord * strides[i];
    }
    
    idx
}

// Operator overloads
impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        self.add(other).unwrap()
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        self.sub(other).unwrap()
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        self.mul(other).unwrap()
    }
}

impl std::ops::Div for &Tensor {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Tensor {
        self.div(other).unwrap()
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.data_f32(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_add() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0], &[1, 2]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.dims(), &[3, 2]);
    }

    #[test]
    fn test_scalar_ops() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = a.mul_scalar(2.0);
        assert_eq!(b.data_f32(), vec![2.0, 4.0, 6.0]);
    }
}

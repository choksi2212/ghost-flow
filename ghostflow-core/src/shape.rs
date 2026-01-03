//! Shape and stride handling for tensors

use crate::error::{GhostError, Result};
use smallvec::SmallVec;

/// Maximum dimensions for stack allocation (most tensors are <= 6D)
const MAX_INLINE_DIMS: usize = 6;

/// Shape of a tensor - dimensions along each axis
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(SmallVec<[usize; MAX_INLINE_DIMS]>);

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: &[usize]) -> Self {
        Shape(SmallVec::from_slice(dims))
    }

    /// Create a scalar shape (0 dimensions)
    pub fn scalar() -> Self {
        Shape(SmallVec::new())
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Get dimension at index
    pub fn dim(&self, idx: usize) -> Option<usize> {
        self.0.get(idx).copied()
    }

    /// Get dimensions as slice
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Check if this is a scalar (0D tensor)
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if shapes are broadcastable
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result = SmallVec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let a = if i < self.ndim() {
                self.0[self.ndim() - 1 - i]
            } else {
                1
            };
            let b = if i < other.ndim() {
                other.0[other.ndim() - 1 - i]
            } else {
                1
            };

            if a == b {
                result.push(a);
            } else if a == 1 {
                result.push(b);
            } else if b == 1 {
                result.push(a);
            } else {
                return Err(GhostError::BroadcastError {
                    a: self.0.to_vec(),
                    b: other.0.to_vec(),
                });
            }
        }

        result.reverse();
        Ok(Shape(result))
    }

    /// Compute default (contiguous) strides for this shape
    pub fn default_strides(&self) -> Strides {
        if self.is_scalar() {
            return Strides::new(&[]);
        }

        let mut strides = SmallVec::with_capacity(self.ndim());
        let mut stride = 1usize;

        for &dim in self.0.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }

        strides.reverse();
        Strides(strides)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape(SmallVec::from_vec(dims))
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// Strides for memory layout - bytes to skip for each dimension
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(SmallVec<[usize; MAX_INLINE_DIMS]>);

impl Strides {
    /// Create new strides
    pub fn new(strides: &[usize]) -> Self {
        Strides(SmallVec::from_slice(strides))
    }

    /// Get stride at index
    pub fn stride(&self, idx: usize) -> Option<usize> {
        self.0.get(idx).copied()
    }

    /// Get strides as slice
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Check if strides represent contiguous memory
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        if shape.is_scalar() {
            return true;
        }

        let expected = shape.default_strides();
        self.0 == expected.0
    }

    /// Compute linear offset from multi-dimensional indices
    pub fn offset(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.0.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }
}

impl From<&[usize]> for Strides {
    fn from(strides: &[usize]) -> Self {
        Strides::new(strides)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_numel() {
        assert_eq!(Shape::new(&[2, 3, 4]).numel(), 24);
        assert_eq!(Shape::new(&[1]).numel(), 1);
        assert_eq!(Shape::scalar().numel(), 1);
    }

    #[test]
    fn test_broadcast() {
        let a = Shape::new(&[3, 1]);
        let b = Shape::new(&[1, 4]);
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[3, 4]);
    }

    #[test]
    fn test_strides() {
        let shape = Shape::new(&[2, 3, 4]);
        let strides = shape.default_strides();
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }
}

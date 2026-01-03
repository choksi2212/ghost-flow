//! Python bindings for GhostFlow ML framework
//! 
//! This module provides zero-cost Python bindings to the Rust GhostFlow library,
//! enabling Python users to leverage the full performance of Rust while maintaining
//! a Pythonic API similar to PyTorch and TensorFlow.

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::PyValueError;
use pyo3::wrap_pymodule;
use ghostflow_core::Tensor as RustTensor;
use ghostflow_nn::Module;

/// Python wrapper for GhostFlow Tensor
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    inner: RustTensor,
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from a Python list
    #[new]
    fn new(data: &PyAny, shape: Vec<usize>) -> PyResult<Self> {
        // Convert Python data to Rust Vec<f32>
        let flat_data: Vec<f32> = if let Ok(list) = data.downcast::<PyList>() {
            Self::flatten_list(list)?
        } else {
            return Err(PyValueError::new_err("Data must be a list"));
        };
        
        let tensor = RustTensor::from_slice(&flat_data, &shape)
            .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?;
        
        Ok(PyTensor { inner: tensor })
    }
    
    /// Create a tensor filled with zeros
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = RustTensor::zeros(&shape);
        Ok(PyTensor { inner: tensor })
    }
    
    /// Create a tensor filled with ones
    #[staticmethod]
    fn ones(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = RustTensor::ones(&shape);
        Ok(PyTensor { inner: tensor })
    }
    
    /// Create a tensor with random values from normal distribution
    #[staticmethod]
    fn randn(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = RustTensor::randn(&shape);
        Ok(PyTensor { inner: tensor })
    }
    
    /// Create a tensor with random values from uniform distribution
    #[staticmethod]
    fn rand(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = RustTensor::rand(&shape);
        Ok(PyTensor { inner: tensor })
    }
    
    /// Get tensor shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }
    
    /// Get number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.dims().len()
    }
    
    /// Get total number of elements
    #[getter]
    fn size(&self) -> usize {
        self.inner.numel()
    }
    
    /// Matrix multiplication
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.matmul(&other.inner)
            .map_err(|e| PyValueError::new_err(format!("Matmul failed: {}", e)))?;
        Ok(PyTensor { inner: result })
    }
    
    /// Element-wise addition
    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.add(&other.inner)
            .map_err(|e| PyValueError::new_err(format!("Addition failed: {}", e)))?;
        Ok(PyTensor { inner: result })
    }
    
    /// Element-wise subtraction
    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.sub(&other.inner)
            .map_err(|e| PyValueError::new_err(format!("Subtraction failed: {}", e)))?;
        Ok(PyTensor { inner: result })
    }
    
    /// Element-wise multiplication
    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.mul(&other.inner)
            .map_err(|e| PyValueError::new_err(format!("Multiplication failed: {}", e)))?;
        Ok(PyTensor { inner: result })
    }
    
    /// Matrix multiplication operator (@)
    fn __matmul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }
    
    /// Transpose
    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
        let result = self.inner.transpose(dim0, dim1)
            .map_err(|e| PyValueError::new_err(format!("Transpose failed: {}", e)))?;
        Ok(PyTensor { inner: result })
    }
    
    /// Reshape tensor
    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        let result = self.inner.reshape(&shape)
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))?;
        Ok(PyTensor { inner: result })
    }
    
    /// ReLU activation
    fn relu(&self) -> PyTensor {
        PyTensor { inner: self.inner.relu() }
    }
    
    /// Sigmoid activation
    fn sigmoid(&self) -> PyTensor {
        PyTensor { inner: self.inner.sigmoid() }
    }
    
    /// GELU activation
    fn gelu(&self) -> PyTensor {
        PyTensor { inner: self.inner.gelu() }
    }
    
    /// Softmax
    fn softmax(&self, dim: i32) -> PyTensor {
        PyTensor { inner: self.inner.softmax(dim) }
    }
    
    /// Sum all elements
    fn sum(&self) -> PyTensor {
        PyTensor { inner: self.inner.sum() }
    }
    
    /// Mean of all elements
    fn mean(&self) -> PyTensor {
        PyTensor { inner: self.inner.mean() }
    }
    
    /// Convert to Python list
    fn tolist(&self) -> Vec<f32> {
        self.inner.data_f32()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, dtype=f32)", self.inner.dims())
    }
}

impl PyTensor {
    fn flatten_list(list: &PyList) -> PyResult<Vec<f32>> {
        let mut result = Vec::new();
        for item in list.iter() {
            if let Ok(val) = item.extract::<f32>() {
                result.push(val);
            } else if let Ok(sublist) = item.downcast::<PyList>() {
                result.extend(Self::flatten_list(sublist)?);
            } else {
                return Err(PyValueError::new_err("Invalid data type in list"));
            }
        }
        Ok(result)
    }
}

/// Neural network module
#[pymodule]
fn nn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLinear>()?;
    m.add_class::<PyReLU>()?;
    m.add_class::<PySigmoid>()?;
    Ok(())
}

/// Linear layer
#[pyclass(name = "Linear")]
struct PyLinear {
    inner: ghostflow_nn::Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize) -> Self {
        PyLinear {
            inner: ghostflow_nn::Linear::new(in_features, out_features),
        }
    }
    
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner);
        Ok(PyTensor { inner: output })
    }
    
    fn __call__(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }
}

/// ReLU activation layer
#[pyclass(name = "ReLU")]
struct PyReLU {
    inner: ghostflow_nn::ReLU,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> Self {
        PyReLU {
            inner: ghostflow_nn::ReLU::new(),
        }
    }
    
    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor { inner: self.inner.forward(&input.inner) }
    }
    
    fn __call__(&self, input: &PyTensor) -> PyTensor {
        self.forward(input)
    }
}

/// Sigmoid activation layer
#[pyclass(name = "Sigmoid")]
struct PySigmoid {
    inner: ghostflow_nn::Sigmoid,
}

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> Self {
        PySigmoid {
            inner: ghostflow_nn::Sigmoid::new(),
        }
    }
    
    fn forward(&self, input: &PyTensor) -> PyTensor {
        PyTensor { inner: self.inner.forward(&input.inner) }
    }
    
    fn __call__(&self, input: &PyTensor) -> PyTensor {
        self.forward(input)
    }
}

/// Main Python module
#[pymodule]
fn _ghost_flow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_wrapped(wrap_pymodule!(nn))?;
    
    // Add version
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}

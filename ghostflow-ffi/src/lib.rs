//! C FFI bindings for GhostFlow
//!
//! This module provides C-compatible APIs for using GhostFlow from other languages
//! like Python, C++, Java, Go, etc.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;
use std::slice;
use ghostflow_core::Tensor;

/// Opaque handle to a GhostFlow tensor
#[repr(C)]
pub struct GhostFlowTensor {
    _private: [u8; 0],
}

/// Opaque handle to a GhostFlow model
#[repr(C)]
pub struct GhostFlowModel {
    _private: [u8; 0],
}

/// Error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GhostFlowError {
    Success = 0,
    InvalidShape = 1,
    InvalidData = 2,
    NullPointer = 3,
    AllocationFailed = 4,
    ComputationFailed = 5,
    Unknown = 99,
}

/// Convert Rust Tensor to opaque pointer
fn tensor_to_ptr(tensor: Tensor) -> *mut GhostFlowTensor {
    Box::into_raw(Box::new(tensor)) as *mut GhostFlowTensor
}

/// Convert opaque pointer back to Rust Tensor reference
unsafe fn ptr_to_tensor<'a>(ptr: *const GhostFlowTensor) -> Option<&'a Tensor> {
    if ptr.is_null() {
        None
    } else {
        Some(&*(ptr as *const Tensor))
    }
}

/// Convert opaque pointer back to mutable Rust Tensor reference
unsafe fn ptr_to_tensor_mut<'a>(ptr: *mut GhostFlowTensor) -> Option<&'a mut Tensor> {
    if ptr.is_null() {
        None
    } else {
        Some(&mut *(ptr as *mut Tensor))
    }
}

/// Initialize GhostFlow library
#[no_mangle]
pub extern "C" fn ghostflow_init() -> GhostFlowError {
    GhostFlowError::Success
}

/// Get GhostFlow version string
#[no_mangle]
pub extern "C" fn ghostflow_version() -> *const c_char {
    let version = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
    version.into_raw()
}

/// Free a version string
#[no_mangle]
pub unsafe extern "C" fn ghostflow_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Create a new tensor from data
///
/// # Arguments
/// * `data` - Pointer to float array
/// * `data_len` - Length of data array
/// * `shape` - Pointer to shape array
/// * `shape_len` - Length of shape array
/// * `out` - Output pointer to store the created tensor
///
/// # Returns
/// Error code
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_create(
    data: *const c_float,
    data_len: usize,
    shape: *const usize,
    shape_len: usize,
    out: *mut *mut GhostFlowTensor,
) -> GhostFlowError {
    if data.is_null() || shape.is_null() || out.is_null() {
        return GhostFlowError::NullPointer;
    }

    let data_slice = slice::from_raw_parts(data, data_len);
    let shape_slice = slice::from_raw_parts(shape, shape_len);

    match Tensor::from_slice(data_slice, shape_slice) {
        Ok(tensor) => {
            *out = tensor_to_ptr(tensor);
            GhostFlowError::Success
        }
        Err(_) => GhostFlowError::InvalidShape,
    }
}

/// Create a tensor filled with zeros
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_zeros(
    shape: *const usize,
    shape_len: usize,
    out: *mut *mut GhostFlowTensor,
) -> GhostFlowError {
    if shape.is_null() || out.is_null() {
        return GhostFlowError::NullPointer;
    }

    let shape_slice = slice::from_raw_parts(shape, shape_len);
    let numel: usize = shape_slice.iter().product();
    let data = vec![0.0f32; numel];

    match Tensor::from_slice(&data, shape_slice) {
        Ok(tensor) => {
            *out = tensor_to_ptr(tensor);
            GhostFlowError::Success
        }
        Err(_) => GhostFlowError::InvalidShape,
    }
}

/// Create a tensor filled with ones
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_ones(
    shape: *const usize,
    shape_len: usize,
    out: *mut *mut GhostFlowTensor,
) -> GhostFlowError {
    if shape.is_null() || out.is_null() {
        return GhostFlowError::NullPointer;
    }

    let shape_slice = slice::from_raw_parts(shape, shape_len);
    let numel: usize = shape_slice.iter().product();
    let data = vec![1.0f32; numel];

    match Tensor::from_slice(&data, shape_slice) {
        Ok(tensor) => {
            *out = tensor_to_ptr(tensor);
            GhostFlowError::Success
        }
        Err(_) => GhostFlowError::InvalidShape,
    }
}

/// Free a tensor
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_free(tensor: *mut GhostFlowTensor) {
    if !tensor.is_null() {
        let _ = Box::from_raw(tensor as *mut Tensor);
    }
}

/// Get tensor shape
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_shape(
    tensor: *const GhostFlowTensor,
    out_shape: *mut usize,
    out_ndim: *mut usize,
) -> GhostFlowError {
    if tensor.is_null() || out_shape.is_null() || out_ndim.is_null() {
        return GhostFlowError::NullPointer;
    }

    if let Some(t) = ptr_to_tensor(tensor) {
        let shape = t.dims();
        *out_ndim = shape.len();
        ptr::copy_nonoverlapping(shape.as_ptr(), out_shape, shape.len());
        GhostFlowError::Success
    } else {
        GhostFlowError::NullPointer
    }
}

/// Get tensor data
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_data(
    tensor: *const GhostFlowTensor,
    out_data: *mut c_float,
    out_len: *mut usize,
) -> GhostFlowError {
    if tensor.is_null() || out_data.is_null() || out_len.is_null() {
        return GhostFlowError::NullPointer;
    }

    if let Some(t) = ptr_to_tensor(tensor) {
        let data = t.data_f32();
        *out_len = data.len();
        ptr::copy_nonoverlapping(data.as_ptr(), out_data, data.len());
        GhostFlowError::Success
    } else {
        GhostFlowError::NullPointer
    }
}

/// Add two tensors
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_add(
    a: *const GhostFlowTensor,
    b: *const GhostFlowTensor,
    out: *mut *mut GhostFlowTensor,
) -> GhostFlowError {
    if a.is_null() || b.is_null() || out.is_null() {
        return GhostFlowError::NullPointer;
    }

    let tensor_a = match ptr_to_tensor(a) {
        Some(t) => t,
        None => return GhostFlowError::NullPointer,
    };

    let tensor_b = match ptr_to_tensor(b) {
        Some(t) => t,
        None => return GhostFlowError::NullPointer,
    };

    match tensor_a.add(&tensor_b) {
        Ok(result) => {
            unsafe { *out = tensor_to_ptr(result); }
            GhostFlowError::Success
        }
        Err(_) => GhostFlowError::ComputationFailed,
    }
}

/// Multiply two tensors element-wise
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_mul(
    a: *const GhostFlowTensor,
    b: *const GhostFlowTensor,
    out: *mut *mut GhostFlowTensor,
) -> GhostFlowError {
    if a.is_null() || b.is_null() || out.is_null() {
        return GhostFlowError::NullPointer;
    }

    let tensor_a = match ptr_to_tensor(a) {
        Some(t) => t,
        None => return GhostFlowError::NullPointer,
    };

    let tensor_b = match ptr_to_tensor(b) {
        Some(t) => t,
        None => return GhostFlowError::NullPointer,
    };

    match tensor_a.mul(&tensor_b) {
        Ok(result) => {
            unsafe { *out = tensor_to_ptr(result); }
            GhostFlowError::Success
        }
        Err(_) => GhostFlowError::ComputationFailed,
    }
}

/// Matrix multiplication
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_matmul(
    a: *const GhostFlowTensor,
    b: *const GhostFlowTensor,
    out: *mut *mut GhostFlowTensor,
) -> GhostFlowError {
    if a.is_null() || b.is_null() || out.is_null() {
        return GhostFlowError::NullPointer;
    }

    let tensor_a = match ptr_to_tensor(a) {
        Some(t) => t,
        None => return GhostFlowError::NullPointer,
    };

    let tensor_b = match ptr_to_tensor(b) {
        Some(t) => t,
        None => return GhostFlowError::NullPointer,
    };

    match tensor_a.matmul(tensor_b) {
        Ok(result) => {
            *out = tensor_to_ptr(result);
            GhostFlowError::Success
        }
        Err(_) => GhostFlowError::ComputationFailed,
    }
}

/// Reshape a tensor
#[no_mangle]
pub unsafe extern "C" fn ghostflow_tensor_reshape(
    tensor: *const GhostFlowTensor,
    new_shape: *const usize,
    new_shape_len: usize,
    out: *mut *mut GhostFlowTensor,
) -> GhostFlowError {
    if tensor.is_null() || new_shape.is_null() || out.is_null() {
        return GhostFlowError::NullPointer;
    }

    let t = match ptr_to_tensor(tensor) {
        Some(t) => t,
        None => return GhostFlowError::NullPointer,
    };

    let shape_slice = slice::from_raw_parts(new_shape, new_shape_len);

    match t.reshape(shape_slice) {
        Ok(result) => {
            *out = tensor_to_ptr(result);
            GhostFlowError::Success
        }
        Err(_) => GhostFlowError::InvalidShape,
    }
}

/// Get last error message
#[no_mangle]
pub extern "C" fn ghostflow_get_last_error() -> *const c_char {
    let msg = CString::new("No error").unwrap();
    msg.into_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_create() {
        unsafe {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];
            let mut tensor: *mut GhostFlowTensor = ptr::null_mut();

            let result = ghostflow_tensor_create(
                data.as_ptr(),
                data.len(),
                shape.as_ptr(),
                shape.len(),
                &mut tensor,
            );

            assert_eq!(result, GhostFlowError::Success);
            assert!(!tensor.is_null());

            ghostflow_tensor_free(tensor);
        }
    }

    #[test]
    fn test_tensor_zeros() {
        unsafe {
            let shape = vec![2, 3];
            let mut tensor: *mut GhostFlowTensor = ptr::null_mut();

            let result = ghostflow_tensor_zeros(
                shape.as_ptr(),
                shape.len(),
                &mut tensor,
            );

            assert_eq!(result, GhostFlowError::Success);
            assert!(!tensor.is_null());

            ghostflow_tensor_free(tensor);
        }
    }
}

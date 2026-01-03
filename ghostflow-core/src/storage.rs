//! Storage backend for tensor data

use std::sync::Arc;
use parking_lot::RwLock;
use crate::dtype::{DType, TensorElement};

/// Raw storage for tensor data
/// Separate from Tensor to enable zero-copy views
#[derive(Debug)]
pub struct Storage {
    /// Raw bytes
    data: Arc<RwLock<Vec<u8>>>,
    /// Data type
    dtype: DType,
    /// Number of elements
    len: usize,
}

impl Storage {
    /// Create new storage with given capacity
    pub fn new(dtype: DType, len: usize) -> Self {
        let byte_len = len * dtype.size_bytes();
        let data = vec![0u8; byte_len];
        Storage {
            data: Arc::new(RwLock::new(data)),
            dtype,
            len,
        }
    }

    /// Create storage from typed data
    pub fn from_slice<T: TensorElement>(data: &[T]) -> Self {
        let byte_len = data.len() * std::mem::size_of::<T>();
        let mut bytes = vec![0u8; byte_len];
        
        // Safe copy from typed slice to bytes
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                byte_len,
            );
        }
        
        Storage {
            data: Arc::new(RwLock::new(bytes)),
            dtype: T::DTYPE,
            len: data.len(),
        }
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.len * self.dtype.size_bytes()
    }

    /// Get read access to data as typed slice
    pub fn as_slice<T: TensorElement>(&self) -> StorageReadGuard<'_, T> {
        debug_assert_eq!(T::DTYPE, self.dtype);
        StorageReadGuard {
            guard: self.data.read(),
            len: self.len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get write access to data as typed slice
    pub fn as_slice_mut<T: TensorElement>(&self) -> StorageWriteGuard<'_, T> {
        debug_assert_eq!(T::DTYPE, self.dtype);
        StorageWriteGuard {
            guard: self.data.write(),
            len: self.len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Clone the storage (deep copy)
    pub fn deep_clone(&self) -> Self {
        let data = self.data.read().clone();
        Storage {
            data: Arc::new(RwLock::new(data)),
            dtype: self.dtype,
            len: self.len,
        }
    }

    /// Check if this storage is shared (has multiple references)
    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.data) > 1
    }
}

impl Clone for Storage {
    /// Shallow clone - shares underlying data
    fn clone(&self) -> Self {
        Storage {
            data: Arc::clone(&self.data),
            dtype: self.dtype,
            len: self.len,
        }
    }
}

/// Read guard for typed access to storage
pub struct StorageReadGuard<'a, T> {
    guard: parking_lot::RwLockReadGuard<'a, Vec<u8>>,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: TensorElement> StorageReadGuard<'a, T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.guard.as_ptr() as *const T, self.len)
        }
    }
}

impl<'a, T: TensorElement> std::ops::Deref for StorageReadGuard<'a, T> {
    type Target = [T];
    
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Write guard for typed access to storage
pub struct StorageWriteGuard<'a, T> {
    guard: parking_lot::RwLockWriteGuard<'a, Vec<u8>>,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: TensorElement> StorageWriteGuard<'a, T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.guard.as_ptr() as *const T, self.len)
        }
    }
    
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.guard.as_mut_ptr() as *mut T, self.len)
        }
    }
}

impl<'a, T: TensorElement> std::ops::Deref for StorageWriteGuard<'a, T> {
    type Target = [T];
    
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T: TensorElement> std::ops::DerefMut for StorageWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        let storage = Storage::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.dtype(), DType::F32);
    }

    #[test]
    fn test_storage_read() {
        let storage = Storage::from_slice(&[1.0f32, 2.0, 3.0]);
        let data = storage.as_slice::<f32>();
        assert_eq!(&*data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_storage_write() {
        let storage = Storage::from_slice(&[1.0f32, 2.0, 3.0]);
        {
            let mut data = storage.as_slice_mut::<f32>();
            data[0] = 10.0;
        }
        let data = storage.as_slice::<f32>();
        assert_eq!(data[0], 10.0);
    }
}

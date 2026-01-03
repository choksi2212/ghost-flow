//! CUDA FFI bindings - Real CUDA Runtime API
//!
//! These are the actual CUDA C API bindings. When compiled with the `cuda` feature,
//! these will link against the real CUDA runtime library.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_int, c_void, c_char};

/// CUDA error codes
pub type cudaError_t = c_int;

/// CUDA stream handle
pub type cudaStream_t = *mut c_void;

/// CUDA event handle  
pub type cudaEvent_t = *mut c_void;

/// cuBLAS handle
pub type cublasHandle_t = *mut c_void;

/// cuBLAS status
pub type cublasStatus_t = c_int;

/// cuBLAS operation type
pub type cublasOperation_t = c_int;

pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

/// CUDA memory copy kind
#[repr(C)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

/// CUDA device properties
#[repr(C)]
#[derive(Debug, Clone)]
pub struct cudaDeviceProp {
    pub name: [c_char; 256],
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture2DMipmap: [c_int; 2],
    pub maxTexture2DLinear: [c_int; 3],
    pub maxTexture2DGather: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub maxTexture3DAlt: [c_int; 3],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2],
    pub maxTexture2DLayered: [c_int; 3],
    pub maxTextureCubemapLayered: [c_int; 2],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2],
    pub maxSurface3D: [c_int; 3],
    pub maxSurface1DLayered: [c_int; 2],
    pub maxSurface2DLayered: [c_int; 3],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2],
    pub surfaceAlignment: usize,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: usize,
}

impl Default for cudaDeviceProp {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// CUDA Runtime API
#[cfg(feature = "cuda")]
#[link(name = "cudart")]
extern "C" {
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaDeviceReset() -> cudaError_t;
    
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t;
    pub fn cudaMemsetAsync(devPtr: *mut c_void, value: c_int, count: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    
    pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;
    
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
    
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaPeekAtLastError() -> cudaError_t;
}

// cuBLAS API
#[cfg(feature = "cuda")]
#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
    
    // SGEMM: C = alpha * op(A) * op(B) + beta * C
    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        B: *const f32,
        ldb: c_int,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
    ) -> cublasStatus_t;
    
    // Batched SGEMM
    pub fn cublasSgemmBatched(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        Aarray: *const *const f32,
        lda: c_int,
        Barray: *const *const f32,
        ldb: c_int,
        beta: *const f32,
        Carray: *mut *mut f32,
        ldc: c_int,
        batchCount: c_int,
    ) -> cublasStatus_t;
    
    // SAXPY: y = alpha * x + y
    pub fn cublasSaxpy_v2(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;
    
    // SDOT: result = x . y
    pub fn cublasSdot_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *const f32,
        incy: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;
    
    // SNRM2: result = ||x||_2
    pub fn cublasSnrm2_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;
    
    // SSCAL: x = alpha * x
    pub fn cublasSscal_v2(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *mut f32,
        incx: c_int,
    ) -> cublasStatus_t;
}

// Stub implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub mod stubs {
    use super::*;
    
    pub unsafe fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t {
        *count = 0;
        0 // cudaSuccess
    }
    
    pub unsafe fn cudaSetDevice(_device: c_int) -> cudaError_t {
        1 // cudaErrorInvalidDevice
    }
    
    pub unsafe fn cudaGetDevice(device: *mut c_int) -> cudaError_t {
        *device = -1;
        1
    }
    
    pub unsafe fn cudaDeviceSynchronize() -> cudaError_t {
        0
    }
    
    pub unsafe fn cudaMalloc(_devPtr: *mut *mut c_void, _size: usize) -> cudaError_t {
        2 // cudaErrorMemoryAllocation
    }
    
    pub unsafe fn cudaFree(_devPtr: *mut c_void) -> cudaError_t {
        0
    }
    
    pub unsafe fn cudaMemcpy(_dst: *mut c_void, _src: *const c_void, _count: usize, _kind: cudaMemcpyKind) -> cudaError_t {
        0
    }
    
    pub unsafe fn cudaMemset(_devPtr: *mut c_void, _value: c_int, _count: usize) -> cudaError_t {
        0
    }
    
    pub unsafe fn cudaStreamCreate(_pStream: *mut cudaStream_t) -> cudaError_t {
        0
    }
    
    pub unsafe fn cudaStreamDestroy(_stream: cudaStream_t) -> cudaError_t {
        0
    }
    
    pub unsafe fn cudaStreamSynchronize(_stream: cudaStream_t) -> cudaError_t {
        0
    }
}

#[cfg(not(feature = "cuda"))]
pub use stubs::*;

/// Check CUDA error and convert to Result
pub fn check_cuda(err: cudaError_t) -> Result<(), cudaError_t> {
    if err == 0 {
        Ok(())
    } else {
        Err(err)
    }
}

/// Check cuBLAS error
pub fn check_cublas(status: cublasStatus_t) -> Result<(), cublasStatus_t> {
    if status == 0 {
        Ok(())
    } else {
        Err(status)
    }
}

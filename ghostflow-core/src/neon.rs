//! ARM NEON SIMD optimizations
//!
//! Provides SIMD acceleration for ARM processors (mobile, Apple Silicon, etc.)

use crate::tensor::Tensor;
use crate::error::Result;

/// Check if NEON is available on this platform
pub fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is always available on AArch64
    }
    #[cfg(all(target_arch = "arm", target_feature = "neon"))]
    {
        true
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "arm", target_feature = "neon"))))]
    {
        false
    }
}

/// NEON-optimized vector addition
pub fn add_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            add_neon_impl(a, b, result);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback to scalar
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn add_neon_impl(a: &[f32], b: &[f32], result: &mut [f32]) {
    use std::arch::aarch64::*;
    
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;
    
    // Process 4 elements at a time using NEON
    for i in 0..chunks {
        let idx = i * 4;
        
        // Load 4 floats from a and b
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        
        // Add vectors
        let vc = vaddq_f32(va, vb);
        
        // Store result
        vst1q_f32(result.as_mut_ptr().add(idx), vc);
    }
    
    // Handle remainder
    for i in (chunks * 4)..len {
        result[i] = a[i] + b[i];
    }
}

/// NEON-optimized vector multiplication
pub fn mul_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            mul_neon_impl(a, b, result);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn mul_neon_impl(a: &[f32], b: &[f32], result: &mut [f32]) {
    use std::arch::aarch64::*;
    
    let len = a.len();
    let chunks = len / 4;
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let vc = vmulq_f32(va, vb);
        vst1q_f32(result.as_mut_ptr().add(idx), vc);
    }
    
    for i in (chunks * 4)..len {
        result[i] = a[i] * b[i];
    }
}

/// NEON-optimized dot product
pub fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon_impl(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    
    let len = a.len();
    let chunks = len / 4;
    
    // Accumulator vector
    let mut acc = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        
        // Multiply and accumulate
        acc = vfmaq_f32(acc, va, vb);
    }
    
    // Horizontal sum of accumulator
    let mut sum = vaddvq_f32(acc);
    
    // Handle remainder
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    
    sum
}

/// NEON-optimized ReLU
pub fn relu_neon(data: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            relu_neon_impl(data);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for x in data.iter_mut() {
            *x = x.max(0.0);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn relu_neon_impl(data: &mut [f32]) {
    use std::arch::aarch64::*;
    
    let len = data.len();
    let chunks = len / 4;
    let zero = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let idx = i * 4;
        let v = vld1q_f32(data.as_ptr().add(idx));
        let result = vmaxq_f32(v, zero);
        vst1q_f32(data.as_mut_ptr().add(idx), result);
    }
    
    for i in (chunks * 4)..len {
        data[i] = data[i].max(0.0);
    }
}

/// NEON-optimized sigmoid
pub fn sigmoid_neon(data: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            sigmoid_neon_impl(data);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for x in data.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sigmoid_neon_impl(data: &mut [f32]) {
    // NEON doesn't have native exp, so we use scalar for now
    // In production, would use a fast approximation
    for x in data.iter_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

/// NEON-optimized matrix multiplication (simplified)
pub fn matmul_neon(
    a: &[f32],
    b: &[f32],
    result: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            matmul_neon_impl(a, b, result, m, n, k);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback to scalar
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn matmul_neon_impl(
    a: &[f32],
    b: &[f32],
    result: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    
    // Simplified NEON matmul - production would use blocking and better optimization
    for i in 0..m {
        for j in 0..n {
            let mut acc = vdupq_n_f32(0.0);
            let chunks = k / 4;
            
            for p in 0..chunks {
                let idx = p * 4;
                let va = vld1q_f32(a.as_ptr().add(i * k + idx));
                let vb = vld1q_f32(b.as_ptr().add(idx * n + j));
                acc = vfmaq_f32(acc, va, vb);
            }
            
            let mut sum = vaddvq_f32(acc);
            
            // Handle remainder
            for p in (chunks * 4)..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            
            result[i * n + j] = sum;
        }
    }
}

/// NEON-optimized convolution (simplified 2D)
pub fn conv2d_neon(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
) {
    let output_h = input_h - kernel_h + 1;
    let output_w = input_w - kernel_w + 1;
    
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            conv2d_neon_impl(input, kernel, output, input_h, input_w, kernel_h, kernel_w, output_h, output_w);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Scalar fallback
        for i in 0..output_h {
            for j in 0..output_w {
                let mut sum = 0.0;
                for ki in 0..kernel_h {
                    for kj in 0..kernel_w {
                        sum += input[(i + ki) * input_w + (j + kj)] * kernel[ki * kernel_w + kj];
                    }
                }
                output[i * output_w + j] = sum;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn conv2d_neon_impl(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
) {
    use std::arch::aarch64::*;
    
    // Simplified - production would use im2col or Winograd
    for i in 0..output_h {
        for j in 0..output_w {
            let mut acc = vdupq_n_f32(0.0);
            
            for ki in 0..kernel_h {
                for kj in 0..kernel_w {
                    let input_val = input[(i + ki) * input_w + (j + kj)];
                    let kernel_val = kernel[ki * kernel_w + kj];
                    let v_input = vdupq_n_f32(input_val);
                    let v_kernel = vdupq_n_f32(kernel_val);
                    acc = vfmaq_f32(acc, v_input, v_kernel);
                }
            }
            
            output[i * output_w + j] = vaddvq_f32(acc);
        }
    }
}

/// Tensor operations with NEON acceleration
impl Tensor {
    /// Add two tensors using NEON
    pub fn add_neon(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data_f32();
        let b = other.data_f32();
        let mut result = vec![0.0; a.len()];
        
        add_neon(&a, &b, &mut result);
        
        Tensor::from_slice(&result, self.dims())
    }
    
    /// Multiply two tensors using NEON
    pub fn mul_neon(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data_f32();
        let b = other.data_f32();
        let mut result = vec![0.0; a.len()];
        
        mul_neon(&a, &b, &mut result);
        
        Tensor::from_slice(&result, self.dims())
    }
    
    /// ReLU activation using NEON
    pub fn relu_neon(&self) -> Tensor {
        let mut data = self.data_f32();
        relu_neon(&mut data);
        Tensor::from_slice(&data, self.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neon_availability() {
        let available = is_neon_available();
        #[cfg(target_arch = "aarch64")]
        assert!(available);
    }
    
    #[test]
    fn test_add_neon() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0; 8];
        
        add_neon(&a, &b, &mut result);
        
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
    
    #[test]
    fn test_dot_neon() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        
        let result = dot_neon(&a, &b);
        assert_eq!(result, 10.0);
    }
    
    #[test]
    fn test_relu_neon() {
        let mut data = vec![-1.0, 2.0, -3.0, 4.0];
        relu_neon(&mut data);
        assert_eq!(data, vec![0.0, 2.0, 0.0, 4.0]);
    }
}

//! Advanced SIMD optimizations for tensor operations
//!
//! This module provides highly optimized SIMD implementations for common operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized vector addition
#[inline]
pub fn simd_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_add_f32_avx2(a, b, out) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { simd_add_f32_sse(a, b, out) }
        } else {
            scalar_add_f32(a, b, out)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        scalar_add_f32(a, b, out)
    }
}

/// AVX2 implementation of vector addition
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    // Process 8 elements at a time with AVX2
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vout = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), vout);
        i += 8;
    }
    
    // Handle remaining elements
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// SSE implementation of vector addition
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_add_f32_sse(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time with SSE
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vout = _mm_add_ps(va, vb);
        _mm_storeu_ps(out.as_mut_ptr().add(i), vout);
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Scalar fallback for vector addition
#[inline]
fn scalar_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// SIMD-optimized vector multiplication
#[inline]
pub fn simd_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_mul_f32_avx2(a, b, out) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { simd_mul_f32_sse(a, b, out) }
        } else {
            scalar_mul_f32(a, b, out)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        scalar_mul_f32(a, b, out)
    }
}

/// AVX2 implementation of vector multiplication
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vout = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), vout);
        i += 8;
    }
    
    while i < len {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

/// SSE implementation of vector multiplication
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_mul_f32_sse(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vout = _mm_mul_ps(va, vb);
        _mm_storeu_ps(out.as_mut_ptr().add(i), vout);
        i += 4;
    }
    
    while i < len {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

/// Scalar fallback for vector multiplication
#[inline]
fn scalar_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// SIMD-optimized dot product
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_dot_f32_avx2(a, b) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { simd_dot_f32_sse(a, b) }
        } else {
            scalar_dot_f32(a, b)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        scalar_dot_f32(a, b)
    }
}

/// AVX2 implementation of dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut sum = _mm256_setzero_ps();
    
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vprod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, vprod);
        i += 8;
    }
    
    // Horizontal sum
    let mut result = 0.0f32;
    let sum_array: [f32; 8] = std::mem::transmute(sum);
    for &val in &sum_array {
        result += val;
    }
    
    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// SSE implementation of dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_dot_f32_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut sum = _mm_setzero_ps();
    
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vprod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, vprod);
        i += 4;
    }
    
    // Horizontal sum
    let mut result = 0.0f32;
    let sum_array: [f32; 4] = std::mem::transmute(sum);
    for &val in &sum_array {
        result += val;
    }
    
    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// Scalar fallback for dot product
#[inline]
fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// SIMD-optimized ReLU activation
#[inline]
pub fn simd_relu_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_relu_f32_avx2(input, output) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { simd_relu_f32_sse(input, output) }
        } else {
            scalar_relu_f32(input, output)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        scalar_relu_f32(input, output)
    }
}

/// AVX2 implementation of ReLU
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_relu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.as_ptr().add(i));
        let vout = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), vout);
        i += 8;
    }
    
    while i < len {
        output[i] = input[i].max(0.0);
        i += 1;
    }
}

/// SSE implementation of ReLU
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_relu_f32_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let mut i = 0;
    let zero = _mm_setzero_ps();
    
    while i + 4 <= len {
        let v = _mm_loadu_ps(input.as_ptr().add(i));
        let vout = _mm_max_ps(v, zero);
        _mm_storeu_ps(output.as_mut_ptr().add(i), vout);
        i += 4;
    }
    
    while i < len {
        output[i] = input[i].max(0.0);
        i += 1;
    }
}

/// Scalar fallback for ReLU
#[inline]
fn scalar_relu_f32(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = vec![0.0f32; 8];
        
        simd_add_f32(&a, &b, &mut out);
        
        for i in 0..8 {
            assert_eq!(out[i], 9.0);
        }
    }

    #[test]
    fn test_simd_mul() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 4];
        
        simd_mul_f32(&a, &b, &mut out);
        
        assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_simd_dot() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        
        let result = simd_dot_f32(&a, &b);
        
        assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
    }

    #[test]
    fn test_simd_relu() {
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 5];
        
        simd_relu_f32(&input, &mut output);
        
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
}

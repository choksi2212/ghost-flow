//! Backward pass implementation

use ghostflow_core::Tensor;

/// Compute gradients via backpropagation
/// 
/// This is a simplified implementation. A full implementation would:
/// 1. Build computation graph during forward pass
/// 2. Topologically sort the graph
/// 3. Propagate gradients backward
pub fn backward(loss: &Tensor) {
    // For now, this is a placeholder
    // Full autograd requires tracking the computation graph
    
    // The loss should be a scalar
    assert!(loss.numel() == 1, "backward() requires scalar loss");
    
    // In a full implementation:
    // 1. Start with grad_output = 1.0 for the loss
    // 2. For each operation in reverse order:
    //    - Compute gradients w.r.t. inputs using chain rule
    //    - Accumulate gradients for each parameter
    
    // This would be implemented with the GradTape
}

/// Gradient functions for common operations
pub mod grad_fns {
    /// Gradient of addition: d(a+b)/da = 1, d(a+b)/db = 1
    pub fn add_backward(grad_output: &[f32], _inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        vec![grad_output.to_vec(), grad_output.to_vec()]
    }

    /// Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a
    pub fn mul_backward(grad_output: &[f32], inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        let a = inputs[0];
        let b = inputs[1];
        
        let grad_a: Vec<f32> = grad_output.iter()
            .zip(b.iter())
            .map(|(&g, &bi)| g * bi)
            .collect();
        
        let grad_b: Vec<f32> = grad_output.iter()
            .zip(a.iter())
            .map(|(&g, &ai)| g * ai)
            .collect();
        
        vec![grad_a, grad_b]
    }

    /// Gradient of ReLU: d(relu(x))/dx = 1 if x > 0 else 0
    pub fn relu_backward(grad_output: &[f32], inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        let x = inputs[0];
        
        let grad: Vec<f32> = grad_output.iter()
            .zip(x.iter())
            .map(|(&g, &xi)| if xi > 0.0 { g } else { 0.0 })
            .collect();
        
        vec![grad]
    }

    /// Gradient of sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
    pub fn sigmoid_backward(grad_output: &[f32], inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        let x = inputs[0];
        
        let grad: Vec<f32> = grad_output.iter()
            .zip(x.iter())
            .map(|(&g, &xi)| {
                let s = 1.0 / (1.0 + (-xi).exp());
                g * s * (1.0 - s)
            })
            .collect();
        
        vec![grad]
    }

    /// Gradient of tanh: d(tanh(x))/dx = 1 - tanh(x)^2
    pub fn tanh_backward(grad_output: &[f32], inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        let x = inputs[0];
        
        let grad: Vec<f32> = grad_output.iter()
            .zip(x.iter())
            .map(|(&g, &xi)| {
                let t = xi.tanh();
                g * (1.0 - t * t)
            })
            .collect();
        
        vec![grad]
    }

    /// Gradient of exp: d(exp(x))/dx = exp(x)
    pub fn exp_backward(grad_output: &[f32], inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        let x = inputs[0];
        
        let grad: Vec<f32> = grad_output.iter()
            .zip(x.iter())
            .map(|(&g, &xi)| g * xi.exp())
            .collect();
        
        vec![grad]
    }

    /// Gradient of log: d(log(x))/dx = 1/x
    pub fn log_backward(grad_output: &[f32], inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        let x = inputs[0];
        
        let grad: Vec<f32> = grad_output.iter()
            .zip(x.iter())
            .map(|(&g, &xi)| g / xi)
            .collect();
        
        vec![grad]
    }

    /// Gradient of pow: d(x^n)/dx = n * x^(n-1)
    pub fn pow_backward(grad_output: &[f32], inputs: &[&[f32]], exp: f32) -> Vec<Vec<f32>> {
        let x = inputs[0];
        
        let grad: Vec<f32> = grad_output.iter()
            .zip(x.iter())
            .map(|(&g, &xi)| g * exp * xi.powf(exp - 1.0))
            .collect();
        
        vec![grad]
    }

    /// Gradient of matmul: d(AB)/dA = grad @ B^T, d(AB)/dB = A^T @ grad
    pub fn matmul_backward(
        grad_output: &[f32],
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // grad_output: [m, n]
        // a: [m, k]
        // b: [k, n]
        
        // grad_a = grad_output @ b^T  -> [m, n] @ [n, k] = [m, k]
        let mut grad_a = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0f32;
                for l in 0..n {
                    sum += grad_output[i * n + l] * b[j * n + l];
                }
                grad_a[i * k + j] = sum;
            }
        }
        
        // grad_b = a^T @ grad_output  -> [k, m] @ [m, n] = [k, n]
        let mut grad_b = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..m {
                    sum += a[l * k + i] * grad_output[l * n + j];
                }
                grad_b[i * n + j] = sum;
            }
        }
        
        (grad_a, grad_b)
    }

    /// Gradient of softmax (combined with cross-entropy for numerical stability)
    pub fn softmax_cross_entropy_backward(
        softmax_output: &[f32],
        target: &[f32],
        batch_size: usize,
        num_classes: usize,
    ) -> Vec<f32> {
        // grad = softmax - one_hot(target)
        let mut grad = softmax_output.to_vec();
        
        for b in 0..batch_size {
            let target_class = target[b] as usize;
            grad[b * num_classes + target_class] -= 1.0;
        }
        
        // Average over batch
        for g in &mut grad {
            *g /= batch_size as f32;
        }
        
        grad
    }
}

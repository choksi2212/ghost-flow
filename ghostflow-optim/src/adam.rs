//! Adam and AdamW optimizers

use ghostflow_core::Tensor;
use crate::optimizer::Optimizer;

/// Adam optimizer
pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,  // First moment
    v: Vec<Vec<f32>>,  // Second moment
    t: usize,          // Time step
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        
        Adam {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            m,
            v,
            t: 0,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.betas = (beta1, beta2);
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let (beta1, beta2) = self.betas;
        
        // Bias correction
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
        
        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let mut grad_data = grad.data_f32();
                let param_data = param.data_f32();
                
                // L2 regularization (not decoupled)
                if self.weight_decay != 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += self.weight_decay * p;
                    }
                }
                
                // Update biased first moment estimate
                for (j, &g) in grad_data.iter().enumerate() {
                    self.m[i][j] = beta1 * self.m[i][j] + (1.0 - beta1) * g;
                }
                
                // Update biased second moment estimate
                for (j, &g) in grad_data.iter().enumerate() {
                    self.v[i][j] = beta2 * self.v[i][j] + (1.0 - beta2) * g * g;
                }
                
                // Compute bias-corrected estimates and update
                let new_data: Vec<f32> = param_data.iter()
                    .enumerate()
                    .map(|(j, &p)| {
                        let m_hat = self.m[i][j] / bias_correction1;
                        let v_hat = self.v[i][j] / bias_correction2;
                        p - self.lr * m_hat / (v_hat.sqrt() + self.eps)
                    })
                    .collect();
                
                *param = Tensor::from_slice(&new_data, param.dims()).unwrap();
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.zero_grad();
        }
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn parameters(&self) -> &[Tensor] {
        &self.params
    }
}

/// AdamW optimizer (decoupled weight decay)
pub struct AdamW {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        
        AdamW {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,  // Default for AdamW
            m,
            v,
            t: 0,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.betas = (beta1, beta2);
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.t += 1;
        let (beta1, beta2) = self.betas;
        
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
        
        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let grad_data = grad.data_f32();
                let param_data = param.data_f32();
                
                // Update moments (without weight decay in gradient)
                for (j, &g) in grad_data.iter().enumerate() {
                    self.m[i][j] = beta1 * self.m[i][j] + (1.0 - beta1) * g;
                    self.v[i][j] = beta2 * self.v[i][j] + (1.0 - beta2) * g * g;
                }
                
                // Update with decoupled weight decay
                let new_data: Vec<f32> = param_data.iter()
                    .enumerate()
                    .map(|(j, &p)| {
                        let m_hat = self.m[i][j] / bias_correction1;
                        let v_hat = self.v[i][j] / bias_correction2;
                        
                        // Decoupled weight decay
                        let p_decayed = p * (1.0 - self.lr * self.weight_decay);
                        
                        p_decayed - self.lr * m_hat / (v_hat.sqrt() + self.eps)
                    })
                    .collect();
                
                *param = Tensor::from_slice(&new_data, param.dims()).unwrap();
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.zero_grad();
        }
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn parameters(&self) -> &[Tensor] {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_step() {
        let mut param = Tensor::ones(&[3]);
        param.set_requires_grad(true);
        param.set_grad(Tensor::full(&[3], 0.1f32));
        
        let mut adam = Adam::new(vec![param], 0.001);
        
        // Multiple steps
        for _ in 0..10 {
            adam.step();
        }
        
        // Parameters should have changed
        let updated = &adam.params[0];
        assert!(updated.data_f32()[0] < 1.0);
    }
}

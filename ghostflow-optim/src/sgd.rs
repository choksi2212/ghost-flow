//! Stochastic Gradient Descent optimizer

use ghostflow_core::Tensor;
use crate::optimizer::Optimizer;

/// SGD optimizer with optional momentum and weight decay
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let velocity = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        
        SGD {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity,
        }
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn dampening(mut self, dampening: f32) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let mut grad_data = grad.data_f32();
                let param_data = param.data_f32();
                
                // Weight decay (L2 regularization)
                if self.weight_decay != 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += self.weight_decay * p;
                    }
                }
                
                // Momentum
                if self.momentum != 0.0 {
                    let v = &mut self.velocity[i];
                    
                    for (j, g) in grad_data.iter().enumerate() {
                        v[j] = self.momentum * v[j] + (1.0 - self.dampening) * g;
                    }
                    
                    if self.nesterov {
                        for (j, g) in grad_data.iter_mut().enumerate() {
                            *g += self.momentum * self.velocity[i][j];
                        }
                    } else {
                        grad_data = self.velocity[i].clone();
                    }
                }
                
                // Update parameters
                let new_data: Vec<f32> = param_data.iter()
                    .zip(grad_data.iter())
                    .map(|(&p, &g)| p - self.lr * g)
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
    fn test_sgd_step() {
        let mut param = Tensor::ones(&[3]);
        param.set_requires_grad(true);
        param.set_grad(Tensor::full(&[3], 0.1f32));
        
        let mut sgd = SGD::new(vec![param], 0.1);
        sgd.step();
        
        let updated = &sgd.params[0];
        // 1.0 - 0.1 * 0.1 = 0.99
        assert!((updated.data_f32()[0] - 0.99).abs() < 1e-6);
    }
}

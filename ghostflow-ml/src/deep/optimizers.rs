//! Optimizers - SGD, Adam, RMSprop, AdaGrad, etc.

/// Optimizer trait
pub trait Optimizer: Send + Sync {
    fn step(&mut self, params: &mut [f32], grads: &[f32]);
    fn reset(&mut self);
}

/// Stochastic Gradient Descent with momentum
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
    velocity: Vec<f32>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: Vec::new(),
        }
    }

    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn nesterov(mut self, n: bool) -> Self {
        self.nesterov = n;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if self.velocity.len() != params.len() {
            self.velocity = vec![0.0; params.len()];
        }

        for i in 0..params.len() {
            let mut grad = grads[i];
            
            // Weight decay
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * params[i];
            }

            // Momentum
            self.velocity[i] = self.momentum * self.velocity[i] + grad;

            // Update
            if self.nesterov {
                params[i] -= self.lr * (grad + self.momentum * self.velocity[i]);
            } else {
                params[i] -= self.lr * self.velocity[i];
            }
        }
    }

    fn reset(&mut self) {
        self.velocity.clear();
    }
}

/// Adam optimizer
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }

        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let mut grad = grads[i];
            
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * params[i];
            }

            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            
            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            // Bias-corrected estimates
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Update parameters
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }

        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let grad = grads[i];

            // Decoupled weight decay
            params[i] -= self.lr * self.weight_decay * params[i];

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

/// RMSprop optimizer
pub struct RMSprop {
    pub lr: f32,
    pub alpha: f32,
    pub eps: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    v: Vec<f32>,
    buffer: Vec<f32>,
}

impl RMSprop {
    pub fn new(lr: f32) -> Self {
        RMSprop {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
            v: Vec::new(),
            buffer: Vec::new(),
        }
    }

    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }

    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if self.v.len() != params.len() {
            self.v = vec![0.0; params.len()];
            self.buffer = vec![0.0; params.len()];
        }

        for i in 0..params.len() {
            let mut grad = grads[i];
            
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * params[i];
            }

            self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * grad * grad;

            let avg = self.v[i].sqrt() + self.eps;

            if self.momentum > 0.0 {
                self.buffer[i] = self.momentum * self.buffer[i] + grad / avg;
                params[i] -= self.lr * self.buffer[i];
            } else {
                params[i] -= self.lr * grad / avg;
            }
        }
    }

    fn reset(&mut self) {
        self.v.clear();
        self.buffer.clear();
    }
}

/// AdaGrad optimizer
pub struct AdaGrad {
    pub lr: f32,
    pub eps: f32,
    pub weight_decay: f32,
    sum_sq: Vec<f32>,
}

impl AdaGrad {
    pub fn new(lr: f32) -> Self {
        AdaGrad {
            lr,
            eps: 1e-10,
            weight_decay: 0.0,
            sum_sq: Vec::new(),
        }
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if self.sum_sq.len() != params.len() {
            self.sum_sq = vec![0.0; params.len()];
        }

        for i in 0..params.len() {
            let mut grad = grads[i];
            
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * params[i];
            }

            self.sum_sq[i] += grad * grad;
            params[i] -= self.lr * grad / (self.sum_sq[i].sqrt() + self.eps);
        }
    }

    fn reset(&mut self) {
        self.sum_sq.clear();
    }
}

/// Adadelta optimizer
pub struct Adadelta {
    pub rho: f32,
    pub eps: f32,
    pub weight_decay: f32,
    avg_sq_grad: Vec<f32>,
    avg_sq_delta: Vec<f32>,
}

impl Adadelta {
    pub fn new() -> Self {
        Adadelta {
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
            avg_sq_grad: Vec::new(),
            avg_sq_delta: Vec::new(),
        }
    }
}

impl Default for Adadelta {
    fn default() -> Self { Self::new() }
}

impl Optimizer for Adadelta {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if self.avg_sq_grad.len() != params.len() {
            self.avg_sq_grad = vec![0.0; params.len()];
            self.avg_sq_delta = vec![0.0; params.len()];
        }

        for i in 0..params.len() {
            let mut grad = grads[i];
            
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * params[i];
            }

            self.avg_sq_grad[i] = self.rho * self.avg_sq_grad[i] + (1.0 - self.rho) * grad * grad;

            let std_grad = (self.avg_sq_grad[i] + self.eps).sqrt();
            let std_delta = (self.avg_sq_delta[i] + self.eps).sqrt();

            let delta = std_delta / std_grad * grad;
            params[i] -= delta;

            self.avg_sq_delta[i] = self.rho * self.avg_sq_delta[i] + (1.0 - self.rho) * delta * delta;
        }
    }

    fn reset(&mut self) {
        self.avg_sq_grad.clear();
        self.avg_sq_delta.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd() {
        let mut opt = SGD::new(0.1);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        opt.step(&mut params, &grads);
        assert!(params[0] < 1.0);
    }

    #[test]
    fn test_adam() {
        let mut opt = Adam::new(0.001);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        opt.step(&mut params, &grads);
        assert!(params[0] < 1.0);
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }

        self.t += 1;

        for i in 0..params.len() {
            let mut grad = grads[i];
            
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * params[i];
            }

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));

            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)

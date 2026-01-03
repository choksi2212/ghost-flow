//! Learning rate schedulers

use crate::optimizer::Optimizer;

/// Learning rate scheduler trait
pub trait LRScheduler {
    fn step(&mut self);
    fn get_lr(&self) -> f32;
}

/// Step decay scheduler
pub struct StepLR<O: Optimizer> {
    optimizer: O,
    step_size: usize,
    gamma: f32,
    current_step: usize,
    base_lr: f32,
}

impl<O: Optimizer> StepLR<O> {
    pub fn new(optimizer: O, step_size: usize, gamma: f32) -> Self {
        let base_lr = optimizer.get_lr();
        StepLR {
            optimizer,
            step_size,
            gamma,
            current_step: 0,
            base_lr,
        }
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

impl<O: Optimizer> LRScheduler for StepLR<O> {
    fn step(&mut self) {
        self.current_step += 1;
        let num_decays = self.current_step / self.step_size;
        let new_lr = self.base_lr * self.gamma.powi(num_decays as i32);
        self.optimizer.set_lr(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.get_lr()
    }
}

/// Exponential decay scheduler
pub struct ExponentialLR<O: Optimizer> {
    optimizer: O,
    gamma: f32,
}

impl<O: Optimizer> ExponentialLR<O> {
    pub fn new(optimizer: O, gamma: f32) -> Self {
        ExponentialLR { optimizer, gamma }
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

impl<O: Optimizer> LRScheduler for ExponentialLR<O> {
    fn step(&mut self) {
        let current_lr = self.optimizer.get_lr();
        self.optimizer.set_lr(current_lr * self.gamma);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.get_lr()
    }
}

/// Cosine annealing scheduler
pub struct CosineAnnealingLR<O: Optimizer> {
    optimizer: O,
    t_max: usize,
    eta_min: f32,
    base_lr: f32,
    current_step: usize,
}

impl<O: Optimizer> CosineAnnealingLR<O> {
    pub fn new(optimizer: O, t_max: usize, eta_min: f32) -> Self {
        let base_lr = optimizer.get_lr();
        CosineAnnealingLR {
            optimizer,
            t_max,
            eta_min,
            base_lr,
            current_step: 0,
        }
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

impl<O: Optimizer> LRScheduler for CosineAnnealingLR<O> {
    fn step(&mut self) {
        self.current_step += 1;
        let t = self.current_step % self.t_max;
        let cos_val = (std::f32::consts::PI * t as f32 / self.t_max as f32).cos();
        let new_lr = self.eta_min + (self.base_lr - self.eta_min) * (1.0 + cos_val) / 2.0;
        self.optimizer.set_lr(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.get_lr()
    }
}

/// Linear warmup scheduler
pub struct LinearWarmup<O: Optimizer> {
    optimizer: O,
    warmup_steps: usize,
    target_lr: f32,
    current_step: usize,
}

impl<O: Optimizer> LinearWarmup<O> {
    pub fn new(mut optimizer: O, warmup_steps: usize) -> Self {
        let target_lr = optimizer.get_lr();
        optimizer.set_lr(0.0);
        LinearWarmup {
            optimizer,
            warmup_steps,
            target_lr,
            current_step: 0,
        }
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

impl<O: Optimizer> LRScheduler for LinearWarmup<O> {
    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step <= self.warmup_steps {
            let new_lr = self.target_lr * (self.current_step as f32 / self.warmup_steps as f32);
            self.optimizer.set_lr(new_lr);
        }
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.get_lr()
    }
}

/// Reduce on plateau scheduler
pub struct ReduceLROnPlateau<O: Optimizer> {
    optimizer: O,
    factor: f32,
    patience: usize,
    min_lr: f32,
    best_loss: f32,
    num_bad_epochs: usize,
}

impl<O: Optimizer> ReduceLROnPlateau<O> {
    pub fn new(optimizer: O, factor: f32, patience: usize) -> Self {
        ReduceLROnPlateau {
            optimizer,
            factor,
            patience,
            min_lr: 1e-8,
            best_loss: f32::INFINITY,
            num_bad_epochs: 0,
        }
    }

    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn step_with_loss(&mut self, loss: f32) {
        if loss < self.best_loss {
            self.best_loss = loss;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
            
            if self.num_bad_epochs >= self.patience {
                let current_lr = self.optimizer.get_lr();
                let new_lr = (current_lr * self.factor).max(self.min_lr);
                self.optimizer.set_lr(new_lr);
                self.num_bad_epochs = 0;
            }
        }
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

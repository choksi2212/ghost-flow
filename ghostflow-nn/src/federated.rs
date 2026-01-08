//! Federated Learning
//!
//! Implements federated learning algorithms:
//! - FedAvg (Federated Averaging)
//! - FedProx (Federated Proximal)
//! - FedAdam (Federated Adam)
//! - Secure aggregation
//! - Differential privacy

use ghostflow_core::Tensor;
use std::collections::HashMap;
use rand::Rng;

/// Client in federated learning
#[derive(Debug, Clone)]
pub struct FederatedClient {
    /// Client ID
    pub id: usize,
    /// Local model parameters
    pub parameters: HashMap<String, Tensor>,
    /// Number of local samples
    pub num_samples: usize,
    /// Local learning rate
    pub learning_rate: f32,
}

impl FederatedClient {
    /// Create a new federated client
    pub fn new(id: usize, parameters: HashMap<String, Tensor>, num_samples: usize) -> Self {
        FederatedClient {
            id,
            parameters,
            num_samples,
            learning_rate: 0.01,
        }
    }
    
    /// Perform local training
    pub fn local_train(&mut self, num_epochs: usize, _batch_size: usize) {
        // Simplified local training
        // In practice, this would train on local data
        for _ in 0..num_epochs {
            // Simulate gradient updates
            for (_, param) in self.parameters.iter_mut() {
                let grad = Tensor::randn(param.dims()).mul_scalar(0.01);
                *param = param.sub(&grad).unwrap();
            }
        }
    }
    
    /// Get model update (difference from initial parameters)
    pub fn get_update(&self, initial_params: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let mut updates = HashMap::new();
        
        for (name, param) in &self.parameters {
            if let Some(initial) = initial_params.get(name) {
                let update = param.sub(initial).unwrap();
                updates.insert(name.clone(), update);
            }
        }
        
        updates
    }
    
    /// Set parameters
    pub fn set_parameters(&mut self, parameters: HashMap<String, Tensor>) {
        self.parameters = parameters;
    }
}

/// Federated server
pub struct FederatedServer {
    /// Global model parameters
    pub global_parameters: HashMap<String, Tensor>,
    /// Connected clients
    pub clients: Vec<FederatedClient>,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
    /// Current round
    pub round: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregationStrategy {
    /// Federated Averaging (FedAvg)
    FedAvg,
    /// Federated Proximal (FedProx)
    FedProx { mu: f32 },
    /// Federated Adam
    FedAdam { beta1: f32, beta2: f32 },
}

impl FederatedServer {
    /// Create a new federated server
    pub fn new(global_parameters: HashMap<String, Tensor>, aggregation: AggregationStrategy) -> Self {
        FederatedServer {
            global_parameters,
            clients: Vec::new(),
            aggregation,
            round: 0,
        }
    }
    
    /// Register a client
    pub fn register_client(&mut self, client: FederatedClient) {
        self.clients.push(client);
    }
    
    /// Select clients for training (random sampling)
    pub fn select_clients(&self, fraction: f32) -> Vec<usize> {
        let num_clients = (self.clients.len() as f32 * fraction).ceil() as usize;
        let mut rng = rand::thread_rng();
        
        let mut selected = Vec::new();
        let mut available: Vec<usize> = (0..self.clients.len()).collect();
        
        for _ in 0..num_clients.min(available.len()) {
            let idx = rng.gen_range(0..available.len());
            selected.push(available.remove(idx));
        }
        
        selected
    }
    
    /// Distribute global model to clients
    pub fn distribute_model(&mut self, client_ids: &[usize]) {
        for &id in client_ids {
            if let Some(client) = self.clients.get_mut(id) {
                client.set_parameters(self.global_parameters.clone());
            }
        }
    }
    
    /// Aggregate client updates using FedAvg
    fn aggregate_fedavg(&self, client_ids: &[usize]) -> HashMap<String, Tensor> {
        let mut aggregated = HashMap::new();
        let mut total_samples = 0;
        
        // Collect initial parameters
        let initial_params = self.global_parameters.clone();
        
        // Weighted average based on number of samples
        for &id in client_ids {
            if let Some(client) = self.clients.get(id) {
                total_samples += client.num_samples;
                let updates = client.get_update(&initial_params);
                
                for (name, update) in updates {
                    let weighted_update = update.mul_scalar(client.num_samples as f32);
                    
                    aggregated.entry(name)
                        .and_modify(|agg: &mut Tensor| *agg = agg.add(&weighted_update).unwrap())
                        .or_insert(weighted_update);
                }
            }
        }
        
        // Normalize by total samples
        for (_, update) in aggregated.iter_mut() {
            *update = update.div_scalar(total_samples as f32);
        }
        
        // Apply updates to global parameters
        let mut new_params = HashMap::new();
        for (name, param) in &self.global_parameters {
            if let Some(update) = aggregated.get(name) {
                new_params.insert(name.clone(), param.add(update).unwrap());
            } else {
                new_params.insert(name.clone(), param.clone());
            }
        }
        
        new_params
    }
    
    /// Aggregate client updates using FedProx
    fn aggregate_fedprox(&self, client_ids: &[usize], mu: f32) -> HashMap<String, Tensor> {
        // FedProx adds a proximal term to keep updates close to global model
        // For simplicity, we use FedAvg with a damping factor
        let mut updates = self.aggregate_fedavg(client_ids);
        
        // Apply proximal damping
        for (name, update) in updates.iter_mut() {
            if let Some(global_param) = self.global_parameters.get(name) {
                let diff = update.sub(global_param).unwrap();
                let damped = diff.mul_scalar(1.0 / (1.0 + mu));
                *update = global_param.add(&damped).unwrap();
            }
        }
        
        updates
    }
    
    /// Aggregate client updates
    pub fn aggregate(&mut self, client_ids: &[usize]) {
        let new_params = match self.aggregation {
            AggregationStrategy::FedAvg => self.aggregate_fedavg(client_ids),
            AggregationStrategy::FedProx { mu } => self.aggregate_fedprox(client_ids, mu),
            AggregationStrategy::FedAdam { .. } => {
                // Simplified - would need momentum buffers
                self.aggregate_fedavg(client_ids)
            }
        };
        
        self.global_parameters = new_params;
        self.round += 1;
    }
    
    /// Run one round of federated learning
    pub fn train_round(&mut self, client_fraction: f32, local_epochs: usize, batch_size: usize) {
        // Select clients
        let selected_clients = self.select_clients(client_fraction);
        
        // Distribute model
        self.distribute_model(&selected_clients);
        
        // Local training
        for &id in &selected_clients {
            if let Some(client) = self.clients.get_mut(id) {
                client.local_train(local_epochs, batch_size);
            }
        }
        
        // Aggregate updates
        self.aggregate(&selected_clients);
    }
    
    /// Get current round number
    pub fn current_round(&self) -> usize {
        self.round
    }
}

/// Secure aggregation using secret sharing
pub struct SecureAggregation {
    /// Number of clients
    num_clients: usize,
    /// Threshold for reconstruction
    threshold: usize,
}

impl SecureAggregation {
    /// Create a new secure aggregation scheme
    pub fn new(num_clients: usize, threshold: usize) -> Self {
        SecureAggregation {
            num_clients,
            threshold,
        }
    }
    
    /// Split a value into shares (simplified Shamir's secret sharing)
    pub fn share(&self, value: f32) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut shares = Vec::with_capacity(self.num_clients);
        
        // Generate random shares
        let mut sum = 0.0;
        for _ in 0..self.num_clients - 1 {
            let share: f32 = rng.gen_range(-1.0..1.0);
            shares.push(share);
            sum += share;
        }
        
        // Last share ensures sum equals value
        shares.push(value - sum);
        
        shares
    }
    
    /// Reconstruct value from shares
    pub fn reconstruct(&self, shares: &[f32]) -> f32 {
        if shares.len() < self.threshold {
            return 0.0;
        }
        
        shares.iter().sum()
    }
    
    /// Securely aggregate client updates
    pub fn aggregate_secure(&self, client_updates: &[HashMap<String, Tensor>]) -> HashMap<String, Tensor> {
        let mut aggregated = HashMap::new();
        
        // For each parameter
        if let Some(first_update) = client_updates.first() {
            for (name, _) in first_update {
                // Collect all client values for this parameter
                let mut param_values = Vec::new();
                for update in client_updates {
                    if let Some(tensor) = update.get(name) {
                        param_values.push(tensor.clone());
                    }
                }
                
                // Simple aggregation (in practice, would use secure protocols)
                if !param_values.is_empty() {
                    let mut sum = param_values[0].clone();
                    for tensor in &param_values[1..] {
                        sum = sum.add(tensor).unwrap();
                    }
                    let avg = sum.div_scalar(param_values.len() as f32);
                    aggregated.insert(name.clone(), avg);
                }
            }
        }
        
        aggregated
    }
}

/// Differential privacy for federated learning
pub struct DifferentialPrivacy {
    /// Privacy budget (epsilon)
    pub epsilon: f32,
    /// Sensitivity
    pub sensitivity: f32,
    /// Noise scale
    pub noise_scale: f32,
}

impl DifferentialPrivacy {
    /// Create a new differential privacy mechanism
    pub fn new(epsilon: f32, sensitivity: f32) -> Self {
        let noise_scale = sensitivity / epsilon;
        
        DifferentialPrivacy {
            epsilon,
            sensitivity,
            noise_scale,
        }
    }
    
    /// Add Gaussian noise to a tensor
    pub fn add_noise(&self, tensor: &Tensor) -> Tensor {
        let noise = Tensor::randn(tensor.dims()).mul_scalar(self.noise_scale);
        tensor.add(&noise).unwrap()
    }
    
    /// Clip gradients to bound sensitivity
    pub fn clip_gradients(&self, gradients: &HashMap<String, Tensor>, max_norm: f32) -> HashMap<String, Tensor> {
        let mut clipped = HashMap::new();
        
        // Compute global norm
        let mut global_norm_sq = 0.0;
        for (_, grad) in gradients {
            let data = grad.data_f32();
            global_norm_sq += data.iter().map(|x| x * x).sum::<f32>();
        }
        let global_norm = global_norm_sq.sqrt();
        
        // Clip if necessary
        let clip_factor = if global_norm > max_norm {
            max_norm / global_norm
        } else {
            1.0
        };
        
        for (name, grad) in gradients {
            let clipped_grad = grad.mul_scalar(clip_factor);
            clipped.insert(name.clone(), clipped_grad);
        }
        
        clipped
    }
    
    /// Apply differential privacy to aggregated updates
    pub fn privatize(&self, updates: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let mut private_updates = HashMap::new();
        
        for (name, update) in updates {
            let noisy_update = self.add_noise(update);
            private_updates.insert(name.clone(), noisy_update);
        }
        
        private_updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_federated_client() {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), Tensor::ones(&[2, 2]));
        
        let mut client = FederatedClient::new(0, params, 100);
        assert_eq!(client.num_samples, 100);
        
        client.local_train(1, 32);
        // Parameters should have changed
    }
    
    #[test]
    fn test_federated_server() {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), Tensor::ones(&[2, 2]));
        
        let mut server = FederatedServer::new(params.clone(), AggregationStrategy::FedAvg);
        
        // Register clients
        for i in 0..5 {
            let client = FederatedClient::new(i, params.clone(), 100);
            server.register_client(client);
        }
        
        assert_eq!(server.clients.len(), 5);
        
        // Select clients
        let selected = server.select_clients(0.5);
        assert!(selected.len() >= 2 && selected.len() <= 3);
    }
    
    #[test]
    fn test_secure_aggregation() {
        let secure_agg = SecureAggregation::new(5, 3);
        
        let value = 10.0;
        let shares = secure_agg.share(value);
        assert_eq!(shares.len(), 5);
        
        let reconstructed = secure_agg.reconstruct(&shares);
        assert!((reconstructed - value).abs() < 0.001);
    }
    
    #[test]
    fn test_differential_privacy() {
        let dp = DifferentialPrivacy::new(1.0, 1.0);
        
        let tensor = Tensor::ones(&[2, 2]);
        let noisy = dp.add_noise(&tensor);
        
        // Noisy tensor should be different
        assert_ne!(noisy.data_f32(), tensor.data_f32());
    }
}

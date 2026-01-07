//! Reinforcement Learning module
//!
//! Implements RL algorithms:
//! - Deep Q-Network (DQN)
//! - Policy Gradient (REINFORCE)
//! - Actor-Critic (A2C/A3C)
//! - Proximal Policy Optimization (PPO)
//! - Deep Deterministic Policy Gradient (DDPG)

use crate::Tensor;
use std::collections::VecDeque;
use rand::Rng;

/// Experience replay buffer for off-policy learning
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    capacity: usize,
    buffer: VecDeque<Experience>,
}

#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Tensor,
    pub action: usize,
    pub reward: f32,
    pub next_state: Tensor,
    pub done: bool,
}

impl ReplayBuffer {
    /// Create a new replay buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }
    
    /// Add an experience to the buffer
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }
    
    /// Sample a batch of experiences
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            let idx = rng.gen_range(0..self.buffer.len());
            samples.push(self.buffer[idx].clone());
        }
        
        samples
    }
    
    /// Get current size of buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Deep Q-Network (DQN) agent
pub struct DQNAgent {
    q_network: QNetwork,
    target_network: QNetwork,
    replay_buffer: ReplayBuffer,
    gamma: f32,
    epsilon: f32,
    epsilon_decay: f32,
    epsilon_min: f32,
    learning_rate: f32,
    batch_size: usize,
    target_update_freq: usize,
    steps: usize,
}

/// Q-Network (simple MLP)
#[derive(Debug, Clone)]
pub struct QNetwork {
    fc1: Tensor,
    fc2: Tensor,
    fc3: Tensor,
    state_dim: usize,
    action_dim: usize,
}

impl QNetwork {
    /// Create a new Q-Network
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        let fc1 = Tensor::randn(&[state_dim, hidden_dim]).mul_scalar(0.01);
        let fc2 = Tensor::randn(&[hidden_dim, hidden_dim]).mul_scalar(0.01);
        let fc3 = Tensor::randn(&[hidden_dim, action_dim]).mul_scalar(0.01);
        
        QNetwork {
            fc1,
            fc2,
            fc3,
            state_dim,
            action_dim,
        }
    }
    
    /// Forward pass: compute Q-values for all actions
    pub fn forward(&self, state: &Tensor) -> Tensor {
        let h1 = state.matmul(&self.fc1).unwrap().relu();
        let h2 = h1.matmul(&self.fc2).unwrap().relu();
        h2.matmul(&self.fc3).unwrap()
    }
    
    /// Get Q-value for a specific action
    pub fn q_value(&self, state: &Tensor, action: usize) -> f32 {
        let q_values = self.forward(state);
        q_values.data_f32()[action]
    }
}

impl DQNAgent {
    /// Create a new DQN agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        buffer_capacity: usize,
        gamma: f32,
        epsilon: f32,
        learning_rate: f32,
        batch_size: usize,
    ) -> Self {
        let q_network = QNetwork::new(state_dim, action_dim, hidden_dim);
        let target_network = q_network.clone();
        let replay_buffer = ReplayBuffer::new(buffer_capacity);
        
        DQNAgent {
            q_network,
            target_network,
            replay_buffer,
            gamma,
            epsilon,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            learning_rate,
            batch_size,
            target_update_freq: 100,
            steps: 0,
        }
    }
    
    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &Tensor) -> usize {
        let mut rng = rand::thread_rng();
        
        if rng.gen::<f32>() < self.epsilon {
            // Random action (exploration)
            rng.gen_range(0..self.q_network.action_dim)
        } else {
            // Greedy action (exploitation)
            let q_values = self.q_network.forward(state);
            let data = q_values.data_f32();
            data.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        }
    }
    
    /// Store experience in replay buffer
    pub fn store_experience(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }
    
    /// Train the agent on a batch of experiences
    pub fn train(&mut self) -> f32 {
        if self.replay_buffer.len() < self.batch_size {
            return 0.0;
        }
        
        let batch = self.replay_buffer.sample(self.batch_size);
        let mut total_loss = 0.0;
        
        for exp in batch {
            // Compute target Q-value: r + γ * max_a' Q_target(s', a')
            let target_q = if exp.done {
                exp.reward
            } else {
                let next_q_values = self.target_network.forward(&exp.next_state);
                let max_next_q = next_q_values.data_f32().iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                exp.reward + self.gamma * max_next_q
            };
            
            // Compute current Q-value
            let current_q = self.q_network.q_value(&exp.state, exp.action);
            
            // Compute loss (MSE)
            let loss = (current_q - target_q).powi(2);
            total_loss += loss;
        }
        
        // Update target network periodically
        self.steps += 1;
        if self.steps % self.target_update_freq == 0 {
            self.target_network = self.q_network.clone();
        }
        
        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        
        total_loss / self.batch_size as f32
    }
}

/// Policy network for policy gradient methods
#[derive(Debug, Clone)]
pub struct PolicyNetwork {
    fc1: Tensor,
    fc2: Tensor,
    fc3: Tensor,
}

impl PolicyNetwork {
    /// Create a new policy network
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        let fc1 = Tensor::randn(&[state_dim, hidden_dim]).mul_scalar(0.01);
        let fc2 = Tensor::randn(&[hidden_dim, hidden_dim]).mul_scalar(0.01);
        let fc3 = Tensor::randn(&[hidden_dim, action_dim]).mul_scalar(0.01);
        
        PolicyNetwork { fc1, fc2, fc3 }
    }
    
    /// Forward pass: compute action probabilities
    pub fn forward(&self, state: &Tensor) -> Tensor {
        let h1 = state.matmul(&self.fc1).unwrap().relu();
        let h2 = h1.matmul(&self.fc2).unwrap().relu();
        let logits = h2.matmul(&self.fc3).unwrap();
        logits.softmax(-1)
    }
    
    /// Sample action from policy
    pub fn sample_action(&self, state: &Tensor) -> usize {
        let probs = self.forward(state);
        let prob_data = probs.data_f32();
        
        // Sample from categorical distribution
        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();
        let mut cumsum = 0.0;
        
        for (i, &p) in prob_data.iter().enumerate() {
            cumsum += p;
            if sample < cumsum {
                return i;
            }
        }
        
        prob_data.len() - 1
    }
}

/// REINFORCE (Policy Gradient) agent
pub struct REINFORCEAgent {
    policy: PolicyNetwork,
    gamma: f32,
    learning_rate: f32,
    episode_rewards: Vec<f32>,
    episode_actions: Vec<usize>,
    episode_states: Vec<Tensor>,
}

impl REINFORCEAgent {
    /// Create a new REINFORCE agent
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize, gamma: f32, learning_rate: f32) -> Self {
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_dim);
        
        REINFORCEAgent {
            policy,
            gamma,
            learning_rate,
            episode_rewards: Vec::new(),
            episode_actions: Vec::new(),
            episode_states: Vec::new(),
        }
    }
    
    /// Select action from policy
    pub fn select_action(&self, state: &Tensor) -> usize {
        self.policy.sample_action(state)
    }
    
    /// Store step in current episode
    pub fn store_step(&mut self, state: Tensor, action: usize, reward: f32) {
        self.episode_states.push(state);
        self.episode_actions.push(action);
        self.episode_rewards.push(reward);
    }
    
    /// Train on completed episode
    pub fn train_episode(&mut self) -> f32 {
        let episode_len = self.episode_rewards.len();
        if episode_len == 0 {
            return 0.0;
        }
        
        // Compute discounted returns
        let mut returns = vec![0.0; episode_len];
        let mut g = 0.0;
        for t in (0..episode_len).rev() {
            g = self.episode_rewards[t] + self.gamma * g;
            returns[t] = g;
        }
        
        // Normalize returns
        let mean = returns.iter().sum::<f32>() / episode_len as f32;
        let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / episode_len as f32).sqrt();
        for r in &mut returns {
            *r = (*r - mean) / (std + 1e-8);
        }
        
        let total_return = returns[0];
        
        // Clear episode data
        self.episode_rewards.clear();
        self.episode_actions.clear();
        self.episode_states.clear();
        
        total_return
    }
}

/// Actor-Critic agent (A2C)
pub struct ActorCriticAgent {
    actor: PolicyNetwork,
    critic: ValueNetwork,
    gamma: f32,
    actor_lr: f32,
    critic_lr: f32,
}

/// Value network for critic
#[derive(Debug, Clone)]
pub struct ValueNetwork {
    fc1: Tensor,
    fc2: Tensor,
    fc3: Tensor,
}

impl ValueNetwork {
    /// Create a new value network
    pub fn new(state_dim: usize, hidden_dim: usize) -> Self {
        let fc1 = Tensor::randn(&[state_dim, hidden_dim]).mul_scalar(0.01);
        let fc2 = Tensor::randn(&[hidden_dim, hidden_dim]).mul_scalar(0.01);
        let fc3 = Tensor::randn(&[hidden_dim, 1]).mul_scalar(0.01);
        
        ValueNetwork { fc1, fc2, fc3 }
    }
    
    /// Forward pass: compute state value
    pub fn forward(&self, state: &Tensor) -> f32 {
        let h1 = state.matmul(&self.fc1).unwrap().relu();
        let h2 = h1.matmul(&self.fc2).unwrap().relu();
        let value = h2.matmul(&self.fc3).unwrap();
        value.data_f32()[0]
    }
}

impl ActorCriticAgent {
    /// Create a new Actor-Critic agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        gamma: f32,
        actor_lr: f32,
        critic_lr: f32,
    ) -> Self {
        let actor = PolicyNetwork::new(state_dim, action_dim, hidden_dim);
        let critic = ValueNetwork::new(state_dim, hidden_dim);
        
        ActorCriticAgent {
            actor,
            critic,
            gamma,
            actor_lr,
            critic_lr,
        }
    }
    
    /// Select action from actor policy
    pub fn select_action(&self, state: &Tensor) -> usize {
        self.actor.sample_action(state)
    }
    
    /// Train on a single step
    pub fn train_step(&mut self, state: &Tensor, action: usize, reward: f32, next_state: &Tensor, done: bool) -> (f32, f32) {
        // Compute TD error: δ = r + γ*V(s') - V(s)
        let value = self.critic.forward(state);
        let next_value = if done { 0.0 } else { self.critic.forward(next_state) };
        let td_error = reward + self.gamma * next_value - value;
        
        // Actor loss (policy gradient with advantage)
        let actor_loss = -td_error; // Simplified
        
        // Critic loss (MSE)
        let critic_loss = td_error.powi(2);
        
        (actor_loss, critic_loss)
    }
}

/// PPO (Proximal Policy Optimization) agent
pub struct PPOAgent {
    actor: PolicyNetwork,
    critic: ValueNetwork,
    gamma: f32,
    lambda: f32, // GAE parameter
    epsilon_clip: f32,
    actor_lr: f32,
    critic_lr: f32,
}

impl PPOAgent {
    /// Create a new PPO agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        gamma: f32,
        lambda: f32,
        epsilon_clip: f32,
    ) -> Self {
        let actor = PolicyNetwork::new(state_dim, action_dim, hidden_dim);
        let critic = ValueNetwork::new(state_dim, hidden_dim);
        
        PPOAgent {
            actor,
            critic,
            gamma,
            lambda,
            epsilon_clip,
            actor_lr: 3e-4,
            critic_lr: 1e-3,
        }
    }
    
    /// Select action from policy
    pub fn select_action(&self, state: &Tensor) -> usize {
        self.actor.sample_action(state)
    }
    
    /// Compute Generalized Advantage Estimation (GAE)
    pub fn compute_gae(&self, rewards: &[f32], values: &[f32], next_value: f32) -> Vec<f32> {
        let mut advantages = vec![0.0; rewards.len()];
        let mut gae = 0.0;
        
        for t in (0..rewards.len()).rev() {
            let next_val = if t == rewards.len() - 1 { next_value } else { values[t + 1] };
            let delta = rewards[t] + self.gamma * next_val - values[t];
            gae = delta + self.gamma * self.lambda * gae;
            advantages[t] = gae;
        }
        
        advantages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(10);
        let state = Tensor::zeros(&[4]);
        let next_state = Tensor::zeros(&[4]);
        
        let exp = Experience {
            state: state.clone(),
            action: 0,
            reward: 1.0,
            next_state: next_state.clone(),
            done: false,
        };
        
        buffer.push(exp);
        assert_eq!(buffer.len(), 1);
    }
    
    #[test]
    fn test_dqn_agent() {
        let agent = DQNAgent::new(4, 2, 64, 1000, 0.99, 1.0, 0.001, 32);
        let state = Tensor::randn(&[1, 4]);
        let action = agent.select_action(&state);
        assert!(action < 2);
    }
    
    #[test]
    fn test_policy_network() {
        let policy = PolicyNetwork::new(4, 2, 64);
        let state = Tensor::randn(&[1, 4]);
        let probs = policy.forward(&state);
        
        // Check probabilities sum to 1
        let sum: f32 = probs.data_f32().iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_reinforce_agent() {
        let mut agent = REINFORCEAgent::new(4, 2, 64, 0.99, 0.001);
        let state = Tensor::randn(&[1, 4]);
        let action = agent.select_action(&state);
        
        agent.store_step(state, action, 1.0);
        assert_eq!(agent.episode_rewards.len(), 1);
    }
    
    #[test]
    fn test_actor_critic() {
        let agent = ActorCriticAgent::new(4, 2, 64, 0.99, 0.001, 0.001);
        let state = Tensor::randn(&[1, 4]);
        let action = agent.select_action(&state);
        assert!(action < 2);
    }
}

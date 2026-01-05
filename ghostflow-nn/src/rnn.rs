//! Recurrent Neural Network Layers
//!
//! Implements LSTM, GRU, and basic RNN cells for sequence modeling.

use ghostflow_core::Tensor;
use crate::module::Module;
use crate::linear::Linear;

/// LSTM Cell - Long Short-Term Memory
///
/// Implements the LSTM equations:
/// - i_t = σ(W_ii * x_t + b_ii + W_hi * h_(t-1) + b_hi)  [input gate]
/// - f_t = σ(W_if * x_t + b_if + W_hf * h_(t-1) + b_hf)  [forget gate]
/// - g_t = tanh(W_ig * x_t + b_ig + W_hg * h_(t-1) + b_hg)  [cell gate]
/// - o_t = σ(W_io * x_t + b_io + W_ho * h_(t-1) + b_ho)  [output gate]
/// - c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t  [cell state]
/// - h_t = o_t ⊙ tanh(c_t)  [hidden state]
pub struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    
    // Input-to-hidden weights (combined for all gates)
    w_ih: Linear,
    // Hidden-to-hidden weights (combined for all gates)
    w_hh: Linear,
    
    training: bool,
}

impl LSTMCell {
    /// Create a new LSTM cell
    ///
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LSTMCell {
            input_size,
            hidden_size,
            // 4 * hidden_size for i, f, g, o gates
            w_ih: Linear::new(input_size, 4 * hidden_size),
            w_hh: Linear::new(hidden_size, 4 * hidden_size),
            training: true,
        }
    }

    /// Forward pass through LSTM cell
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, input_size]
    /// * `hidden` - Previous hidden state [batch, hidden_size]
    /// * `cell` - Previous cell state [batch, hidden_size]
    ///
    /// # Returns
    /// Tuple of (new_hidden, new_cell)
    pub fn forward_cell(&self, input: &Tensor, hidden: &Tensor, cell: &Tensor) -> (Tensor, Tensor) {
        let batch_size = input.dims()[0];
        
        // Compute all gates at once
        let gates = self.w_ih.forward(input)
            .add(&self.w_hh.forward(hidden))
            .unwrap();
        
        let gates_data = gates.data_f32();
        let hidden_data = cell.data_f32();
        
        let mut new_cell_data = vec![0.0f32; batch_size * self.hidden_size];
        let mut new_hidden_data = vec![0.0f32; batch_size * self.hidden_size];
        
        for b in 0..batch_size {
            for h in 0..self.hidden_size {
                let base_idx = b * 4 * self.hidden_size;
                
                // Extract gates
                let i = sigmoid(gates_data[base_idx + h]);  // input gate
                let f = sigmoid(gates_data[base_idx + self.hidden_size + h]);  // forget gate
                let g = tanh(gates_data[base_idx + 2 * self.hidden_size + h]);  // cell gate
                let o = sigmoid(gates_data[base_idx + 3 * self.hidden_size + h]);  // output gate
                
                // Update cell state
                let c_prev = hidden_data[b * self.hidden_size + h];
                let c_new = f * c_prev + i * g;
                new_cell_data[b * self.hidden_size + h] = c_new;
                
                // Update hidden state
                new_hidden_data[b * self.hidden_size + h] = o * tanh(c_new);
            }
        }
        
        let new_hidden = Tensor::from_slice(&new_hidden_data, &[batch_size, self.hidden_size]).unwrap();
        let new_cell = Tensor::from_slice(&new_cell_data, &[batch_size, self.hidden_size]).unwrap();
        
        (new_hidden, new_cell)
    }
}

impl Module for LSTMCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = input.dims()[0];
        let hidden = Tensor::zeros(&[batch_size, self.hidden_size]);
        let cell = Tensor::zeros(&[batch_size, self.hidden_size]);
        let (h, _) = self.forward_cell(input, &hidden, &cell);
        h
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.w_ih.parameters();
        params.extend(self.w_hh.parameters());
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// LSTM Layer - processes entire sequences
pub struct LSTM {
    cell: LSTMCell,
    num_layers: usize,
    bidirectional: bool,
    dropout: f32,
    training: bool,
}

impl LSTM {
    /// Create a new LSTM layer
    ///
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `num_layers` - Number of stacked LSTM layers
    /// * `bidirectional` - Whether to use bidirectional LSTM
    /// * `dropout` - Dropout probability between layers
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout: f32,
    ) -> Self {
        LSTM {
            cell: LSTMCell::new(input_size, hidden_size),
            num_layers,
            bidirectional,
            dropout,
            training: true,
        }
    }

    /// Forward pass through LSTM
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, seq_len, input_size]
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size * num_directions]
    pub fn forward_sequence(&self, input: &Tensor) -> Tensor {
        let batch_size = input.dims()[0];
        let seq_len = input.dims()[1];
        let input_size = input.dims()[2];
        
        let hidden_size = self.cell.hidden_size;
        let num_directions = if self.bidirectional { 2 } else { 1 };
        
        // Initialize hidden and cell states
        let mut h = Tensor::zeros(&[batch_size, hidden_size]);
        let mut c = Tensor::zeros(&[batch_size, hidden_size]);
        
        let input_data = input.data_f32();
        let mut output_data = vec![0.0f32; batch_size * seq_len * hidden_size * num_directions];
        
        // Forward direction
        for t in 0..seq_len {
            // Extract input at time t
            let mut x_t_data = vec![0.0f32; batch_size * input_size];
            for b in 0..batch_size {
                for i in 0..input_size {
                    x_t_data[b * input_size + i] = 
                        input_data[b * seq_len * input_size + t * input_size + i];
                }
            }
            let x_t = Tensor::from_slice(&x_t_data, &[batch_size, input_size]).unwrap();
            
            // LSTM cell forward
            let (h_new, c_new) = self.cell.forward_cell(&x_t, &h, &c);
            h = h_new;
            c = c_new;
            
            // Store output
            let h_data = h.data_f32();
            for b in 0..batch_size {
                for h_idx in 0..hidden_size {
                    output_data[b * seq_len * hidden_size * num_directions + 
                               t * hidden_size * num_directions + h_idx] = h_data[b * hidden_size + h_idx];
                }
            }
        }
        
        // Backward direction (if bidirectional)
        if self.bidirectional {
            let mut h_back = Tensor::zeros(&[batch_size, hidden_size]);
            let mut c_back = Tensor::zeros(&[batch_size, hidden_size]);
            
            for t in (0..seq_len).rev() {
                let mut x_t_data = vec![0.0f32; batch_size * input_size];
                for b in 0..batch_size {
                    for i in 0..input_size {
                        x_t_data[b * input_size + i] = 
                            input_data[b * seq_len * input_size + t * input_size + i];
                    }
                }
                let x_t = Tensor::from_slice(&x_t_data, &[batch_size, input_size]).unwrap();
                
                let (h_new, c_new) = self.cell.forward_cell(&x_t, &h_back, &c_back);
                h_back = h_new;
                c_back = c_new;
                
                let h_data = h_back.data_f32();
                for b in 0..batch_size {
                    for h_idx in 0..hidden_size {
                        output_data[b * seq_len * hidden_size * num_directions + 
                                   t * hidden_size * num_directions + hidden_size + h_idx] = 
                            h_data[b * hidden_size + h_idx];
                    }
                }
            }
        }
        
        Tensor::from_slice(
            &output_data,
            &[batch_size, seq_len, hidden_size * num_directions]
        ).unwrap()
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_sequence(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.cell.parameters()
    }

    fn train(&mut self) {
        self.training = true;
        self.cell.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.cell.eval();
    }

    fn is_training(&self) -> bool { self.training }
}

/// GRU Cell - Gated Recurrent Unit
///
/// Implements the GRU equations:
/// - r_t = σ(W_ir * x_t + b_ir + W_hr * h_(t-1) + b_hr)  [reset gate]
/// - z_t = σ(W_iz * x_t + b_iz + W_hz * h_(t-1) + b_hz)  [update gate]
/// - n_t = tanh(W_in * x_t + b_in + r_t ⊙ (W_hn * h_(t-1) + b_hn))  [new gate]
/// - h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_(t-1)  [hidden state]
pub struct GRUCell {
    input_size: usize,
    hidden_size: usize,
    
    // Input-to-hidden weights (combined for all gates)
    w_ih: Linear,
    // Hidden-to-hidden weights (combined for all gates)
    w_hh: Linear,
    
    training: bool,
}

impl GRUCell {
    /// Create a new GRU cell
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        GRUCell {
            input_size,
            hidden_size,
            // 3 * hidden_size for r, z, n gates
            w_ih: Linear::new(input_size, 3 * hidden_size),
            w_hh: Linear::new(hidden_size, 3 * hidden_size),
            training: true,
        }
    }

    /// Forward pass through GRU cell
    pub fn forward_cell(&self, input: &Tensor, hidden: &Tensor) -> Tensor {
        let batch_size = input.dims()[0];
        
        // Compute gates
        let gi = self.w_ih.forward(input);
        let gh = self.w_hh.forward(hidden);
        
        let gi_data = gi.data_f32();
        let gh_data = gh.data_f32();
        let h_data = hidden.data_f32();
        
        let mut new_hidden_data = vec![0.0f32; batch_size * self.hidden_size];
        
        for b in 0..batch_size {
            for h in 0..self.hidden_size {
                // Reset gate
                let r = sigmoid(
                    gi_data[b * 3 * self.hidden_size + h] + 
                    gh_data[b * 3 * self.hidden_size + h]
                );
                
                // Update gate
                let z = sigmoid(
                    gi_data[b * 3 * self.hidden_size + self.hidden_size + h] + 
                    gh_data[b * 3 * self.hidden_size + self.hidden_size + h]
                );
                
                // New gate
                let n = tanh(
                    gi_data[b * 3 * self.hidden_size + 2 * self.hidden_size + h] + 
                    r * gh_data[b * 3 * self.hidden_size + 2 * self.hidden_size + h]
                );
                
                // New hidden state
                let h_prev = h_data[b * self.hidden_size + h];
                new_hidden_data[b * self.hidden_size + h] = (1.0 - z) * n + z * h_prev;
            }
        }
        
        Tensor::from_slice(&new_hidden_data, &[batch_size, self.hidden_size]).unwrap()
    }
}

impl Module for GRUCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = input.dims()[0];
        let hidden = Tensor::zeros(&[batch_size, self.hidden_size]);
        self.forward_cell(input, &hidden)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.w_ih.parameters();
        params.extend(self.w_hh.parameters());
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// GRU Layer - processes entire sequences
pub struct GRU {
    cell: GRUCell,
    num_layers: usize,
    bidirectional: bool,
    dropout: f32,
    training: bool,
}

impl GRU {
    /// Create a new GRU layer
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout: f32,
    ) -> Self {
        GRU {
            cell: GRUCell::new(input_size, hidden_size),
            num_layers,
            bidirectional,
            dropout,
            training: true,
        }
    }

    /// Forward pass through GRU
    pub fn forward_sequence(&self, input: &Tensor) -> Tensor {
        let batch_size = input.dims()[0];
        let seq_len = input.dims()[1];
        let input_size = input.dims()[2];
        
        let hidden_size = self.cell.hidden_size;
        let num_directions = if self.bidirectional { 2 } else { 1 };
        
        let mut h = Tensor::zeros(&[batch_size, hidden_size]);
        
        let input_data = input.data_f32();
        let mut output_data = vec![0.0f32; batch_size * seq_len * hidden_size * num_directions];
        
        // Forward direction
        for t in 0..seq_len {
            let mut x_t_data = vec![0.0f32; batch_size * input_size];
            for b in 0..batch_size {
                for i in 0..input_size {
                    x_t_data[b * input_size + i] = 
                        input_data[b * seq_len * input_size + t * input_size + i];
                }
            }
            let x_t = Tensor::from_slice(&x_t_data, &[batch_size, input_size]).unwrap();
            
            h = self.cell.forward_cell(&x_t, &h);
            
            let h_data = h.data_f32();
            for b in 0..batch_size {
                for h_idx in 0..hidden_size {
                    output_data[b * seq_len * hidden_size * num_directions + 
                               t * hidden_size * num_directions + h_idx] = h_data[b * hidden_size + h_idx];
                }
            }
        }
        
        // Backward direction (if bidirectional)
        if self.bidirectional {
            let mut h_back = Tensor::zeros(&[batch_size, hidden_size]);
            
            for t in (0..seq_len).rev() {
                let mut x_t_data = vec![0.0f32; batch_size * input_size];
                for b in 0..batch_size {
                    for i in 0..input_size {
                        x_t_data[b * input_size + i] = 
                            input_data[b * seq_len * input_size + t * input_size + i];
                    }
                }
                let x_t = Tensor::from_slice(&x_t_data, &[batch_size, input_size]).unwrap();
                
                h_back = self.cell.forward_cell(&x_t, &h_back);
                
                let h_data = h_back.data_f32();
                for b in 0..batch_size {
                    for h_idx in 0..hidden_size {
                        output_data[b * seq_len * hidden_size * num_directions + 
                                   t * hidden_size * num_directions + hidden_size + h_idx] = 
                            h_data[b * hidden_size + h_idx];
                    }
                }
            }
        }
        
        Tensor::from_slice(
            &output_data,
            &[batch_size, seq_len, hidden_size * num_directions]
        ).unwrap()
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_sequence(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.cell.parameters()
    }

    fn train(&mut self) {
        self.training = true;
        self.cell.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.cell.eval();
    }

    fn is_training(&self) -> bool { self.training }
}

// Helper functions
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_cell() {
        let cell = LSTMCell::new(10, 20);
        let input = Tensor::randn(&[2, 10]);
        let hidden = Tensor::zeros(&[2, 20]);
        let cell_state = Tensor::zeros(&[2, 20]);
        
        let (h, c) = cell.forward_cell(&input, &hidden, &cell_state);
        
        assert_eq!(h.dims(), &[2, 20]);
        assert_eq!(c.dims(), &[2, 20]);
    }

    #[test]
    fn test_lstm_sequence() {
        let lstm = LSTM::new(10, 20, 1, false, 0.0);
        let input = Tensor::randn(&[2, 5, 10]); // [batch, seq, features]
        
        let output = lstm.forward_sequence(&input);
        
        assert_eq!(output.dims(), &[2, 5, 20]);
    }

    #[test]
    fn test_lstm_bidirectional() {
        let lstm = LSTM::new(10, 20, 1, true, 0.0);
        let input = Tensor::randn(&[2, 5, 10]);
        
        let output = lstm.forward_sequence(&input);
        
        assert_eq!(output.dims(), &[2, 5, 40]); // 20 * 2 directions
    }

    #[test]
    fn test_gru_cell() {
        let cell = GRUCell::new(10, 20);
        let input = Tensor::randn(&[2, 10]);
        let hidden = Tensor::zeros(&[2, 20]);
        
        let h = cell.forward_cell(&input, &hidden);
        
        assert_eq!(h.dims(), &[2, 20]);
    }

    #[test]
    fn test_gru_sequence() {
        let gru = GRU::new(10, 20, 1, false, 0.0);
        let input = Tensor::randn(&[2, 5, 10]);
        
        let output = gru.forward_sequence(&input);
        
        assert_eq!(output.dims(), &[2, 5, 20]);
    }

    #[test]
    fn test_gru_bidirectional() {
        let gru = GRU::new(10, 20, 1, true, 0.0);
        let input = Tensor::randn(&[2, 5, 10]);
        
        let output = gru.forward_sequence(&input);
        
        assert_eq!(output.dims(), &[2, 5, 40]); // 20 * 2 directions
    }
}

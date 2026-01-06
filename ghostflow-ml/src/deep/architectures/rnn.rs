//! RNN Architectures - LSTM, GRU, BiLSTM, Seq2Seq, Attention, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::Dense;
use crate::deep::activations::{Tanh, Sigmoid};

/// LSTM Cell
pub struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    w_ih: Dense, // input to hidden
    w_hh: Dense, // hidden to hidden
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LSTMCell {
            input_size,
            hidden_size,
            w_ih: Dense::new(input_size, hidden_size * 4),
            w_hh: Dense::new(hidden_size, hidden_size * 4),
        }
    }

    pub fn forward(&mut self, x: &Tensor, h: &Tensor, c: &Tensor, training: bool) -> (Tensor, Tensor) {
        let gates_ih = self.w_ih.forward(x, training);
        let gates_hh = self.w_hh.forward(h, training);
        
        // Add gates
        let gates_data_ih = gates_ih.data_f32();
        let gates_data_hh = gates_hh.data_f32();
        let gates: Vec<f32> = gates_data_ih.iter()
            .zip(gates_data_hh.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        let batch_size = x.dims()[0];
        let gates_tensor = Tensor::from_slice(&gates, &[batch_size, self.hidden_size * 4]).unwrap();
        
        // Split into 4 gates: input, forget, cell, output
        let (i_gate, f_gate, g_gate, o_gate) = self.split_gates(&gates_tensor);
        
        // Apply activations
        let i = Sigmoid::new().forward(&i_gate);
        let f = Sigmoid::new().forward(&f_gate);
        let g = Tanh::new().forward(&g_gate);
        let o = Sigmoid::new().forward(&o_gate);
        
        // Compute new cell state: c_new = f * c + i * g
        let c_data = c.data_f32();
        let f_data = f.data_f32();
        let i_data = i.data_f32();
        let g_data = g.data_f32();
        
        let c_new_data: Vec<f32> = (0..c_data.len())
            .map(|idx| f_data[idx] * c_data[idx] + i_data[idx] * g_data[idx])
            .collect();
        let c_new = Tensor::from_slice(&c_new_data, c.dims()).unwrap();
        
        // Compute new hidden state: h_new = o * tanh(c_new)
        let c_new_tanh = Tanh::new().forward(&c_new);
        let o_data = o.data_f32();
        let c_tanh_data = c_new_tanh.data_f32();
        
        let h_new_data: Vec<f32> = (0..o_data.len())
            .map(|idx| o_data[idx] * c_tanh_data[idx])
            .collect();
        let h_new = Tensor::from_slice(&h_new_data, h.dims()).unwrap();
        
        (h_new, c_new)
    }

    fn split_gates(&self, gates: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        let data = gates.data_f32();
        let batch_size = gates.dims()[0];
        
        let mut i_gate = Vec::new();
        let mut f_gate = Vec::new();
        let mut g_gate = Vec::new();
        let mut o_gate = Vec::new();
        
        for b in 0..batch_size {
            let offset = b * self.hidden_size * 4;
            for h in 0..self.hidden_size {
                i_gate.push(data[offset + h]);
                f_gate.push(data[offset + self.hidden_size + h]);
                g_gate.push(data[offset + self.hidden_size * 2 + h]);
                o_gate.push(data[offset + self.hidden_size * 3 + h]);
            }
        }
        
        (
            Tensor::from_slice(&i_gate, &[batch_size, self.hidden_size]).unwrap(),
            Tensor::from_slice(&f_gate, &[batch_size, self.hidden_size]).unwrap(),
            Tensor::from_slice(&g_gate, &[batch_size, self.hidden_size]).unwrap(),
            Tensor::from_slice(&o_gate, &[batch_size, self.hidden_size]).unwrap(),
        )
    }
}

/// LSTM Layer
pub struct LSTM {
    cell: LSTMCell,
    hidden_size: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LSTM {
            cell: LSTMCell::new(input_size, hidden_size),
            hidden_size,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];
        let input_size = x.dims()[2];
        
        // Initialize hidden and cell states
        let mut h = Tensor::from_slice(&vec![0.0f32; batch_size * self.hidden_size], 
                                       &[batch_size, self.hidden_size]).unwrap();
        let mut c = Tensor::from_slice(&vec![0.0f32; batch_size * self.hidden_size], 
                                       &[batch_size, self.hidden_size]).unwrap();
        
        let mut outputs = Vec::new();
        
        // Process sequence
        for t in 0..seq_len {
            let x_t = self.get_timestep(x, t, batch_size, input_size);
            let (h_new, c_new) = self.cell.forward(&x_t, &h, &c, training);
            h = h_new;
            c = c_new;
            outputs.push(h.clone());
        }
        
        // Stack outputs
        self.stack_outputs(&outputs, batch_size, seq_len)
    }

    fn get_timestep(&self, x: &Tensor, t: usize, batch_size: usize, input_size: usize) -> Tensor {
        let data = x.data_f32();
        let mut timestep_data = Vec::new();
        
        for b in 0..batch_size {
            let offset = (b * x.dims()[1] + t) * input_size;
            for i in 0..input_size {
                timestep_data.push(data[offset + i]);
            }
        }
        
        Tensor::from_slice(&timestep_data, &[batch_size, input_size]).unwrap()
    }

    fn stack_outputs(&self, outputs: &[Tensor], batch_size: usize, seq_len: usize) -> Tensor {
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let h_data = outputs[t].data_f32();
                let offset = b * self.hidden_size;
                for i in 0..self.hidden_size {
                    result.push(h_data[offset + i]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_len, self.hidden_size]).unwrap()
    }
}

/// GRU Cell
pub struct GRUCell {
    input_size: usize,
    hidden_size: usize,
    w_ih: Dense,
    w_hh: Dense,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        GRUCell {
            input_size,
            hidden_size,
            w_ih: Dense::new(input_size, hidden_size * 3),
            w_hh: Dense::new(hidden_size, hidden_size * 3),
        }
    }

    pub fn forward(&mut self, x: &Tensor, h: &Tensor, training: bool) -> Tensor {
        let gates_ih = self.w_ih.forward(x, training);
        let gates_hh = self.w_hh.forward(h, training);
        
        let gates_data_ih = gates_ih.data_f32();
        let gates_data_hh = gates_hh.data_f32();
        let gates: Vec<f32> = gates_data_ih.iter()
            .zip(gates_data_hh.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        let batch_size = x.dims()[0];
        let gates_tensor = Tensor::from_slice(&gates, &[batch_size, self.hidden_size * 3]).unwrap();
        
        let (r_gate, z_gate, n_gate) = self.split_gates(&gates_tensor);
        
        let r = Sigmoid::new().forward(&r_gate);
        let z = Sigmoid::new().forward(&z_gate);
        let n = Tanh::new().forward(&n_gate);
        
        // h_new = (1 - z) * n + z * h
        let h_data = h.data_f32();
        let z_data = z.data_f32();
        let n_data = n.data_f32();
        
        let h_new_data: Vec<f32> = (0..h_data.len())
            .map(|idx| (1.0 - z_data[idx]) * n_data[idx] + z_data[idx] * h_data[idx])
            .collect();
        
        Tensor::from_slice(&h_new_data, h.dims()).unwrap()
    }

    fn split_gates(&self, gates: &Tensor) -> (Tensor, Tensor, Tensor) {
        let data = gates.data_f32();
        let batch_size = gates.dims()[0];
        
        let mut r_gate = Vec::new();
        let mut z_gate = Vec::new();
        let mut n_gate = Vec::new();
        
        for b in 0..batch_size {
            let offset = b * self.hidden_size * 3;
            for h in 0..self.hidden_size {
                r_gate.push(data[offset + h]);
                z_gate.push(data[offset + self.hidden_size + h]);
                n_gate.push(data[offset + self.hidden_size * 2 + h]);
            }
        }
        
        (
            Tensor::from_slice(&r_gate, &[batch_size, self.hidden_size]).unwrap(),
            Tensor::from_slice(&z_gate, &[batch_size, self.hidden_size]).unwrap(),
            Tensor::from_slice(&n_gate, &[batch_size, self.hidden_size]).unwrap(),
        )
    }
}

/// GRU Layer
pub struct GRU {
    cell: GRUCell,
    hidden_size: usize,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        GRU {
            cell: GRUCell::new(input_size, hidden_size),
            hidden_size,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];
        let input_size = x.dims()[2];
        
        let mut h = Tensor::from_slice(&vec![0.0f32; batch_size * self.hidden_size], 
                                       &[batch_size, self.hidden_size]).unwrap();
        
        let mut outputs = Vec::new();
        
        for t in 0..seq_len {
            let x_t = self.get_timestep(x, t, batch_size, input_size);
            h = self.cell.forward(&x_t, &h, training);
            outputs.push(h.clone());
        }
        
        self.stack_outputs(&outputs, batch_size, seq_len)
    }

    fn get_timestep(&self, x: &Tensor, t: usize, batch_size: usize, input_size: usize) -> Tensor {
        let data = x.data_f32();
        let mut timestep_data = Vec::new();
        
        for b in 0..batch_size {
            let offset = (b * x.dims()[1] + t) * input_size;
            for i in 0..input_size {
                timestep_data.push(data[offset + i]);
            }
        }
        
        Tensor::from_slice(&timestep_data, &[batch_size, input_size]).unwrap()
    }

    fn stack_outputs(&self, outputs: &[Tensor], batch_size: usize, seq_len: usize) -> Tensor {
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let h_data = outputs[t].data_f32();
                let offset = b * self.hidden_size;
                for i in 0..self.hidden_size {
                    result.push(h_data[offset + i]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_len, self.hidden_size]).unwrap()
    }
}

/// Bidirectional LSTM
pub struct BiLSTM {
    forward_lstm: LSTM,
    backward_lstm: LSTM,
    hidden_size: usize,
}

impl BiLSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        BiLSTM {
            forward_lstm: LSTM::new(input_size, hidden_size),
            backward_lstm: LSTM::new(input_size, hidden_size),
            hidden_size,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let forward_out = self.forward_lstm.forward(x, training);
        let x_reversed = self.reverse_sequence(x);
        let backward_out = self.backward_lstm.forward(&x_reversed, training);
        let backward_out_reversed = self.reverse_sequence(&backward_out);
        
        self.concatenate(&forward_out, &backward_out_reversed)
    }

    fn reverse_sequence(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let feature_size = dims[2];
        let data = x.data_f32();
        
        let mut reversed = Vec::new();
        
        for b in 0..batch_size {
            for t in (0..seq_len).rev() {
                let offset = (b * seq_len + t) * feature_size;
                for f in 0..feature_size {
                    reversed.push(data[offset + f]);
                }
            }
        }
        
        Tensor::from_slice(&reversed, dims).unwrap()
    }

    fn concatenate(&self, x1: &Tensor, x2: &Tensor) -> Tensor {
        let dims = x1.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let feature_size = dims[2];
        
        let data1 = x1.data_f32();
        let data2 = x2.data_f32();
        
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let offset = (b * seq_len + t) * feature_size;
                for f in 0..feature_size {
                    result.push(data1[offset + f]);
                }
                for f in 0..feature_size {
                    result.push(data2[offset + f]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_len, feature_size * 2]).unwrap()
    }
}

/// Seq2Seq Encoder
pub struct Seq2SeqEncoder {
    lstm: LSTM,
}

impl Seq2SeqEncoder {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Seq2SeqEncoder {
            lstm: LSTM::new(input_size, hidden_size),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        self.lstm.forward(x, training)
    }
}

/// Seq2Seq Decoder
pub struct Seq2SeqDecoder {
    lstm: LSTM,
    output_layer: Dense,
}

impl Seq2SeqDecoder {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Seq2SeqDecoder {
            lstm: LSTM::new(input_size, hidden_size),
            output_layer: Dense::new(hidden_size, output_size),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let lstm_out = self.lstm.forward(x, training);
        
        // Apply output layer to each timestep
        let batch_size = lstm_out.dims()[0];
        let seq_len = lstm_out.dims()[1];
        let hidden_size = lstm_out.dims()[2];
        
        let mut outputs = Vec::new();
        
        for t in 0..seq_len {
            let timestep = self.get_timestep(&lstm_out, t, batch_size, hidden_size);
            let out = self.output_layer.forward(&timestep, training);
            outputs.push(out);
        }
        
        self.stack_outputs(&outputs, batch_size, seq_len)
    }

    fn get_timestep(&self, x: &Tensor, t: usize, batch_size: usize, hidden_size: usize) -> Tensor {
        let data = x.data_f32();
        let mut timestep_data = Vec::new();
        
        for b in 0..batch_size {
            let offset = (b * x.dims()[1] + t) * hidden_size;
            for i in 0..hidden_size {
                timestep_data.push(data[offset + i]);
            }
        }
        
        Tensor::from_slice(&timestep_data, &[batch_size, hidden_size]).unwrap()
    }

    fn stack_outputs(&self, outputs: &[Tensor], batch_size: usize, seq_len: usize) -> Tensor {
        let output_size = outputs[0].dims()[1];
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let out_data = outputs[t].data_f32();
                let offset = b * output_size;
                for i in 0..output_size {
                    result.push(out_data[offset + i]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_len, output_size]).unwrap()
    }
}

/// Seq2Seq Model
pub struct Seq2Seq {
    encoder: Seq2SeqEncoder,
    decoder: Seq2SeqDecoder,
}

impl Seq2Seq {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Seq2Seq {
            encoder: Seq2SeqEncoder::new(input_size, hidden_size),
            decoder: Seq2SeqDecoder::new(output_size, hidden_size, output_size),
        }
    }

    pub fn forward(&mut self, src: &Tensor, tgt: &Tensor, training: bool) -> Tensor {
        let _context = self.encoder.forward(src, training);
        self.decoder.forward(tgt, training)
    }
}

/// Attention Mechanism
pub struct Attention {
    query_proj: Dense,
    key_proj: Dense,
    value_proj: Dense,
    hidden_size: usize,
}

impl Attention {
    pub fn new(hidden_size: usize) -> Self {
        Attention {
            query_proj: Dense::new(hidden_size, hidden_size),
            key_proj: Dense::new(hidden_size, hidden_size),
            value_proj: Dense::new(hidden_size, hidden_size),
            hidden_size,
        }
    }

    pub fn forward(&mut self, query: &Tensor, key: &Tensor, value: &Tensor, training: bool) -> Tensor {
        let q = self.query_proj.forward(query, training);
        let k = self.key_proj.forward(key, training);
        let v = self.value_proj.forward(value, training);
        
        // Compute attention scores
        let scores = self.matmul(&q, &k);
        let scores_scaled = self.scale(&scores, (self.hidden_size as f32).sqrt());
        let attention_weights = self.softmax(&scores_scaled);
        
        // Apply attention to values
        self.matmul(&attention_weights, &v)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        // Simplified matrix multiplication
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        
        let batch_size = a.dims()[0];
        let m = a.dims()[1];
        let n = b.dims()[1];
        
        let mut result = vec![0.0f32; batch_size * m * n];
        
        for batch in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for k in 0..self.hidden_size {
                        let a_idx = (batch * m + i) * self.hidden_size + k;
                        let b_idx = (batch * self.hidden_size + k) * n + j;
                        sum += a_data[a_idx] * b_data[b_idx];
                    }
                    result[(batch * m + i) * n + j] = sum;
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, m, n]).unwrap()
    }

    fn scale(&self, x: &Tensor, scale: f32) -> Tensor {
        let data = x.data_f32();
        let scaled: Vec<f32> = data.iter().map(|&v| v / scale).collect();
        Tensor::from_slice(&scaled, x.dims()).unwrap()
    }

    fn softmax(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let feature_size = dims[2];
        
        let mut result = vec![0.0f32; data.len()];
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                let offset = (b * seq_len + i) * feature_size;
                
                // Find max for numerical stability
                let mut max_val = data[offset];
                for j in 1..feature_size {
                    max_val = max_val.max(data[offset + j]);
                }
                
                // Compute exp and sum
                let mut sum = 0.0f32;
                for j in 0..feature_size {
                    let exp_val = (data[offset + j] - max_val).exp();
                    result[offset + j] = exp_val;
                    sum += exp_val;
                }
                
                // Normalize
                for j in 0..feature_size {
                    result[offset + j] /= sum;
                }
            }
        }
        
        Tensor::from_slice(&result, dims).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm() {
        let mut lstm = LSTM::new(10, 20);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 5 * 10], &[2, 5, 10]).unwrap();
        let output = lstm.forward(&input, false);
        assert_eq!(output.dims(), &[2, 5, 20]);
    }

    #[test]
    fn test_gru() {
        let mut gru = GRU::new(10, 20);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 5 * 10], &[2, 5, 10]).unwrap();
        let output = gru.forward(&input, false);
        assert_eq!(output.dims(), &[2, 5, 20]);
    }

    #[test]
    fn test_bilstm() {
        let mut bilstm = BiLSTM::new(10, 20);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 5 * 10], &[2, 5, 10]).unwrap();
        let output = bilstm.forward(&input, false);
        assert_eq!(output.dims(), &[2, 5, 40]);
    }
}



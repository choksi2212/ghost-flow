//! Recurrent Neural Network Layers - RNN, LSTM, GRU

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Simple RNN Cell
pub struct RNNCell {
    pub input_size: usize,
    pub hidden_size: usize,
    w_ih: Vec<f32>,  // Input to hidden weights
    w_hh: Vec<f32>,  // Hidden to hidden weights
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    grad_w_ih: Vec<f32>,
    grad_w_hh: Vec<f32>,
    grad_b_ih: Vec<f32>,
    grad_b_hh: Vec<f32>,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (1.0 / hidden_size as f32).sqrt();

        let w_ih: Vec<f32> = (0..input_size * hidden_size)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let w_hh: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();

        RNNCell {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih: vec![0.0; hidden_size],
            b_hh: vec![0.0; hidden_size],
            grad_w_ih: vec![0.0; input_size * hidden_size],
            grad_w_hh: vec![0.0; hidden_size * hidden_size],
            grad_b_ih: vec![0.0; hidden_size],
            grad_b_hh: vec![0.0; hidden_size],
        }
    }

    pub fn forward(&self, input: &[f32], hidden: &[f32]) -> Vec<f32> {
        let mut new_hidden = vec![0.0f32; self.hidden_size];

        for h in 0..self.hidden_size {
            let mut sum = self.b_ih[h] + self.b_hh[h];
            
            for i in 0..self.input_size {
                sum += input[i] * self.w_ih[i * self.hidden_size + h];
            }
            for hh in 0..self.hidden_size {
                sum += hidden[hh] * self.w_hh[hh * self.hidden_size + h];
            }
            
            new_hidden[h] = sum.tanh();
        }

        new_hidden
    }
}

/// RNN Layer (many-to-many or many-to-one)
pub struct RNN {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub batch_first: bool,
    cells: Vec<RNNCell>,
    reverse_cells: Vec<RNNCell>,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::with_capacity(num_layers);
        cells.push(RNNCell::new(input_size, hidden_size));
        for _ in 1..num_layers {
            cells.push(RNNCell::new(hidden_size, hidden_size));
        }

        RNN {
            input_size,
            hidden_size,
            num_layers,
            bidirectional: false,
            batch_first: true,
            cells,
            reverse_cells: Vec::new(),
        }
    }

    pub fn bidirectional(mut self, bi: bool) -> Self {
        if bi && self.reverse_cells.is_empty() {
            self.reverse_cells.push(RNNCell::new(self.input_size, self.hidden_size));
            for _ in 1..self.num_layers {
                self.reverse_cells.push(RNNCell::new(self.hidden_size, self.hidden_size));
            }
        }
        self.bidirectional = bi;
        self
    }

    pub fn forward(&self, input: &Tensor, h0: Option<&Tensor>) -> (Tensor, Tensor) {
        let input_data = input.data_f32();
        let dims = input.dims();
        
        let (batch_size, seq_len) = if self.batch_first {
            (dims[0], dims[1])
        } else {
            (dims[1], dims[0])
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let output_size = self.hidden_size * num_directions;

        // Initialize hidden state
        let mut hidden: Vec<Vec<f32>> = if let Some(h) = h0 {
            let h_data = h.data_f32();
            (0..self.num_layers * num_directions)
                .map(|l| {
                    (0..batch_size)
                        .flat_map(|b| {
                            let start = (l * batch_size + b) * self.hidden_size;
                            h_data[start..start + self.hidden_size].to_vec()
                        })
                        .collect()
                })
                .collect()
        } else {
            vec![vec![0.0f32; batch_size * self.hidden_size]; self.num_layers * num_directions]
        };

        let mut output = vec![0.0f32; batch_size * seq_len * output_size];

        // Forward pass
        for b in 0..batch_size {
            let mut layer_input: Vec<Vec<f32>> = (0..seq_len)
                .map(|t| {
                    let start = if self.batch_first {
                        b * seq_len * self.input_size + t * self.input_size
                    } else {
                        t * batch_size * self.input_size + b * self.input_size
                    };
                    input_data[start..start + self.input_size].to_vec()
                })
                .collect();

            for layer in 0..self.num_layers {
                let mut h = hidden[layer][b * self.hidden_size..(b + 1) * self.hidden_size].to_vec();
                let mut layer_output = Vec::with_capacity(seq_len);

                // Forward direction
                for t in 0..seq_len {
                    h = self.cells[layer].forward(&layer_input[t], &h);
                    layer_output.push(h.clone());
                }

                // Backward direction if bidirectional
                if self.bidirectional {
                    let mut h_rev = hidden[self.num_layers + layer][b * self.hidden_size..(b + 1) * self.hidden_size].to_vec();
                    
                    for t in (0..seq_len).rev() {
                        h_rev = self.reverse_cells[layer].forward(&layer_input[t], &h_rev);
                        layer_output[t].extend(h_rev.clone());
                    }
                }

                layer_input = layer_output.clone();

                // Store final hidden state
                let h_start = b * self.hidden_size;
                hidden[layer][h_start..h_start + self.hidden_size].copy_from_slice(&h);
            }

            // Copy to output
            for t in 0..seq_len {
                let out_start = if self.batch_first {
                    b * seq_len * output_size + t * output_size
                } else {
                    t * batch_size * output_size + b * output_size
                };
                output[out_start..out_start + output_size].copy_from_slice(&layer_input[t]);
            }
        }

        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, output_size]
        } else {
            vec![seq_len, batch_size, output_size]
        };

        let hidden_flat: Vec<f32> = hidden.into_iter().flatten().collect();
        let hidden_shape = vec![self.num_layers * num_directions, batch_size, self.hidden_size];

        (
            Tensor::from_slice(&output, &output_shape).unwrap(),
            Tensor::from_slice(&hidden_flat, &hidden_shape).unwrap()
        )
    }
}

/// LSTM Cell
pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Weights for input gate, forget gate, cell gate, output gate
    w_ii: Vec<f32>, w_if: Vec<f32>, w_ig: Vec<f32>, w_io: Vec<f32>,
    w_hi: Vec<f32>, w_hf: Vec<f32>, w_hg: Vec<f32>, w_ho: Vec<f32>,
    b_i: Vec<f32>, b_f: Vec<f32>, b_g: Vec<f32>, b_o: Vec<f32>,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (1.0 / hidden_size as f32).sqrt();

        let init_weights = |size: usize| -> Vec<f32> {
            (0..size).map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale).collect()
        };

        LSTMCell {
            input_size,
            hidden_size,
            w_ii: init_weights(input_size * hidden_size),
            w_if: init_weights(input_size * hidden_size),
            w_ig: init_weights(input_size * hidden_size),
            w_io: init_weights(input_size * hidden_size),
            w_hi: init_weights(hidden_size * hidden_size),
            w_hf: init_weights(hidden_size * hidden_size),
            w_hg: init_weights(hidden_size * hidden_size),
            w_ho: init_weights(hidden_size * hidden_size),
            b_i: vec![0.0; hidden_size],
            b_f: vec![1.0; hidden_size], // Initialize forget gate bias to 1
            b_g: vec![0.0; hidden_size],
            b_o: vec![0.0; hidden_size],
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn forward(&self, input: &[f32], hidden: &[f32], cell: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut new_hidden = vec![0.0f32; self.hidden_size];
        let mut new_cell = vec![0.0f32; self.hidden_size];

        for h in 0..self.hidden_size {
            // Input gate
            let mut i_gate = self.b_i[h];
            // Forget gate
            let mut f_gate = self.b_f[h];
            // Cell gate
            let mut g_gate = self.b_g[h];
            // Output gate
            let mut o_gate = self.b_o[h];

            for i in 0..self.input_size {
                i_gate += input[i] * self.w_ii[i * self.hidden_size + h];
                f_gate += input[i] * self.w_if[i * self.hidden_size + h];
                g_gate += input[i] * self.w_ig[i * self.hidden_size + h];
                o_gate += input[i] * self.w_io[i * self.hidden_size + h];
            }

            for hh in 0..self.hidden_size {
                i_gate += hidden[hh] * self.w_hi[hh * self.hidden_size + h];
                f_gate += hidden[hh] * self.w_hf[hh * self.hidden_size + h];
                g_gate += hidden[hh] * self.w_hg[hh * self.hidden_size + h];
                o_gate += hidden[hh] * self.w_ho[hh * self.hidden_size + h];
            }

            let i_gate = Self::sigmoid(i_gate);
            let f_gate = Self::sigmoid(f_gate);
            let g_gate = g_gate.tanh();
            let o_gate = Self::sigmoid(o_gate);

            new_cell[h] = f_gate * cell[h] + i_gate * g_gate;
            new_hidden[h] = o_gate * new_cell[h].tanh();
        }

        (new_hidden, new_cell)
    }
}

/// LSTM Layer
pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub batch_first: bool,
    cells: Vec<LSTMCell>,
    reverse_cells: Vec<LSTMCell>,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::with_capacity(num_layers);
        cells.push(LSTMCell::new(input_size, hidden_size));
        for _ in 1..num_layers {
            cells.push(LSTMCell::new(hidden_size, hidden_size));
        }

        LSTM {
            input_size,
            hidden_size,
            num_layers,
            bidirectional: false,
            batch_first: true,
            cells,
            reverse_cells: Vec::new(),
        }
    }

    pub fn bidirectional(mut self, bi: bool) -> Self {
        if bi && self.reverse_cells.is_empty() {
            self.reverse_cells.push(LSTMCell::new(self.input_size, self.hidden_size));
            for _ in 1..self.num_layers {
                self.reverse_cells.push(LSTMCell::new(self.hidden_size, self.hidden_size));
            }
        }
        self.bidirectional = bi;
        self
    }

    pub fn forward(&self, input: &Tensor, initial_state: Option<(&Tensor, &Tensor)>) 
        -> (Tensor, (Tensor, Tensor)) 
    {
        let input_data = input.data_f32();
        let dims = input.dims();
        
        let (batch_size, seq_len) = if self.batch_first {
            (dims[0], dims[1])
        } else {
            (dims[1], dims[0])
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let output_size = self.hidden_size * num_directions;

        // Initialize states
        let total_layers = self.num_layers * num_directions;
        let mut h_states: Vec<Vec<f32>> = vec![vec![0.0f32; batch_size * self.hidden_size]; total_layers];
        let mut c_states: Vec<Vec<f32>> = vec![vec![0.0f32; batch_size * self.hidden_size]; total_layers];

        if let Some((h0, c0)) = initial_state {
            let h_data = h0.data_f32();
            let c_data = c0.data_f32();
            for l in 0..total_layers {
                let start = l * batch_size * self.hidden_size;
                let end = start + batch_size * self.hidden_size;
                h_states[l] = h_data[start..end].to_vec();
                c_states[l] = c_data[start..end].to_vec();
            }
        }

        let mut output = vec![0.0f32; batch_size * seq_len * output_size];

        for b in 0..batch_size {
            let mut layer_input: Vec<Vec<f32>> = (0..seq_len)
                .map(|t| {
                    let start = if self.batch_first {
                        b * seq_len * self.input_size + t * self.input_size
                    } else {
                        t * batch_size * self.input_size + b * self.input_size
                    };
                    input_data[start..start + self.input_size].to_vec()
                })
                .collect();

            for layer in 0..self.num_layers {
                let h_start = b * self.hidden_size;
                let mut h = h_states[layer][h_start..h_start + self.hidden_size].to_vec();
                let mut c = c_states[layer][h_start..h_start + self.hidden_size].to_vec();
                let mut layer_output = Vec::with_capacity(seq_len);

                for t in 0..seq_len {
                    let (new_h, new_c) = self.cells[layer].forward(&layer_input[t], &h, &c);
                    h = new_h;
                    c = new_c;
                    layer_output.push(h.clone());
                }

                if self.bidirectional {
                    let rev_layer = self.num_layers + layer;
                    let mut h_rev = h_states[rev_layer][h_start..h_start + self.hidden_size].to_vec();
                    let mut c_rev = c_states[rev_layer][h_start..h_start + self.hidden_size].to_vec();

                    for t in (0..seq_len).rev() {
                        let (new_h, new_c) = self.reverse_cells[layer].forward(&layer_input[t], &h_rev, &c_rev);
                        h_rev = new_h;
                        c_rev = new_c;
                        layer_output[t].extend(h_rev.clone());
                    }
                }

                layer_input = layer_output;
                h_states[layer][h_start..h_start + self.hidden_size].copy_from_slice(&h);
                c_states[layer][h_start..h_start + self.hidden_size].copy_from_slice(&c);
            }

            for t in 0..seq_len {
                let out_start = if self.batch_first {
                    b * seq_len * output_size + t * output_size
                } else {
                    t * batch_size * output_size + b * output_size
                };
                output[out_start..out_start + output_size].copy_from_slice(&layer_input[t]);
            }
        }

        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, output_size]
        } else {
            vec![seq_len, batch_size, output_size]
        };

        let h_flat: Vec<f32> = h_states.into_iter().flatten().collect();
        let c_flat: Vec<f32> = c_states.into_iter().flatten().collect();
        let state_shape = vec![total_layers, batch_size, self.hidden_size];

        (
            Tensor::from_slice(&output, &output_shape).unwrap(),
            (
                Tensor::from_slice(&h_flat, &state_shape).unwrap(),
                Tensor::from_slice(&c_flat, &state_shape).unwrap()
            )
        )
    }
}

/// GRU Cell
pub struct GRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    w_ir: Vec<f32>, w_iz: Vec<f32>, w_in: Vec<f32>,
    w_hr: Vec<f32>, w_hz: Vec<f32>, w_hn: Vec<f32>,
    b_r: Vec<f32>, b_z: Vec<f32>, b_n: Vec<f32>,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (1.0 / hidden_size as f32).sqrt();

        let init_weights = |size: usize| -> Vec<f32> {
            (0..size).map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale).collect()
        };

        GRUCell {
            input_size,
            hidden_size,
            w_ir: init_weights(input_size * hidden_size),
            w_iz: init_weights(input_size * hidden_size),
            w_in: init_weights(input_size * hidden_size),
            w_hr: init_weights(hidden_size * hidden_size),
            w_hz: init_weights(hidden_size * hidden_size),
            w_hn: init_weights(hidden_size * hidden_size),
            b_r: vec![0.0; hidden_size],
            b_z: vec![0.0; hidden_size],
            b_n: vec![0.0; hidden_size],
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn forward(&self, input: &[f32], hidden: &[f32]) -> Vec<f32> {
        let mut new_hidden = vec![0.0f32; self.hidden_size];

        for h in 0..self.hidden_size {
            let mut r_gate = self.b_r[h];
            let mut z_gate = self.b_z[h];
            let mut n_gate = self.b_n[h];

            for i in 0..self.input_size {
                r_gate += input[i] * self.w_ir[i * self.hidden_size + h];
                z_gate += input[i] * self.w_iz[i * self.hidden_size + h];
                n_gate += input[i] * self.w_in[i * self.hidden_size + h];
            }

            for hh in 0..self.hidden_size {
                r_gate += hidden[hh] * self.w_hr[hh * self.hidden_size + h];
                z_gate += hidden[hh] * self.w_hz[hh * self.hidden_size + h];
            }

            let r_gate = Self::sigmoid(r_gate);
            let z_gate = Self::sigmoid(z_gate);

            for hh in 0..self.hidden_size {
                n_gate += r_gate * hidden[hh] * self.w_hn[hh * self.hidden_size + h];
            }
            let n_gate = n_gate.tanh();

            new_hidden[h] = (1.0 - z_gate) * n_gate + z_gate * hidden[h];
        }

        new_hidden
    }
}

/// GRU Layer
pub struct GRU {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub batch_first: bool,
    cells: Vec<GRUCell>,
    reverse_cells: Vec<GRUCell>,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::with_capacity(num_layers);
        cells.push(GRUCell::new(input_size, hidden_size));
        for _ in 1..num_layers {
            cells.push(GRUCell::new(hidden_size, hidden_size));
        }

        GRU {
            input_size,
            hidden_size,
            num_layers,
            bidirectional: false,
            batch_first: true,
            cells,
            reverse_cells: Vec::new(),
        }
    }

    pub fn bidirectional(mut self, bi: bool) -> Self {
        if bi && self.reverse_cells.is_empty() {
            self.reverse_cells.push(GRUCell::new(self.input_size, self.hidden_size));
            for _ in 1..self.num_layers {
                self.reverse_cells.push(GRUCell::new(self.hidden_size, self.hidden_size));
            }
        }
        self.bidirectional = bi;
        self
    }

    pub fn forward(&self, input: &Tensor, h0: Option<&Tensor>) -> (Tensor, Tensor) {
        let input_data = input.data_f32();
        let dims = input.dims();
        
        let (batch_size, seq_len) = if self.batch_first {
            (dims[0], dims[1])
        } else {
            (dims[1], dims[0])
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let output_size = self.hidden_size * num_directions;
        let total_layers = self.num_layers * num_directions;

        let mut h_states: Vec<Vec<f32>> = if let Some(h) = h0 {
            let h_data = h.data_f32();
            (0..total_layers)
                .map(|l| {
                    let start = l * batch_size * self.hidden_size;
                    h_data[start..start + batch_size * self.hidden_size].to_vec()
                })
                .collect()
        } else {
            vec![vec![0.0f32; batch_size * self.hidden_size]; total_layers]
        };

        let mut output = vec![0.0f32; batch_size * seq_len * output_size];

        for b in 0..batch_size {
            let mut layer_input: Vec<Vec<f32>> = (0..seq_len)
                .map(|t| {
                    let start = if self.batch_first {
                        b * seq_len * self.input_size + t * self.input_size
                    } else {
                        t * batch_size * self.input_size + b * self.input_size
                    };
                    input_data[start..start + self.input_size].to_vec()
                })
                .collect();

            for layer in 0..self.num_layers {
                let h_start = b * self.hidden_size;
                let mut h = h_states[layer][h_start..h_start + self.hidden_size].to_vec();
                let mut layer_output = Vec::with_capacity(seq_len);

                for t in 0..seq_len {
                    h = self.cells[layer].forward(&layer_input[t], &h);
                    layer_output.push(h.clone());
                }

                if self.bidirectional {
                    let rev_layer = self.num_layers + layer;
                    let mut h_rev = h_states[rev_layer][h_start..h_start + self.hidden_size].to_vec();

                    for t in (0..seq_len).rev() {
                        h_rev = self.reverse_cells[layer].forward(&layer_input[t], &h_rev);
                        layer_output[t].extend(h_rev.clone());
                    }
                }

                layer_input = layer_output;
                h_states[layer][h_start..h_start + self.hidden_size].copy_from_slice(&h);
            }

            for t in 0..seq_len {
                let out_start = if self.batch_first {
                    b * seq_len * output_size + t * output_size
                } else {
                    t * batch_size * output_size + b * output_size
                };
                output[out_start..out_start + output_size].copy_from_slice(&layer_input[t]);
            }
        }

        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, output_size]
        } else {
            vec![seq_len, batch_size, output_size]
        };

        let h_flat: Vec<f32> = h_states.into_iter().flatten().collect();
        let state_shape = vec![total_layers, batch_size, self.hidden_size];

        (
            Tensor::from_slice(&output, &output_shape).unwrap(),
            Tensor::from_slice(&h_flat, &state_shape).unwrap()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn() {
        let x = Tensor::from_slice(&vec![1.0f32; 2 * 5 * 10], &[2, 5, 10]).unwrap();
        let rnn = RNN::new(10, 20, 2);
        let (output, hidden) = rnn.forward(&x, None);
        assert_eq!(output.dims(), &[2, 5, 20]);
        assert_eq!(hidden.dims(), &[2, 2, 20]);
    }

    #[test]
    fn test_lstm() {
        let x = Tensor::from_slice(&vec![1.0f32; 2 * 5 * 10], &[2, 5, 10]).unwrap();
        let lstm = LSTM::new(10, 20, 2);
        let (output, (h, c)) = lstm.forward(&x, None);
        assert_eq!(output.dims(), &[2, 5, 20]);
        assert_eq!(h.dims(), &[2, 2, 20]);
        assert_eq!(c.dims(), &[2, 2, 20]);
    }

    #[test]
    fn test_gru() {
        let x = Tensor::from_slice(&vec![1.0f32; 2 * 5 * 10], &[2, 5, 10]).unwrap();
        let gru = GRU::new(10, 20, 2);
        let (output, hidden) = gru.forward(&x, None);
        assert_eq!(output.dims(), &[2, 5, 20]);
        assert_eq!(hidden.dims(), &[2, 2, 20]);
    }
}



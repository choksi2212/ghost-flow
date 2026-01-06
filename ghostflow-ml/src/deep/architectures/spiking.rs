//! Spiking Neural Network Architectures - LIF, Izhikevich, STDP, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::Dense;

/// Leaky Integrate-and-Fire (LIF) Neuron
pub struct LIFNeuron {
    threshold: f32,
    decay: f32,
    reset_potential: f32,
    membrane_potential: Vec<f32>,
}

impl LIFNeuron {
    pub fn new(num_neurons: usize, threshold: f32, decay: f32) -> Self {
        LIFNeuron {
            threshold,
            decay,
            reset_potential: 0.0,
            membrane_potential: vec![0.0f32; num_neurons],
        }
    }

    pub fn forward(&mut self, input: &Tensor, dt: f32) -> Tensor {
        let batch_size = input.dims()[0];
        let num_neurons = input.dims()[1];
        let input_data = input.data_f32();
        
        let mut spikes = Vec::new();
        
        for b in 0..batch_size {
            for n in 0..num_neurons {
                let input_current = input_data[b * num_neurons + n];
                
                // Update membrane potential
                self.membrane_potential[n] = self.membrane_potential[n] * (1.0 - self.decay * dt) 
                    + input_current * dt;
                
                // Check for spike
                if self.membrane_potential[n] >= self.threshold {
                    spikes.push(1.0f32);
                    self.membrane_potential[n] = self.reset_potential;
                } else {
                    spikes.push(0.0f32);
                }
            }
        }
        
        Tensor::from_slice(&spikes, &[batch_size, num_neurons]).unwrap()
    }

    pub fn reset(&mut self) {
        for v in &mut self.membrane_potential {
            *v = self.reset_potential;
        }
    }
}

/// Izhikevich Neuron Model
pub struct IzhikevichNeuron {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    v: Vec<f32>,  // membrane potential
    u: Vec<f32>,  // recovery variable
}

impl IzhikevichNeuron {
    pub fn new(num_neurons: usize, a: f32, b: f32, c: f32, d: f32) -> Self {
        IzhikevichNeuron {
            a,
            b,
            c,
            d,
            v: vec![-65.0f32; num_neurons],
            u: vec![b * -65.0; num_neurons],
        }
    }

    pub fn forward(&mut self, input: &Tensor, dt: f32) -> Tensor {
        let batch_size = input.dims()[0];
        let num_neurons = input.dims()[1];
        let input_data = input.data_f32();
        
        let mut spikes = Vec::new();
        
        for b in 0..batch_size {
            for n in 0..num_neurons {
                let i = input_data[b * num_neurons + n];
                
                // Update equations
                let dv = (0.04 * self.v[n] * self.v[n] + 5.0 * self.v[n] + 140.0 - self.u[n] + i) * dt;
                let du = self.a * (self.b * self.v[n] - self.u[n]) * dt;
                
                self.v[n] += dv;
                self.u[n] += du;
                
                // Check for spike
                if self.v[n] >= 30.0 {
                    spikes.push(1.0f32);
                    self.v[n] = self.c;
                    self.u[n] += self.d;
                } else {
                    spikes.push(0.0f32);
                }
            }
        }
        
        Tensor::from_slice(&spikes, &[batch_size, num_neurons]).unwrap()
    }
}

/// Spiking Neural Network Layer
pub struct SpikingLayer {
    weights: Dense,
    neurons: LIFNeuron,
}

impl SpikingLayer {
    pub fn new(input_size: usize, output_size: usize, threshold: f32, decay: f32) -> Self {
        SpikingLayer {
            weights: Dense::new(input_size, output_size),
            neurons: LIFNeuron::new(output_size, threshold, decay),
        }
    }

    pub fn forward(&mut self, spikes: &Tensor, dt: f32, training: bool) -> Tensor {
        let weighted_input = self.weights.forward(spikes, training);
        self.neurons.forward(&weighted_input, dt)
    }

    pub fn reset(&mut self) {
        self.neurons.reset();
    }
}

/// Spiking Neural Network
pub struct SpikingNeuralNetwork {
    layers: Vec<SpikingLayer>,
    num_timesteps: usize,
}

impl SpikingNeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>, threshold: f32, decay: f32, num_timesteps: usize) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(SpikingLayer::new(layer_sizes[i], layer_sizes[i + 1], threshold, decay));
        }
        
        SpikingNeuralNetwork {
            layers,
            num_timesteps,
        }
    }

    pub fn forward(&mut self, input: &Tensor, dt: f32, training: bool) -> Tensor {
        // Reset all neurons
        for layer in &mut self.layers {
            layer.reset();
        }
        
        let mut spike_counts = vec![0.0f32; input.dims()[0] * self.layers.last().unwrap().neurons.membrane_potential.len()];
        
        // Simulate over time
        for _ in 0..self.num_timesteps {
            let mut spikes = input.clone();
            
            for layer in &mut self.layers {
                spikes = layer.forward(&spikes, dt, training);
            }
            
            // Accumulate spikes
            let spike_data = spikes.data_f32();
            for (i, &s) in spike_data.iter().enumerate() {
                spike_counts[i] += s;
            }
        }
        
        Tensor::from_slice(&spike_counts, spikes.dims()).unwrap()
    }
}

/// STDP (Spike-Timing-Dependent Plasticity) Learning
pub struct STDPLayer {
    weights: Vec<Vec<f32>>,
    pre_spike_times: Vec<f32>,
    post_spike_times: Vec<f32>,
    a_plus: f32,
    a_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
    input_size: usize,
    output_size: usize,
}

impl STDPLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let mut weights = Vec::new();
        for _ in 0..output_size {
            let row: Vec<f32> = (0..input_size)
                .map(|_| rng.gen::<f32>() * 0.1)
                .collect();
            weights.push(row);
        }
        
        STDPLayer {
            weights,
            pre_spike_times: vec![0.0f32; input_size],
            post_spike_times: vec![0.0f32; output_size],
            a_plus: 0.01,
            a_minus: 0.01,
            tau_plus: 20.0,
            tau_minus: 20.0,
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input_spikes: &Tensor) -> Tensor {
        let batch_size = input_spikes.dims()[0];
        let spike_data = input_spikes.data_f32();
        
        let mut output = vec![0.0f32; batch_size * self.output_size];
        
        for b in 0..batch_size {
            for j in 0..self.output_size {
                let mut sum = 0.0f32;
                for i in 0..self.input_size {
                    let spike = spike_data[b * self.input_size + i];
                    sum += self.weights[j][i] * spike;
                }
                output[b * self.output_size + j] = sum;
            }
        }
        
        Tensor::from_slice(&output, &[batch_size, self.output_size]).unwrap()
    }

    pub fn update_weights(&mut self, pre_spikes: &[f32], post_spikes: &[f32], current_time: f32) {
        // Update spike times
        for (i, &spike) in pre_spikes.iter().enumerate() {
            if spike > 0.5 {
                self.pre_spike_times[i] = current_time;
            }
        }
        
        for (j, &spike) in post_spikes.iter().enumerate() {
            if spike > 0.5 {
                self.post_spike_times[j] = current_time;
            }
        }
        
        // STDP weight update
        for j in 0..self.output_size {
            if post_spikes[j] > 0.5 {
                for i in 0..self.input_size {
                    let dt = current_time - self.pre_spike_times[i];
                    if dt > 0.0 && dt < 100.0 {
                        // LTP (Long-Term Potentiation)
                        let dw = self.a_plus * (-dt / self.tau_plus).exp();
                        self.weights[j][i] += dw;
                        self.weights[j][i] = self.weights[j][i].min(1.0);
                    }
                }
            }
        }
        
        for i in 0..self.input_size {
            if pre_spikes[i] > 0.5 {
                for j in 0..self.output_size {
                    let dt = current_time - self.post_spike_times[j];
                    if dt > 0.0 && dt < 100.0 {
                        // LTD (Long-Term Depression)
                        let dw = -self.a_minus * (-dt / self.tau_minus).exp();
                        self.weights[j][i] += dw;
                        self.weights[j][i] = self.weights[j][i].max(0.0);
                    }
                }
            }
        }
    }
}

/// Liquid State Machine (LSM)
pub struct LiquidStateMachine {
    reservoir: Vec<LIFNeuron>,
    connections: Vec<Vec<f32>>,
    readout: Dense,
    num_neurons: usize,
}

impl LiquidStateMachine {
    pub fn new(input_size: usize, reservoir_size: usize, output_size: usize, connectivity: f32) -> Self {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        // Create reservoir neurons
        let mut reservoir = Vec::new();
        for _ in 0..reservoir_size {
            reservoir.push(LIFNeuron::new(1, 1.0, 0.1));
        }
        
        // Create random connections
        let mut connections = vec![vec![0.0f32; reservoir_size]; reservoir_size];
        for i in 0..reservoir_size {
            for j in 0..reservoir_size {
                if rng.gen::<f32>() < connectivity {
                    connections[i][j] = rng.gen::<f32>() * 0.2 - 0.1;
                }
            }
        }
        
        LiquidStateMachine {
            reservoir,
            connections,
            readout: Dense::new(reservoir_size, output_size),
            num_neurons: reservoir_size,
        }
    }

    pub fn forward(&mut self, input: &Tensor, num_steps: usize, dt: f32, training: bool) -> Tensor {
        let batch_size = input.dims()[0];
        let mut states = vec![0.0f32; batch_size * self.num_neurons];
        
        // Simulate reservoir dynamics
        for _ in 0..num_steps {
            // Update each neuron
            for n in 0..self.num_neurons {
                let mut input_current = 0.0f32;
                
                // Recurrent connections
                for m in 0..self.num_neurons {
                    input_current += self.connections[n][m] * states[m];
                }
                
                // External input (simplified)
                let input_tensor = Tensor::from_slice(&[input_current], &[1, 1]).unwrap();
                let spike = self.reservoir[n].forward(&input_tensor, dt);
                states[n] = spike.data_f32()[0];
            }
        }
        
        // Readout
        let state_tensor = Tensor::from_slice(&states, &[batch_size, self.num_neurons]).unwrap();
        self.readout.forward(&state_tensor, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_neuron() {
        let mut neuron = LIFNeuron::new(10, 1.0, 0.1);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 10], &[1, 10]).unwrap();
        let spikes = neuron.forward(&input, 0.1);
        assert_eq!(spikes.dims(), &[1, 10]);
    }

    #[test]
    fn test_spiking_network() {
        let mut snn = SpikingNeuralNetwork::new(vec![784, 256, 10], 1.0, 0.1, 100);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 784], &[1, 784]).unwrap();
        let output = snn.forward(&input, 0.1, false);
        assert_eq!(output.dims()[1], 10);
    }

    #[test]
    fn test_izhikevich_neuron() {
        let mut neuron = IzhikevichNeuron::new(10, 0.02, 0.2, -65.0, 8.0);
        let input = Tensor::from_slice(&vec![10.0f32; 1 * 10], &[1, 10]).unwrap();
        let spikes = neuron.forward(&input, 0.5);
        assert_eq!(spikes.dims(), &[1, 10]);
    }
}



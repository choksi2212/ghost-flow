//! Autoencoder Architectures - Standard, Denoising, Sparse, Contractive, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense, BatchNorm2d};
use crate::deep::activations::{ReLU, Sigmoid, Tanh};

/// Standard Autoencoder
pub struct Autoencoder {
    encoder: Encoder,
    decoder: Decoder,
    latent_dim: usize,
}

struct Encoder {
    layers: Vec<Dense>,
}

impl Encoder {
    fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize) -> Self {
        let mut layers = Vec::new();
        
        layers.push(Dense::new(input_dim, hidden_dims[0]));
        for i in 0..hidden_dims.len() - 1 {
            layers.push(Dense::new(hidden_dims[i], hidden_dims[i + 1]));
        }
        layers.push(Dense::new(hidden_dims[hidden_dims.len() - 1], latent_dim));
        
        Encoder { layers }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            out = layer.forward(&out, training);
            if i < self.layers.len() - 1 {
                out = ReLU::new().forward(&out);
            }
        }
        
        out
    }
}

struct Decoder {
    layers: Vec<Dense>,
}

impl Decoder {
    fn new(latent_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        let mut layers = Vec::new();
        
        layers.push(Dense::new(latent_dim, hidden_dims[0]));
        for i in 0..hidden_dims.len() - 1 {
            layers.push(Dense::new(hidden_dims[i], hidden_dims[i + 1]));
        }
        layers.push(Dense::new(hidden_dims[hidden_dims.len() - 1], output_dim));
        
        Decoder { layers }
    }

    fn forward(&mut self, z: &Tensor, training: bool) -> Tensor {
        let mut out = z.clone();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            out = layer.forward(&out, training);
            if i < self.layers.len() - 1 {
                out = ReLU::new().forward(&out);
            } else {
                out = Sigmoid::new().forward(&out);
            }
        }
        
        out
    }
}

impl Autoencoder {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize) -> Self {
        let mut decoder_hidden = hidden_dims.clone();
        decoder_hidden.reverse();
        
        Autoencoder {
            encoder: Encoder::new(input_dim, hidden_dims, latent_dim),
            decoder: Decoder::new(latent_dim, decoder_hidden, input_dim),
            latent_dim,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let z = self.encoder.forward(x, training);
        self.decoder.forward(&z, training)
    }

    pub fn encode(&mut self, x: &Tensor, training: bool) -> Tensor {
        self.encoder.forward(x, training)
    }

    pub fn decode(&mut self, z: &Tensor, training: bool) -> Tensor {
        self.decoder.forward(z, training)
    }
}

/// Denoising Autoencoder
pub struct DenoisingAutoencoder {
    encoder: Encoder,
    decoder: Decoder,
    noise_factor: f32,
}

impl DenoisingAutoencoder {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize, noise_factor: f32) -> Self {
        let mut decoder_hidden = hidden_dims.clone();
        decoder_hidden.reverse();
        
        DenoisingAutoencoder {
            encoder: Encoder::new(input_dim, hidden_dims, latent_dim),
            decoder: Decoder::new(latent_dim, decoder_hidden, input_dim),
            noise_factor,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let noisy_x = if training {
            self.add_noise(x)
        } else {
            x.clone()
        };
        
        let z = self.encoder.forward(&noisy_x, training);
        self.decoder.forward(&z, training)
    }

    fn add_noise(&self, x: &Tensor) -> Tensor {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let data = x.data_f32();
        let noisy: Vec<f32> = data.iter()
            .map(|&val| {
                let noise = (rng.gen::<f32>() * 2.0 - 1.0) * self.noise_factor;
                (val + noise).max(0.0).min(1.0)
            })
            .collect();
        
        Tensor::from_slice(&noisy, x.dims()).unwrap()
    }
}

/// Sparse Autoencoder
pub struct SparseAutoencoder {
    encoder: Encoder,
    decoder: Decoder,
    sparsity_param: f32,
    sparsity_weight: f32,
}

impl SparseAutoencoder {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize, 
               sparsity_param: f32, sparsity_weight: f32) -> Self {
        let mut decoder_hidden = hidden_dims.clone();
        decoder_hidden.reverse();
        
        SparseAutoencoder {
            encoder: Encoder::new(input_dim, hidden_dims, latent_dim),
            decoder: Decoder::new(latent_dim, decoder_hidden, input_dim),
            sparsity_param,
            sparsity_weight,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let z = self.encoder.forward(x, training);
        let recon = self.decoder.forward(&z, training);
        (recon, z)
    }

    pub fn compute_sparsity_loss(&self, z: &Tensor) -> f32 {
        let z_data = z.data_f32();
        let batch_size = z.dims()[0];
        let latent_dim = z.dims()[1];
        
        // Compute average activation
        let mut avg_activation = vec![0.0f32; latent_dim];
        for b in 0..batch_size {
            for d in 0..latent_dim {
                avg_activation[d] += z_data[b * latent_dim + d];
            }
        }
        
        for d in 0..latent_dim {
            avg_activation[d] /= batch_size as f32;
        }
        
        // KL divergence
        let mut kl_div = 0.0f32;
        for &rho_hat in &avg_activation {
            if rho_hat > 0.0 && rho_hat < 1.0 {
                kl_div += self.sparsity_param * (self.sparsity_param / rho_hat).ln()
                    + (1.0 - self.sparsity_param) * ((1.0 - self.sparsity_param) / (1.0 - rho_hat)).ln();
            }
        }
        
        self.sparsity_weight * kl_div
    }
}

/// Contractive Autoencoder
pub struct ContractiveAutoencoder {
    encoder: Encoder,
    decoder: Decoder,
    contraction_weight: f32,
}

impl ContractiveAutoencoder {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize, contraction_weight: f32) -> Self {
        let mut decoder_hidden = hidden_dims.clone();
        decoder_hidden.reverse();
        
        ContractiveAutoencoder {
            encoder: Encoder::new(input_dim, hidden_dims, latent_dim),
            decoder: Decoder::new(latent_dim, decoder_hidden, input_dim),
            contraction_weight,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let z = self.encoder.forward(x, training);
        let recon = self.decoder.forward(&z, training);
        (recon, z)
    }

    pub fn compute_contraction_loss(&self, z: &Tensor) -> f32 {
        // Simplified Frobenius norm of Jacobian
        let z_data = z.data_f32();
        let mut norm = 0.0f32;
        
        for &val in z_data {
            norm += val * val;
        }
        
        self.contraction_weight * norm
    }
}

/// Convolutional Autoencoder
pub struct ConvAutoencoder {
    encoder: ConvEncoder,
    decoder: ConvDecoder,
}

struct ConvEncoder {
    conv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
}

impl ConvEncoder {
    fn new() -> Self {
        ConvEncoder {
            conv_layers: vec![
                Conv2d::new(1, 32, (3, 3)).stride((2, 2)).padding((1, 1)),
                Conv2d::new(32, 64, (3, 3)).stride((2, 2)).padding((1, 1)),
                Conv2d::new(64, 128, (3, 3)).stride((2, 2)).padding((1, 1)),
            ],
            bn_layers: vec![
                BatchNorm2d::new(32),
                BatchNorm2d::new(64),
                BatchNorm2d::new(128),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (conv, bn) in self.conv_layers.iter_mut().zip(self.bn_layers.iter_mut()) {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = ReLU::new().forward(&out);
        }
        
        out
    }
}

struct ConvDecoder {
    deconv_layers: Vec<Conv2d>,
    bn_layers: Vec<BatchNorm2d>,
}

impl ConvDecoder {
    fn new() -> Self {
        ConvDecoder {
            deconv_layers: vec![
                Conv2d::new(128, 64, (3, 3)).padding((1, 1)),
                Conv2d::new(64, 32, (3, 3)).padding((1, 1)),
                Conv2d::new(32, 1, (3, 3)).padding((1, 1)),
            ],
            bn_layers: vec![
                BatchNorm2d::new(64),
                BatchNorm2d::new(32),
            ],
        }
    }

    fn forward(&mut self, z: &Tensor, training: bool) -> Tensor {
        let mut out = z.clone();
        
        for i in 0..self.deconv_layers.len() {
            out = self.upsample(&out);
            out = self.deconv_layers[i].forward(&out, training);
            
            if i < self.bn_layers.len() {
                out = self.bn_layers[i].forward(&out, training);
                out = ReLU::new().forward(&out);
            } else {
                out = Sigmoid::new().forward(&out);
            }
        }
        
        out
    }

    fn upsample(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();

        let new_height = height * 2;
        let new_width = width * 2;
        let mut result = vec![0.0f32; batch * channels * new_height * new_width];

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let val = data[((b * channels + c) * height + h) * width + w];
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let new_h = h * 2 + dh;
                                let new_w = w * 2 + dw;
                                let idx = ((b * channels + c) * new_height + new_h) * new_width + new_w;
                                result[idx] = val;
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, new_height, new_width]).unwrap()
    }
}

impl ConvAutoencoder {
    pub fn new() -> Self {
        ConvAutoencoder {
            encoder: ConvEncoder::new(),
            decoder: ConvDecoder::new(),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let z = self.encoder.forward(x, training);
        self.decoder.forward(&z, training)
    }
}

/// Variational Autoencoder (already in vae.rs, but adding here for completeness)
pub struct VariationalAutoencoder {
    encoder_mean: Encoder,
    encoder_logvar: Encoder,
    decoder: Decoder,
}

impl VariationalAutoencoder {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize) -> Self {
        let mut decoder_hidden = hidden_dims.clone();
        decoder_hidden.reverse();
        
        VariationalAutoencoder {
            encoder_mean: Encoder::new(input_dim, hidden_dims.clone(), latent_dim),
            encoder_logvar: Encoder::new(input_dim, hidden_dims, latent_dim),
            decoder: Decoder::new(latent_dim, decoder_hidden, input_dim),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let mu = self.encoder_mean.forward(x, training);
        let logvar = self.encoder_logvar.forward(x, training);
        let z = self.reparameterize(&mu, &logvar);
        let recon = self.decoder.forward(&z, training);
        (recon, mu, logvar)
    }

    fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Tensor {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let mu_data = mu.data_f32();
        let logvar_data = logvar.data_f32();
        
        let z: Vec<f32> = mu_data.iter()
            .zip(logvar_data.iter())
            .map(|(&m, &lv)| {
                let std = (lv * 0.5).exp();
                let eps: f32 = rng.gen::<f32>() * 2.0 - 1.0;
                m + std * eps
            })
            .collect();
        
        Tensor::from_slice(&z, mu.dims()).unwrap()
    }
}

/// Adversarial Autoencoder
pub struct AdversarialAutoencoder {
    encoder: Encoder,
    decoder: Decoder,
    discriminator: Discriminator,
}

struct Discriminator {
    layers: Vec<Dense>,
}

impl Discriminator {
    fn new(latent_dim: usize) -> Self {
        Discriminator {
            layers: vec![
                Dense::new(latent_dim, 128),
                Dense::new(128, 64),
                Dense::new(64, 1),
            ],
        }
    }

    fn forward(&mut self, z: &Tensor, training: bool) -> Tensor {
        let mut out = z.clone();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            out = layer.forward(&out, training);
            if i < self.layers.len() - 1 {
                out = ReLU::new().forward(&out);
            } else {
                out = Sigmoid::new().forward(&out);
            }
        }
        
        out
    }
}

impl AdversarialAutoencoder {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize) -> Self {
        let mut decoder_hidden = hidden_dims.clone();
        decoder_hidden.reverse();
        
        AdversarialAutoencoder {
            encoder: Encoder::new(input_dim, hidden_dims, latent_dim),
            decoder: Decoder::new(latent_dim, decoder_hidden, input_dim),
            discriminator: Discriminator::new(latent_dim),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let z = self.encoder.forward(x, training);
        let recon = self.decoder.forward(&z, training);
        let disc_out = self.discriminator.forward(&z, training);
        (recon, z, disc_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoencoder() {
        let mut ae = Autoencoder::new(784, vec![512, 256], 64);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 784], &[2, 784]).unwrap();
        let output = ae.forward(&input, false);
        assert_eq!(output.dims()[1], 784);
    }

    #[test]
    fn test_denoising_autoencoder() {
        let mut dae = DenoisingAutoencoder::new(784, vec![512, 256], 64, 0.3);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 784], &[2, 784]).unwrap();
        let output = dae.forward(&input, false);
        assert_eq!(output.dims()[1], 784);
    }

    #[test]
    fn test_sparse_autoencoder() {
        let mut sae = SparseAutoencoder::new(784, vec![512, 256], 64, 0.05, 0.1);
        let input = Tensor::from_slice(&vec![0.5f32; 2 * 784], &[2, 784]).unwrap();
        let (recon, z) = sae.forward(&input, false);
        assert_eq!(recon.dims()[1], 784);
        assert_eq!(z.dims()[1], 64);
    }
}

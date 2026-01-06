//! VAE Architectures - Variational Autoencoders and variants

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense, BatchNorm2d};
use crate::deep::activations::{ReLU, Sigmoid, Tanh};

/// VAE Encoder
pub struct VAEEncoder {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    fc_mu: Dense,
    fc_logvar: Dense,
    latent_dim: usize,
}

impl VAEEncoder {
    pub fn new(in_channels: usize, latent_dim: usize) -> Self {
        VAEEncoder {
            conv1: Conv2d::new(in_channels, 32, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(32),
            conv2: Conv2d::new(32, 64, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn2: BatchNorm2d::new(64),
            conv3: Conv2d::new(64, 128, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn3: BatchNorm2d::new(128),
            fc_mu: Dense::new(128 * 4 * 4, latent_dim),
            fc_logvar: Dense::new(128 * 4 * 4, latent_dim),
            latent_dim,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv3.forward(&out, training);
        out = self.bn3.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        // Flatten
        let batch_size = out.dims()[0];
        let flat_size = out.data_f32().len() / batch_size;
        out = Tensor::from_slice(out.data_f32(), &[batch_size, flat_size]).unwrap();
        
        let mu = self.fc_mu.forward(&out, training);
        let logvar = self.fc_logvar.forward(&out, training);
        
        (mu, logvar)
    }
}

/// VAE Decoder
pub struct VAEDecoder {
    fc: Dense,
    deconv1: Conv2d,
    bn1: BatchNorm2d,
    deconv2: Conv2d,
    bn2: BatchNorm2d,
    deconv3: Conv2d,
    out_channels: usize,
}

impl VAEDecoder {
    pub fn new(latent_dim: usize, out_channels: usize) -> Self {
        VAEDecoder {
            fc: Dense::new(latent_dim, 128 * 4 * 4),
            deconv1: Conv2d::new(128, 64, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(64),
            deconv2: Conv2d::new(64, 32, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn2: BatchNorm2d::new(32),
            deconv3: Conv2d::new(32, out_channels, (4, 4)).stride((2, 2)).padding((1, 1)),
            out_channels,
        }
    }

    pub fn forward(&mut self, z: &Tensor, training: bool) -> Tensor {
        let mut out = self.fc.forward(z, training);
        
        // Reshape
        let batch_size = z.dims()[0];
        out = Tensor::from_slice(out.data_f32(), &[batch_size, 128, 4, 4]).unwrap();
        
        out = self.deconv1.forward(&out, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.deconv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.deconv3.forward(&out, training);
        Sigmoid::new().forward(&out)
    }
}

/// Standard VAE
pub struct VAE {
    encoder: VAEEncoder,
    decoder: VAEDecoder,
    latent_dim: usize,
}

impl VAE {
    pub fn new(in_channels: usize, latent_dim: usize) -> Self {
        VAE {
            encoder: VAEEncoder::new(in_channels, latent_dim),
            decoder: VAEDecoder::new(latent_dim, in_channels),
            latent_dim,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let (mu, logvar) = self.encoder.forward(x, training);
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

/// Beta-VAE (with adjustable beta parameter)
pub struct BetaVAE {
    encoder: VAEEncoder,
    decoder: VAEDecoder,
    latent_dim: usize,
    beta: f32,
}

impl BetaVAE {
    pub fn new(in_channels: usize, latent_dim: usize, beta: f32) -> Self {
        BetaVAE {
            encoder: VAEEncoder::new(in_channels, latent_dim),
            decoder: VAEDecoder::new(latent_dim, in_channels),
            latent_dim,
            beta,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let (mu, logvar) = self.encoder.forward(x, training);
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
                m + std * eps * self.beta
            })
            .collect();
        
        Tensor::from_slice(&z, mu.dims()).unwrap()
    }
}

/// Conditional VAE
pub struct ConditionalVAE {
    encoder: ConditionalVAEEncoder,
    decoder: ConditionalVAEDecoder,
    latent_dim: usize,
    num_classes: usize,
}

struct ConditionalVAEEncoder {
    label_embedding: Dense,
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    fc_mu: Dense,
    fc_logvar: Dense,
}

impl ConditionalVAEEncoder {
    fn new(in_channels: usize, latent_dim: usize, num_classes: usize) -> Self {
        ConditionalVAEEncoder {
            label_embedding: Dense::new(num_classes, in_channels * 32 * 32),
            conv1: Conv2d::new(in_channels * 2, 32, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(32),
            conv2: Conv2d::new(32, 64, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn2: BatchNorm2d::new(64),
            conv3: Conv2d::new(64, 128, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn3: BatchNorm2d::new(128),
            fc_mu: Dense::new(128 * 4 * 4, latent_dim),
            fc_logvar: Dense::new(128 * 4 * 4, latent_dim),
        }
    }

    fn forward(&mut self, x: &Tensor, labels: &Tensor, training: bool) -> (Tensor, Tensor) {
        // Embed labels and reshape
        let label_embed = self.label_embedding.forward(labels, training);
        let batch_size = x.dims()[0];
        let in_channels = x.dims()[1];
        let height = x.dims()[2];
        let width = x.dims()[3];
        
        let label_reshaped = Tensor::from_slice(label_embed.data_f32(), 
                                                 &[batch_size, in_channels, height, width]).unwrap();
        
        // Concatenate input with label embedding
        let x_cond = self.concatenate(x, &label_reshaped);
        
        let mut out = self.conv1.forward(&x_cond, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv3.forward(&out, training);
        out = self.bn3.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        let flat_size = out.data_f32().len() / batch_size;
        out = Tensor::from_slice(out.data_f32(), &[batch_size, flat_size]).unwrap();
        
        let mu = self.fc_mu.forward(&out, training);
        let logvar = self.fc_logvar.forward(&out, training);
        
        (mu, logvar)
    }

    fn concatenate(&self, x1: &Tensor, x2: &Tensor) -> Tensor {
        let dims1 = x1.dims();
        let dims2 = x2.dims();
        let batch = dims1[0];
        let channels1 = dims1[1];
        let channels2 = dims2[1];
        let height = dims1[2];
        let width = dims1[3];

        let total_channels = channels1 + channels2;
        let mut result = Vec::new();

        for b in 0..batch {
            for c in 0..channels1 {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels1 + c) * height + h) * width + w;
                        result.push(x1.data_f32()[idx]);
                    }
                }
            }
            for c in 0..channels2 {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels2 + c) * height + h) * width + w;
                        result.push(x2.data_f32()[idx]);
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, total_channels, height, width]).unwrap()
    }
}

struct ConditionalVAEDecoder {
    label_embedding: Dense,
    fc: Dense,
    deconv1: Conv2d,
    bn1: BatchNorm2d,
    deconv2: Conv2d,
    bn2: BatchNorm2d,
    deconv3: Conv2d,
}

impl ConditionalVAEDecoder {
    fn new(latent_dim: usize, out_channels: usize, num_classes: usize) -> Self {
        ConditionalVAEDecoder {
            label_embedding: Dense::new(num_classes, latent_dim),
            fc: Dense::new(latent_dim * 2, 128 * 4 * 4),
            deconv1: Conv2d::new(128, 64, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(64),
            deconv2: Conv2d::new(64, 32, (4, 4)).stride((2, 2)).padding((1, 1)),
            bn2: BatchNorm2d::new(32),
            deconv3: Conv2d::new(32, out_channels, (4, 4)).stride((2, 2)).padding((1, 1)),
        }
    }

    fn forward(&mut self, z: &Tensor, labels: &Tensor, training: bool) -> Tensor {
        let label_embed = self.label_embedding.forward(labels, training);
        let z_cond = self.concatenate_vectors(z, &label_embed);
        
        let mut out = self.fc.forward(&z_cond, training);
        
        let batch_size = z.dims()[0];
        out = Tensor::from_slice(out.data_f32(), &[batch_size, 128, 4, 4]).unwrap();
        
        out = self.deconv1.forward(&out, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.deconv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.deconv3.forward(&out, training);
        Sigmoid::new().forward(&out)
    }

    fn concatenate_vectors(&self, x1: &Tensor, x2: &Tensor) -> Tensor {
        let data1 = x1.data_f32();
        let data2 = x2.data_f32();
        let batch_size = x1.dims()[0];
        let dim1 = x1.dims()[1];
        let dim2 = x2.dims()[1];
        
        let mut result = Vec::new();
        for b in 0..batch_size {
            for i in 0..dim1 {
                result.push(data1[b * dim1 + i]);
            }
            for i in 0..dim2 {
                result.push(data2[b * dim2 + i]);
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, dim1 + dim2]).unwrap()
    }
}

impl ConditionalVAE {
    pub fn new(in_channels: usize, latent_dim: usize, num_classes: usize) -> Self {
        ConditionalVAE {
            encoder: ConditionalVAEEncoder::new(in_channels, latent_dim, num_classes),
            decoder: ConditionalVAEDecoder::new(latent_dim, in_channels, num_classes),
            latent_dim,
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, labels: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let (mu, logvar) = self.encoder.forward(x, labels, training);
        let z = self.reparameterize(&mu, &logvar);
        let recon = self.decoder.forward(&z, labels, training);
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

/// Vector Quantized VAE (VQ-VAE)
pub struct VQVAE {
    encoder: VAEEncoder,
    decoder: VAEDecoder,
    codebook: Vec<Vec<f32>>,
    latent_dim: usize,
    num_embeddings: usize,
}

impl VQVAE {
    pub fn new(in_channels: usize, latent_dim: usize, num_embeddings: usize) -> Self {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        // Initialize codebook
        let mut codebook = Vec::new();
        for _ in 0..num_embeddings {
            let embedding: Vec<f32> = (0..latent_dim)
                .map(|_| rng.gen::<f32>() * 0.02 - 0.01)
                .collect();
            codebook.push(embedding);
        }
        
        VQVAE {
            encoder: VAEEncoder::new(in_channels, latent_dim),
            decoder: VAEDecoder::new(latent_dim, in_channels),
            codebook,
            latent_dim,
            num_embeddings,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let (z_e, _) = self.encoder.forward(x, training);
        let z_q = self.quantize(&z_e);
        self.decoder.forward(&z_q, training)
    }

    fn quantize(&self, z_e: &Tensor) -> Tensor {
        let z_e_data = z_e.data_f32();
        let batch_size = z_e.dims()[0];
        let latent_dim = z_e.dims()[1];
        
        let mut z_q_data = Vec::new();
        
        for b in 0..batch_size {
            let offset = b * latent_dim;
            let z_vec: Vec<f32> = (0..latent_dim)
                .map(|i| z_e_data[offset + i])
                .collect();
            
            // Find nearest codebook entry
            let mut min_dist = f32::MAX;
            let mut best_idx = 0;
            
            for (idx, embedding) in self.codebook.iter().enumerate() {
                let dist: f32 = z_vec.iter()
                    .zip(embedding.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = idx;
                }
            }
            
            z_q_data.extend_from_slice(&self.codebook[best_idx]);
        }
        
        Tensor::from_slice(&z_q_data, z_e.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae() {
        let mut vae = VAE::new(3, 128);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let (recon, mu, logvar) = vae.forward(&input, false);
        assert_eq!(recon.dims()[1], 3);
        assert_eq!(mu.dims()[1], 128);
        assert_eq!(logvar.dims()[1], 128);
    }

    #[test]
    fn test_beta_vae() {
        let mut vae = BetaVAE::new(3, 128, 4.0);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let (recon, _, _) = vae.forward(&input, false);
        assert_eq!(recon.dims()[1], 3);
    }

    #[test]
    fn test_vqvae() {
        let mut vqvae = VQVAE::new(3, 128, 512);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let output = vqvae.forward(&input, false);
        assert_eq!(output.dims()[1], 3);
    }
}



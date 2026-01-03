//! Diffusion Models - DDPM, DDIM, Latent Diffusion, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense, BatchNorm2d};
use crate::deep::activations::ReLU;

/// Time Embedding for Diffusion Models
pub struct TimeEmbedding {
    fc1: Dense,
    fc2: Dense,
    dim: usize,
}

impl TimeEmbedding {
    pub fn new(dim: usize) -> Self {
        TimeEmbedding {
            fc1: Dense::new(dim, dim * 4),
            fc2: Dense::new(dim * 4, dim),
            dim,
        }
    }

    pub fn forward(&mut self, t: &Tensor, training: bool) -> Tensor {
        let mut out = self.fc1.forward(t, training);
        out = ReLU::new().forward(&out);
        self.fc2.forward(&out, training)
    }

    pub fn sinusoidal_embedding(&self, timesteps: &[usize], max_period: usize) -> Tensor {
        let half_dim = self.dim / 2;
        let mut embeddings = Vec::new();
        
        for &t in timesteps {
            for i in 0..half_dim {
                let freq = (-(i as f32) * (max_period as f32).ln() / (half_dim as f32)).exp();
                let arg = t as f32 * freq;
                embeddings.push(arg.sin());
            }
            for i in 0..half_dim {
                let freq = (-(i as f32) * (max_period as f32).ln() / (half_dim as f32)).exp();
                let arg = t as f32 * freq;
                embeddings.push(arg.cos());
            }
        }
        
        Tensor::from_slice(&embeddings, &[timesteps.len(), self.dim]).unwrap()
    }
}

/// Residual Block for U-Net in Diffusion
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    time_mlp: Dense,
}

impl ResidualBlock {
    pub fn new(channels: usize, time_emb_dim: usize) -> Self {
        ResidualBlock {
            conv1: Conv2d::new(channels, channels, (3, 3)).padding((1, 1)),
            bn1: BatchNorm2d::new(channels),
            conv2: Conv2d::new(channels, channels, (3, 3)).padding((1, 1)),
            bn2: BatchNorm2d::new(channels),
            time_mlp: Dense::new(time_emb_dim, channels),
        }
    }

    pub fn forward(&mut self, x: &Tensor, time_emb: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        // Add time embedding
        let time_out = self.time_mlp.forward(time_emb, training);
        out = self.add_time_embedding(&out, &time_out);
        
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        
        // Add residual
        let out_data = out.data_f32();
        let id_data = identity.data_f32();
        let result: Vec<f32> = out_data.iter()
            .zip(id_data.iter())
            .map(|(&o, &i)| o + i)
            .collect();
        
        Tensor::from_slice(&result, out.dims()).unwrap()
    }

    fn add_time_embedding(&self, x: &Tensor, time_emb: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let time_data = time_emb.data_f32();
        
        let batch_size = x.dims()[0];
        let channels = x.dims()[1];
        let height = x.dims()[2];
        let width = x.dims()[3];
        
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            for c in 0..channels {
                let time_val = time_data[b * channels + c];
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        result.push(x_data[idx] + time_val);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// U-Net for Diffusion Models
pub struct DiffusionUNet {
    time_embedding: TimeEmbedding,
    
    // Encoder
    enc_conv1: Conv2d,
    enc_res1: ResidualBlock,
    enc_conv2: Conv2d,
    enc_res2: ResidualBlock,
    enc_conv3: Conv2d,
    enc_res3: ResidualBlock,
    
    // Bottleneck
    bottleneck: ResidualBlock,
    
    // Decoder
    dec_conv3: Conv2d,
    dec_res3: ResidualBlock,
    dec_conv2: Conv2d,
    dec_res2: ResidualBlock,
    dec_conv1: Conv2d,
    dec_res1: ResidualBlock,
    
    // Output
    out_conv: Conv2d,
}

impl DiffusionUNet {
    pub fn new(in_channels: usize, out_channels: usize, time_emb_dim: usize) -> Self {
        DiffusionUNet {
            time_embedding: TimeEmbedding::new(time_emb_dim),
            
            enc_conv1: Conv2d::new(in_channels, 64, (3, 3)).padding((1, 1)),
            enc_res1: ResidualBlock::new(64, time_emb_dim),
            enc_conv2: Conv2d::new(64, 128, (3, 3)).stride((2, 2)).padding((1, 1)),
            enc_res2: ResidualBlock::new(128, time_emb_dim),
            enc_conv3: Conv2d::new(128, 256, (3, 3)).stride((2, 2)).padding((1, 1)),
            enc_res3: ResidualBlock::new(256, time_emb_dim),
            
            bottleneck: ResidualBlock::new(256, time_emb_dim),
            
            dec_conv3: Conv2d::new(256, 128, (3, 3)).padding((1, 1)),
            dec_res3: ResidualBlock::new(128, time_emb_dim),
            dec_conv2: Conv2d::new(128, 64, (3, 3)).padding((1, 1)),
            dec_res2: ResidualBlock::new(64, time_emb_dim),
            dec_conv1: Conv2d::new(64, 64, (3, 3)).padding((1, 1)),
            dec_res1: ResidualBlock::new(64, time_emb_dim),
            
            out_conv: Conv2d::new(64, out_channels, (1, 1)),
        }
    }

    pub fn forward(&mut self, x: &Tensor, timesteps: &[usize], training: bool) -> Tensor {
        // Get time embeddings
        let t_emb = self.time_embedding.sinusoidal_embedding(timesteps, 10000);
        let time_emb = self.time_embedding.forward(&t_emb, training);
        
        // Encoder
        let mut enc1 = self.enc_conv1.forward(x, training);
        enc1 = self.enc_res1.forward(&enc1, &time_emb, training);
        
        let mut enc2 = self.enc_conv2.forward(&enc1, training);
        enc2 = self.enc_res2.forward(&enc2, &time_emb, training);
        
        let mut enc3 = self.enc_conv3.forward(&enc2, training);
        enc3 = self.enc_res3.forward(&enc3, &time_emb, training);
        
        // Bottleneck
        let mut bottleneck = self.bottleneck.forward(&enc3, &time_emb, training);
        
        // Decoder
        bottleneck = self.upsample(&bottleneck);
        let mut dec3 = self.dec_conv3.forward(&bottleneck, training);
        dec3 = self.add_skip(&dec3, &enc2);
        dec3 = self.dec_res3.forward(&dec3, &time_emb, training);
        
        dec3 = self.upsample(&dec3);
        let mut dec2 = self.dec_conv2.forward(&dec3, training);
        dec2 = self.add_skip(&dec2, &enc1);
        dec2 = self.dec_res2.forward(&dec2, &time_emb, training);
        
        let mut dec1 = self.dec_conv1.forward(&dec2, training);
        dec1 = self.dec_res1.forward(&dec1, &time_emb, training);
        
        self.out_conv.forward(&dec1, training)
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

    fn add_skip(&self, x: &Tensor, skip: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let skip_data = skip.data_f32();
        let result: Vec<f32> = x_data.iter()
            .zip(skip_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// DDPM (Denoising Diffusion Probabilistic Model)
pub struct DDPM {
    unet: DiffusionUNet,
    num_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
}

impl DDPM {
    pub fn new(in_channels: usize, num_timesteps: usize) -> Self {
        DDPM {
            unet: DiffusionUNet::new(in_channels, in_channels, 256),
            num_timesteps,
            beta_start: 0.0001,
            beta_end: 0.02,
        }
    }

    pub fn forward(&mut self, x: &Tensor, timesteps: &[usize], training: bool) -> Tensor {
        self.unet.forward(x, timesteps, training)
    }

    pub fn get_beta_schedule(&self) -> Vec<f32> {
        let mut betas = Vec::new();
        for t in 0..self.num_timesteps {
            let beta = self.beta_start + (self.beta_end - self.beta_start) * 
                       (t as f32 / self.num_timesteps as f32);
            betas.push(beta);
        }
        betas
    }

    pub fn add_noise(&self, x: &Tensor, t: usize) -> Tensor {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let betas = self.get_beta_schedule();
        let alpha = 1.0 - betas[t];
        let alpha_bar: f32 = (0..=t).map(|i| 1.0 - betas[i]).product();
        
        let x_data = x.data_f32();
        let noise: Vec<f32> = (0..x_data.len())
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        
        let noisy: Vec<f32> = x_data.iter()
            .zip(noise.iter())
            .map(|(&x_val, &n)| alpha_bar.sqrt() * x_val + (1.0 - alpha_bar).sqrt() * n)
            .collect();
        
        Tensor::from_slice(&noisy, x.dims()).unwrap()
    }
}

/// DDIM (Denoising Diffusion Implicit Model)
pub struct DDIM {
    unet: DiffusionUNet,
    num_timesteps: usize,
    eta: f32,
}

impl DDIM {
    pub fn new(in_channels: usize, num_timesteps: usize, eta: f32) -> Self {
        DDIM {
            unet: DiffusionUNet::new(in_channels, in_channels, 256),
            num_timesteps,
            eta,
        }
    }

    pub fn forward(&mut self, x: &Tensor, timesteps: &[usize], training: bool) -> Tensor {
        self.unet.forward(x, timesteps, training)
    }
}

/// Latent Diffusion Model
pub struct LatentDiffusion {
    vae_encoder: VAEEncoder,
    vae_decoder: VAEDecoder,
    diffusion: DDPM,
    latent_dim: usize,
}

struct VAEEncoder {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc_mu: Dense,
    fc_logvar: Dense,
}

impl VAEEncoder {
    fn new(in_channels: usize, latent_dim: usize) -> Self {
        VAEEncoder {
            conv1: Conv2d::new(in_channels, 32, (4, 4)).stride((2, 2)).padding((1, 1)),
            conv2: Conv2d::new(32, 64, (4, 4)).stride((2, 2)).padding((1, 1)),
            conv3: Conv2d::new(64, 128, (4, 4)).stride((2, 2)).padding((1, 1)),
            fc_mu: Dense::new(128 * 4 * 4, latent_dim),
            fc_logvar: Dense::new(128 * 4 * 4, latent_dim),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv3.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        let batch_size = out.dims()[0];
        let flat_size = out.data_f32().len() / batch_size;
        out = Tensor::from_slice(out.data_f32(), &[batch_size, flat_size]).unwrap();
        
        self.fc_mu.forward(&out, training)
    }
}

struct VAEDecoder {
    fc: Dense,
    deconv1: Conv2d,
    deconv2: Conv2d,
    deconv3: Conv2d,
}

impl VAEDecoder {
    fn new(latent_dim: usize, out_channels: usize) -> Self {
        VAEDecoder {
            fc: Dense::new(latent_dim, 128 * 4 * 4),
            deconv1: Conv2d::new(128, 64, (4, 4)).stride((2, 2)).padding((1, 1)),
            deconv2: Conv2d::new(64, 32, (4, 4)).stride((2, 2)).padding((1, 1)),
            deconv3: Conv2d::new(32, out_channels, (4, 4)).stride((2, 2)).padding((1, 1)),
        }
    }

    fn forward(&mut self, z: &Tensor, training: bool) -> Tensor {
        let mut out = self.fc.forward(z, training);
        
        let batch_size = z.dims()[0];
        out = Tensor::from_slice(out.data_f32(), &[batch_size, 128, 4, 4]).unwrap();
        
        out = self.deconv1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.deconv2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        self.deconv3.forward(&out, training)
    }
}

impl LatentDiffusion {
    pub fn new(in_channels: usize, latent_dim: usize, num_timesteps: usize) -> Self {
        LatentDiffusion {
            vae_encoder: VAEEncoder::new(in_channels, latent_dim),
            vae_decoder: VAEDecoder::new(latent_dim, in_channels),
            diffusion: DDPM::new(latent_dim, num_timesteps),
            latent_dim,
        }
    }

    pub fn encode(&mut self, x: &Tensor, training: bool) -> Tensor {
        self.vae_encoder.forward(x, training)
    }

    pub fn decode(&mut self, z: &Tensor, training: bool) -> Tensor {
        self.vae_decoder.forward(z, training)
    }

    pub fn forward(&mut self, x: &Tensor, timesteps: &[usize], training: bool) -> Tensor {
        let z = self.encode(x, training);
        let z_denoised = self.diffusion.forward(&z, timesteps, training);
        self.decode(&z_denoised, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddpm() {
        let mut ddpm = DDPM::new(3, 1000);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let timesteps = vec![100];
        let output = ddpm.forward(&input, &timesteps, false);
        assert_eq!(output.dims()[1], 3);
    }

    #[test]
    fn test_latent_diffusion() {
        let mut ld = LatentDiffusion::new(3, 256, 1000);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let timesteps = vec![100];
        let output = ld.forward(&input, &timesteps, false);
        assert_eq!(output.dims()[1], 3);
    }
}

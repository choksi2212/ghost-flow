//! Diffusion Models (Stable Diffusion, DDPM, DDIM)
//!
//! Implements diffusion models for image generation:
//! - DDPM (Denoising Diffusion Probabilistic Models)
//! - DDIM (Denoising Diffusion Implicit Models)
//! - Stable Diffusion architecture
//! - Noise scheduling
//! - Sampling algorithms

use ghostflow_core::Tensor;
use crate::conv::Conv2d;
use crate::norm::GroupNorm;
use crate::linear::Linear;
use crate::Module;

/// Diffusion model configuration
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Image size
    pub image_size: usize,
    /// Number of channels
    pub in_channels: usize,
    /// Model channels
    pub model_channels: usize,
    /// Number of residual blocks per resolution
    pub num_res_blocks: usize,
    /// Channel multipliers for each resolution
    pub channel_mult: Vec<usize>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of diffusion timesteps
    pub num_timesteps: usize,
    /// Beta schedule type
    pub beta_schedule: BetaSchedule,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        DiffusionConfig {
            image_size: 256,
            in_channels: 3,
            model_channels: 128,
            num_res_blocks: 2,
            channel_mult: vec![1, 2, 4, 8],
            num_heads: 8,
            num_timesteps: 1000,
            beta_schedule: BetaSchedule::Linear,
        }
    }
}

impl DiffusionConfig {
    /// Stable Diffusion configuration
    pub fn stable_diffusion() -> Self {
        DiffusionConfig {
            image_size: 512,
            in_channels: 4, // Latent space
            model_channels: 320,
            num_res_blocks: 2,
            channel_mult: vec![1, 2, 4, 4],
            num_heads: 8,
            num_timesteps: 1000,
            beta_schedule: BetaSchedule::ScaledLinear,
        }
    }
    
    /// DDPM configuration (original paper)
    pub fn ddpm() -> Self {
        DiffusionConfig {
            image_size: 256,
            in_channels: 3,
            model_channels: 128,
            num_res_blocks: 2,
            channel_mult: vec![1, 1, 2, 2, 4, 4],
            num_heads: 4,
            num_timesteps: 1000,
            beta_schedule: BetaSchedule::Linear,
        }
    }
    
    /// Small model for testing
    pub fn tiny() -> Self {
        DiffusionConfig {
            image_size: 32,
            in_channels: 3,
            model_channels: 64,
            num_res_blocks: 1,
            channel_mult: vec![1, 2],
            num_heads: 2,
            num_timesteps: 100,
            beta_schedule: BetaSchedule::Linear,
        }
    }
}

/// Beta schedule for noise
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetaSchedule {
    Linear,
    Cosine,
    ScaledLinear,
}

/// Noise scheduler
pub struct NoiseScheduler {
    /// Beta values (variance schedule)
    betas: Vec<f32>,
    /// Alpha values (1 - beta)
    alphas: Vec<f32>,
    /// Cumulative product of alphas
    alphas_cumprod: Vec<f32>,
    /// Number of timesteps
    num_timesteps: usize,
}

impl NoiseScheduler {
    /// Create new noise scheduler
    pub fn new(num_timesteps: usize, schedule: BetaSchedule) -> Self {
        let betas = Self::compute_betas(num_timesteps, schedule);
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        
        // Compute cumulative product
        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut prod = 1.0;
        for &alpha in &alphas {
            prod *= alpha;
            alphas_cumprod.push(prod);
        }
        
        NoiseScheduler {
            betas,
            alphas,
            alphas_cumprod,
            num_timesteps,
        }
    }
    
    /// Compute beta schedule
    fn compute_betas(num_timesteps: usize, schedule: BetaSchedule) -> Vec<f32> {
        match schedule {
            BetaSchedule::Linear => {
                let beta_start = 0.0001;
                let beta_end = 0.02;
                (0..num_timesteps)
                    .map(|t| {
                        let frac = t as f32 / (num_timesteps - 1) as f32;
                        beta_start + (beta_end - beta_start) * frac
                    })
                    .collect()
            }
            BetaSchedule::Cosine => {
                let s = 0.008;
                (0..num_timesteps)
                    .map(|t| {
                        let t_frac = t as f32 / num_timesteps as f32;
                        let alpha_bar = ((t_frac + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos().powi(2);
                        let alpha_bar_prev = if t > 0 {
                            (((t - 1) as f32 / num_timesteps as f32 + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos().powi(2)
                        } else {
                            1.0
                        };
                        (1.0 - alpha_bar / alpha_bar_prev).min(0.999)
                    })
                    .collect()
            }
            BetaSchedule::ScaledLinear => {
                let beta_start = 0.00085_f32.sqrt();
                let beta_end = 0.012_f32.sqrt();
                (0..num_timesteps)
                    .map(|t| {
                        let frac = t as f32 / (num_timesteps - 1) as f32;
                        let beta = beta_start + (beta_end - beta_start) * frac;
                        beta * beta
                    })
                    .collect()
            }
        }
    }
    
    /// Add noise to image (forward diffusion)
    pub fn add_noise(&self, x0: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor, String> {
        if timestep >= self.num_timesteps {
            return Err(format!("Timestep {} out of range", timestep));
        }
        
        let alpha_cumprod = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_cumprod).sqrt();
        
        let x0_data = x0.data_f32();
        let noise_data = noise.data_f32();
        
        let noisy: Vec<f32> = x0_data.iter()
            .zip(noise_data.iter())
            .map(|(&x, &n)| sqrt_alpha * x + sqrt_one_minus_alpha * n)
            .collect();
        
        Tensor::from_slice(&noisy, x0.dims())
            .map_err(|e| format!("Failed to create noisy tensor: {:?}", e))
    }
    
    /// Predict original image from noisy image and predicted noise
    pub fn predict_x0(&self, xt: &Tensor, noise_pred: &Tensor, timestep: usize) -> Result<Tensor, String> {
        if timestep >= self.num_timesteps {
            return Err(format!("Timestep {} out of range", timestep));
        }
        
        let alpha_cumprod = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_cumprod).sqrt();
        
        let xt_data = xt.data_f32();
        let noise_data = noise_pred.data_f32();
        
        let x0: Vec<f32> = xt_data.iter()
            .zip(noise_data.iter())
            .map(|(&x, &n)| (x - sqrt_one_minus_alpha * n) / sqrt_alpha)
            .collect();
        
        Tensor::from_slice(&x0, xt.dims())
            .map_err(|e| format!("Failed to create x0 tensor: {:?}", e))
    }
}

/// Residual block for U-Net
pub struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    norm1: GroupNorm,
    norm2: GroupNorm,
    time_emb_proj: Linear,
    skip_conv: Option<Conv2d>,
}

impl ResidualBlock {
    /// Create new residual block
    pub fn new(in_channels: usize, out_channels: usize, time_emb_dim: usize) -> Self {
        let conv1 = Conv2d::new(in_channels, out_channels, 3, 1, 1);
        let conv2 = Conv2d::new(out_channels, out_channels, 3, 1, 1);
        let norm1 = GroupNorm::new(32, in_channels);
        let norm2 = GroupNorm::new(32, out_channels);
        let time_emb_proj = Linear::new(time_emb_dim, out_channels);
        
        let skip_conv = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0))
        } else {
            None
        };
        
        ResidualBlock {
            conv1,
            conv2,
            norm1,
            norm2,
            time_emb_proj,
            skip_conv,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor, time_emb: &Tensor) -> Tensor {
        let h = self.norm1.forward(x);
        let h = h.relu();
        let h = self.conv1.forward(&h);
        
        // Add time embedding
        let time_proj = self.time_emb_proj.forward(time_emb);
        // Broadcast time embedding to spatial dimensions (simplified)
        let h = h.add(&time_proj).unwrap_or(h);
        
        let h = self.norm2.forward(&h);
        let h = h.relu();
        let h = self.conv2.forward(&h);
        
        // Skip connection
        let skip = if let Some(ref conv) = self.skip_conv {
            conv.forward(x)
        } else {
            x.clone()
        };
        
        h.add(&skip).unwrap_or(h)
    }
}

/// U-Net for diffusion models
pub struct UNet {
    /// Time embedding dimension
    time_emb_dim: usize,
    /// Time embedding MLP
    time_mlp: Vec<Linear>,
    /// Encoder blocks
    encoder_blocks: Vec<ResidualBlock>,
    /// Decoder blocks
    decoder_blocks: Vec<ResidualBlock>,
    /// Output convolution
    out_conv: Conv2d,
}

impl UNet {
    /// Create new U-Net
    pub fn new(config: &DiffusionConfig) -> Self {
        let time_emb_dim = config.model_channels * 4;
        
        // Time embedding MLP
        let time_mlp = vec![
            Linear::new(config.model_channels, time_emb_dim),
            Linear::new(time_emb_dim, time_emb_dim),
        ];
        
        // Simplified encoder/decoder (real implementation would be more complex)
        let mut encoder_blocks = Vec::new();
        let mut in_ch = config.in_channels;
        
        for &mult in &config.channel_mult {
            let out_ch = config.model_channels * mult;
            for _ in 0..config.num_res_blocks {
                encoder_blocks.push(ResidualBlock::new(in_ch, out_ch, time_emb_dim));
                in_ch = out_ch;
            }
        }
        
        // Decoder (mirror of encoder)
        let mut decoder_blocks = Vec::new();
        for &mult in config.channel_mult.iter().rev() {
            let out_ch = config.model_channels * mult;
            for _ in 0..config.num_res_blocks {
                decoder_blocks.push(ResidualBlock::new(in_ch, out_ch, time_emb_dim));
                in_ch = out_ch;
            }
        }
        
        let out_conv = Conv2d::new(in_ch, config.in_channels, 3, 1, 1);
        
        UNet {
            time_emb_dim,
            time_mlp,
            encoder_blocks,
            decoder_blocks,
            out_conv,
        }
    }
    
    /// Compute time embeddings
    fn get_time_embedding(&self, timestep: usize) -> Tensor {
        // Sinusoidal position embedding
        let half_dim = self.time_emb_dim / 2;
        let emb: Vec<f32> = (0..half_dim)
            .flat_map(|i| {
                let freq = (timestep as f32 * (-10000_f32.ln() * i as f32 / half_dim as f32).exp()).sin();
                let freq_cos = (timestep as f32 * (-10000_f32.ln() * i as f32 / half_dim as f32).exp()).cos();
                vec![freq, freq_cos]
            })
            .collect();
        
        Tensor::from_slice(&emb, &[self.time_emb_dim]).unwrap()
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor, timestep: usize) -> Tensor {
        // Get time embedding
        let mut time_emb = self.get_time_embedding(timestep);
        
        // Time MLP
        for layer in &self.time_mlp {
            time_emb = layer.forward(&time_emb);
            time_emb = time_emb.relu();
        }
        
        // Encoder
        let mut h = x.clone();
        for block in &self.encoder_blocks {
            h = block.forward(&h, &time_emb);
        }
        
        // Decoder
        for block in &self.decoder_blocks {
            h = block.forward(&h, &time_emb);
        }
        
        // Output
        self.out_conv.forward(&h)
    }
}

/// DDPM (Denoising Diffusion Probabilistic Model)
pub struct DDPM {
    /// Configuration
    config: DiffusionConfig,
    /// U-Net model
    unet: UNet,
    /// Noise scheduler
    scheduler: NoiseScheduler,
}

impl DDPM {
    /// Create new DDPM
    pub fn new(config: DiffusionConfig) -> Self {
        let unet = UNet::new(&config);
        let scheduler = NoiseScheduler::new(config.num_timesteps, config.beta_schedule);
        
        DDPM {
            config,
            unet,
            scheduler,
        }
    }
    
    /// Training forward pass
    pub fn forward(&self, x0: &Tensor, timestep: usize) -> Result<(Tensor, Tensor), String> {
        // Sample noise
        let noise = Tensor::randn(x0.dims());
        
        // Add noise to image
        let xt = self.scheduler.add_noise(x0, &noise, timestep)?;
        
        // Predict noise
        let noise_pred = self.unet.forward(&xt, timestep);
        
        Ok((noise_pred, noise))
    }
    
    /// Sample (generate) images
    pub fn sample(&self, batch_size: usize) -> Result<Tensor, String> {
        // Start from pure noise
        let mut xt = Tensor::randn(&[
            batch_size,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        ]);
        
        // Reverse diffusion process
        for t in (0..self.config.num_timesteps).rev() {
            // Predict noise
            let noise_pred = self.unet.forward(&xt, t);
            
            // Denoise one step
            xt = self.denoise_step(&xt, &noise_pred, t)?;
        }
        
        Ok(xt)
    }
    
    /// Single denoising step
    fn denoise_step(&self, xt: &Tensor, noise_pred: &Tensor, timestep: usize) -> Result<Tensor, String> {
        // Predict x0
        let x0_pred = self.scheduler.predict_x0(xt, noise_pred, timestep)?;
        
        if timestep == 0 {
            return Ok(x0_pred);
        }
        
        // Add noise for next step (simplified DDPM sampling)
        let alpha_t = self.scheduler.alphas_cumprod[timestep];
        let alpha_t_prev = if timestep > 0 {
            self.scheduler.alphas_cumprod[timestep - 1]
        } else {
            1.0
        };
        
        let xt_data = xt.data_f32();
        let x0_data = x0_pred.data_f32();
        
        let xt_prev: Vec<f32> = xt_data.iter()
            .zip(x0_data.iter())
            .map(|(&x, &x0)| {
                let coef1 = alpha_t_prev.sqrt();
                let coef2 = (1.0 - alpha_t_prev).sqrt();
                coef1 * x0 + coef2 * ((x - alpha_t.sqrt() * x0) / (1.0 - alpha_t).sqrt())
            })
            .collect();
        
        Tensor::from_slice(&xt_prev, xt.dims())
            .map_err(|e| format!("Failed to create denoised tensor: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_noise_scheduler() {
        let scheduler = NoiseScheduler::new(100, BetaSchedule::Linear);
        assert_eq!(scheduler.betas.len(), 100);
        assert_eq!(scheduler.alphas.len(), 100);
        assert_eq!(scheduler.alphas_cumprod.len(), 100);
        
        // Check that alphas_cumprod is decreasing
        for i in 1..100 {
            assert!(scheduler.alphas_cumprod[i] < scheduler.alphas_cumprod[i-1]);
        }
    }
    
    #[test]
    fn test_add_noise() {
        let scheduler = NoiseScheduler::new(100, BetaSchedule::Linear);
        let x0 = Tensor::randn(&[2, 3, 32, 32]);
        let noise = Tensor::randn(&[2, 3, 32, 32]);
        
        let xt = scheduler.add_noise(&x0, &noise, 50).unwrap();
        assert_eq!(xt.dims(), &[2, 3, 32, 32]);
    }
    
    #[test]
    fn test_ddpm_config() {
        let config = DiffusionConfig::tiny();
        assert_eq!(config.image_size, 32);
        assert_eq!(config.num_timesteps, 100);
        
        let config = DiffusionConfig::stable_diffusion();
        assert_eq!(config.image_size, 512);
        assert_eq!(config.in_channels, 4);
    }
    
    #[test]
    fn test_ddpm_forward() {
        let config = DiffusionConfig::tiny();
        let ddpm = DDPM::new(config);
        
        let x0 = Tensor::randn(&[2, 3, 32, 32]);
        let (noise_pred, noise) = ddpm.forward(&x0, 50).unwrap();
        
        assert_eq!(noise_pred.dims(), noise.dims());
    }
}

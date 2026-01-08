//! NeRF (Neural Radiance Fields)
//!
//! Implements Neural Radiance Fields for 3D scene representation:
//! - Volumetric scene representation
//! - Novel view synthesis
//! - Positional encoding
//! - Hierarchical sampling
//! - Volume rendering

use ghostflow_core::Tensor;
use crate::linear::Linear;
use crate::Module;

/// NeRF configuration
#[derive(Debug, Clone)]
pub struct NeRFConfig {
    /// Number of samples per ray (coarse)
    pub num_samples_coarse: usize,
    /// Number of samples per ray (fine)
    pub num_samples_fine: usize,
    /// Number of frequency bands for positional encoding
    pub num_freq_bands: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_layers: usize,
    /// Skip connection layer
    pub skip_layer: usize,
    /// Use view direction
    pub use_view_dirs: bool,
    /// Near plane distance
    pub near: f32,
    /// Far plane distance
    pub far: f32,
}

impl Default for NeRFConfig {
    fn default() -> Self {
        NeRFConfig {
            num_samples_coarse: 64,
            num_samples_fine: 128,
            num_freq_bands: 10,
            hidden_size: 256,
            num_layers: 8,
            skip_layer: 4,
            use_view_dirs: true,
            near: 2.0,
            far: 6.0,
        }
    }
}

impl NeRFConfig {
    /// Tiny NeRF for testing
    pub fn tiny() -> Self {
        NeRFConfig {
            num_samples_coarse: 32,
            num_samples_fine: 64,
            num_freq_bands: 6,
            hidden_size: 128,
            num_layers: 4,
            skip_layer: 2,
            use_view_dirs: true,
            near: 2.0,
            far: 6.0,
        }
    }
    
    /// Large NeRF for high quality
    pub fn large() -> Self {
        NeRFConfig {
            num_samples_coarse: 128,
            num_samples_fine: 256,
            num_freq_bands: 15,
            hidden_size: 512,
            num_layers: 10,
            skip_layer: 5,
            use_view_dirs: true,
            near: 2.0,
            far: 6.0,
        }
    }
}

/// Positional encoding for NeRF
pub struct PositionalEncoder {
    num_freq_bands: usize,
    include_input: bool,
}

impl PositionalEncoder {
    /// Create new positional encoder
    pub fn new(num_freq_bands: usize, include_input: bool) -> Self {
        PositionalEncoder {
            num_freq_bands,
            include_input,
        }
    }
    
    /// Get output dimension
    pub fn output_dim(&self, input_dim: usize) -> usize {
        let encoded_dim = input_dim * self.num_freq_bands * 2; // sin and cos
        if self.include_input {
            encoded_dim + input_dim
        } else {
            encoded_dim
        }
    }
    
    /// Encode positions
    pub fn encode(&self, x: &Tensor) -> Result<Tensor, String> {
        let x_data = x.data_f32();
        let dims = x.dims();
        let input_dim = dims[dims.len() - 1];
        let batch_size = x_data.len() / input_dim;
        
        let output_dim = self.output_dim(input_dim);
        let mut result = Vec::with_capacity(batch_size * output_dim);
        
        for i in 0..batch_size {
            let start = i * input_dim;
            let end = start + input_dim;
            let input_slice = &x_data[start..end];
            
            // Include original input if requested
            if self.include_input {
                result.extend_from_slice(input_slice);
            }
            
            // Apply positional encoding: [sin(2^0*pi*x), cos(2^0*pi*x), sin(2^1*pi*x), cos(2^1*pi*x), ...]
            for freq in 0..self.num_freq_bands {
                let freq_scale = 2.0_f32.powi(freq as i32) * std::f32::consts::PI;
                for &val in input_slice.iter() {
                    let scaled = val * freq_scale;
                    result.push(scaled.sin());
                    result.push(scaled.cos());
                }
            }
        }
        
        let mut output_dims = dims.to_vec();
        output_dims[dims.len() - 1] = output_dim;
        
        Tensor::from_slice(&result, &output_dims)
            .map_err(|e| format!("Failed to create encoded tensor: {:?}", e))
    }
}

/// NeRF MLP network
pub struct NeRFMLP {
    config: NeRFConfig,
    pos_encoder: PositionalEncoder,
    dir_encoder: Option<PositionalEncoder>,
    layers: Vec<Linear>,
    density_layer: Linear,
    rgb_layers: Vec<Linear>,
}

impl NeRFMLP {
    /// Create new NeRF MLP
    pub fn new(config: NeRFConfig) -> Self {
        let pos_encoder = PositionalEncoder::new(config.num_freq_bands, true);
        let pos_dim = pos_encoder.output_dim(3); // 3D positions
        
        let dir_encoder = if config.use_view_dirs {
            Some(PositionalEncoder::new(config.num_freq_bands / 2, true))
        } else {
            None
        };
        
        // Main MLP layers
        let mut layers = Vec::new();
        let mut in_dim = pos_dim;
        
        for i in 0..config.num_layers {
            let out_dim = config.hidden_size;
            layers.push(Linear::new(in_dim, out_dim));
            
            // Skip connection
            if i == config.skip_layer {
                in_dim = config.hidden_size + pos_dim;
            } else {
                in_dim = config.hidden_size;
            }
        }
        
        // Density output
        let density_layer = Linear::new(config.hidden_size, 1);
        
        // RGB layers
        let mut rgb_layers = Vec::new();
        if config.use_view_dirs {
            let dir_dim = dir_encoder.as_ref().unwrap().output_dim(3);
            rgb_layers.push(Linear::new(config.hidden_size + dir_dim, config.hidden_size / 2));
            rgb_layers.push(Linear::new(config.hidden_size / 2, 3));
        } else {
            rgb_layers.push(Linear::new(config.hidden_size, 3));
        }
        
        NeRFMLP {
            config,
            pos_encoder,
            dir_encoder,
            layers,
            density_layer,
            rgb_layers,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, positions: &Tensor, directions: Option<&Tensor>) -> Result<(Tensor, Tensor), String> {
        // Encode positions
        let mut x = self.pos_encoder.encode(positions)?;
        let encoded_pos = x.clone();
        
        // Pass through main MLP
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            x = x.relu();
            
            // Skip connection
            if i == self.config.skip_layer {
                x = self.concatenate(&x, &encoded_pos)?;
            }
        }
        
        // Density output (with ReLU activation)
        let density = self.density_layer.forward(&x);
        let density = density.relu();
        
        // RGB output
        let rgb = if self.config.use_view_dirs && directions.is_some() {
            let dirs = directions.unwrap();
            let encoded_dirs = self.dir_encoder.as_ref().unwrap().encode(dirs)?;
            let mut rgb_input = self.concatenate(&x, &encoded_dirs)?;
            
            for (i, layer) in self.rgb_layers.iter().enumerate() {
                rgb_input = layer.forward(&rgb_input);
                if i < self.rgb_layers.len() - 1 {
                    rgb_input = rgb_input.relu();
                }
            }
            rgb_input.sigmoid() // RGB values in [0, 1]
        } else {
            let mut rgb = self.rgb_layers[0].forward(&x);
            rgb = rgb.sigmoid();
            rgb
        };
        
        Ok((rgb, density))
    }
    
    /// Concatenate tensors along last dimension
    fn concatenate(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let a_dims = a.dims();
        let b_dims = b.dims();
        
        if a_dims.len() != b_dims.len() {
            return Err("Tensors must have same number of dimensions".to_string());
        }
        
        for i in 0..a_dims.len() - 1 {
            if a_dims[i] != b_dims[i] {
                return Err(format!("Dimension mismatch at axis {}", i));
            }
        }
        
        let a_last = a_dims[a_dims.len() - 1];
        let b_last = b_dims[b_dims.len() - 1];
        let batch_size = a_data.len() / a_last;
        
        let mut result = Vec::with_capacity(batch_size * (a_last + b_last));
        
        for i in 0..batch_size {
            let a_start = i * a_last;
            let b_start = i * b_last;
            result.extend_from_slice(&a_data[a_start..a_start + a_last]);
            result.extend_from_slice(&b_data[b_start..b_start + b_last]);
        }
        
        let mut output_dims = a_dims.to_vec();
        output_dims[a_dims.len() - 1] = a_last + b_last;
        
        Tensor::from_slice(&result, &output_dims)
            .map_err(|e| format!("Failed to concatenate: {:?}", e))
    }
}

/// Ray sampler for NeRF
pub struct RaySampler {
    near: f32,
    far: f32,
}

impl RaySampler {
    /// Create new ray sampler
    pub fn new(near: f32, far: f32) -> Self {
        RaySampler { near, far }
    }
    
    /// Sample points along rays
    pub fn sample_along_rays(&self, ray_origins: &Tensor, ray_directions: &Tensor, num_samples: usize) -> Result<(Tensor, Tensor), String> {
        let origins_data = ray_origins.data_f32();
        let directions_data = ray_directions.data_f32();
        let dims = ray_origins.dims();
        let num_rays = dims[0];
        
        // Generate sample depths
        let mut depths = Vec::with_capacity(num_rays * num_samples);
        let step = (self.far - self.near) / (num_samples - 1) as f32;
        
        for _ in 0..num_rays {
            for i in 0..num_samples {
                let depth = self.near + step * i as f32;
                depths.push(depth);
            }
        }
        
        // Compute 3D sample positions: origin + depth * direction
        let mut positions = Vec::with_capacity(num_rays * num_samples * 3);
        
        for ray_idx in 0..num_rays {
            let o_start = ray_idx * 3;
            let d_start = ray_idx * 3;
            
            for sample_idx in 0..num_samples {
                let depth = depths[ray_idx * num_samples + sample_idx];
                
                for dim in 0..3 {
                    let pos = origins_data[o_start + dim] + depth * directions_data[d_start + dim];
                    positions.push(pos);
                }
            }
        }
        
        let positions_tensor = Tensor::from_slice(&positions, &[num_rays, num_samples, 3])
            .map_err(|e| format!("Failed to create positions: {:?}", e))?;
        
        let depths_tensor = Tensor::from_slice(&depths, &[num_rays, num_samples])
            .map_err(|e| format!("Failed to create depths: {:?}", e))?;
        
        Ok((positions_tensor, depths_tensor))
    }
}

/// Volume renderer for NeRF
pub struct VolumeRenderer;

impl VolumeRenderer {
    /// Render rays using volume rendering equation
    pub fn render_rays(rgb: &Tensor, density: &Tensor, depths: &Tensor) -> Result<Tensor, String> {
        let rgb_data = rgb.data_f32();
        let density_data = density.data_f32();
        let depths_data = depths.data_f32();
        
        let dims = rgb.dims();
        let num_rays = dims[0];
        let num_samples = dims[1];
        
        let mut rendered = Vec::with_capacity(num_rays * 3);
        
        for ray_idx in 0..num_rays {
            let mut accumulated_rgb = [0.0f32; 3];
            let mut accumulated_alpha = 0.0f32;
            
            for sample_idx in 0..num_samples {
                // Get depth interval
                let delta = if sample_idx < num_samples - 1 {
                    depths_data[ray_idx * num_samples + sample_idx + 1] 
                        - depths_data[ray_idx * num_samples + sample_idx]
                } else {
                    1e10 // Large value for last sample
                };
                
                // Get density and RGB
                let density_idx = ray_idx * num_samples + sample_idx;
                let sigma = density_data[density_idx];
                
                // Compute alpha (opacity)
                let alpha = 1.0 - (-sigma * delta).exp();
                
                // Compute transmittance
                let transmittance = 1.0 - accumulated_alpha;
                
                // Accumulate color
                let rgb_start = (ray_idx * num_samples + sample_idx) * 3;
                for c in 0..3 {
                    accumulated_rgb[c] += transmittance * alpha * rgb_data[rgb_start + c];
                }
                
                // Accumulate alpha
                accumulated_alpha += transmittance * alpha;
                
                // Early stopping if fully opaque
                if accumulated_alpha > 0.999 {
                    break;
                }
            }
            
            rendered.extend_from_slice(&accumulated_rgb);
        }
        
        Tensor::from_slice(&rendered, &[num_rays, 3])
            .map_err(|e| format!("Failed to create rendered image: {:?}", e))
    }
}

/// Complete NeRF model
pub struct NeRF {
    config: NeRFConfig,
    coarse_network: NeRFMLP,
    fine_network: Option<NeRFMLP>,
    sampler: RaySampler,
}

impl NeRF {
    /// Create new NeRF model
    pub fn new(config: NeRFConfig) -> Self {
        let coarse_network = NeRFMLP::new(config.clone());
        let fine_network = if config.num_samples_fine > 0 {
            Some(NeRFMLP::new(config.clone()))
        } else {
            None
        };
        let sampler = RaySampler::new(config.near, config.far);
        
        NeRF {
            config,
            coarse_network,
            fine_network,
            sampler,
        }
    }
    
    /// Render rays
    pub fn render(&self, ray_origins: &Tensor, ray_directions: &Tensor) -> Result<Tensor, String> {
        // Coarse sampling
        let (positions_coarse, depths_coarse) = self.sampler.sample_along_rays(
            ray_origins,
            ray_directions,
            self.config.num_samples_coarse,
        )?;
        
        // Reshape for network
        let num_rays = ray_origins.dims()[0];
        let positions_flat = self.reshape_for_network(&positions_coarse)?;
        let directions_flat = self.repeat_directions(ray_directions, self.config.num_samples_coarse)?;
        
        // Coarse network forward pass
        let (rgb_coarse, density_coarse) = self.coarse_network.forward(&positions_flat, Some(&directions_flat))?;
        
        // Reshape back
        let rgb_coarse = self.reshape_from_network(&rgb_coarse, num_rays, self.config.num_samples_coarse, 3)?;
        let density_coarse = self.reshape_from_network(&density_coarse, num_rays, self.config.num_samples_coarse, 1)?;
        
        // Render coarse
        let rendered_coarse = VolumeRenderer::render_rays(&rgb_coarse, &density_coarse, &depths_coarse)?;
        
        // Fine sampling (if enabled)
        if let Some(ref fine_net) = self.fine_network {
            // For simplicity, use uniform sampling for fine network too
            // In practice, you'd use importance sampling based on coarse weights
            let (positions_fine, depths_fine) = self.sampler.sample_along_rays(
                ray_origins,
                ray_directions,
                self.config.num_samples_fine,
            )?;
            
            let positions_flat = self.reshape_for_network(&positions_fine)?;
            let directions_flat = self.repeat_directions(ray_directions, self.config.num_samples_fine)?;
            
            let (rgb_fine, density_fine) = fine_net.forward(&positions_flat, Some(&directions_flat))?;
            
            let rgb_fine = self.reshape_from_network(&rgb_fine, num_rays, self.config.num_samples_fine, 3)?;
            let density_fine = self.reshape_from_network(&density_fine, num_rays, self.config.num_samples_fine, 1)?;
            
            VolumeRenderer::render_rays(&rgb_fine, &density_fine, &depths_fine)
        } else {
            Ok(rendered_coarse)
        }
    }
    
    /// Reshape tensor for network input
    fn reshape_for_network(&self, x: &Tensor) -> Result<Tensor, String> {
        let data = x.data_f32();
        let dims = x.dims();
        let new_dims = vec![dims[0] * dims[1], dims[2]];
        Tensor::from_slice(&data, &new_dims)
            .map_err(|e| format!("Failed to reshape: {:?}", e))
    }
    
    /// Reshape tensor from network output
    fn reshape_from_network(&self, x: &Tensor, num_rays: usize, num_samples: usize, channels: usize) -> Result<Tensor, String> {
        let data = x.data_f32();
        let new_dims = vec![num_rays, num_samples, channels];
        Tensor::from_slice(&data, &new_dims)
            .map_err(|e| format!("Failed to reshape: {:?}", e))
    }
    
    /// Repeat directions for each sample
    fn repeat_directions(&self, directions: &Tensor, num_samples: usize) -> Result<Tensor, String> {
        let data = directions.data_f32();
        let dims = directions.dims();
        let num_rays = dims[0];
        
        let mut result = Vec::with_capacity(num_rays * num_samples * 3);
        
        for ray_idx in 0..num_rays {
            let start = ray_idx * 3;
            for _ in 0..num_samples {
                result.extend_from_slice(&data[start..start + 3]);
            }
        }
        
        Tensor::from_slice(&result, &[num_rays * num_samples, 3])
            .map_err(|e| format!("Failed to repeat directions: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nerf_config() {
        let config = NeRFConfig::default();
        assert_eq!(config.num_samples_coarse, 64);
        assert_eq!(config.num_samples_fine, 128);
        
        let tiny = NeRFConfig::tiny();
        assert_eq!(tiny.hidden_size, 128);
    }
    
    #[test]
    fn test_positional_encoder() {
        let encoder = PositionalEncoder::new(4, true);
        assert_eq!(encoder.output_dim(3), 3 + 3 * 4 * 2); // input + encoded
        
        let x = Tensor::from_slice(&[0.5f32, 0.3, 0.1], &[1, 3]).unwrap();
        let encoded = encoder.encode(&x).unwrap();
        assert_eq!(encoded.dims(), &[1, 27]); // 3 + 24
    }
    
    #[test]
    fn test_ray_sampler() {
        let sampler = RaySampler::new(2.0, 6.0);
        
        let origins = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap();
        let directions = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]).unwrap();
        
        let (positions, depths) = sampler.sample_along_rays(&origins, &directions, 8).unwrap();
        
        assert_eq!(positions.dims(), &[2, 8, 3]);
        assert_eq!(depths.dims(), &[2, 8]);
    }
    
    #[test]
    fn test_nerf_mlp() {
        let config = NeRFConfig::tiny();
        let mlp = NeRFMLP::new(config);
        
        let positions = Tensor::randn(&[4, 3]);
        let directions = Tensor::randn(&[4, 3]);
        
        let (rgb, density) = mlp.forward(&positions, Some(&directions)).unwrap();
        
        assert_eq!(rgb.dims(), &[4, 3]);
        assert_eq!(density.dims(), &[4, 1]);
    }
    
    #[test]
    fn test_volume_renderer() {
        let rgb = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[1, 3, 3]).unwrap();
        let density = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3, 1]).unwrap();
        let depths = Tensor::from_slice(&[2.0f32, 3.0, 4.0], &[1, 3]).unwrap();
        
        let rendered = VolumeRenderer::render_rays(&rgb, &density, &depths).unwrap();
        assert_eq!(rendered.dims(), &[1, 3]);
    }
    
    #[test]
    fn test_nerf_model() {
        let config = NeRFConfig::tiny();
        let nerf = NeRF::new(config);
        
        let ray_origins = Tensor::from_slice(&[0.0f32, 0.0, 0.0], &[1, 3]).unwrap();
        let ray_directions = Tensor::from_slice(&[0.0f32, 0.0, 1.0], &[1, 3]).unwrap();
        
        let rendered = nerf.render(&ray_origins, &ray_directions).unwrap();
        assert_eq!(rendered.dims(), &[1, 3]); // RGB output
    }
}

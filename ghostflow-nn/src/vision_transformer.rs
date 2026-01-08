//! Vision Transformer (ViT) Implementation
//!
//! Implements Vision Transformers as described in "An Image is Worth 16x16 Words"
//! - Patch embedding
//! - Position embedding
//! - Transformer encoder blocks
//! - Classification head

use ghostflow_core::Tensor;
use crate::transformer::{TransformerEncoder, PositionalEncoding};
use crate::linear::Linear;
use crate::norm::LayerNorm;
use crate::Module;

/// Vision Transformer configuration
#[derive(Debug, Clone)]
pub struct ViTConfig {
    /// Image size (assumed square)
    pub image_size: usize,
    /// Patch size (assumed square)
    pub patch_size: usize,
    /// Number of input channels
    pub in_channels: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP hidden dimension
    pub mlp_dim: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for ViTConfig {
    fn default() -> Self {
        ViTConfig {
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            num_classes: 1000,
            dropout: 0.1,
        }
    }
}

impl ViTConfig {
    /// ViT-Base configuration
    pub fn vit_base() -> Self {
        Self::default()
    }
    
    /// ViT-Large configuration
    pub fn vit_large() -> Self {
        ViTConfig {
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            mlp_dim: 4096,
            ..Default::default()
        }
    }
    
    /// ViT-Huge configuration
    pub fn vit_huge() -> Self {
        ViTConfig {
            embed_dim: 1280,
            num_layers: 32,
            num_heads: 16,
            mlp_dim: 5120,
            ..Default::default()
        }
    }
    
    /// Get number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }
}

/// Patch embedding layer
pub struct PatchEmbedding {
    /// Projection layer
    projection: Linear,
    /// Patch size
    patch_size: usize,
    /// Number of patches
    num_patches: usize,
}

impl PatchEmbedding {
    /// Create new patch embedding
    pub fn new(config: &ViTConfig) -> Self {
        let patch_dim = config.patch_size * config.patch_size * config.in_channels;
        let projection = Linear::new(patch_dim, config.embed_dim);
        
        PatchEmbedding {
            projection,
            patch_size: config.patch_size,
            num_patches: config.num_patches(),
        }
    }
    
    /// Extract patches from image
    fn extract_patches(&self, x: &Tensor) -> Result<Tensor, String> {
        // Input: [batch, channels, height, width]
        // Output: [batch, num_patches, patch_dim]
        
        let dims = x.dims();
        if dims.len() != 4 {
            return Err(format!("Expected 4D input, got {}D", dims.len()));
        }
        
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        
        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        let patch_dim = self.patch_size * self.patch_size * channels;
        
        // Extract patches
        let x_data = x.data_f32();
        let mut patches = Vec::with_capacity(batch_size * num_patches_h * num_patches_w * patch_dim);
        
        for b in 0..batch_size {
            for ph in 0..num_patches_h {
                for pw in 0..num_patches_w {
                    // Extract single patch
                    for c in 0..channels {
                        for h in 0..self.patch_size {
                            for w in 0..self.patch_size {
                                let y = ph * self.patch_size + h;
                                let x_pos = pw * self.patch_size + w;
                                let idx = b * (channels * height * width) +
                                         c * (height * width) +
                                         y * width +
                                         x_pos;
                                patches.push(x_data[idx]);
                            }
                        }
                    }
                }
            }
        }
        
        Tensor::from_slice(&patches, &[batch_size, num_patches_h * num_patches_w, patch_dim])
            .map_err(|e| format!("Failed to create patches tensor: {:?}", e))
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        // Extract patches
        let patches = self.extract_patches(x)?;
        
        // Project patches to embedding dimension
        Ok(self.projection.forward(&patches))
    }
}

// Note: PatchEmbedding doesn't implement Module trait due to Result return type
// Use patch_embed.forward() directly instead

/// Vision Transformer model
pub struct VisionTransformer {
    /// Configuration
    config: ViTConfig,
    /// Patch embedding
    patch_embed: PatchEmbedding,
    /// Class token
    cls_token: Tensor,
    /// Position embedding
    pos_embed: Tensor,
    /// Transformer encoder
    encoder: TransformerEncoder,
    /// Layer normalization
    norm: LayerNorm,
    /// Classification head
    head: Linear,
}

impl VisionTransformer {
    /// Create new Vision Transformer
    pub fn new(config: ViTConfig) -> Self {
        let patch_embed = PatchEmbedding::new(&config);
        
        // Create class token [1, 1, embed_dim]
        let cls_token = Tensor::randn(&[1, 1, config.embed_dim]);
        
        // Create position embedding [1, num_patches + 1, embed_dim]
        let num_positions = config.num_patches() + 1; // +1 for class token
        let pos_embed = Tensor::randn(&[1, num_positions, config.embed_dim]);
        
        // Create transformer encoder
        let encoder = TransformerEncoder::new(
            config.embed_dim,
            config.num_heads,
            config.mlp_dim,
            config.num_layers,
            config.dropout,
        );
        
        // Layer norm
        let norm = LayerNorm::new(&[config.embed_dim]);
        
        // Classification head
        let head = Linear::new(config.embed_dim, config.num_classes);
        
        VisionTransformer {
            config,
            patch_embed,
            cls_token,
            pos_embed,
            encoder,
            norm,
            head,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let batch_size = x.dims()[0];
        
        // Patch embedding: [batch, num_patches, embed_dim]
        let x = self.patch_embed.forward(x)?;
        
        // Expand class token for batch: [batch, 1, embed_dim]
        let cls_tokens = self.expand_cls_token(batch_size)?;
        
        // Concatenate class token: [batch, num_patches + 1, embed_dim]
        let x = self.concat_cls_token(&x, &cls_tokens)?;
        
        // Add position embedding
        let x = self.add_position_embedding(&x)?;
        
        // Transformer encoder
        let x = self.encoder.forward(&x);
        
        // Layer norm
        let x = self.norm.forward(&x);
        
        // Extract class token: [batch, embed_dim]
        let cls_output = self.extract_cls_token(&x)?;
        
        // Classification head: [batch, num_classes]
        Ok(self.head.forward(&cls_output))
    }
    
    /// Expand class token for batch
    fn expand_cls_token(&self, batch_size: usize) -> Result<Tensor, String> {
        let cls_data = self.cls_token.data_f32();
        let embed_dim = self.config.embed_dim;
        
        let mut expanded = Vec::with_capacity(batch_size * embed_dim);
        for _ in 0..batch_size {
            expanded.extend_from_slice(&cls_data);
        }
        
        Tensor::from_slice(&expanded, &[batch_size, 1, embed_dim])
            .map_err(|e| format!("Failed to expand class token: {:?}", e))
    }
    
    /// Concatenate class token with patches
    fn concat_cls_token(&self, patches: &Tensor, cls_tokens: &Tensor) -> Result<Tensor, String> {
        let patches_data = patches.data_f32();
        let cls_data = cls_tokens.data_f32();
        
        let dims = patches.dims();
        let batch_size = dims[0];
        let num_patches = dims[1];
        let embed_dim = dims[2];
        
        let mut concatenated = Vec::with_capacity(batch_size * (num_patches + 1) * embed_dim);
        
        for b in 0..batch_size {
            // Add class token
            let cls_start = b * embed_dim;
            concatenated.extend_from_slice(&cls_data[cls_start..cls_start + embed_dim]);
            
            // Add patches
            let patch_start = b * num_patches * embed_dim;
            let patch_end = patch_start + num_patches * embed_dim;
            concatenated.extend_from_slice(&patches_data[patch_start..patch_end]);
        }
        
        Tensor::from_slice(&concatenated, &[batch_size, num_patches + 1, embed_dim])
            .map_err(|e| format!("Failed to concatenate tokens: {:?}", e))
    }
    
    /// Add position embedding
    fn add_position_embedding(&self, x: &Tensor) -> Result<Tensor, String> {
        let x_data = x.data_f32();
        let pos_data = self.pos_embed.data_f32();
        
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let embed_dim = dims[2];
        
        let mut result = Vec::with_capacity(x_data.len());
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..embed_dim {
                    let x_idx = b * seq_len * embed_dim + s * embed_dim + d;
                    let pos_idx = s * embed_dim + d;
                    result.push(x_data[x_idx] + pos_data[pos_idx]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, seq_len, embed_dim])
            .map_err(|e| format!("Failed to add position embedding: {:?}", e))
    }
    
    /// Extract class token from sequence
    fn extract_cls_token(&self, x: &Tensor) -> Result<Tensor, String> {
        let x_data = x.data_f32();
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let embed_dim = dims[2];
        
        let mut cls_output = Vec::with_capacity(batch_size * embed_dim);
        
        for b in 0..batch_size {
            let start = b * seq_len * embed_dim;
            let end = start + embed_dim;
            cls_output.extend_from_slice(&x_data[start..end]);
        }
        
        Tensor::from_slice(&cls_output, &[batch_size, embed_dim])
            .map_err(|e| format!("Failed to extract class token: {:?}", e))
    }
}

// Note: VisionTransformer doesn't implement Module trait due to Result return type
// Use vit.forward() directly instead

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vit_config() {
        let config = ViTConfig::vit_base();
        assert_eq!(config.num_patches(), 196); // (224/16)^2
        
        let config = ViTConfig::vit_large();
        assert_eq!(config.embed_dim, 1024);
        
        let config = ViTConfig::vit_huge();
        assert_eq!(config.embed_dim, 1280);
    }
    
    #[test]
    fn test_patch_embedding() {
        let config = ViTConfig {
            image_size: 32,
            patch_size: 8,
            in_channels: 3,
            embed_dim: 64,
            ..Default::default()
        };
        
        let patch_embed = PatchEmbedding::new(&config);
        let input = Tensor::randn(&[2, 3, 32, 32]); // batch=2, channels=3, 32x32
        
        let output = patch_embed.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 16, 64]); // 16 patches, 64 embed_dim
    }
    
    #[test]
    fn test_vision_transformer() {
        let config = ViTConfig {
            image_size: 32,
            patch_size: 8,
            in_channels: 3,
            embed_dim: 64,
            num_layers: 2,
            num_heads: 4,
            mlp_dim: 128,
            num_classes: 10,
            dropout: 0.1,
        };
        
        let vit = VisionTransformer::new(config);
        let input = Tensor::randn(&[2, 3, 32, 32]);
        
        let output = vit.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 10]); // batch=2, num_classes=10
    }
}

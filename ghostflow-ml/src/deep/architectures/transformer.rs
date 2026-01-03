//! Transformer Architectures - BERT, GPT, ViT, T5, and variants

use ghostflow_core::Tensor;
use crate::deep::layers::{Dense, LayerNorm, Dropout, Embedding};
use crate::deep::transformer::{MultiHeadAttention, PositionalEncoding};
use crate::deep::activations::{GELU, ReLU};

/// BERT Encoder Layer
pub struct BERTEncoderLayer {
    attention: MultiHeadAttention,
    attention_norm: LayerNorm,
    ffn1: Dense,
    ffn2: Dense,
    ffn_norm: LayerNorm,
    dropout: f32,
}

impl BERTEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        BERTEncoderLayer {
            attention: MultiHeadAttention::new(d_model, num_heads),
            attention_norm: LayerNorm::new(vec![d_model]),
            ffn1: Dense::new(d_model, d_ff),
            ffn2: Dense::new(d_ff, d_model),
            ffn_norm: LayerNorm::new(vec![d_model]),
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, training: bool) -> Tensor {
        // Multi-head self-attention
        let attn_out = self.attention.forward(x, x, x, mask);
        let attn_out = self.apply_dropout(&attn_out, self.dropout, training);
        
        // Add & Norm
        let x1 = self.add_tensors(x, &attn_out);
        let x1 = self.attention_norm.forward(&x1, training);

        // Feed-forward network
        let mut ffn_out = self.ffn1.forward(&x1, training);
        ffn_out = GELU::new().forward(&ffn_out);
        ffn_out = self.ffn2.forward(&ffn_out, training);
        ffn_out = self.apply_dropout(&ffn_out, self.dropout, training);

        // Add & Norm
        let x2 = self.add_tensors(&x1, &ffn_out);
        self.ffn_norm.forward(&x2, training)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// BERT Model (Base)
pub struct BERTBase {
    embedding: Embedding,
    token_type_embedding: Embedding,
    position_encoding: PositionalEncoding,
    embedding_norm: LayerNorm,
    encoder_layers: Vec<BERTEncoderLayer>,
    pooler: Dense,
    dropout: f32,
}

impl BERTBase {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize, 
               num_layers: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        let mut encoder_layers = Vec::new();
        for _ in 0..num_layers {
            encoder_layers.push(BERTEncoderLayer::new(d_model, num_heads, d_ff, dropout));
        }

        BERTBase {
            embedding: Embedding::new(vocab_size, d_model),
            token_type_embedding: Embedding::new(2, d_model), // For segment A/B
            position_encoding: PositionalEncoding::new(d_model, max_seq_len),
            embedding_norm: LayerNorm::new(vec![d_model]),
            encoder_layers,
            pooler: Dense::new(d_model, d_model),
            dropout,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, token_type_ids: Option<&Tensor>, 
                   attention_mask: Option<&Tensor>, training: bool) -> Tensor {
        // Embeddings
        let mut embeddings = self.embedding.forward(input_ids, training);
        
        // Add token type embeddings
        if let Some(token_types) = token_type_ids {
            let token_type_emb = self.token_type_embedding.forward(token_types, training);
            embeddings = self.add_tensors(&embeddings, &token_type_emb);
        }

        // Add positional encoding
        embeddings = self.position_encoding.forward(&embeddings);
        embeddings = self.embedding_norm.forward(&embeddings, training);
        embeddings = self.apply_dropout(&embeddings, self.dropout, training);

        // Pass through encoder layers
        let mut hidden_states = embeddings;
        for layer in &mut self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, attention_mask, training);
        }

        // Pool the first token ([CLS]) for classification
        let pooled = self.extract_cls_token(&hidden_states);
        let pooled = self.pooler.forward(&pooled, training);
        self.apply_tanh(&pooled)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn extract_cls_token(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch_size = dims[0];
        let d_model = dims[2];
        let data = x.data_f32();
        
        let mut cls_tokens = Vec::new();
        for b in 0..batch_size {
            for d in 0..d_model {
                cls_tokens.push(data[b * dims[1] * d_model + d]);
            }
        }

        Tensor::from_slice(&cls_tokens, &[batch_size, d_model]).unwrap()
    }

    fn apply_tanh(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter().map(|&val| val.tanh()).collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// GPT Decoder Layer
pub struct GPTDecoderLayer {
    attention: MultiHeadAttention,
    attention_norm: LayerNorm,
    ffn1: Dense,
    ffn2: Dense,
    ffn_norm: LayerNorm,
    dropout: f32,
}

impl GPTDecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        GPTDecoderLayer {
            attention: MultiHeadAttention::new(d_model, num_heads),
            attention_norm: LayerNorm::new(vec![d_model]),
            ffn1: Dense::new(d_model, d_ff),
            ffn2: Dense::new(d_ff, d_model),
            ffn_norm: LayerNorm::new(vec![d_model]),
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Tensor, causal_mask: &Tensor, training: bool) -> Tensor {
        // Masked multi-head self-attention
        let attn_out = self.attention.forward(x, x, x, Some(causal_mask));
        let attn_out = self.apply_dropout(&attn_out, self.dropout, training);
        
        // Add & Norm
        let x1 = self.add_tensors(x, &attn_out);
        let x1 = self.attention_norm.forward(&x1, training);

        // Feed-forward network
        let mut ffn_out = self.ffn1.forward(&x1, training);
        ffn_out = GELU::new().forward(&ffn_out);
        ffn_out = self.ffn2.forward(&ffn_out, training);
        ffn_out = self.apply_dropout(&ffn_out, self.dropout, training);

        // Add & Norm
        let x2 = self.add_tensors(&x1, &ffn_out);
        self.ffn_norm.forward(&x2, training)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// GPT-2 Model
pub struct GPT2 {
    embedding: Embedding,
    position_encoding: PositionalEncoding,
    decoder_layers: Vec<GPTDecoderLayer>,
    final_norm: LayerNorm,
    lm_head: Dense,
    dropout: f32,
}

impl GPT2 {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize,
               num_layers: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        let mut decoder_layers = Vec::new();
        for _ in 0..num_layers {
            decoder_layers.push(GPTDecoderLayer::new(d_model, num_heads, d_ff, dropout));
        }

        GPT2 {
            embedding: Embedding::new(vocab_size, d_model),
            position_encoding: PositionalEncoding::new(d_model, max_seq_len),
            decoder_layers,
            final_norm: LayerNorm::new(vec![d_model]),
            lm_head: Dense::new(d_model, vocab_size),
            dropout,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        // Embeddings
        let mut embeddings = self.embedding.forward(input_ids, training);
        embeddings = self.position_encoding.forward(&embeddings);
        embeddings = self.apply_dropout(&embeddings, self.dropout, training);

        // Create causal mask
        let seq_len = input_ids.dims()[1];
        let causal_mask = self.create_causal_mask(seq_len);

        // Pass through decoder layers
        let mut hidden_states = embeddings;
        for layer in &mut self.decoder_layers {
            hidden_states = layer.forward(&hidden_states, &causal_mask, training);
        }

        // Final layer norm
        hidden_states = self.final_norm.forward(&hidden_states, training);

        // Language modeling head
        self.lm_head.forward(&hidden_states, training)
    }

    fn create_causal_mask(&self, seq_len: usize) -> Tensor {
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    mask[i * seq_len + j] = 1.0;
                }
            }
        }
        Tensor::from_slice(&mask, &[seq_len, seq_len]).unwrap()
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Vision Transformer (ViT) Patch Embedding
pub struct PatchEmbedding {
    proj: Dense,
    patch_size: usize,
}

impl PatchEmbedding {
    pub fn new(img_size: usize, patch_size: usize, in_channels: usize, embed_dim: usize) -> Self {
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        let patch_dim = in_channels * patch_size * patch_size;

        PatchEmbedding {
            proj: Dense::new(patch_dim, embed_dim),
            patch_size,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // Extract patches and flatten
        let patches = self.extract_patches(x);
        self.proj.forward(&patches, training)
    }

    fn extract_patches(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let p = self.patch_size;

        let num_patches_h = height / p;
        let num_patches_w = width / p;
        let num_patches = num_patches_h * num_patches_w;
        let patch_dim = channels * p * p;

        let data = x.data_f32();
        let mut patches = vec![0.0f32; batch_size * num_patches * patch_dim];

        for b in 0..batch_size {
            for ph in 0..num_patches_h {
                for pw in 0..num_patches_w {
                    let patch_idx = ph * num_patches_w + pw;
                    for c in 0..channels {
                        for i in 0..p {
                            for j in 0..p {
                                let h = ph * p + i;
                                let w = pw * p + j;
                                let img_idx = ((b * channels + c) * height + h) * width + w;
                                let patch_pos = ((c * p + i) * p + j);
                                let out_idx = (b * num_patches + patch_idx) * patch_dim + patch_pos;
                                patches[out_idx] = data[img_idx];
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&patches, &[batch_size, num_patches, patch_dim]).unwrap()
    }
}

/// Vision Transformer (ViT)
pub struct VisionTransformer {
    patch_embed: PatchEmbedding,
    cls_token: Vec<f32>,
    pos_embed: Vec<f32>,
    encoder_layers: Vec<BERTEncoderLayer>,
    norm: LayerNorm,
    head: Dense,
    dropout: f32,
}

impl VisionTransformer {
    pub fn new(img_size: usize, patch_size: usize, in_channels: usize,
               num_classes: usize, embed_dim: usize, depth: usize,
               num_heads: usize, mlp_ratio: usize, dropout: f32) -> Self {
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        let d_ff = embed_dim * mlp_ratio;

        let mut encoder_layers = Vec::new();
        for _ in 0..depth {
            encoder_layers.push(BERTEncoderLayer::new(embed_dim, num_heads, d_ff, dropout));
        }

        // Initialize CLS token and positional embeddings
        use rand::prelude::*;
        let mut rng = thread_rng();
        let cls_token: Vec<f32> = (0..embed_dim).map(|_| rng.gen::<f32>() * 0.02 - 0.01).collect();
        let pos_embed: Vec<f32> = (0..(num_patches + 1) * embed_dim)
            .map(|_| rng.gen::<f32>() * 0.02 - 0.01).collect();

        VisionTransformer {
            patch_embed: PatchEmbedding::new(img_size, patch_size, in_channels, embed_dim),
            cls_token,
            pos_embed,
            encoder_layers,
            norm: LayerNorm::new(vec![embed_dim]),
            head: Dense::new(embed_dim, num_classes),
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let batch_size = x.dims()[0];

        // Patch embedding
        let mut x = self.patch_embed.forward(x, training);

        // Prepend CLS token
        x = self.prepend_cls_token(&x, batch_size);

        // Add positional embedding
        x = self.add_pos_embed(&x);

        // Dropout
        x = self.apply_dropout(&x, self.dropout, training);

        // Transformer encoder
        for layer in &mut self.encoder_layers {
            x = layer.forward(&x, None, training);
        }

        // Layer norm
        x = self.norm.forward(&x, training);

        // Extract CLS token
        let cls = self.extract_cls_token(&x);

        // Classification head
        self.head.forward(&cls, training)
    }

    fn prepend_cls_token(&self, x: &Tensor, batch_size: usize) -> Tensor {
        let dims = x.dims();
        let num_patches = dims[1];
        let embed_dim = dims[2];
        let data = x.data_f32();

        let mut result = Vec::new();
        for b in 0..batch_size {
            // Add CLS token
            result.extend_from_slice(&self.cls_token);
            // Add patch embeddings
            for p in 0..num_patches {
                for d in 0..embed_dim {
                    result.push(data[(b * num_patches + p) * embed_dim + d]);
                }
            }
        }

        Tensor::from_slice(&result, &[batch_size, num_patches + 1, embed_dim]).unwrap()
    }

    fn add_pos_embed(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .zip(self.pos_embed.iter().cycle())
            .map(|(&x, &p)| x + p)
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn extract_cls_token(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch_size = dims[0];
        let embed_dim = dims[2];
        let data = x.data_f32();

        let mut cls_tokens = Vec::new();
        for b in 0..batch_size {
            for d in 0..embed_dim {
                cls_tokens.push(data[b * dims[1] * embed_dim + d]);
            }
        }

        Tensor::from_slice(&cls_tokens, &[batch_size, embed_dim]).unwrap()
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_base() {
        let mut model = BERTBase::new(30000, 512, 768, 12, 12, 3072, 0.1);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let output = model.forward(&input_ids, None, None, false);
        assert_eq!(output.dims()[0], 2);
    }

    #[test]
    fn test_gpt2() {
        let mut model = GPT2::new(50257, 1024, 768, 12, 12, 3072, 0.1);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let output = model.forward(&input_ids, false);
        assert_eq!(output.dims()[0], 2);
    }

    #[test]
    fn test_vit() {
        let mut model = VisionTransformer::new(224, 16, 3, 1000, 768, 12, 12, 4, 0.1);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}


/// RoBERTa (Robustly Optimized BERT)
/// Same architecture as BERT but with different training procedure
pub type RoBERTa = BERTBase;

/// ALBERT (A Lite BERT) - Parameter sharing across layers
pub struct ALBERT {
    embedding: Embedding,
    position_encoding: PositionalEncoding,
    embedding_norm: LayerNorm,
    shared_layer: BERTEncoderLayer,
    num_layers: usize,
    pooler: Dense,
    dropout: f32,
}

impl ALBERT {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize,
               num_layers: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        ALBERT {
            embedding: Embedding::new(vocab_size, d_model),
            position_encoding: PositionalEncoding::new(d_model, max_seq_len),
            embedding_norm: LayerNorm::new(vec![d_model]),
            shared_layer: BERTEncoderLayer::new(d_model, num_heads, d_ff, dropout),
            num_layers,
            pooler: Dense::new(d_model, d_model),
            dropout,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, attention_mask: Option<&Tensor>, training: bool) -> Tensor {
        let mut embeddings = self.embedding.forward(input_ids, training);
        embeddings = self.position_encoding.forward(&embeddings);
        embeddings = self.embedding_norm.forward(&embeddings, training);
        embeddings = self.apply_dropout(&embeddings, self.dropout, training);

        // Share the same layer across all depths
        let mut hidden_states = embeddings;
        for _ in 0..self.num_layers {
            hidden_states = self.shared_layer.forward(&hidden_states, attention_mask, training);
        }

        let pooled = self.extract_cls_token(&hidden_states);
        let pooled = self.pooler.forward(&pooled, training);
        self.apply_tanh(&pooled)
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn extract_cls_token(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch_size = dims[0];
        let d_model = dims[2];
        let data = x.data_f32();
        
        let mut cls_tokens = Vec::new();
        for b in 0..batch_size {
            for d in 0..d_model {
                cls_tokens.push(data[b * dims[1] * d_model + d]);
            }
        }

        Tensor::from_slice(&cls_tokens, &[batch_size, d_model]).unwrap()
    }

    fn apply_tanh(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter().map(|&val| val.tanh()).collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// DistilBERT - Distilled version of BERT
pub struct DistilBERT {
    embedding: Embedding,
    position_encoding: PositionalEncoding,
    encoder_layers: Vec<BERTEncoderLayer>,
    dropout: f32,
}

impl DistilBERT {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize,
               num_layers: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        let mut encoder_layers = Vec::new();
        for _ in 0..num_layers {
            encoder_layers.push(BERTEncoderLayer::new(d_model, num_heads, d_ff, dropout));
        }

        DistilBERT {
            embedding: Embedding::new(vocab_size, d_model),
            position_encoding: PositionalEncoding::new(d_model, max_seq_len),
            encoder_layers,
            dropout,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, attention_mask: Option<&Tensor>, training: bool) -> Tensor {
        let mut embeddings = self.embedding.forward(input_ids, training);
        embeddings = self.position_encoding.forward(&embeddings);
        embeddings = self.apply_dropout(&embeddings, self.dropout, training);

        let mut hidden_states = embeddings;
        for layer in &mut self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, attention_mask, training);
        }

        hidden_states
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// DeiT (Data-efficient Image Transformer)
pub struct DeiT {
    patch_embed: PatchEmbedding,
    cls_token: Vec<f32>,
    dist_token: Vec<f32>, // Distillation token
    pos_embed: Vec<f32>,
    encoder_layers: Vec<BERTEncoderLayer>,
    norm: LayerNorm,
    head: Dense,
    head_dist: Dense, // Distillation head
    dropout: f32,
}

impl DeiT {
    pub fn new(img_size: usize, patch_size: usize, in_channels: usize,
               num_classes: usize, embed_dim: usize, depth: usize,
               num_heads: usize, mlp_ratio: usize, dropout: f32) -> Self {
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        let d_ff = embed_dim * mlp_ratio;

        let mut encoder_layers = Vec::new();
        for _ in 0..depth {
            encoder_layers.push(BERTEncoderLayer::new(embed_dim, num_heads, d_ff, dropout));
        }

        use rand::prelude::*;
        let mut rng = thread_rng();
        let cls_token: Vec<f32> = (0..embed_dim).map(|_| rng.gen::<f32>() * 0.02 - 0.01).collect();
        let dist_token: Vec<f32> = (0..embed_dim).map(|_| rng.gen::<f32>() * 0.02 - 0.01).collect();
        let pos_embed: Vec<f32> = (0..(num_patches + 2) * embed_dim)
            .map(|_| rng.gen::<f32>() * 0.02 - 0.01).collect();

        DeiT {
            patch_embed: PatchEmbedding::new(img_size, patch_size, in_channels, embed_dim),
            cls_token,
            dist_token,
            pos_embed,
            encoder_layers,
            norm: LayerNorm::new(vec![embed_dim]),
            head: Dense::new(embed_dim, num_classes),
            head_dist: Dense::new(embed_dim, num_classes),
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let batch_size = x.dims()[0];

        let mut x = self.patch_embed.forward(x, training);
        x = self.prepend_tokens(&x, batch_size);
        x = self.add_pos_embed(&x);
        x = self.apply_dropout(&x, self.dropout, training);

        for layer in &mut self.encoder_layers {
            x = layer.forward(&x, None, training);
        }

        x = self.norm.forward(&x, training);

        let (cls, dist) = self.extract_tokens(&x);
        let cls_out = self.head.forward(&cls, training);
        let dist_out = self.head_dist.forward(&dist, training);

        (cls_out, dist_out)
    }

    fn prepend_tokens(&self, x: &Tensor, batch_size: usize) -> Tensor {
        let dims = x.dims();
        let num_patches = dims[1];
        let embed_dim = dims[2];
        let data = x.data_f32();

        let mut result = Vec::new();
        for b in 0..batch_size {
            result.extend_from_slice(&self.cls_token);
            result.extend_from_slice(&self.dist_token);
            for p in 0..num_patches {
                for d in 0..embed_dim {
                    result.push(data[(b * num_patches + p) * embed_dim + d]);
                }
            }
        }

        Tensor::from_slice(&result, &[batch_size, num_patches + 2, embed_dim]).unwrap()
    }

    fn add_pos_embed(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .zip(self.pos_embed.iter().cycle())
            .map(|(&x, &p)| x + p)
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn extract_tokens(&self, x: &Tensor) -> (Tensor, Tensor) {
        let dims = x.dims();
        let batch_size = dims[0];
        let embed_dim = dims[2];
        let data = x.data_f32();

        let mut cls_tokens = Vec::new();
        let mut dist_tokens = Vec::new();

        for b in 0..batch_size {
            for d in 0..embed_dim {
                cls_tokens.push(data[b * dims[1] * embed_dim + d]);
                dist_tokens.push(data[b * dims[1] * embed_dim + embed_dim + d]);
            }
        }

        (
            Tensor::from_slice(&cls_tokens, &[batch_size, embed_dim]).unwrap(),
            Tensor::from_slice(&dist_tokens, &[batch_size, embed_dim]).unwrap(),
        )
    }

    fn apply_dropout(&self, x: &Tensor, p: f32, training: bool) -> Tensor {
        if !training || p == 0.0 {
            return x.clone();
        }
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Swin Transformer Block (Shifted Window)
pub struct SwinTransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    mlp1: Dense,
    mlp2: Dense,
    norm2: LayerNorm,
    window_size: usize,
    shift_size: usize,
}

impl SwinTransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, window_size: usize, shift_size: usize) -> Self {
        SwinTransformerBlock {
            attention: MultiHeadAttention::new(d_model, num_heads),
            norm1: LayerNorm::new(vec![d_model]),
            mlp1: Dense::new(d_model, d_model * 4),
            mlp2: Dense::new(d_model * 4, d_model),
            norm2: LayerNorm::new(vec![d_model]),
            window_size,
            shift_size,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // Window-based self-attention
        let shortcut = x.clone();
        let mut x = self.norm1.forward(x, training);
        
        // Apply windowed attention (simplified)
        x = self.attention.forward(&x, &x, &x, None);
        
        // Add shortcut
        x = self.add_tensors(&x, &shortcut);

        // MLP
        let shortcut = x.clone();
        let mut x = self.norm2.forward(&x, training);
        x = self.mlp1.forward(&x, training);
        x = GELU::new().forward(&x);
        x = self.mlp2.forward(&x, training);
        
        self.add_tensors(&x, &shortcut)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

/// Swin Transformer
pub struct SwinTransformer {
    patch_embed: PatchEmbedding,
    layers: Vec<Vec<SwinTransformerBlock>>,
    norm: LayerNorm,
    avgpool: GlobalAvgPool2d,
    head: Dense,
}

impl SwinTransformer {
    pub fn new(img_size: usize, patch_size: usize, in_channels: usize,
               num_classes: usize, embed_dim: usize, depths: Vec<usize>,
               num_heads: Vec<usize>, window_size: usize) -> Self {
        let mut layers = Vec::new();
        
        for (depth, &heads) in depths.iter().zip(num_heads.iter()) {
            let mut stage = Vec::new();
            for i in 0..*depth {
                let shift_size = if i % 2 == 0 { 0 } else { window_size / 2 };
                stage.push(SwinTransformerBlock::new(embed_dim, heads, window_size, shift_size));
            }
            layers.push(stage);
        }

        SwinTransformer {
            patch_embed: PatchEmbedding::new(img_size, patch_size, in_channels, embed_dim),
            layers,
            norm: LayerNorm::new(vec![embed_dim]),
            avgpool: GlobalAvgPool2d::new(),
            head: Dense::new(embed_dim, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut x = self.patch_embed.forward(x, training);

        for stage in &mut self.layers {
            for block in stage {
                x = block.forward(&x, training);
            }
        }

        x = self.norm.forward(&x, training);
        
        // Global average pooling (simplified)
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let d_model = dims[2];
        let data = x.data_f32();
        
        let mut pooled = vec![0.0f32; batch_size * d_model];
        for b in 0..batch_size {
            for d in 0..d_model {
                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    sum += data[(b * seq_len + s) * d_model + d];
                }
                pooled[b * d_model + d] = sum / seq_len as f32;
            }
        }

        let pooled_tensor = Tensor::from_slice(&pooled, &[batch_size, d_model]).unwrap();
        self.head.forward(&pooled_tensor, training)
    }
}

#[cfg(test)]
mod tests_extended {
    use super::*;

    #[test]
    fn test_albert() {
        let mut model = ALBERT::new(30000, 512, 768, 12, 12, 3072, 0.1);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let output = model.forward(&input_ids, None, false);
        assert_eq!(output.dims()[0], 2);
    }

    #[test]
    fn test_distilbert() {
        let mut model = DistilBERT::new(30000, 512, 768, 6, 12, 3072, 0.1);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let output = model.forward(&input_ids, None, false);
        assert_eq!(output.dims()[0], 2);
    }

    #[test]
    fn test_deit() {
        let mut model = DeiT::new(224, 16, 3, 1000, 768, 12, 12, 4, 0.1);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let (cls_out, dist_out) = model.forward(&input, false);
        assert_eq!(cls_out.dims()[1], 1000);
        assert_eq!(dist_out.dims()[1], 1000);
    }

    #[test]
    fn test_swin_transformer() {
        let mut model = SwinTransformer::new(
            224, 4, 3, 1000, 96,
            vec![2, 2, 6, 2],
            vec![3, 6, 12, 24],
            7
        );
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}

/// T5 (Text-to-Text Transfer Transformer)
pub struct T5 {
    encoder: T5Encoder,
    decoder: T5Decoder,
    vocab_size: usize,
    d_model: usize,
}

struct T5Encoder {
    embedding: Dense,
    layers: Vec<T5EncoderLayer>,
    final_norm: LayerNorm,
}

struct T5EncoderLayer {
    self_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl T5EncoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        T5EncoderLayer {
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            ffn: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.norm1.forward(x);
        let attn_out = self.self_attn.forward(&normed, &normed, &normed, training);
        let x = self.add_tensors(x, &attn_out);
        
        let normed = self.norm2.forward(&x);
        let ffn_out = self.ffn.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl T5Encoder {
    fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        T5Encoder {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| T5EncoderLayer::new(d_model, num_heads, d_ff)).collect(),
            final_norm: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        self.final_norm.forward(&x)
    }
}

struct T5Decoder {
    embedding: Dense,
    layers: Vec<T5DecoderLayer>,
    final_norm: LayerNorm,
    lm_head: Dense,
}

struct T5DecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl T5DecoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        T5DecoderLayer {
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            cross_attn: MultiHeadAttention::new(d_model, num_heads),
            ffn: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            norm3: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, encoder_output: &Tensor, training: bool) -> Tensor {
        let normed = self.norm1.forward(x);
        let self_attn_out = self.self_attn.forward(&normed, &normed, &normed, training);
        let x = self.add_tensors(x, &self_attn_out);
        
        let normed = self.norm2.forward(&x);
        let cross_attn_out = self.cross_attn.forward(&normed, encoder_output, encoder_output, training);
        let x = self.add_tensors(&x, &cross_attn_out);
        
        let normed = self.norm3.forward(&x);
        let ffn_out = self.ffn.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl T5Decoder {
    fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        T5Decoder {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| T5DecoderLayer::new(d_model, num_heads, d_ff)).collect(),
            final_norm: LayerNorm::new(d_model),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    fn forward(&mut self, input_ids: &Tensor, encoder_output: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, encoder_output, training);
        }
        
        x = self.final_norm.forward(&x);
        self.lm_head.forward(&x, training)
    }
}

impl T5 {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        T5 {
            encoder: T5Encoder::new(vocab_size, d_model, num_layers, num_heads, d_ff),
            decoder: T5Decoder::new(vocab_size, d_model, num_layers, num_heads, d_ff),
            vocab_size,
            d_model,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, decoder_input_ids: &Tensor, training: bool) -> Tensor {
        let encoder_output = self.encoder.forward(input_ids, training);
        self.decoder.forward(decoder_input_ids, &encoder_output, training)
    }
}

/// BART (Bidirectional and Auto-Regressive Transformers)
pub struct BART {
    encoder: BARTEncoder,
    decoder: BARTDecoder,
    vocab_size: usize,
}

struct BARTEncoder {
    embedding: Dense,
    layers: Vec<BERTEncoderLayer>,
}

impl BARTEncoder {
    fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        BARTEncoder {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| BERTEncoderLayer::new(d_model, num_heads, d_ff, 0.1)).collect(),
        }
    }

    fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, None, training);
        }
        
        x
    }
}

struct BARTDecoder {
    embedding: Dense,
    layers: Vec<T5DecoderLayer>,
    lm_head: Dense,
}

impl BARTDecoder {
    fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        BARTDecoder {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| T5DecoderLayer::new(d_model, num_heads, d_ff)).collect(),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    fn forward(&mut self, input_ids: &Tensor, encoder_output: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, encoder_output, training);
        }
        
        self.lm_head.forward(&x, training)
    }
}

impl BART {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        BART {
            encoder: BARTEncoder::new(vocab_size, d_model, num_layers, num_heads, d_ff),
            decoder: BARTDecoder::new(vocab_size, d_model, num_layers, num_heads, d_ff),
            vocab_size,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, decoder_input_ids: &Tensor, training: bool) -> Tensor {
        let encoder_output = self.encoder.forward(input_ids, training);
        self.decoder.forward(decoder_input_ids, &encoder_output, training)
    }
}

/// XLNet
pub struct XLNet {
    embedding: Dense,
    layers: Vec<XLNetLayer>,
    lm_head: Dense,
    vocab_size: usize,
    d_model: usize,
}

struct XLNetLayer {
    rel_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl XLNetLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        XLNetLayer {
            rel_attn: MultiHeadAttention::new(d_model, num_heads),
            ffn: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.norm1.forward(x);
        let attn_out = self.rel_attn.forward(&normed, &normed, &normed, training);
        let x = self.add_tensors(x, &attn_out);
        
        let normed = self.norm2.forward(&x);
        let ffn_out = self.ffn.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl XLNet {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        XLNet {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| XLNetLayer::new(d_model, num_heads, d_ff)).collect(),
            lm_head: Dense::new(d_model, vocab_size),
            vocab_size,
            d_model,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        self.lm_head.forward(&x, training)
    }
}

/// ELECTRA
pub struct ELECTRA {
    generator: ELECTRAGenerator,
    discriminator: ELECTRADiscriminator,
}

struct ELECTRAGenerator {
    embedding: Dense,
    layers: Vec<BERTEncoderLayer>,
    lm_head: Dense,
}

impl ELECTRAGenerator {
    fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        ELECTRAGenerator {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| BERTEncoderLayer::new(d_model, num_heads, d_ff, 0.1)).collect(),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, None, training);
        }
        
        self.lm_head.forward(&x, training)
    }
}

struct ELECTRADiscriminator {
    embedding: Dense,
    layers: Vec<BERTEncoderLayer>,
    classifier: Dense,
}

impl ELECTRADiscriminator {
    fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        ELECTRADiscriminator {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| BERTEncoderLayer::new(d_model, num_heads, d_ff, 0.1)).collect(),
            classifier: Dense::new(d_model, 1),
        }
    }

    fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, None, training);
        }
        
        self.classifier.forward(&x, training)
    }
}

impl ELECTRA {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize) -> Self {
        ELECTRA {
            generator: ELECTRAGenerator::new(vocab_size, d_model / 4, num_layers / 3, num_heads, d_ff / 4),
            discriminator: ELECTRADiscriminator::new(vocab_size, d_model, num_layers, num_heads, d_ff),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> (Tensor, Tensor) {
        let gen_output = self.generator.forward(input_ids, training);
        let disc_output = self.discriminator.forward(input_ids, training);
        (gen_output, disc_output)
    }
}

/// Reformer (Efficient Transformer)
pub struct Reformer {
    embedding: Dense,
    layers: Vec<ReformerLayer>,
    lm_head: Dense,
}

struct ReformerLayer {
    lsh_attn: LSHAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

struct LSHAttention {
    query_proj: Dense,
    key_proj: Dense,
    value_proj: Dense,
    num_hashes: usize,
}

impl LSHAttention {
    fn new(d_model: usize, num_hashes: usize) -> Self {
        LSHAttention {
            query_proj: Dense::new(d_model, d_model),
            key_proj: Dense::new(d_model, d_model),
            value_proj: Dense::new(d_model, d_model),
            num_hashes,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let q = self.query_proj.forward(x, training);
        let k = self.key_proj.forward(x, training);
        let v = self.value_proj.forward(x, training);
        
        // Simplified LSH attention (full implementation would use locality-sensitive hashing)
        v
    }
}

impl ReformerLayer {
    fn new(d_model: usize, d_ff: usize, num_hashes: usize) -> Self {
        ReformerLayer {
            lsh_attn: LSHAttention::new(d_model, num_hashes),
            ffn: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.norm1.forward(x);
        let attn_out = self.lsh_attn.forward(&normed, training);
        let x = self.add_tensors(x, &attn_out);
        
        let normed = self.norm2.forward(&x);
        let ffn_out = self.ffn.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl Reformer {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize, d_ff: usize, num_hashes: usize) -> Self {
        Reformer {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..num_layers).map(|_| ReformerLayer::new(d_model, d_ff, num_hashes)).collect(),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        self.lm_head.forward(&x, training)
    }
}

#[cfg(test)]
mod tests_new {
    use super::*;

    #[test]
    fn test_t5() {
        let mut model = T5::new(32000, 512, 6, 8, 2048);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let decoder_ids = Tensor::from_slice(&vec![1.0f32; 2 * 64], &[2, 64]).unwrap();
        let output = model.forward(&input_ids, &decoder_ids, false);
        assert_eq!(output.dims()[0], 2);
    }

    #[test]
    fn test_bart() {
        let mut model = BART::new(50000, 768, 6, 12, 3072);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let decoder_ids = Tensor::from_slice(&vec![1.0f32; 2 * 64], &[2, 64]).unwrap();
        let output = model.forward(&input_ids, &decoder_ids, false);
        assert_eq!(output.dims()[0], 2);
    }

    #[test]
    fn test_xlnet() {
        let mut model = XLNet::new(32000, 768, 12, 12, 3072);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let output = model.forward(&input_ids, false);
        assert_eq!(output.dims()[0], 2);
    }
}


/// LLaMA (Large Language Model Meta AI)
pub struct LLaMA {
    embedding: Dense,
    layers: Vec<LLaMALayer>,
    norm: RMSNorm,
    lm_head: Dense,
}

struct LLaMALayer {
    attention: LLaMAAttention,
    feed_forward: LLaMAFeedForward,
    attention_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

struct LLaMAAttention {
    wq: Dense,
    wk: Dense,
    wv: Dense,
    wo: Dense,
    n_heads: usize,
}

impl LLaMAAttention {
    fn new(d_model: usize, n_heads: usize) -> Self {
        LLaMAAttention {
            wq: Dense::new(d_model, d_model),
            wk: Dense::new(d_model, d_model),
            wv: Dense::new(d_model, d_model),
            wo: Dense::new(d_model, d_model),
            n_heads,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let q = self.wq.forward(x, training);
        let k = self.wk.forward(x, training);
        let v = self.wv.forward(x, training);
        
        // Simplified attention
        let attn_out = v;
        self.wo.forward(&attn_out, training)
    }
}

struct LLaMAFeedForward {
    w1: Dense,
    w2: Dense,
    w3: Dense,
}

impl LLaMAFeedForward {
    fn new(d_model: usize, d_ff: usize) -> Self {
        LLaMAFeedForward {
            w1: Dense::new(d_model, d_ff),
            w2: Dense::new(d_ff, d_model),
            w3: Dense::new(d_model, d_ff),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let gate = self.w1.forward(x, training);
        let gate = self.silu(&gate);
        
        let up = self.w3.forward(x, training);
        
        let combined = self.multiply(&gate, &up);
        self.w2.forward(&combined, training)
    }

    fn silu(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn multiply(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x * y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

struct RMSNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RMSNorm {
    fn new(dim: usize) -> Self {
        RMSNorm {
            weight: vec![1.0f32; dim],
            eps: 1e-6,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let batch_size = x.dims()[0];
        let dim = x.dims()[1];
        
        let mut result = Vec::new();
        
        for b in 0..batch_size {
            let offset = b * dim;
            
            // Compute RMS
            let mut sum_sq = 0.0f32;
            for i in 0..dim {
                sum_sq += data[offset + i].powi(2);
            }
            let rms = (sum_sq / dim as f32 + self.eps).sqrt();
            
            // Normalize and scale
            for i in 0..dim {
                result.push(data[offset + i] / rms * self.weight[i]);
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

impl LLaMALayer {
    fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        LLaMALayer {
            attention: LLaMAAttention::new(d_model, n_heads),
            feed_forward: LLaMAFeedForward::new(d_model, d_ff),
            attention_norm: RMSNorm::new(d_model),
            ffn_norm: RMSNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, training);
        let x = self.add_tensors(x, &attn_out);
        
        let normed = self.ffn_norm.forward(&x);
        let ffn_out = self.feed_forward.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl LLaMA {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize, n_heads: usize, d_ff: usize) -> Self {
        LLaMA {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..n_layers).map(|_| LLaMALayer::new(d_model, n_heads, d_ff)).collect(),
            norm: RMSNorm::new(d_model),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        x = self.norm.forward(&x);
        self.lm_head.forward(&x, training)
    }
}

/// Mistral
pub struct Mistral {
    embedding: Dense,
    layers: Vec<MistralLayer>,
    norm: RMSNorm,
    lm_head: Dense,
}

struct MistralLayer {
    attention: MistralAttention,
    feed_forward: LLaMAFeedForward,
    attention_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

struct MistralAttention {
    wq: Dense,
    wk: Dense,
    wv: Dense,
    wo: Dense,
    n_heads: usize,
    sliding_window: usize,
}

impl MistralAttention {
    fn new(d_model: usize, n_heads: usize, sliding_window: usize) -> Self {
        MistralAttention {
            wq: Dense::new(d_model, d_model),
            wk: Dense::new(d_model, d_model),
            wv: Dense::new(d_model, d_model),
            wo: Dense::new(d_model, d_model),
            n_heads,
            sliding_window,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let q = self.wq.forward(x, training);
        let k = self.wk.forward(x, training);
        let v = self.wv.forward(x, training);
        
        // Simplified sliding window attention
        let attn_out = v;
        self.wo.forward(&attn_out, training)
    }
}

impl MistralLayer {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, sliding_window: usize) -> Self {
        MistralLayer {
            attention: MistralAttention::new(d_model, n_heads, sliding_window),
            feed_forward: LLaMAFeedForward::new(d_model, d_ff),
            attention_norm: RMSNorm::new(d_model),
            ffn_norm: RMSNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, training);
        let x = self.add_tensors(x, &attn_out);
        
        let normed = self.ffn_norm.forward(&x);
        let ffn_out = self.feed_forward.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl Mistral {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize, n_heads: usize, d_ff: usize) -> Self {
        Mistral {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..n_layers).map(|_| MistralLayer::new(d_model, n_heads, d_ff, 4096)).collect(),
            norm: RMSNorm::new(d_model),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        x = self.norm.forward(&x);
        self.lm_head.forward(&x, training)
    }
}

/// GPT-3 Architecture
pub struct GPT3 {
    embedding: Dense,
    position_embedding: Dense,
    layers: Vec<GPT3Layer>,
    norm: LayerNorm,
    lm_head: Dense,
}

struct GPT3Layer {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl GPT3Layer {
    fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        GPT3Layer {
            attention: MultiHeadAttention::new(d_model, n_heads),
            feed_forward: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.norm1.forward(x);
        let attn_out = self.attention.forward(&normed, &normed, &normed, training);
        let x = self.add_tensors(x, &attn_out);
        
        let normed = self.norm2.forward(&x);
        let ffn_out = self.feed_forward.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl GPT3 {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize, n_layers: usize, n_heads: usize, d_ff: usize) -> Self {
        GPT3 {
            embedding: Dense::new(vocab_size, d_model),
            position_embedding: Dense::new(max_seq_len, d_model),
            layers: (0..n_layers).map(|_| GPT3Layer::new(d_model, n_heads, d_ff)).collect(),
            norm: LayerNorm::new(d_model),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let token_emb = self.embedding.forward(input_ids, training);
        
        // Simplified position embedding
        let mut x = token_emb;
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        x = self.norm.forward(&x);
        self.lm_head.forward(&x, training)
    }
}

/// Falcon
pub struct Falcon {
    embedding: Dense,
    layers: Vec<FalconLayer>,
    norm: LayerNorm,
    lm_head: Dense,
}

struct FalconLayer {
    attention: FalconAttention,
    mlp: Dense,
    input_layernorm: LayerNorm,
}

struct FalconAttention {
    query_key_value: Dense,
    dense: Dense,
    n_heads: usize,
}

impl FalconAttention {
    fn new(d_model: usize, n_heads: usize) -> Self {
        FalconAttention {
            query_key_value: Dense::new(d_model, d_model * 3),
            dense: Dense::new(d_model, d_model),
            n_heads,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let qkv = self.query_key_value.forward(x, training);
        
        // Simplified attention
        self.dense.forward(&qkv, training)
    }
}

impl FalconLayer {
    fn new(d_model: usize, n_heads: usize) -> Self {
        FalconLayer {
            attention: FalconAttention::new(d_model, n_heads),
            mlp: Dense::new(d_model, d_model * 4),
            input_layernorm: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.input_layernorm.forward(x);
        
        let attn_out = self.attention.forward(&normed, training);
        let mlp_out = self.mlp.forward(&normed, training);
        
        let combined = self.add_tensors(&attn_out, &mlp_out);
        self.add_tensors(x, &combined)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl Falcon {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize, n_heads: usize) -> Self {
        Falcon {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..n_layers).map(|_| FalconLayer::new(d_model, n_heads)).collect(),
            norm: LayerNorm::new(d_model),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        x = self.norm.forward(&x);
        self.lm_head.forward(&x, training)
    }
}

/// MPT (MosaicML Pretrained Transformer)
pub struct MPT {
    embedding: Dense,
    layers: Vec<MPTLayer>,
    norm: LayerNorm,
    lm_head: Dense,
}

struct MPTLayer {
    attention: MPTAttention,
    ffn: Dense,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

struct MPTAttention {
    wqkv: Dense,
    out_proj: Dense,
}

impl MPTAttention {
    fn new(d_model: usize) -> Self {
        MPTAttention {
            wqkv: Dense::new(d_model, d_model * 3),
            out_proj: Dense::new(d_model, d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let qkv = self.wqkv.forward(x, training);
        self.out_proj.forward(&qkv, training)
    }
}

impl MPTLayer {
    fn new(d_model: usize, d_ff: usize) -> Self {
        MPTLayer {
            attention: MPTAttention::new(d_model),
            ffn: Dense::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let normed = self.norm1.forward(x);
        let attn_out = self.attention.forward(&normed, training);
        let x = self.add_tensors(x, &attn_out);
        
        let normed = self.norm2.forward(&x);
        let ffn_out = self.ffn.forward(&normed, training);
        self.add_tensors(&x, &ffn_out)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl MPT {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize, d_ff: usize) -> Self {
        MPT {
            embedding: Dense::new(vocab_size, d_model),
            layers: (0..n_layers).map(|_| MPTLayer::new(d_model, d_ff)).collect(),
            norm: LayerNorm::new(d_model),
            lm_head: Dense::new(d_model, vocab_size),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, training: bool) -> Tensor {
        let mut x = self.embedding.forward(input_ids, training);
        
        for layer in &mut self.layers {
            x = layer.forward(&x, training);
        }
        
        x = self.norm.forward(&x);
        self.lm_head.forward(&x, training)
    }
}

#[cfg(test)]
mod tests_llm {
    use super::*;

    #[test]
    fn test_llama() {
        let mut model = LLaMA::new(32000, 4096, 32, 32, 11008);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let output = model.forward(&input_ids, false);
        assert_eq!(output.dims()[0], 2);
    }

    #[test]
    fn test_mistral() {
        let mut model = Mistral::new(32000, 4096, 32, 32, 14336);
        let input_ids = Tensor::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]).unwrap();
        let output = model.forward(&input_ids, false);
        assert_eq!(output.dims()[0], 2);
    }
}

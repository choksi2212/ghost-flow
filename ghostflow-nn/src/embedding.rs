//! Embedding layers

use ghostflow_core::Tensor;
use crate::module::Module;
use crate::init;

/// Embedding layer - lookup table for discrete tokens
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    padding_idx: Option<usize>,
    training: bool,
}

impl Embedding {
    /// Create new embedding layer
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = init::normal(&[num_embeddings, embedding_dim], 0.0, 1.0);
        
        Embedding {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            training: true,
        }
    }

    /// Create embedding with padding index (embedding at padding_idx will be zeros)
    pub fn with_padding(num_embeddings: usize, embedding_dim: usize, padding_idx: usize) -> Self {
        let mut emb = Self::new(num_embeddings, embedding_dim);
        emb.padding_idx = Some(padding_idx);
        
        // Zero out padding embedding
        let mut weight_data = emb.weight.data_f32();
        let start = padding_idx * embedding_dim;
        for i in 0..embedding_dim {
            weight_data[start + i] = 0.0;
        }
        emb.weight = Tensor::from_slice(&weight_data, &[num_embeddings, embedding_dim]).unwrap();
        
        emb
    }

    /// Create embedding from pretrained weights
    pub fn from_pretrained(weight: Tensor, freeze: bool) -> Self {
        let dims = weight.dims();
        let num_embeddings = dims[0];
        let embedding_dim = dims[1];
        
        let emb = Embedding {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            training: !freeze,
        };
        
        emb
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get number of embeddings
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Forward pass with integer indices
    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        let weight_data = self.weight.data_f32();
        let mut output = Vec::with_capacity(indices.len() * self.embedding_dim);
        
        for &idx in indices {
            let start = idx * self.embedding_dim;
            output.extend_from_slice(&weight_data[start..start + self.embedding_dim]);
        }
        
        Tensor::from_slice(&output, &[indices.len(), self.embedding_dim]).unwrap()
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Input should be integer indices (stored as f32)
        let indices: Vec<usize> = input.data_f32()
            .iter()
            .map(|&x| x as usize)
            .collect();
        
        let input_shape = input.dims();
        let batch_dims: Vec<usize> = input_shape.to_vec();
        
        let weight_data = self.weight.data_f32();
        let mut output = Vec::with_capacity(indices.len() * self.embedding_dim);
        
        for &idx in &indices {
            if idx >= self.num_embeddings {
                // Out of bounds - use zeros
                output.extend(vec![0.0f32; self.embedding_dim]);
            } else {
                let start = idx * self.embedding_dim;
                output.extend_from_slice(&weight_data[start..start + self.embedding_dim]);
            }
        }
        
        // Output shape: input_shape + [embedding_dim]
        let mut output_shape = batch_dims;
        output_shape.push(self.embedding_dim);
        
        Tensor::from_slice(&output, &output_shape).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        if self.training {
            vec![self.weight.clone()]
        } else {
            vec![] // Frozen
        }
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}

/// Token + Position Embedding (common in transformers)
pub struct TokenPositionEmbedding {
    token_embedding: Embedding,
    position_embedding: Embedding,
    #[allow(dead_code)]
    dropout_p: f32,
    #[allow(dead_code)]
    max_seq_len: usize,
}

impl TokenPositionEmbedding {
    pub fn new(vocab_size: usize, embed_dim: usize, max_seq_len: usize, dropout: f32) -> Self {
        TokenPositionEmbedding {
            token_embedding: Embedding::new(vocab_size, embed_dim),
            position_embedding: Embedding::new(max_seq_len, embed_dim),
            dropout_p: dropout,
            max_seq_len,
        }
    }
}

impl Module for TokenPositionEmbedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.dims()[input.ndim() - 1];
        
        // Token embeddings
        let token_emb = self.token_embedding.forward(input);
        
        // Position indices
        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let pos_tensor = Tensor::from_slice(&positions, &[seq_len]).unwrap();
        let pos_emb = self.position_embedding.forward(&pos_tensor);
        
        // Add token and position embeddings
        token_emb.add(&pos_emb).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.token_embedding.parameters();
        params.extend(self.position_embedding.parameters());
        params
    }

    fn train(&mut self) {
        self.token_embedding.train();
        self.position_embedding.train();
    }

    fn eval(&mut self) {
        self.token_embedding.eval();
        self.position_embedding.eval();
    }

    fn is_training(&self) -> bool {
        self.token_embedding.is_training()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(100, 64);
        let indices = Tensor::from_slice(&[0.0f32, 5.0, 10.0], &[3]).unwrap();
        let output = emb.forward(&indices);
        
        assert_eq!(output.dims(), &[3, 64]);
    }

    #[test]
    fn test_embedding_batch() {
        let emb = Embedding::new(100, 64);
        let indices = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
        let output = emb.forward(&indices);
        
        assert_eq!(output.dims(), &[2, 3, 64]);
    }
}

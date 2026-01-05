# Sequence Models Guide

Complete guide to using LSTM, GRU, and Transformers in GhostFlow.

## Table of Contents

1. [Overview](#overview)
2. [LSTM (Long Short-Term Memory)](#lstm)
3. [GRU (Gated Recurrent Unit)](#gru)
4. [Transformers](#transformers)
5. [Comparison](#comparison)
6. [Use Cases](#use-cases)
7. [Best Practices](#best-practices)

---

## Overview

GhostFlow provides three powerful architectures for sequence modeling:

- **LSTM**: Best for long-term dependencies, complex patterns
- **GRU**: Faster than LSTM, good for most tasks
- **Transformer**: State-of-the-art for NLP, parallel processing

---

## LSTM

### What is LSTM?

Long Short-Term Memory networks are designed to remember information for long periods. They solve the vanishing gradient problem of traditional RNNs.

### Architecture

LSTM uses four gates:
- **Input Gate** (i): Controls what new information to store
- **Forget Gate** (f): Controls what information to discard
- **Cell Gate** (g): Creates new candidate values
- **Output Gate** (o): Controls what to output

### Usage

```rust
use ghostflow_nn::{LSTM, Module};
use ghostflow_core::Tensor;

// Create LSTM
let lstm = LSTM::new(
    input_size: 128,      // Input feature dimension
    hidden_size: 256,     // Hidden state dimension
    num_layers: 2,        // Number of stacked layers
    bidirectional: false, // Unidirectional
    dropout: 0.1,         // Dropout between layers
);

// Input: [batch, sequence_length, input_size]
let input = Tensor::randn(&[32, 50, 128]);

// Forward pass
let output = lstm.forward(&input);
// Output: [batch, sequence_length, hidden_size]
```

### Bidirectional LSTM

```rust
let bilstm = LSTM::new(128, 256, 1, true, 0.0);
let input = Tensor::randn(&[32, 50, 128]);
let output = bilstm.forward(&input);
// Output: [32, 50, 512] (256 * 2 directions)
```

### When to Use LSTM

✅ **Use LSTM when:**
- You need to capture long-term dependencies
- Sequence order is crucial
- You have complex temporal patterns
- You need bidirectional context

❌ **Avoid LSTM when:**
- You need real-time processing (use GRU)
- Sequences are very long (use Transformer)
- You have limited compute (use GRU)

---

## GRU

### What is GRU?

Gated Recurrent Unit is a simpler alternative to LSTM with fewer parameters and faster training.

### Architecture

GRU uses three gates:
- **Reset Gate** (r): Controls how much past information to forget
- **Update Gate** (z): Controls how much past information to keep
- **New Gate** (n): Creates new candidate hidden state

### Usage

```rust
use ghostflow_nn::{GRU, Module};
use ghostflow_core::Tensor;

// Create GRU
let gru = GRU::new(
    input_size: 128,
    hidden_size: 256,
    num_layers: 2,
    bidirectional: false,
    dropout: 0.1,
);

let input = Tensor::randn(&[32, 50, 128]);
let output = gru.forward(&input);
```

### GRU vs LSTM

| Feature | GRU | LSTM |
|---------|-----|------|
| Parameters | Fewer (3 gates) | More (4 gates) |
| Speed | Faster | Slower |
| Memory | Less | More |
| Performance | Similar | Slightly better on complex tasks |

### When to Use GRU

✅ **Use GRU when:**
- You want faster training
- You have limited memory
- LSTM and GRU perform similarly on your task
- You need a good baseline quickly

---

## Transformers

### What are Transformers?

Transformers use self-attention mechanisms to process sequences in parallel, making them much faster than RNNs.

### Architecture Components

1. **Multi-Head Attention**: Attends to different parts of the sequence
2. **Feed-Forward Network**: Processes each position independently
3. **Layer Normalization**: Stabilizes training
4. **Positional Encoding**: Adds position information

### Usage

```rust
use ghostflow_nn::{TransformerEncoder, Module};
use ghostflow_core::Tensor;

// Create Transformer Encoder
let transformer = TransformerEncoder::new(
    d_model: 512,      // Model dimension
    nhead: 8,          // Number of attention heads
    d_ff: 2048,        // Feed-forward dimension
    num_layers: 6,     // Number of encoder layers
    dropout: 0.1,      // Dropout probability
);

let input = Tensor::randn(&[32, 100, 512]);
let output = transformer.forward(&input);
```

### Multi-Head Attention

```rust
use ghostflow_nn::{MultiHeadAttention, Module};

let mha = MultiHeadAttention::new(
    embed_dim: 512,
    num_heads: 8,
    dropout: 0.1,
);

let input = Tensor::randn(&[32, 100, 512]);
let output = mha.forward(&input);
```

### Positional Encoding

```rust
use ghostflow_nn::{PositionalEncoding, Module};

let pe = PositionalEncoding::new(
    d_model: 512,
    max_len: 5000,
    dropout: 0.1,
);

let input = Tensor::randn(&[32, 100, 512]);
let output = pe.forward(&input); // Adds positional information
```

### When to Use Transformers

✅ **Use Transformers when:**
- You need state-of-the-art performance
- You have sufficient compute (GPU)
- Sequences are long (>100 tokens)
- You can train in parallel
- You're working on NLP tasks

❌ **Avoid Transformers when:**
- You have limited compute
- Sequences are very short (<20 tokens)
- You need real-time inference on CPU
- You have small datasets

---

## Comparison

### Performance

| Model | Speed | Memory | Long-term Deps | Parallelization |
|-------|-------|--------|----------------|-----------------|
| LSTM | Slow | High | Excellent | No |
| GRU | Medium | Medium | Good | No |
| Transformer | Fast* | Very High | Excellent | Yes |

*Fast during training with GPU, but has quadratic complexity with sequence length

### Parameter Count

For hidden_size=256:

- **GRU**: ~200K parameters per layer
- **LSTM**: ~260K parameters per layer
- **Transformer** (d_model=512, 8 heads): ~2.3M parameters per layer

---

## Use Cases

### Natural Language Processing

**Sentiment Analysis**
```rust
// Use LSTM or GRU for sentence classification
let lstm = LSTM::new(300, 128, 2, false, 0.1);
```

**Machine Translation**
```rust
// Use Transformer for best results
let encoder = TransformerEncoder::new(512, 8, 2048, 6, 0.1);
```

**Named Entity Recognition**
```rust
// Use Bidirectional LSTM
let bilstm = LSTM::new(100, 256, 1, true, 0.0);
```

### Time Series

**Stock Price Prediction**
```rust
// Use GRU for efficiency
let gru = GRU::new(10, 64, 1, false, 0.0);
```

**Weather Forecasting**
```rust
// Use LSTM for long-term patterns
let lstm = LSTM::new(20, 128, 2, false, 0.1);
```

### Speech Recognition

```rust
// Use Bidirectional LSTM or Transformer
let bilstm = LSTM::new(40, 256, 3, true, 0.2);
```

### Video Analysis

```rust
// Use LSTM for temporal modeling
let lstm = LSTM::new(2048, 512, 2, false, 0.1);
```

---

## Best Practices

### 1. Choosing the Right Model

```
Start with GRU → If not good enough → Try LSTM → If still not enough → Use Transformer
```

### 2. Hyperparameter Tuning

**Hidden Size**
- Start with 128-256 for most tasks
- Use 512-1024 for complex tasks
- Larger is not always better (overfitting risk)

**Number of Layers**
- 1-2 layers: Simple tasks
- 2-4 layers: Most tasks
- 4-6 layers: Complex tasks
- 6+ layers: Only with large datasets

**Dropout**
- 0.0-0.1: Small datasets
- 0.1-0.3: Medium datasets
- 0.3-0.5: Large datasets

### 3. Training Tips

**Gradient Clipping**
```rust
// Prevent exploding gradients
let max_norm = 1.0;
// Apply gradient clipping in your training loop
```

**Learning Rate**
- LSTM/GRU: 0.001 - 0.01
- Transformer: Use learning rate warmup

**Batch Size**
- LSTM/GRU: 32-128
- Transformer: 64-256 (larger is better with GPU)

### 4. Sequence Length

**LSTM/GRU**
- Optimal: 20-100 tokens
- Maximum: 500 tokens
- Beyond 500: Consider truncation or Transformer

**Transformer**
- Optimal: 100-512 tokens
- Maximum: 2048 tokens (with sufficient memory)
- Memory usage: O(n²) where n is sequence length

### 5. Bidirectional vs Unidirectional

**Use Bidirectional when:**
- You have access to the full sequence
- Task: Classification, NER, POS tagging

**Use Unidirectional when:**
- You need online/streaming processing
- Task: Language modeling, generation

### 6. Debugging

**Check shapes**
```rust
println!("Input: {:?}", input.dims());
println!("Output: {:?}", output.dims());
```

**Monitor gradients**
```rust
// Check for vanishing/exploding gradients
for param in model.parameters() {
    let grad_norm = param.grad().unwrap().norm();
    println!("Gradient norm: {}", grad_norm);
}
```

---

## Examples

### Complete Sentiment Analysis

```rust
use ghostflow_nn::{LSTM, Linear, Module};
use ghostflow_core::Tensor;

// 1. Embedding layer (vocabulary → vectors)
let vocab_size = 10000;
let embed_dim = 128;

// 2. LSTM encoder
let lstm = LSTM::new(embed_dim, 256, 2, false, 0.1);

// 3. Classification head
let classifier = Linear::new(256, 2); // Binary classification

// 4. Forward pass
let input = Tensor::randn(&[32, 50, embed_dim]); // [batch, seq, embed]
let encoded = lstm.forward(&input);              // [32, 50, 256]

// Take last hidden state
let last_hidden = encoded.slice(1, 49, 50).unwrap(); // [32, 1, 256]
let last_hidden = last_hidden.reshape(&[32, 256]).unwrap();

// Classify
let logits = classifier.forward(&last_hidden);   // [32, 2]
```

### Sequence-to-Sequence Translation

```rust
// Encoder
let encoder = LSTM::new(256, 512, 2, false, 0.1);

// Decoder
let decoder = LSTM::new(256, 512, 2, false, 0.1);

// Encode source
let source = Tensor::randn(&[32, 20, 256]);
let encoder_output = encoder.forward(&source);

// Decode target (teacher forcing)
let target = Tensor::randn(&[32, 25, 256]);
let decoder_output = decoder.forward(&target);
```

### Transformer Language Model

```rust
use ghostflow_nn::{TransformerEncoder, PositionalEncoding, Linear, Module};

// 1. Positional encoding
let pe = PositionalEncoding::new(512, 5000, 0.1);

// 2. Transformer encoder
let transformer = TransformerEncoder::new(512, 8, 2048, 6, 0.1);

// 3. Output projection
let output_proj = Linear::new(512, vocab_size);

// Forward
let input = Tensor::randn(&[32, 100, 512]);
let pos_encoded = pe.forward(&input);
let encoded = transformer.forward(&pos_encoded);
let logits = output_proj.forward(&encoded);
```

---

## Performance Optimization

### 1. Use GPU

```rust
// Enable CUDA feature
// Cargo.toml: ghostflow = { version = "0.1", features = ["cuda"] }
```

### 2. Batch Processing

```rust
// Process multiple sequences together
let batch_size = 64; // Larger batches = better GPU utilization
```

### 3. Sequence Packing

```rust
// Pack sequences of similar length together
// Reduces padding overhead
```

### 4. Mixed Precision Training

```rust
// Use FP16 for faster training (coming in v0.4.0)
```

---

## Troubleshooting

### Issue: Vanishing Gradients

**Solution:**
- Use LSTM/GRU instead of vanilla RNN
- Use gradient clipping
- Use residual connections
- Reduce number of layers

### Issue: Exploding Gradients

**Solution:**
```rust
// Clip gradients
let max_norm = 1.0;
// Apply clipping in training loop
```

### Issue: Slow Training

**Solution:**
- Use GRU instead of LSTM
- Reduce hidden size
- Reduce number of layers
- Use GPU
- Increase batch size

### Issue: Out of Memory

**Solution:**
- Reduce batch size
- Reduce sequence length
- Reduce hidden size
- Use gradient checkpointing (coming soon)

---

## References

- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [GRU Paper](https://arxiv.org/abs/1406.1078)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [GhostFlow Documentation](../README.md)

---

**Next Steps:**
- Try the [examples](../examples/sequence_models.rs)
- Read the [API documentation](https://docs.rs/ghostflow)
- Join our [community](https://github.com/choksi2212/ghost-flow/discussions)

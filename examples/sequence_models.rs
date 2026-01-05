//! Sequence Modeling Examples
//!
//! Demonstrates LSTM, GRU, and Transformer usage for sequence tasks.

use ghostflow_core::Tensor;
use ghostflow_nn::{LSTM, GRU, TransformerEncoder, Module};

fn main() {
    println!("=== GhostFlow Sequence Models Demo ===\n");
    
    // Example 1: LSTM for sequence classification
    lstm_example();
    
    // Example 2: GRU for sequence-to-sequence
    gru_example();
    
    // Example 3: Transformer for language modeling
    transformer_example();
    
    // Example 4: Bidirectional LSTM
    bidirectional_lstm_example();
}

fn lstm_example() {
    println!("1. LSTM for Sequence Classification");
    println!("   Task: Sentiment analysis on text sequences\n");
    
    // Create LSTM: input_size=50 (embedding dim), hidden_size=128
    let lstm = LSTM::new(
        50,    // input_size (word embedding dimension)
        128,   // hidden_size
        2,     // num_layers
        false, // bidirectional
        0.1,   // dropout
    );
    
    // Input: [batch=4, seq_len=20, features=50]
    // Represents 4 sentences, each with 20 words, 50-dim embeddings
    let input = Tensor::randn(&[4, 20, 50]);
    
    println!("   Input shape: {:?}", input.dims());
    
    // Forward pass
    let output = lstm.forward(&input);
    
    println!("   Output shape: {:?}", output.dims());
    println!("   ✓ LSTM processes sequences maintaining temporal information\n");
}

fn gru_example() {
    println!("2. GRU for Sequence-to-Sequence");
    println!("   Task: Machine translation\n");
    
    // GRU is lighter than LSTM (fewer parameters)
    let gru = GRU::new(
        256,   // input_size
        512,   // hidden_size
        1,     // num_layers
        false, // bidirectional
        0.0,   // dropout
    );
    
    // Input: [batch=2, seq_len=15, features=256]
    let input = Tensor::randn(&[2, 15, 256]);
    
    println!("   Input shape: {:?}", input.dims());
    
    let output = gru.forward(&input);
    
    println!("   Output shape: {:?}", output.dims());
    println!("   ✓ GRU is faster than LSTM with similar performance\n");
}

fn transformer_example() {
    println!("3. Transformer for Language Modeling");
    println!("   Task: Next token prediction\n");
    
    // Transformer encoder: state-of-the-art for NLP
    let transformer = TransformerEncoder::new(
        512,  // d_model (embedding dimension)
        8,    // num_heads (multi-head attention)
        2048, // d_ff (feed-forward dimension)
        6,    // num_layers
        0.1,  // dropout
    );
    
    // Input: [batch=8, seq_len=128, d_model=512]
    let input = Tensor::randn(&[8, 128, 512]);
    
    println!("   Input shape: {:?}", input.dims());
    
    let output = transformer.forward(&input);
    
    println!("   Output shape: {:?}", output.dims());
    println!("   ✓ Transformer uses self-attention for parallel processing\n");
}

fn bidirectional_lstm_example() {
    println!("4. Bidirectional LSTM");
    println!("   Task: Named Entity Recognition (NER)\n");
    
    // Bidirectional LSTM processes sequence in both directions
    let bilstm = LSTM::new(
        100,   // input_size
        256,   // hidden_size
        1,     // num_layers
        true,  // bidirectional = true
        0.0,   // dropout
    );
    
    // Input: [batch=3, seq_len=25, features=100]
    let input = Tensor::randn(&[3, 25, 100]);
    
    println!("   Input shape: {:?}", input.dims());
    
    let output = bilstm.forward(&input);
    
    println!("   Output shape: {:?}", output.dims());
    println!("   Note: Output has 512 features (256 * 2 directions)");
    println!("   ✓ Bidirectional models capture context from both past and future\n");
}

/// Example: Complete sentiment analysis pipeline
#[allow(dead_code)]
fn sentiment_analysis_pipeline() {
    println!("=== Complete Sentiment Analysis Pipeline ===\n");
    
    // 1. Word embeddings (vocabulary_size=10000, embedding_dim=128)
    let vocab_size = 10000;
    let embed_dim = 128;
    let hidden_size = 256;
    
    // 2. LSTM encoder
    let lstm = LSTM::new(embed_dim, hidden_size, 2, false, 0.1);
    
    // 3. Sample input (batch=32, seq_len=50)
    // In practice, this would be word indices converted to embeddings
    let embeddings = Tensor::randn(&[32, 50, embed_dim]);
    
    // 4. Encode sequence
    let encoded = lstm.forward(&embeddings);
    
    // 5. Take last hidden state for classification
    // In practice, you'd add a Linear layer here for final classification
    println!("Encoded shape: {:?}", encoded.dims());
    println!("Ready for classification layer (Linear(256, num_classes))");
}

/// Example: Sequence-to-sequence translation
#[allow(dead_code)]
fn seq2seq_translation() {
    println!("=== Sequence-to-Sequence Translation ===\n");
    
    let embed_dim = 256;
    let hidden_size = 512;
    
    // Encoder (processes source language)
    let encoder = LSTM::new(embed_dim, hidden_size, 2, false, 0.1);
    
    // Decoder (generates target language)
    let decoder = LSTM::new(embed_dim, hidden_size, 2, false, 0.1);
    
    // Source sentence: "Hello world" (batch=1, seq_len=2)
    let source = Tensor::randn(&[1, 2, embed_dim]);
    
    // Encode
    let encoder_output = encoder.forward(&source);
    
    // Decode (in practice, this would be autoregressive)
    let target_input = Tensor::randn(&[1, 3, embed_dim]); // "Hola mundo"
    let decoder_output = decoder.forward(&target_input);
    
    println!("Encoder output: {:?}", encoder_output.dims());
    println!("Decoder output: {:?}", decoder_output.dims());
}

/// Example: Time series forecasting with GRU
#[allow(dead_code)]
fn time_series_forecasting() {
    println!("=== Time Series Forecasting with GRU ===\n");
    
    let input_features = 10;  // e.g., temperature, humidity, pressure, etc.
    let hidden_size = 64;
    
    let gru = GRU::new(input_features, hidden_size, 1, false, 0.0);
    
    // Historical data: 30 days of measurements
    let history = Tensor::randn(&[1, 30, input_features]);
    
    let forecast_features = gru.forward(&history);
    
    println!("Historical data: {:?}", history.dims());
    println!("Forecast features: {:?}", forecast_features.dims());
    println!("Add Linear layer to predict next values");
}

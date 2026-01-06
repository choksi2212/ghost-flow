//! Natural Language Processing - Tokenizers, Embeddings, Text Processing
//!
//! This module provides NLP utilities for text preprocessing and representation.

use std::collections::HashMap;

/// Word-level tokenizer
pub struct WordTokenizer {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub min_word_length: usize,
    vocab_: Option<HashMap<String, usize>>,
    index_to_word_: Option<Vec<String>>,
}

impl WordTokenizer {
    pub fn new() -> Self {
        WordTokenizer {
            lowercase: true,
            remove_punctuation: true,
            min_word_length: 1,
            vocab_: None,
            index_to_word_: None,
        }
    }

    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    pub fn remove_punctuation(mut self, remove: bool) -> Self {
        self.remove_punctuation = remove;
        self
    }

    pub fn min_word_length(mut self, length: usize) -> Self {
        self.min_word_length = length;
        self
    }

    fn preprocess(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        if self.lowercase {
            processed = processed.to_lowercase();
        }

        if self.remove_punctuation {
            processed = processed.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }

        processed
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let processed = self.preprocess(text);
        
        processed.split_whitespace()
            .filter(|word| word.len() >= self.min_word_length)
            .map(|s| s.to_string())
            .collect()
    }

    pub fn fit(&mut self, texts: &[String]) {
        let mut vocab = HashMap::new();
        let mut index = 0;

        for text in texts {
            let tokens = self.tokenize(text);
            for token in tokens {
                if !vocab.contains_key(&token) {
                    vocab.insert(token.clone(), index);
                    index += 1;
                }
            }
        }

        let mut index_to_word = vec![String::new(); vocab.len()];
        for (word, &idx) in &vocab {
            index_to_word[idx] = word.clone();
        }

        self.vocab_ = Some(vocab);
        self.index_to_word_ = Some(index_to_word);
    }

    pub fn texts_to_sequences(&self, texts: &[String]) -> Vec<Vec<usize>> {
        let vocab = self.vocab_.as_ref().expect("Tokenizer not fitted");
        
        texts.iter()
            .map(|text| {
                self.tokenize(text)
                    .iter()
                    .filter_map(|token| vocab.get(token).copied())
                    .collect()
            })
            .collect()
    }

    pub fn sequences_to_texts(&self, sequences: &[Vec<usize>]) -> Vec<String> {
        let index_to_word = self.index_to_word_.as_ref().expect("Tokenizer not fitted");
        
        sequences.iter()
            .map(|seq| {
                seq.iter()
                    .filter_map(|&idx| {
                        if idx < index_to_word.len() {
                            Some(index_to_word[idx].clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    pub fn vocab(&self) -> Option<&HashMap<String, usize>> {
        self.vocab_.as_ref()
    }
}

impl Default for WordTokenizer {
    fn default() -> Self { Self::new() }
}

/// Character-level tokenizer
pub struct CharTokenizer {
    pub lowercase: bool,
    char_to_idx_: Option<HashMap<char, usize>>,
    idx_to_char_: Option<Vec<char>>,
}

impl CharTokenizer {
    pub fn new() -> Self {
        CharTokenizer {
            lowercase: true,
            char_to_idx_: None,
            idx_to_char_: None,
        }
    }

    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    pub fn fit(&mut self, texts: &[String]) {
        let mut chars = std::collections::HashSet::new();

        for text in texts {
            let processed = if self.lowercase {
                text.to_lowercase()
            } else {
                text.clone()
            };

            for c in processed.chars() {
                chars.insert(c);
            }
        }

        let mut char_to_idx = HashMap::new();
        let mut idx_to_char = Vec::new();

        for (i, c) in chars.into_iter().enumerate() {
            char_to_idx.insert(c, i);
            idx_to_char.push(c);
        }

        self.char_to_idx_ = Some(char_to_idx);
        self.idx_to_char_ = Some(idx_to_char);
    }

    pub fn texts_to_sequences(&self, texts: &[String]) -> Vec<Vec<usize>> {
        let char_to_idx = self.char_to_idx_.as_ref().expect("Tokenizer not fitted");
        
        texts.iter()
            .map(|text| {
                let processed = if self.lowercase {
                    text.to_lowercase()
                } else {
                    text.clone()
                };

                processed.chars()
                    .filter_map(|c| char_to_idx.get(&c).copied())
                    .collect()
            })
            .collect()
    }

    pub fn sequences_to_texts(&self, sequences: &[Vec<usize>]) -> Vec<String> {
        let idx_to_char = self.idx_to_char_.as_ref().expect("Tokenizer not fitted");
        
        sequences.iter()
            .map(|seq| {
                seq.iter()
                    .filter_map(|&idx| {
                        if idx < idx_to_char.len() {
                            Some(idx_to_char[idx])
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.char_to_idx_.as_ref().map(|v| v.len()).unwrap_or(0)
    }
}

impl Default for CharTokenizer {
    fn default() -> Self { Self::new() }
}

/// BPE (Byte Pair Encoding) Tokenizer
pub struct BPETokenizer {
    pub vocab_size: usize,
    merges_: Vec<(String, String)>,
    vocab_: HashMap<String, usize>,
}

impl BPETokenizer {
    pub fn new(vocab_size: usize) -> Self {
        BPETokenizer {
            vocab_size,
            merges_: Vec::new(),
            vocab_: HashMap::new(),
        }
    }

    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        for i in 0..word.len().saturating_sub(1) {
            pairs.push((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    pub fn fit(&mut self, texts: &[String]) {
        // Initialize vocabulary with characters
        let mut vocab: HashMap<Vec<String>, usize> = HashMap::new();
        
        for text in texts {
            for word in text.split_whitespace() {
                let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                *vocab.entry(chars).or_insert(0) += 1;
            }
        }

        // Learn merges
        for _ in 0..self.vocab_size {
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();

            for (word, &freq) in &vocab {
                let pairs = Self::get_pairs(word);
                for pair in pairs {
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_freqs.is_empty() {
                break;
            }

            let best_pair = pair_freqs.iter()
                .max_by_key(|(_, &freq)| freq)
                .map(|(pair, _)| pair.clone());

            if let Some((first, second)) = best_pair {
                self.merges_.push((first.clone(), second.clone()));

                // Update vocabulary
                let mut new_vocab = HashMap::new();
                for (word, freq) in vocab {
                    let new_word = self.merge_pair(&word, &first, &second);
                    new_vocab.insert(new_word, freq);
                }
                vocab = new_vocab;
            } else {
                break;
            }
        }

        // Build final vocabulary
        let mut idx = 0;
        for (word, _) in vocab {
            for token in word {
                if !self.vocab_.contains_key(&token) {
                    self.vocab_.insert(token, idx);
                    idx += 1;
                }
            }
        }
    }

    fn merge_pair(&self, word: &[String], first: &str, second: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < word.len() {
            if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                result.push(format!("{}{}", first, second));
                i += 2;
            } else {
                result.push(word[i].clone());
                i += 1;
            }
        }

        result
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            for (first, second) in &self.merges_ {
                chars = self.merge_pair(&chars, first, second);
            }

            for token in chars {
                if let Some(&idx) = self.vocab_.get(&token) {
                    tokens.push(idx);
                }
            }
        }

        tokens
    }

    pub fn vocab_size_actual(&self) -> usize {
        self.vocab_.len()
    }
}

/// TF-IDF Vectorizer
pub struct TfidfVectorizer {
    pub max_features: Option<usize>,
    pub min_df: usize,
    pub max_df: f32,
    tokenizer: WordTokenizer,
    idf_: Option<Vec<f32>>,
    vocab_: Option<HashMap<String, usize>>,
}

impl TfidfVectorizer {
    pub fn new() -> Self {
        TfidfVectorizer {
            max_features: None,
            min_df: 1,
            max_df: 1.0,
            tokenizer: WordTokenizer::new(),
            idf_: None,
            vocab_: None,
        }
    }

    pub fn max_features(mut self, max: usize) -> Self {
        self.max_features = Some(max);
        self
    }

    pub fn min_df(mut self, min: usize) -> Self {
        self.min_df = min;
        self
    }

    pub fn fit(&mut self, texts: &[String]) {
        // Build vocabulary
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let n_docs = texts.len();

        for text in texts {
            let tokens = self.tokenizer.tokenize(text);
            let unique_tokens: std::collections::HashSet<_> = tokens.into_iter().collect();
            
            for token in unique_tokens {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Filter by document frequency
        let max_df_count = (self.max_df * n_docs as f32) as usize;
        let mut filtered_vocab: Vec<(String, usize)> = doc_freq.into_iter()
            .filter(|(_, freq)| *freq >= self.min_df && *freq <= max_df_count)
            .collect();

        // Limit vocabulary size
        if let Some(max_feat) = self.max_features {
            filtered_vocab.sort_by(|a, b| b.1.cmp(&a.1));
            filtered_vocab.truncate(max_feat);
        }

        // Build vocabulary and IDF
        let mut vocab = HashMap::new();
        let mut idf = Vec::new();

        for (i, (term, df)) in filtered_vocab.iter().enumerate() {
            vocab.insert(term.clone(), i);
            // IDF = log((n_docs + 1) / (df + 1)) + 1
            let idf_value = ((n_docs + 1) as f32 / (*df + 1) as f32).ln() + 1.0;
            idf.push(idf_value);
        }

        self.vocab_ = Some(vocab);
        self.idf_ = Some(idf);
    }

    pub fn transform(&self, texts: &[String]) -> Vec<Vec<f32>> {
        let vocab = self.vocab_.as_ref().expect("Vectorizer not fitted");
        let idf = self.idf_.as_ref().unwrap();
        let vocab_size = vocab.len();

        texts.iter()
            .map(|text| {
                let tokens = self.tokenizer.tokenize(text);
                let mut tf = vec![0.0f32; vocab_size];

                // Count term frequencies
                for token in &tokens {
                    if let Some(&idx) = vocab.get(token) {
                        tf[idx] += 1.0;
                    }
                }

                // Normalize TF
                let total: f32 = tf.iter().sum();
                if total > 0.0 {
                    for t in &mut tf {
                        *t /= total;
                    }
                }

                // Apply IDF
                for (i, t) in tf.iter_mut().enumerate() {
                    *t *= idf[i];
                }

                // L2 normalization
                let norm: f32 = tf.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for t in &mut tf {
                        *t /= norm;
                    }
                }

                tf
            })
            .collect()
    }

    pub fn fit_transform(&mut self, texts: &[String]) -> Vec<Vec<f32>> {
        self.fit(texts);
        self.transform(texts)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_.as_ref().map(|v| v.len()).unwrap_or(0)
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self { Self::new() }
}

/// Word2Vec Skip-gram model (simplified)
pub struct Word2Vec {
    pub embedding_dim: usize,
    pub window_size: usize,
    pub min_count: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    embeddings_: Option<Vec<Vec<f32>>>,
    vocab_: Option<HashMap<String, usize>>,
}

impl Word2Vec {
    pub fn new(embedding_dim: usize) -> Self {
        Word2Vec {
            embedding_dim,
            window_size: 5,
            min_count: 5,
            learning_rate: 0.025,
            epochs: 5,
            embeddings_: None,
            vocab_: None,
        }
    }

    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn min_count(mut self, count: usize) -> Self {
        self.min_count = count;
        self
    }

    pub fn fit(&mut self, texts: &[String]) {
        // Build vocabulary
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let tokenizer = WordTokenizer::new();

        for text in texts {
            let tokens = tokenizer.tokenize(text);
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Filter by min_count
        let mut vocab = HashMap::new();
        let mut idx = 0;
        for (word, count) in word_counts {
            if count >= self.min_count {
                vocab.insert(word, idx);
                idx += 1;
            }
        }

        let vocab_size = vocab.len();

        // Initialize embeddings randomly
        use rand::prelude::*;
        let mut rng = thread_rng();
        let mut embeddings = vec![vec![0.0f32; self.embedding_dim]; vocab_size];
        
        for emb in &mut embeddings {
            for val in emb {
                *val = (rng.gen::<f32>() - 0.5) / self.embedding_dim as f32;
            }
        }

        // Training (simplified skip-gram)
        for _epoch in 0..self.epochs {
            for text in texts {
                let tokens = tokenizer.tokenize(text);
                let indices: Vec<usize> = tokens.iter()
                    .filter_map(|t| vocab.get(t).copied())
                    .collect();

                for (i, &center_idx) in indices.iter().enumerate() {
                    let start = i.saturating_sub(self.window_size);
                    let end = (i + self.window_size + 1).min(indices.len());

                    for j in start..end {
                        if i == j { continue; }
                        let context_idx = indices[j];

                        // Simplified gradient update
                        for d in 0..self.embedding_dim {
                            let grad = embeddings[context_idx][d] - embeddings[center_idx][d];
                            embeddings[center_idx][d] += self.learning_rate * grad * 0.01;
                        }
                    }
                }
            }
        }

        self.embeddings_ = Some(embeddings);
        self.vocab_ = Some(vocab);
    }

    pub fn get_vector(&self, word: &str) -> Option<&[f32]> {
        let vocab = self.vocab_.as_ref()?;
        let embeddings = self.embeddings_.as_ref()?;
        let idx = vocab.get(word)?;
        Some(&embeddings[*idx])
    }

    pub fn similarity(&self, word1: &str, word2: &str) -> Option<f32> {
        let vec1 = self.get_vector(word1)?;
        let vec2 = self.get_vector(word2)?;

        let dot: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        Some(dot / (norm1 * norm2).max(1e-10))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_.as_ref().map(|v| v.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_tokenizer() {
        let mut tokenizer = WordTokenizer::new();
        let texts = vec![
            "Hello world".to_string(),
            "Hello Rust".to_string(),
        ];

        tokenizer.fit(&texts);
        assert_eq!(tokenizer.vocab_size(), 3);

        let sequences = tokenizer.texts_to_sequences(&texts);
        assert_eq!(sequences.len(), 2);
        assert_eq!(sequences[0].len(), 2);
    }

    #[test]
    fn test_char_tokenizer() {
        let mut tokenizer = CharTokenizer::new();
        let texts = vec!["abc".to_string(), "def".to_string()];

        tokenizer.fit(&texts);
        assert_eq!(tokenizer.vocab_size(), 6);

        let sequences = tokenizer.texts_to_sequences(&texts);
        assert_eq!(sequences[0].len(), 3);
    }

    #[test]
    fn test_tfidf() {
        let texts = vec![
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
        ];

        let mut vectorizer = TfidfVectorizer::new();
        let vectors = vectorizer.fit_transform(&texts);

        assert_eq!(vectors.len(), 2);
        assert!(vectors[0].len() > 0);
    }

    #[test]
    fn test_word2vec() {
        let texts = vec![
            "the quick brown fox jumps".to_string(),
            "the lazy dog sleeps".to_string(),
        ];

        let mut w2v = Word2Vec::new(10).min_count(1);
        w2v.epochs = 2;
        w2v.fit(&texts);

        assert!(w2v.vocab_size() > 0);
        assert!(w2v.get_vector("the").is_some());
    }
}



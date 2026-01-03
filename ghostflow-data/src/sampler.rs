//! Samplers for data loading

use rand::seq::SliceRandom;
use rand::thread_rng;

/// Trait for sampling indices
pub trait Sampler: Iterator<Item = usize> {
    fn len(&self) -> usize;
}

/// Sequential sampler - returns indices in order
pub struct SequentialSampler {
    current: usize,
    len: usize,
}

impl SequentialSampler {
    pub fn new(len: usize) -> Self {
        SequentialSampler { current: 0, len }
    }
}

impl Iterator for SequentialSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.len {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

impl Sampler for SequentialSampler {
    fn len(&self) -> usize {
        self.len
    }
}

/// Random sampler - returns indices in random order
pub struct RandomSampler {
    indices: Vec<usize>,
    current: usize,
}

impl RandomSampler {
    pub fn new(len: usize) -> Self {
        let mut indices: Vec<usize> = (0..len).collect();
        indices.shuffle(&mut thread_rng());
        RandomSampler { indices, current: 0 }
    }
}

impl Iterator for RandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

impl Sampler for RandomSampler {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Weighted random sampler
pub struct WeightedRandomSampler {
    indices: Vec<usize>,
    current: usize,
}

impl WeightedRandomSampler {
    pub fn new(weights: &[f32], num_samples: usize, replacement: bool) -> Self {
        let total_weight: f32 = weights.iter().sum();
        let normalized: Vec<f32> = weights.iter().map(|w| w / total_weight).collect();
        
        let mut indices = Vec::with_capacity(num_samples);
        let mut available: Vec<usize> = (0..weights.len()).collect();
        
        for _ in 0..num_samples {
            // Simple weighted selection
            let r: f32 = rand::random();
            let mut cumsum = 0.0f32;
            
            for (i, &w) in normalized.iter().enumerate() {
                cumsum += w;
                if r < cumsum {
                    if replacement {
                        indices.push(i);
                    } else if available.contains(&i) {
                        indices.push(i);
                        available.retain(|&x| x != i);
                    }
                    break;
                }
            }
        }
        
        WeightedRandomSampler { indices, current: 0 }
    }
}

impl Iterator for WeightedRandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

impl Sampler for WeightedRandomSampler {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Batch sampler - yields batches of indices
pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        BatchSampler {
            sampler,
            batch_size,
            drop_last,
        }
    }
}

impl<S: Sampler> Iterator for BatchSampler<S> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        
        for _ in 0..self.batch_size {
            if let Some(idx) = self.sampler.next() {
                batch.push(idx);
            } else {
                break;
            }
        }
        
        if batch.is_empty() {
            None
        } else if self.drop_last && batch.len() < self.batch_size {
            None
        } else {
            Some(batch)
        }
    }
}

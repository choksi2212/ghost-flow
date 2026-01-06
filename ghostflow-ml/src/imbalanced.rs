//! Imbalanced Learning - SMOTE, Random Over/Under Sampling

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Random Over Sampler - duplicates minority class samples
pub struct RandomOverSampler {
    pub sampling_strategy: SamplingStrategy,
    pub random_state: Option<u64>,
}

#[derive(Clone)]
pub enum SamplingStrategy {
    /// Balance all classes to the majority class count
    Auto,
    /// Specific ratio of minority to majority
    Ratio(f32),
    /// Target count for each class
    Counts(Vec<(usize, usize)>),
}

impl RandomOverSampler {
    pub fn new() -> Self {
        RandomOverSampler {
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
        }
    }

    pub fn sampling_strategy(mut self, s: SamplingStrategy) -> Self {
        self.sampling_strategy = s;
        self
    }

    pub fn fit_resample(&self, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let _n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Count classes
        let mut class_counts: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();
        for (i, &label) in y_data.iter().enumerate() {
            class_counts.entry(label as i32).or_default().push(i);
        }

        let max_count = class_counts.values().map(|v| v.len()).max().unwrap_or(0);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let mut new_x = x_data.clone();
        let mut new_y = y_data.clone();

        for (class, indices) in &class_counts {
            let target_count = match &self.sampling_strategy {
                SamplingStrategy::Auto => max_count,
                SamplingStrategy::Ratio(r) => (max_count as f32 * r) as usize,
                SamplingStrategy::Counts(counts) => {
                    counts.iter().find(|(c, _)| *c == *class as usize)
                        .map(|(_, count)| *count)
                        .unwrap_or(indices.len())
                }
            };

            let n_to_add = target_count.saturating_sub(indices.len());
            
            for _ in 0..n_to_add {
                let idx = indices[rng.gen_range(0..indices.len())];
                new_x.extend_from_slice(&x_data[idx * n_features..(idx + 1) * n_features]);
                new_y.push(*class as f32);
            }
        }

        let new_n_samples = new_y.len();
        (
            Tensor::from_slice(&new_x, &[new_n_samples, n_features]).unwrap(),
            Tensor::from_slice(&new_y, &[new_n_samples]).unwrap()
        )
    }
}

impl Default for RandomOverSampler {
    fn default() -> Self { Self::new() }
}

/// Random Under Sampler - removes majority class samples
pub struct RandomUnderSampler {
    pub sampling_strategy: SamplingStrategy,
    pub random_state: Option<u64>,
    pub replacement: bool,
}

impl RandomUnderSampler {
    pub fn new() -> Self {
        RandomUnderSampler {
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
            replacement: false,
        }
    }

    pub fn sampling_strategy(mut self, s: SamplingStrategy) -> Self {
        self.sampling_strategy = s;
        self
    }

    pub fn fit_resample(&self, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let _n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut class_counts: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();
        for (i, &label) in y_data.iter().enumerate() {
            class_counts.entry(label as i32).or_default().push(i);
        }

        let min_count = class_counts.values().map(|v| v.len()).min().unwrap_or(0);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let mut selected_indices = Vec::new();

        for (class, indices) in &class_counts {
            let target_count = match &self.sampling_strategy {
                SamplingStrategy::Auto => min_count,
                SamplingStrategy::Ratio(r) => (min_count as f32 / r) as usize,
                SamplingStrategy::Counts(counts) => {
                    counts.iter().find(|(c, _)| *c == *class as usize)
                        .map(|(_, count)| *count)
                        .unwrap_or(indices.len())
                }
            };

            let n_to_select = target_count.min(indices.len());
            let mut class_indices = indices.clone();
            class_indices.shuffle(&mut rng);
            selected_indices.extend(class_indices.into_iter().take(n_to_select));
        }

        selected_indices.sort();

        let new_x: Vec<f32> = selected_indices.iter()
            .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
            .collect();
        let new_y: Vec<f32> = selected_indices.iter().map(|&i| y_data[i]).collect();

        let new_n_samples = new_y.len();
        (
            Tensor::from_slice(&new_x, &[new_n_samples, n_features]).unwrap(),
            Tensor::from_slice(&new_y, &[new_n_samples]).unwrap()
        )
    }
}

impl Default for RandomUnderSampler {
    fn default() -> Self { Self::new() }
}

/// SMOTE - Synthetic Minority Over-sampling Technique
pub struct SMOTE {
    pub k_neighbors: usize,
    pub sampling_strategy: SamplingStrategy,
    pub random_state: Option<u64>,
}

impl SMOTE {
    pub fn new() -> Self {
        SMOTE {
            k_neighbors: 5,
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
        }
    }

    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    pub fn sampling_strategy(mut self, s: SamplingStrategy) -> Self {
        self.sampling_strategy = s;
        self
    }

    fn find_k_neighbors(&self, point: &[f32], candidates: &[Vec<f32>], k: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = candidates.iter()
            .enumerate()
            .map(|(i, c)| {
                let dist: f32 = point.iter().zip(c.iter())
                    .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt();
                (i, dist)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().skip(1).take(k).map(|(i, _)| i).collect()
    }

    pub fn fit_resample(&self, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let _n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut class_samples: std::collections::HashMap<i32, Vec<Vec<f32>>> = std::collections::HashMap::new();
        for (i, &label) in y_data.iter().enumerate() {
            let sample: Vec<f32> = x_data[i * n_features..(i + 1) * n_features].to_vec();
            class_samples.entry(label as i32).or_default().push(sample);
        }

        let max_count = class_samples.values().map(|v| v.len()).max().unwrap_or(0);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let mut new_x = x_data.clone();
        let mut new_y = y_data.clone();

        for (class, samples) in &class_samples {
            let target_count = match &self.sampling_strategy {
                SamplingStrategy::Auto => max_count,
                SamplingStrategy::Ratio(r) => (max_count as f32 * r) as usize,
                SamplingStrategy::Counts(counts) => {
                    counts.iter().find(|(c, _)| *c == *class as usize)
                        .map(|(_, count)| *count)
                        .unwrap_or(samples.len())
                }
            };

            let n_to_generate = target_count.saturating_sub(samples.len());
            let k = self.k_neighbors.min(samples.len() - 1).max(1);

            for _ in 0..n_to_generate {
                // Pick a random sample
                let idx = rng.gen_range(0..samples.len());
                let sample = &samples[idx];

                // Find k nearest neighbors
                let neighbors = self.find_k_neighbors(sample, samples, k);
                
                if neighbors.is_empty() {
                    // Just duplicate if no neighbors
                    new_x.extend_from_slice(sample);
                } else {
                    // Pick a random neighbor
                    let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                    let neighbor = &samples[neighbor_idx];

                    // Generate synthetic sample
                    let lambda: f32 = rng.gen();
                    let synthetic: Vec<f32> = sample.iter().zip(neighbor.iter())
                        .map(|(&s, &n)| s + lambda * (n - s))
                        .collect();
                    
                    new_x.extend_from_slice(&synthetic);
                }
                new_y.push(*class as f32);
            }
        }

        let new_n_samples = new_y.len();
        (
            Tensor::from_slice(&new_x, &[new_n_samples, n_features]).unwrap(),
            Tensor::from_slice(&new_y, &[new_n_samples]).unwrap()
        )
    }
}

impl Default for SMOTE {
    fn default() -> Self { Self::new() }
}

/// BorderlineSMOTE - SMOTE variant focusing on borderline samples
pub struct BorderlineSMOTE {
    pub k_neighbors: usize,
    pub m_neighbors: usize,
    pub sampling_strategy: SamplingStrategy,
    pub random_state: Option<u64>,
}

impl BorderlineSMOTE {
    pub fn new() -> Self {
        BorderlineSMOTE {
            k_neighbors: 5,
            m_neighbors: 10,
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
        }
    }

    fn find_k_neighbors_with_labels(&self, point: &[f32], all_samples: &[(Vec<f32>, i32)], k: usize) 
        -> Vec<(usize, i32)> 
    {
        let mut distances: Vec<(usize, f32, i32)> = all_samples.iter()
            .enumerate()
            .map(|(i, (s, label))| {
                let dist: f32 = point.iter().zip(s.iter())
                    .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt();
                (i, dist, *label)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().skip(1).take(k).map(|(i, _, l)| (i, l)).collect()
    }

    pub fn fit_resample(&self, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Prepare all samples with labels
        let all_samples: Vec<(Vec<f32>, i32)> = (0..n_samples)
            .map(|i| {
                let sample: Vec<f32> = x_data[i * n_features..(i + 1) * n_features].to_vec();
                (sample, y_data[i] as i32)
            })
            .collect();

        let mut class_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for &label in &y_data {
            *class_counts.entry(label as i32).or_default() += 1;
        }

        let (minority_class, _) = class_counts.iter()
            .min_by_key(|(_, &count)| count)
            .unwrap();
        let max_count = *class_counts.values().max().unwrap();

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Find borderline samples
        let mut borderline_samples: Vec<Vec<f32>> = Vec::new();
        let minority_samples: Vec<Vec<f32>> = all_samples.iter()
            .filter(|(_, l)| l == minority_class)
            .map(|(s, _)| s.clone())
            .collect();

        for (sample, label) in &all_samples {
            if *label != *minority_class { continue; }

            let neighbors = self.find_k_neighbors_with_labels(sample, &all_samples, self.m_neighbors);
            let n_majority = neighbors.iter().filter(|(_, l)| l != minority_class).count();

            // Borderline if half or more neighbors are majority class
            if n_majority >= self.m_neighbors / 2 && n_majority < self.m_neighbors {
                borderline_samples.push(sample.clone());
            }
        }

        // If no borderline samples, fall back to regular SMOTE
        if borderline_samples.is_empty() {
            borderline_samples = minority_samples.clone();
        }

        let mut new_x = x_data.clone();
        let mut new_y = y_data.clone();

        let target_count = match &self.sampling_strategy {
            SamplingStrategy::Auto => max_count,
            SamplingStrategy::Ratio(r) => (max_count as f32 * r) as usize,
            _ => max_count,
        };

        let n_to_generate = target_count.saturating_sub(class_counts[minority_class]);
        let k = self.k_neighbors.min(minority_samples.len() - 1).max(1);

        for _ in 0..n_to_generate {
            let idx = rng.gen_range(0..borderline_samples.len());
            let sample = &borderline_samples[idx];

            // Find k nearest neighbors from minority class
            let mut distances: Vec<(usize, f32)> = minority_samples.iter()
                .enumerate()
                .map(|(i, s)| {
                    let dist: f32 = sample.iter().zip(s.iter())
                        .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt();
                    (i, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let neighbors: Vec<usize> = distances.into_iter().skip(1).take(k).map(|(i, _)| i).collect();

            if !neighbors.is_empty() {
                let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                let neighbor = &minority_samples[neighbor_idx];

                let lambda: f32 = rng.gen();
                let synthetic: Vec<f32> = sample.iter().zip(neighbor.iter())
                    .map(|(&s, &n)| s + lambda * (n - s))
                    .collect();
                
                new_x.extend_from_slice(&synthetic);
                new_y.push(*minority_class as f32);
            }
        }

        let new_n_samples = new_y.len();
        (
            Tensor::from_slice(&new_x, &[new_n_samples, n_features]).unwrap(),
            Tensor::from_slice(&new_y, &[new_n_samples]).unwrap()
        )
    }
}

/// ADASYN - Adaptive Synthetic Sampling
pub struct ADASYN {
    pub k_neighbors: usize,
    pub sampling_strategy: SamplingStrategy,
    pub random_state: Option<u64>,
}

impl ADASYN {
    pub fn new() -> Self {
        ADASYN {
            k_neighbors: 5,
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
        }
    }

    pub fn fit_resample(&self, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let all_samples: Vec<(Vec<f32>, i32)> = (0..n_samples)
            .map(|i| {
                let sample: Vec<f32> = x_data[i * n_features..(i + 1) * n_features].to_vec();
                (sample, y_data[i] as i32)
            })
            .collect();

        let mut class_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for &label in &y_data {
            *class_counts.entry(label as i32).or_default() += 1;
        }

        let (minority_class, minority_count) = class_counts.iter()
            .min_by_key(|(_, &count)| count)
            .map(|(&c, &count)| (c, count))
            .unwrap();
        let max_count = *class_counts.values().max().unwrap();

        let minority_samples: Vec<Vec<f32>> = all_samples.iter()
            .filter(|(_, l)| *l == minority_class)
            .map(|(s, _)| s.clone())
            .collect();

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Compute density ratio for each minority sample
        let k = self.k_neighbors.min(n_samples - 1).max(1);
        let mut ratios: Vec<f32> = Vec::new();

        for sample in &minority_samples {
            let mut distances: Vec<(f32, i32)> = all_samples.iter()
                .map(|(s, label)| {
                    let dist: f32 = sample.iter().zip(s.iter())
                        .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt();
                    (dist, *label)
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            let neighbors: Vec<i32> = distances.into_iter().skip(1).take(k).map(|(_, l)| l).collect();
            let n_majority = neighbors.iter().filter(|&&l| l != minority_class).count();
            ratios.push(n_majority as f32 / k as f32);
        }

        // Normalize ratios
        let ratio_sum: f32 = ratios.iter().sum();
        if ratio_sum > 0.0 {
            for r in &mut ratios {
                *r /= ratio_sum;
            }
        }

        let target_count = match &self.sampling_strategy {
            SamplingStrategy::Auto => max_count,
            SamplingStrategy::Ratio(r) => (max_count as f32 * r) as usize,
            _ => max_count,
        };

        let g = target_count.saturating_sub(minority_count);

        let mut new_x = x_data.clone();
        let mut new_y = y_data.clone();

        let k_synth = self.k_neighbors.min(minority_samples.len() - 1).max(1);

        for (i, sample) in minority_samples.iter().enumerate() {
            let n_to_generate = (ratios[i] * g as f32).round() as usize;

            // Find k nearest minority neighbors
            let mut distances: Vec<(usize, f32)> = minority_samples.iter()
                .enumerate()
                .map(|(j, s)| {
                    let dist: f32 = sample.iter().zip(s.iter())
                        .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt();
                    (j, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let neighbors: Vec<usize> = distances.into_iter().skip(1).take(k_synth).map(|(j, _)| j).collect();

            for _ in 0..n_to_generate {
                if !neighbors.is_empty() {
                    let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                    let neighbor = &minority_samples[neighbor_idx];

                    let lambda: f32 = rng.gen();
                    let synthetic: Vec<f32> = sample.iter().zip(neighbor.iter())
                        .map(|(&s, &n)| s + lambda * (n - s))
                        .collect();
                    
                    new_x.extend_from_slice(&synthetic);
                    new_y.push(minority_class as f32);
                }
            }
        }

        let new_n_samples = new_y.len();
        (
            Tensor::from_slice(&new_x, &[new_n_samples, n_features]).unwrap(),
            Tensor::from_slice(&new_y, &[new_n_samples]).unwrap()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_over_sampler() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0], &[4]).unwrap();
        
        let ros = RandomOverSampler::new();
        let (_x_res, y_res) = ros.fit_resample(&x, &y);
        
        assert!(y_res.dims()[0] >= 4);
    }

    #[test]
    fn test_smote() {
        let x = Tensor::from_slice(&[1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0,
            10.0, 10.0, 11.0, 11.0
        ], &[6, 2]).unwrap();
        let y = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0], &[6]).unwrap();
        
        let smote = SMOTE::new().k_neighbors(1);
        let (_x_res, y_res) = smote.fit_resample(&x, &y);
        
        assert!(y_res.dims()[0] >= 6);
    }
}



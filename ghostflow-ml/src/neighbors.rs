//! K-Nearest Neighbors algorithms

use ghostflow_core::Tensor;
use rayon::prelude::*;

/// K-Nearest Neighbors Classifier
pub struct KNeighborsClassifier {
    pub n_neighbors: usize,
    pub weights: Weights,
    pub metric: Metric,
    pub p: f32,
    x_train_: Option<Vec<f32>>,
    y_train_: Option<Vec<f32>>,
    n_samples_: usize,
    n_features_: usize,
    n_classes_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Weights {
    Uniform,
    Distance,
}

#[derive(Clone, Copy, Debug)]
pub enum Metric {
    Euclidean,
    Manhattan,
    Minkowski,
    Cosine,
}

impl KNeighborsClassifier {
    pub fn new(n_neighbors: usize) -> Self {
        KNeighborsClassifier {
            n_neighbors,
            weights: Weights::Uniform,
            metric: Metric::Euclidean,
            p: 2.0,
            x_train_: None,
            y_train_: None,
            n_samples_: 0,
            n_features_: 0,
            n_classes_: 0,
        }
    }

    pub fn weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::Euclidean => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f32>().sqrt()
            }
            Metric::Manhattan => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).abs()).sum()
            }
            Metric::Minkowski => {
                a.iter().zip(b.iter())
                    .map(|(&ai, &bi)| (ai - bi).abs().powf(self.p))
                    .sum::<f32>()
                    .powf(1.0 / self.p)
            }
            Metric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
                let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (norm_a * norm_b + 1e-10)
            }
        }
    }

    fn find_k_nearest(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let x_train = self.x_train_.as_ref().unwrap();
        
        let mut distances: Vec<(usize, f32)> = (0..self.n_samples_)
            .map(|i| {
                let train_point = &x_train[i * self.n_features_..(i + 1) * self.n_features_];
                let dist = self.distance(query, train_point);
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(self.n_neighbors);
        distances
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        
        self.n_samples_ = x.dims()[0];
        self.n_features_ = x.dims()[1];
        self.n_classes_ = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
        
        self.x_train_ = Some(x_data);
        self.y_train_ = Some(y_data);
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let y_train = self.y_train_.as_ref().expect("Model not fitted");
        
        let predictions: Vec<f32> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query = &x_data[i * n_features..(i + 1) * n_features];
                let neighbors = self.find_k_nearest(query);
                
                match self.weights {
                    Weights::Uniform => {
                        let mut class_counts = vec![0usize; self.n_classes_];
                        for (neighbor_idx, _) in neighbors {
                            let class = y_train[neighbor_idx] as usize;
                            if class < self.n_classes_ {
                                class_counts[class] += 1;
                            }
                        }
                        class_counts.iter()
                            .enumerate()
                            .max_by_key(|(_, &count)| count)
                            .map(|(class, _)| class as f32)
                            .unwrap_or(0.0)
                    }
                    Weights::Distance => {
                        let mut class_weights = vec![0.0f32; self.n_classes_];
                        for (neighbor_idx, dist) in neighbors {
                            let class = y_train[neighbor_idx] as usize;
                            if class < self.n_classes_ {
                                let weight = if dist < 1e-10 { 1e10 } else { 1.0 / dist };
                                class_weights[class] += weight;
                            }
                        }
                        class_weights.iter()
                            .enumerate()
                            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                            .map(|(class, _)| class as f32)
                            .unwrap_or(0.0)
                    }
                }
            })
            .collect();
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn score(&self, x: &Tensor, y: &Tensor) -> f32 {
        let predictions = self.predict(x);
        let pred_data = predictions.data_f32();
        let y_data = y.data_f32();
        
        let correct: usize = pred_data.iter()
            .zip(y_data.iter())
            .filter(|(&p, &y)| (p - y).abs() < 0.5)
            .count();
        
        correct as f32 / y_data.len() as f32
    }
}


/// K-Nearest Neighbors Regressor
pub struct KNeighborsRegressor {
    pub n_neighbors: usize,
    pub weights: Weights,
    pub metric: Metric,
    pub p: f32,
    x_train_: Option<Vec<f32>>,
    y_train_: Option<Vec<f32>>,
    n_samples_: usize,
    n_features_: usize,
}

impl KNeighborsRegressor {
    pub fn new(n_neighbors: usize) -> Self {
        KNeighborsRegressor {
            n_neighbors,
            weights: Weights::Uniform,
            metric: Metric::Euclidean,
            p: 2.0,
            x_train_: None,
            y_train_: None,
            n_samples_: 0,
            n_features_: 0,
        }
    }

    pub fn weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::Euclidean => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f32>().sqrt()
            }
            Metric::Manhattan => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).abs()).sum()
            }
            _ => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f32>().sqrt()
            }
        }
    }

    fn find_k_nearest(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let x_train = self.x_train_.as_ref().unwrap();
        
        let mut distances: Vec<(usize, f32)> = (0..self.n_samples_)
            .map(|i| {
                let train_point = &x_train[i * self.n_features_..(i + 1) * self.n_features_];
                let dist = self.distance(query, train_point);
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(self.n_neighbors);
        distances
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        self.n_samples_ = x.dims()[0];
        self.n_features_ = x.dims()[1];
        self.x_train_ = Some(x.data_f32());
        self.y_train_ = Some(y.data_f32());
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        
        let y_train = self.y_train_.as_ref().expect("Model not fitted");
        
        let predictions: Vec<f32> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query = &x_data[i * n_features..(i + 1) * n_features];
                let neighbors = self.find_k_nearest(query);
                
                match self.weights {
                    Weights::Uniform => {
                        let sum: f32 = neighbors.iter()
                            .map(|(idx, _)| y_train[*idx])
                            .sum();
                        sum / neighbors.len() as f32
                    }
                    Weights::Distance => {
                        let mut weighted_sum = 0.0f32;
                        let mut total_weight = 0.0f32;
                        for (idx, dist) in neighbors {
                            let weight = if dist < 1e-10 { 1e10 } else { 1.0 / dist };
                            weighted_sum += weight * y_train[idx];
                            total_weight += weight;
                        }
                        weighted_sum / total_weight.max(1e-10)
                    }
                }
            })
            .collect();
        
        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }

    pub fn score(&self, x: &Tensor, y: &Tensor) -> f32 {
        let predictions = self.predict(x);
        let pred_data = predictions.data_f32();
        let y_data = y.data_f32();
        
        let y_mean: f32 = y_data.iter().sum::<f32>() / y_data.len() as f32;
        let ss_res: f32 = pred_data.iter()
            .zip(y_data.iter())
            .map(|(&p, &y)| (y - p).powi(2))
            .sum();
        let ss_tot: f32 = y_data.iter()
            .map(|&y| (y - y_mean).powi(2))
            .sum();
        
        1.0 - ss_res / ss_tot.max(1e-10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_classifier() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            1.0, 1.0,
            1.1, 1.1,
        ], &[4, 2]).unwrap();
        
        let y = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4]).unwrap();
        
        let mut knn = KNeighborsClassifier::new(3);
        knn.fit(&x, &y);
        
        let score = knn.score(&x, &y);
        assert!(score >= 0.5);
    }

    #[test]
    fn test_knn_regressor() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap();
        let y = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
        
        let mut knn = KNeighborsRegressor::new(3);
        knn.fit(&x, &y);
        
        let score = knn.score(&x, &y);
        assert!(score >= 0.5);
    }
}

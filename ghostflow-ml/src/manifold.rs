//! Manifold Learning - t-SNE, UMAP, Isomap, LLE, MDS

use ghostflow_core::Tensor;
use rand::prelude::*;

/// t-Distributed Stochastic Neighbor Embedding
pub struct TSNE {
    pub n_components: usize,
    pub perplexity: f32,
    pub learning_rate: f32,
    pub n_iter: usize,
    pub early_exaggeration: f32,
    pub metric: TSNEMetric,
    embedding_: Option<Vec<f32>>,
    kl_divergence_: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub enum TSNEMetric {
    Euclidean,
    Cosine,
    Manhattan,
}

impl TSNE {
    pub fn new(n_components: usize) -> Self {
        TSNE {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            early_exaggeration: 12.0,
            metric: TSNEMetric::Euclidean,
            embedding_: None,
            kl_divergence_: None,
        }
    }

    pub fn perplexity(mut self, p: f32) -> Self {
        self.perplexity = p;
        self
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            TSNEMetric::Euclidean => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f32>().sqrt()
            }
            TSNEMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
                let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (norm_a * norm_b + 1e-10)
            }
            TSNEMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).abs()).sum()
            }
        }
    }

    fn compute_pairwise_distances(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut distances = vec![vec![0.0f32; n_samples]; n_samples];
        
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let xi = &x[i * n_features..(i + 1) * n_features];
                let xj = &x[j * n_features..(j + 1) * n_features];
                let d = self.distance(xi, xj);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }
        
        distances
    }

    fn compute_perplexity_row(&self, distances: &[f32], target_perplexity: f32, i: usize) -> (Vec<f32>, f32) {
        let n = distances.len();
        let target_entropy = target_perplexity.ln();
        
        let mut beta = 1.0f32;
        let mut beta_min = f32::NEG_INFINITY;
        let mut beta_max = f32::INFINITY;
        
        let mut p = vec![0.0f32; n];
        
        for _ in 0..50 {
            // Compute probabilities
            let mut sum_p = 0.0f32;
            for j in 0..n {
                if i != j {
                    p[j] = (-beta * distances[j] * distances[j]).exp();
                    sum_p += p[j];
                } else {
                    p[j] = 0.0;
                }
            }
            
            // Normalize
            if sum_p > 0.0 {
                for pj in &mut p {
                    *pj /= sum_p;
                }
            }
            
            // Compute entropy
            let mut entropy = 0.0f32;
            for j in 0..n {
                if p[j] > 1e-10 {
                    entropy -= p[j] * p[j].ln();
                }
            }
            
            // Binary search for beta
            let diff = entropy - target_entropy;
            if diff.abs() < 1e-5 {
                break;
            }
            
            if diff > 0.0 {
                beta_min = beta;
                beta = if beta_max.is_infinite() { beta * 2.0 } else { (beta + beta_max) / 2.0 };
            } else {
                beta_max = beta;
                beta = if beta_min.is_infinite() { beta / 2.0 } else { (beta + beta_min) / 2.0 };
            }
        }
        
        (p, beta)
    }

    fn compute_joint_probabilities(&self, distances: &[Vec<f32>], n_samples: usize) -> Vec<Vec<f32>> {
        let mut p = vec![vec![0.0f32; n_samples]; n_samples];
        
        // Compute conditional probabilities
        for i in 0..n_samples {
            let (p_row, _) = self.compute_perplexity_row(&distances[i], self.perplexity, i);
            for j in 0..n_samples {
                p[i][j] = p_row[j];
            }
        }
        
        // Symmetrize
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let pij = (p[i][j] + p[j][i]) / (2.0 * n_samples as f32);
                p[i][j] = pij.max(1e-12);
                p[j][i] = pij.max(1e-12);
            }
        }
        
        p
    }

    fn compute_q_distribution(&self, y: &[f32], n_samples: usize) -> Vec<Vec<f32>> {
        let mut q = vec![vec![0.0f32; n_samples]; n_samples];
        let mut sum_q = 0.0f32;
        
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let mut dist_sq = 0.0f32;
                for d in 0..self.n_components {
                    let diff = y[i * self.n_components + d] - y[j * self.n_components + d];
                    dist_sq += diff * diff;
                }
                
                // Student-t distribution with 1 degree of freedom
                let qij = 1.0 / (1.0 + dist_sq);
                q[i][j] = qij;
                q[j][i] = qij;
                sum_q += 2.0 * qij;
            }
        }
        
        // Normalize
        if sum_q > 0.0 {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    q[i][j] = (q[i][j] / sum_q).max(1e-12);
                }
            }
        }
        
        q
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute pairwise distances
        let distances = self.compute_pairwise_distances(&x_data, n_samples, n_features);
        
        // Compute joint probabilities P
        let mut p = self.compute_joint_probabilities(&distances, n_samples);
        
        // Early exaggeration
        for i in 0..n_samples {
            for j in 0..n_samples {
                p[i][j] *= self.early_exaggeration;
            }
        }
        
        // Initialize embedding randomly
        let mut rng = thread_rng();
        let mut y: Vec<f32> = (0..n_samples * self.n_components)
            .map(|_| rng.gen::<f32>() * 0.0001)
            .collect();
        
        // Gradient descent with momentum
        let mut velocity = vec![0.0f32; n_samples * self.n_components];
        let momentum = 0.5f32;
        let final_momentum = 0.8f32;
        let momentum_switch_iter = 250;
        
        for iter in 0..self.n_iter {
            // Remove early exaggeration after some iterations
            if iter == 100 {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        p[i][j] /= self.early_exaggeration;
                    }
                }
            }
            
            // Compute Q distribution
            let q = self.compute_q_distribution(&y, n_samples);
            
            // Compute gradients
            let mut grad = vec![0.0f32; n_samples * self.n_components];
            
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let mut dist_sq = 0.0f32;
                        for d in 0..self.n_components {
                            let diff = y[i * self.n_components + d] - y[j * self.n_components + d];
                            dist_sq += diff * diff;
                        }
                        
                        let pq_diff = p[i][j] - q[i][j];
                        let mult = 4.0 * pq_diff / (1.0 + dist_sq);
                        
                        for d in 0..self.n_components {
                            let diff = y[i * self.n_components + d] - y[j * self.n_components + d];
                            grad[i * self.n_components + d] += mult * diff;
                        }
                    }
                }
            }
            
            // Update with momentum
            let current_momentum = if iter < momentum_switch_iter { momentum } else { final_momentum };
            
            for i in 0..n_samples * self.n_components {
                velocity[i] = current_momentum * velocity[i] - self.learning_rate * grad[i];
                y[i] += velocity[i];
            }
            
            // Center the embedding
            let mut mean = vec![0.0f32; self.n_components];
            for i in 0..n_samples {
                for d in 0..self.n_components {
                    mean[d] += y[i * self.n_components + d];
                }
            }
            for d in 0..self.n_components {
                mean[d] /= n_samples as f32;
            }
            for i in 0..n_samples {
                for d in 0..self.n_components {
                    y[i * self.n_components + d] -= mean[d];
                }
            }
        }
        
        // Compute final KL divergence
        let q = self.compute_q_distribution(&y, n_samples);
        let mut kl = 0.0f32;
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j && p[i][j] > 1e-12 {
                    kl += p[i][j] * (p[i][j] / q[i][j]).ln();
                }
            }
        }
        
        self.embedding_ = Some(y.clone());
        self.kl_divergence_ = Some(kl);
        
        Tensor::from_slice(&y, &[n_samples, self.n_components]).unwrap()
    }
}


/// Multidimensional Scaling
pub struct MDS {
    pub n_components: usize,
    pub metric: bool,
    pub n_init: usize,
    pub max_iter: usize,
    pub eps: f32,
    pub dissimilarity: Dissimilarity,
    embedding_: Option<Vec<f32>>,
    stress_: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub enum Dissimilarity {
    Euclidean,
    Precomputed,
}

impl MDS {
    pub fn new(n_components: usize) -> Self {
        MDS {
            n_components,
            metric: true,
            n_init: 4,
            max_iter: 300,
            eps: 1e-3,
            dissimilarity: Dissimilarity::Euclidean,
            embedding_: None,
            stress_: None,
        }
    }

    pub fn metric(mut self, metric: bool) -> Self {
        self.metric = metric;
        self
    }

    fn compute_distances(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut distances = vec![vec![0.0f32; n_samples]; n_samples];
        
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let mut dist_sq = 0.0f32;
                for k in 0..n_features {
                    let diff = x[i * n_features + k] - x[j * n_features + k];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        
        distances
    }

    fn compute_stress(&self, y: &[f32], distances: &[Vec<f32>], n_samples: usize) -> f32 {
        let mut stress = 0.0f32;
        let mut norm = 0.0f32;
        
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let mut dist_sq = 0.0f32;
                for d in 0..self.n_components {
                    let diff = y[i * self.n_components + d] - y[j * self.n_components + d];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                
                let diff = distances[i][j] - dist;
                stress += diff * diff;
                norm += distances[i][j] * distances[i][j];
            }
        }
        
        (stress / norm.max(1e-10)).sqrt()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute distance matrix
        let distances = match self.dissimilarity {
            Dissimilarity::Euclidean => self.compute_distances(&x_data, n_samples, n_features),
            Dissimilarity::Precomputed => {
                // Assume x is already a distance matrix
                let mut d = vec![vec![0.0f32; n_samples]; n_samples];
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        d[i][j] = x_data[i * n_samples + j];
                    }
                }
                d
            }
        };

        // Classical MDS using double centering
        let mut best_y = None;
        let mut best_stress = f32::INFINITY;

        for _ in 0..self.n_init {
            // Double centering: B = -0.5 * J * D^2 * J where J = I - 1/n * 11^T
            let mut b = vec![vec![0.0f32; n_samples]; n_samples];
            
            // Compute D^2
            for i in 0..n_samples {
                for j in 0..n_samples {
                    b[i][j] = -0.5 * distances[i][j] * distances[i][j];
                }
            }
            
            // Row means
            let row_means: Vec<f32> = (0..n_samples)
                .map(|i| b[i].iter().sum::<f32>() / n_samples as f32)
                .collect();
            
            // Column means
            let col_means: Vec<f32> = (0..n_samples)
                .map(|j| (0..n_samples).map(|i| b[i][j]).sum::<f32>() / n_samples as f32)
                .collect();
            
            // Grand mean
            let grand_mean: f32 = row_means.iter().sum::<f32>() / n_samples as f32;
            
            // Double centering
            for i in 0..n_samples {
                for j in 0..n_samples {
                    b[i][j] = b[i][j] - row_means[i] - col_means[j] + grand_mean;
                }
            }
            
            // Eigendecomposition using power iteration
            let mut y = vec![0.0f32; n_samples * self.n_components];
            let mut rng = thread_rng();
            
            // Flatten B for matrix operations
            let mut b_flat: Vec<f32> = b.iter().flat_map(|row| row.clone()).collect();
            
            for comp in 0..self.n_components {
                // Power iteration
                let mut v: Vec<f32> = (0..n_samples).map(|_| rng.gen::<f32>()).collect();
                let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
                for vi in &mut v {
                    *vi /= norm;
                }
                
                for _ in 0..100 {
                    // w = B * v
                    let mut w = vec![0.0f32; n_samples];
                    for i in 0..n_samples {
                        for j in 0..n_samples {
                            w[i] += b_flat[i * n_samples + j] * v[j];
                        }
                    }
                    
                    // Normalize
                    let norm: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    if norm < 1e-10 {
                        break;
                    }
                    for wi in &mut w {
                        *wi /= norm;
                    }
                    
                    // Check convergence
                    let diff: f32 = v.iter().zip(w.iter()).map(|(&vi, &wi)| (vi - wi).abs()).sum();
                    v = w;
                    
                    if diff < 1e-6 {
                        break;
                    }
                }
                
                // Compute eigenvalue
                let mut eigenvalue = 0.0f32;
                for i in 0..n_samples {
                    let mut bv = 0.0f32;
                    for j in 0..n_samples {
                        bv += b_flat[i * n_samples + j] * v[j];
                    }
                    eigenvalue += v[i] * bv;
                }
                
                // Store component
                let scale = eigenvalue.max(0.0).sqrt();
                for i in 0..n_samples {
                    y[i * self.n_components + comp] = v[i] * scale;
                }
                
                // Deflate B
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        b_flat[i * n_samples + j] -= eigenvalue * v[i] * v[j];
                    }
                }
            }
            
            let stress = self.compute_stress(&y, &distances, n_samples);
            if stress < best_stress {
                best_stress = stress;
                best_y = Some(y);
            }
        }

        let y = best_y.unwrap();
        self.embedding_ = Some(y.clone());
        self.stress_ = Some(best_stress);
        
        Tensor::from_slice(&y, &[n_samples, self.n_components]).unwrap()
    }
}

/// Isomap - Isometric Mapping
pub struct Isomap {
    pub n_components: usize,
    pub n_neighbors: usize,
    embedding_: Option<Vec<f32>>,
    reconstruction_error_: Option<f32>,
}

impl Isomap {
    pub fn new(n_components: usize) -> Self {
        Isomap {
            n_components,
            n_neighbors: 5,
            embedding_: None,
            reconstruction_error_: None,
        }
    }

    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    fn compute_knn_graph(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<(usize, f32)>> {
        let mut graph = vec![Vec::new(); n_samples];
        
        for i in 0..n_samples {
            let mut distances: Vec<(usize, f32)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut dist_sq = 0.0f32;
                    for k in 0..n_features {
                        let diff = x[i * n_features + k] - x[j * n_features + k];
                        dist_sq += diff * diff;
                    }
                    (j, dist_sq.sqrt())
                })
                .collect();
            
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            graph[i] = distances.into_iter().take(self.n_neighbors).collect();
        }
        
        graph
    }

    fn floyd_warshall(&self, graph: &[Vec<(usize, f32)>], n_samples: usize) -> Vec<Vec<f32>> {
        let mut dist = vec![vec![f32::INFINITY; n_samples]; n_samples];
        
        // Initialize with graph edges
        for i in 0..n_samples {
            dist[i][i] = 0.0;
            for &(j, d) in &graph[i] {
                dist[i][j] = d;
                dist[j][i] = d;  // Make symmetric
            }
        }
        
        // Floyd-Warshall
        for k in 0..n_samples {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if dist[i][k] + dist[k][j] < dist[i][j] {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        
        dist
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Build k-NN graph
        let graph = self.compute_knn_graph(&x_data, n_samples, n_features);
        
        // Compute geodesic distances using Floyd-Warshall
        let geodesic_dist = self.floyd_warshall(&graph, n_samples);
        
        // Apply classical MDS to geodesic distances
        let mut mds = MDS::new(self.n_components);
        mds.dissimilarity = Dissimilarity::Precomputed;
        
        // Flatten geodesic distances
        let dist_flat: Vec<f32> = geodesic_dist.iter().flat_map(|row| row.clone()).collect();
        let dist_tensor = Tensor::from_slice(&dist_flat, &[n_samples, n_samples]).unwrap();
        
        let embedding = mds.fit_transform(&dist_tensor);
        
        self.embedding_ = Some(embedding.data_f32());
        self.reconstruction_error_ = mds.stress_;
        
        embedding
    }
}


/// Locally Linear Embedding
pub struct LocallyLinearEmbedding {
    pub n_components: usize,
    pub n_neighbors: usize,
    pub reg: f32,
    pub method: LLEMethod,
    embedding_: Option<Vec<f32>>,
    #[allow(dead_code)]
    reconstruction_error_: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub enum LLEMethod {
    Standard,
    Modified,
    HLLE,
    LTSA,
}

impl LocallyLinearEmbedding {
    pub fn new(n_components: usize) -> Self {
        LocallyLinearEmbedding {
            n_components,
            n_neighbors: 5,
            reg: 1e-3,
            method: LLEMethod::Standard,
            embedding_: None,
            reconstruction_error_: None,
        }
    }

    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    fn find_neighbors(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<usize>> {
        let mut neighbors = vec![Vec::new(); n_samples];
        
        for i in 0..n_samples {
            let mut distances: Vec<(usize, f32)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut dist_sq = 0.0f32;
                    for k in 0..n_features {
                        let diff = x[i * n_features + k] - x[j * n_features + k];
                        dist_sq += diff * diff;
                    }
                    (j, dist_sq)
                })
                .collect();
            
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            neighbors[i] = distances.into_iter().take(self.n_neighbors).map(|(j, _)| j).collect();
        }
        
        neighbors
    }

    fn compute_weights(&self, x: &[f32], neighbors: &[Vec<usize>], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut weights = vec![vec![0.0f32; n_samples]; n_samples];
        
        for i in 0..n_samples {
            let k = neighbors[i].len();
            if k == 0 {
                continue;
            }
            
            // Build local covariance matrix
            let mut c = vec![0.0f32; k * k];
            
            for (a, &ja) in neighbors[i].iter().enumerate() {
                for (b, &jb) in neighbors[i].iter().enumerate() {
                    let mut dot = 0.0f32;
                    for f in 0..n_features {
                        let diff_a = x[ja * n_features + f] - x[i * n_features + f];
                        let diff_b = x[jb * n_features + f] - x[i * n_features + f];
                        dot += diff_a * diff_b;
                    }
                    c[a * k + b] = dot;
                }
            }
            
            // Add regularization
            let trace: f32 = (0..k).map(|a| c[a * k + a]).sum();
            let reg = self.reg * trace / k as f32;
            for a in 0..k {
                c[a * k + a] += reg;
            }
            
            // Solve C * w = 1 for w
            let ones = vec![1.0f32; k];
            let w = self.solve_linear_system(&c, &ones, k);
            
            // Normalize weights
            let sum: f32 = w.iter().sum();
            for (a, &ja) in neighbors[i].iter().enumerate() {
                weights[i][ja] = w[a] / sum.max(1e-10);
            }
        }
        
        weights
    }

    fn solve_linear_system(&self, a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        // Simple Gaussian elimination
        let mut aug = vec![0.0f32; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a[i * n + j];
            }
            aug[i * (n + 1) + n] = b[i];
        }
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[k * (n + 1) + i].abs() > aug[max_row * (n + 1) + i].abs() {
                    max_row = k;
                }
            }
            
            // Swap rows
            for j in 0..=n {
                let tmp = aug[i * (n + 1) + j];
                aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
            
            // Eliminate
            let pivot = aug[i * (n + 1) + i];
            if pivot.abs() < 1e-10 {
                continue;
            }
            
            for k in (i + 1)..n {
                let factor = aug[k * (n + 1) + i] / pivot;
                for j in i..=n {
                    aug[k * (n + 1) + j] -= factor * aug[i * (n + 1) + j];
                }
            }
        }
        
        // Back substitution
        let mut x = vec![0.0f32; n];
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= aug[i * (n + 1) + j] * x[j];
            }
            let pivot = aug[i * (n + 1) + i];
            x[i] = if pivot.abs() > 1e-10 { sum / pivot } else { 0.0 };
        }
        
        x
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Find neighbors
        let neighbors = self.find_neighbors(&x_data, n_samples, n_features);
        
        // Compute reconstruction weights
        let weights = self.compute_weights(&x_data, &neighbors, n_samples, n_features);
        
        // Build M = (I - W)^T (I - W)
        let mut m = vec![0.0f32; n_samples * n_samples];
        
        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut val = 0.0f32;
                
                for k in 0..n_samples {
                    let ik = if i == k { 1.0 } else { 0.0 } - weights[k][i];
                    let jk = if j == k { 1.0 } else { 0.0 } - weights[k][j];
                    val += ik * jk;
                }
                
                m[i * n_samples + j] = val;
            }
        }
        
        // Find smallest eigenvectors of M (skip the first one which is constant)
        let mut y = vec![0.0f32; n_samples * self.n_components];
        let mut rng = thread_rng();
        
        // Use inverse power iteration to find smallest eigenvalues
        // First, shift M to make smallest eigenvalues largest: M' = max_eigenvalue * I - M
        let max_eig = 2.0f32;  // Upper bound estimate
        for i in 0..n_samples {
            m[i * n_samples + i] = max_eig - m[i * n_samples + i];
            for j in 0..n_samples {
                if i != j {
                    m[i * n_samples + j] = -m[i * n_samples + j];
                }
            }
        }
        
        // Skip first eigenvector (constant), find next n_components
        for comp in 0..(self.n_components + 1) {
            let mut v: Vec<f32> = (0..n_samples).map(|_| rng.gen::<f32>()).collect();
            let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            for vi in &mut v {
                *vi /= norm;
            }
            
            for _ in 0..100 {
                let mut w = vec![0.0f32; n_samples];
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        w[i] += m[i * n_samples + j] * v[j];
                    }
                }
                
                let norm: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                for wi in &mut w {
                    *wi /= norm;
                }
                
                let diff: f32 = v.iter().zip(w.iter()).map(|(&vi, &wi)| (vi - wi).abs()).sum();
                v = w;
                
                if diff < 1e-6 {
                    break;
                }
            }
            
            // Store component (skip first)
            if comp > 0 {
                for i in 0..n_samples {
                    y[i * self.n_components + (comp - 1)] = v[i];
                }
            }
            
            // Deflate
            let mut eigenvalue = 0.0f32;
            for i in 0..n_samples {
                let mut mv = 0.0f32;
                for j in 0..n_samples {
                    mv += m[i * n_samples + j] * v[j];
                }
                eigenvalue += v[i] * mv;
            }
            
            for i in 0..n_samples {
                for j in 0..n_samples {
                    m[i * n_samples + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
        
        self.embedding_ = Some(y.clone());
        
        Tensor::from_slice(&y, &[n_samples, self.n_components]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsne() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
        ], &[4, 2]).unwrap();

        let mut tsne = TSNE::new(2).perplexity(2.0).n_iter(100);
        let embedding = tsne.fit_transform(&x);
        
        assert_eq!(embedding.dims(), &[4, 2]);
    }

    #[test]
    fn test_mds() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ], &[4, 2]).unwrap();

        let mut mds = MDS::new(2);
        let embedding = mds.fit_transform(&x);
        
        assert_eq!(embedding.dims(), &[4, 2]);
    }

    #[test]
    fn test_isomap() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            1.0, 0.0,
            2.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ], &[5, 2]).unwrap();

        let mut isomap = Isomap::new(2).n_neighbors(3);
        let embedding = isomap.fit_transform(&x);
        
        assert_eq!(embedding.dims(), &[5, 2]);
    }
}



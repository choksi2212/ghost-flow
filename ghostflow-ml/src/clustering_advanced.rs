//! Advanced Clustering - Spectral, Mean Shift, OPTICS, Birch, Mini-Batch KMeans, Affinity Propagation

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Spectral Clustering
pub struct SpectralClustering {
    pub n_clusters: usize,
    pub affinity: SpectralAffinity,
    pub gamma: f32,
    pub n_neighbors: usize,
    pub assign_labels: AssignLabels,
    labels_: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
pub enum SpectralAffinity {
    RBF,
    NearestNeighbors,
    Precomputed,
}

#[derive(Clone, Copy, Debug)]
pub enum AssignLabels {
    KMeans,
    Discretize,
}

impl SpectralClustering {
    pub fn new(n_clusters: usize) -> Self {
        SpectralClustering {
            n_clusters,
            affinity: SpectralAffinity::RBF,
            gamma: 1.0,
            n_neighbors: 10,
            assign_labels: AssignLabels::KMeans,
            labels_: None,
        }
    }

    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    fn compute_affinity_matrix(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut affinity = vec![vec![0.0f32; n_samples]; n_samples];

        match self.affinity {
            SpectralAffinity::RBF => {
                for i in 0..n_samples {
                    for j in i..n_samples {
                        let mut dist_sq = 0.0f32;
                        for k in 0..n_features {
                            let diff = x[i * n_features + k] - x[j * n_features + k];
                            dist_sq += diff * diff;
                        }
                        let a = (-self.gamma * dist_sq).exp();
                        affinity[i][j] = a;
                        affinity[j][i] = a;
                    }
                }
            }
            SpectralAffinity::NearestNeighbors => {
                // Compute k-NN graph
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
                    
                    for (j, _) in distances.into_iter().take(self.n_neighbors) {
                        affinity[i][j] = 1.0;
                        affinity[j][i] = 1.0;
                    }
                }
            }
            SpectralAffinity::Precomputed => {
                // Assume x is already an affinity matrix
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        affinity[i][j] = x[i * n_samples + j];
                    }
                }
            }
        }

        affinity
    }

    fn compute_laplacian(&self, affinity: &[Vec<f32>], n_samples: usize) -> Vec<f32> {
        // Compute normalized Laplacian: L = D^(-1/2) * (D - A) * D^(-1/2)
        // Or equivalently: L = I - D^(-1/2) * A * D^(-1/2)
        
        // Compute degree matrix
        let degrees: Vec<f32> = (0..n_samples)
            .map(|i| affinity[i].iter().sum::<f32>())
            .collect();

        let mut laplacian = vec![0.0f32; n_samples * n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    laplacian[i * n_samples + j] = 1.0;
                } else {
                    let d_i = degrees[i].max(1e-10).sqrt();
                    let d_j = degrees[j].max(1e-10).sqrt();
                    laplacian[i * n_samples + j] = -affinity[i][j] / (d_i * d_j);
                }
            }
        }

        laplacian
    }

    fn power_iteration_smallest(&self, matrix: &[f32], n: usize, k: usize) -> Vec<Vec<f32>> {
        // Find k smallest eigenvectors using inverse power iteration
        let mut eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut rng = thread_rng();

        // Shift matrix to make smallest eigenvalues largest
        let mut shifted = matrix.to_vec();
        let shift = 2.0f32;  // Laplacian eigenvalues are in [0, 2]
        for i in 0..n {
            shifted[i * n + i] = shift - shifted[i * n + i];
            for j in 0..n {
                if i != j {
                    shifted[i * n + j] = -shifted[i * n + j];
                }
            }
        }

        for _ in 0..k {
            let mut v: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();
            let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            for vi in &mut v {
                *vi /= norm;
            }

            for _ in 0..100 {
                // w = A * v
                let mut w = vec![0.0f32; n];
                for i in 0..n {
                    for j in 0..n {
                        w[i] += shifted[i * n + j] * v[j];
                    }
                }

                // Orthogonalize against previous eigenvectors
                for prev in &eigenvectors {
                    let dot: f32 = w.iter().zip(prev.iter()).map(|(&a, &b)| a * b).sum();
                    for i in 0..n {
                        w[i] -= dot * prev[i];
                    }
                }

                let norm: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                for wi in &mut w {
                    *wi /= norm;
                }

                let diff: f32 = v.iter().zip(w.iter()).map(|(&a, &b)| (a - b).abs()).sum();
                v = w;

                if diff < 1e-6 {
                    break;
                }
            }

            eigenvectors.push(v);

            // Deflate
            let mut eigenvalue = 0.0f32;
            for i in 0..n {
                let mut av = 0.0f32;
                for j in 0..n {
                    av += shifted[i * n + j] * eigenvectors.last().unwrap()[j];
                }
                eigenvalue += eigenvectors.last().unwrap()[i] * av;
            }

            let v = eigenvectors.last().unwrap();
            for i in 0..n {
                for j in 0..n {
                    shifted[i * n + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        eigenvectors
    }

    fn kmeans_on_embedding(&self, embedding: &[Vec<f32>], n_samples: usize) -> Vec<usize> {
        let k = self.n_clusters;
        let n_features = embedding.len();

        // Flatten embedding
        let mut data = vec![0.0f32; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                data[i * n_features + j] = embedding[j][i];
            }
        }

        // Simple k-means
        let mut rng = thread_rng();
        let mut centers: Vec<Vec<f32>> = (0..k)
            .map(|_| {
                let idx = rng.gen_range(0..n_samples);
                (0..n_features).map(|j| data[idx * n_features + j]).collect()
            })
            .collect();

        let mut labels = vec![0usize; n_samples];

        for _ in 0..100 {
            // Assign labels
            for i in 0..n_samples {
                let mut min_dist = f32::INFINITY;
                for c in 0..k {
                    let mut dist = 0.0f32;
                    for j in 0..n_features {
                        let diff = data[i * n_features + j] - centers[c][j];
                        dist += diff * diff;
                    }
                    if dist < min_dist {
                        min_dist = dist;
                        labels[i] = c;
                    }
                }
            }

            // Update centers
            let mut new_centers = vec![vec![0.0f32; n_features]; k];
            let mut counts = vec![0usize; k];

            for i in 0..n_samples {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..n_features {
                    new_centers[c][j] += data[i * n_features + j];
                }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..n_features {
                        new_centers[c][j] /= counts[c] as f32;
                    }
                }
            }

            centers = new_centers;
        }

        labels
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute affinity matrix
        let affinity = self.compute_affinity_matrix(&x_data, n_samples, n_features);

        // Compute normalized Laplacian
        let laplacian = self.compute_laplacian(&affinity, n_samples);

        // Find k smallest eigenvectors
        let eigenvectors = self.power_iteration_smallest(&laplacian, n_samples, self.n_clusters);

        // Cluster in embedding space
        let labels = self.kmeans_on_embedding(&eigenvectors, n_samples);

        self.labels_ = Some(labels);
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }
}


/// Mean Shift Clustering
pub struct MeanShift {
    pub bandwidth: Option<f32>,
    pub max_iter: usize,
    pub bin_seeding: bool,
    cluster_centers_: Option<Vec<Vec<f32>>>,
    labels_: Option<Vec<usize>>,
}

impl MeanShift {
    pub fn new() -> Self {
        MeanShift {
            bandwidth: None,
            max_iter: 300,
            bin_seeding: false,
            cluster_centers_: None,
            labels_: None,
        }
    }

    pub fn bandwidth(mut self, bw: f32) -> Self {
        self.bandwidth = Some(bw);
        self
    }

    fn estimate_bandwidth(&self, x: &[f32], n_samples: usize, n_features: usize) -> f32 {
        // Scott's rule of thumb
        let mut std_sum = 0.0f32;
        
        for j in 0..n_features {
            let mean: f32 = (0..n_samples).map(|i| x[i * n_features + j]).sum::<f32>() / n_samples as f32;
            let variance: f32 = (0..n_samples)
                .map(|i| (x[i * n_features + j] - mean).powi(2))
                .sum::<f32>() / n_samples as f32;
            std_sum += variance.sqrt();
        }
        
        let avg_std = std_sum / n_features as f32;
        avg_std * (n_samples as f32).powf(-1.0 / (n_features as f32 + 4.0))
    }

    fn gaussian_kernel(&self, dist: f32, bandwidth: f32) -> f32 {
        (-0.5 * (dist / bandwidth).powi(2)).exp()
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let bandwidth = self.bandwidth.unwrap_or_else(|| self.estimate_bandwidth(&x_data, n_samples, n_features));

        // Initialize seeds (all points or binned)
        let mut seeds: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| x_data[i * n_features..(i + 1) * n_features].to_vec())
            .collect();

        // Mean shift for each seed
        let mut converged_centers: Vec<Vec<f32>> = Vec::new();

        for seed in &mut seeds {
            for _ in 0..self.max_iter {
                let mut new_center = vec![0.0f32; n_features];
                let mut total_weight = 0.0f32;

                for i in 0..n_samples {
                    let xi = &x_data[i * n_features..(i + 1) * n_features];
                    
                    let dist: f32 = seed.iter().zip(xi.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();

                    let weight = self.gaussian_kernel(dist, bandwidth);
                    total_weight += weight;

                    for j in 0..n_features {
                        new_center[j] += weight * xi[j];
                    }
                }

                if total_weight > 0.0 {
                    for j in 0..n_features {
                        new_center[j] /= total_weight;
                    }
                }

                // Check convergence
                let shift: f32 = seed.iter().zip(new_center.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();

                *seed = new_center;

                if shift < 1e-3 * bandwidth {
                    break;
                }
            }

            // Check if this center is unique
            let is_unique = converged_centers.iter().all(|c| {
                let dist: f32 = c.iter().zip(seed.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                dist > bandwidth / 2.0
            });

            if is_unique {
                converged_centers.push(seed.clone());
            }
        }

        // Assign labels
        let mut labels = vec![0usize; n_samples];
        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            let mut min_dist = f32::INFINITY;
            
            for (c, center) in converged_centers.iter().enumerate() {
                let dist: f32 = xi.iter().zip(center.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                
                if dist < min_dist {
                    min_dist = dist;
                    labels[i] = c;
                }
            }
        }

        self.cluster_centers_ = Some(converged_centers);
        self.labels_ = Some(labels);
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }
}

impl Default for MeanShift {
    fn default() -> Self {
        Self::new()
    }
}

/// Mini-Batch K-Means for large datasets
pub struct MiniBatchKMeans {
    pub n_clusters: usize,
    pub batch_size: usize,
    pub max_iter: usize,
    pub n_init: usize,
    pub init: MiniBatchInit,
    pub reassignment_ratio: f32,
    cluster_centers_: Option<Vec<Vec<f32>>>,
    labels_: Option<Vec<usize>>,
    inertia_: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub enum MiniBatchInit {
    Random,
    KMeansPlusPlus,
}

impl MiniBatchKMeans {
    pub fn new(n_clusters: usize) -> Self {
        MiniBatchKMeans {
            n_clusters,
            batch_size: 100,
            max_iter: 100,
            n_init: 3,
            init: MiniBatchInit::KMeansPlusPlus,
            reassignment_ratio: 0.01,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum()
    }

    fn init_centers(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut rng = thread_rng();

        match self.init {
            MiniBatchInit::Random => {
                let indices: Vec<usize> = (0..n_samples)
                    .choose_multiple(&mut rng, self.n_clusters);
                indices.iter()
                    .map(|&i| x[i * n_features..(i + 1) * n_features].to_vec())
                    .collect()
            }
            MiniBatchInit::KMeansPlusPlus => {
                let mut centers = Vec::with_capacity(self.n_clusters);
                
                // First center randomly
                let first_idx = rng.gen_range(0..n_samples);
                centers.push(x[first_idx * n_features..(first_idx + 1) * n_features].to_vec());

                for _ in 1..self.n_clusters {
                    let distances: Vec<f32> = (0..n_samples)
                        .map(|i| {
                            let point = &x[i * n_features..(i + 1) * n_features];
                            centers.iter()
                                .map(|c| Self::euclidean_distance_sq(point, c))
                                .fold(f32::INFINITY, f32::min)
                        })
                        .collect();

                    let total: f32 = distances.iter().sum();
                    let threshold = rng.gen::<f32>() * total;
                    
                    let mut cumsum = 0.0f32;
                    let mut chosen_idx = 0;
                    for (i, &d) in distances.iter().enumerate() {
                        cumsum += d;
                        if cumsum >= threshold {
                            chosen_idx = i;
                            break;
                        }
                    }

                    centers.push(x[chosen_idx * n_features..(chosen_idx + 1) * n_features].to_vec());
                }

                centers
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut best_centers = None;
        let mut best_inertia = f32::INFINITY;

        for _ in 0..self.n_init {
            let mut centers = self.init_centers(&x_data, n_samples, n_features);
            let mut counts = vec![0usize; self.n_clusters];
            let mut rng = thread_rng();

            for _ in 0..self.max_iter {
                // Sample mini-batch
                let batch_indices: Vec<usize> = (0..n_samples)
                    .choose_multiple(&mut rng, self.batch_size.min(n_samples));

                // Assign batch points to nearest centers
                let assignments: Vec<usize> = batch_indices.iter()
                    .map(|&i| {
                        let point = &x_data[i * n_features..(i + 1) * n_features];
                        centers.iter()
                            .enumerate()
                            .map(|(c, center)| (c, Self::euclidean_distance_sq(point, center)))
                            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
                            .map(|(c, _)| c)
                            .unwrap_or(0)
                    })
                    .collect();

                // Update centers with streaming average
                for (batch_idx, &sample_idx) in batch_indices.iter().enumerate() {
                    let cluster = assignments[batch_idx];
                    counts[cluster] += 1;
                    let eta = 1.0 / counts[cluster] as f32;

                    for j in 0..n_features {
                        centers[cluster][j] = (1.0 - eta) * centers[cluster][j] 
                            + eta * x_data[sample_idx * n_features + j];
                    }
                }
            }

            // Compute inertia
            let inertia: f32 = (0..n_samples)
                .map(|i| {
                    let point = &x_data[i * n_features..(i + 1) * n_features];
                    centers.iter()
                        .map(|c| Self::euclidean_distance_sq(point, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .sum();

            if inertia < best_inertia {
                best_inertia = inertia;
                best_centers = Some(centers);
            }
        }

        let centers = best_centers.unwrap();

        // Final label assignment
        let labels: Vec<usize> = (0..n_samples)
            .map(|i| {
                let point = &x_data[i * n_features..(i + 1) * n_features];
                centers.iter()
                    .enumerate()
                    .map(|(c, center)| (c, Self::euclidean_distance_sq(point, center)))
                    .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
                    .map(|(c, _)| c)
                    .unwrap_or(0)
            })
            .collect();

        self.cluster_centers_ = Some(centers);
        self.labels_ = Some(labels);
        self.inertia_ = Some(best_inertia);
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let centers = self.cluster_centers_.as_ref().expect("Model not fitted");

        let labels: Vec<f32> = (0..n_samples)
            .map(|i| {
                let point = &x_data[i * n_features..(i + 1) * n_features];
                centers.iter()
                    .enumerate()
                    .map(|(c, center)| (c, Self::euclidean_distance_sq(point, center)))
                    .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
                    .map(|(c, _)| c as f32)
                    .unwrap_or(0.0)
            })
            .collect();

        Tensor::from_slice(&labels, &[n_samples]).unwrap()
    }
}

/// Affinity Propagation Clustering
pub struct AffinityPropagation {
    pub damping: f32,
    pub max_iter: usize,
    pub convergence_iter: usize,
    pub preference: Option<f32>,
    cluster_centers_indices_: Option<Vec<usize>>,
    labels_: Option<Vec<usize>>,
}

impl AffinityPropagation {
    pub fn new() -> Self {
        AffinityPropagation {
            damping: 0.5,
            max_iter: 200,
            convergence_iter: 15,
            preference: None,
            cluster_centers_indices_: None,
            labels_: None,
        }
    }

    pub fn damping(mut self, d: f32) -> Self {
        self.damping = d.clamp(0.5, 1.0);
        self
    }

    pub fn preference(mut self, p: f32) -> Self {
        self.preference = Some(p);
        self
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute similarity matrix (negative squared Euclidean distance)
        let mut s = vec![vec![0.0f32; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let mut dist_sq = 0.0f32;
                    for k in 0..n_features {
                        let diff = x_data[i * n_features + k] - x_data[j * n_features + k];
                        dist_sq += diff * diff;
                    }
                    s[i][j] = -dist_sq;
                }
            }
        }

        // Set preference (diagonal of S)
        let preference = self.preference.unwrap_or_else(|| {
            let mut all_similarities: Vec<f32> = s.iter()
                .flat_map(|row| row.iter().cloned())
                .filter(|&x| x != 0.0)
                .collect();
            all_similarities.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all_similarities[all_similarities.len() / 2]  // Median
        });

        for i in 0..n_samples {
            s[i][i] = preference;
        }

        // Initialize responsibility and availability matrices
        let mut r = vec![vec![0.0f32; n_samples]; n_samples];
        let mut a = vec![vec![0.0f32; n_samples]; n_samples];

        let mut prev_exemplars = vec![0usize; n_samples];
        let mut converged_count = 0;

        for _ in 0..self.max_iter {
            // Update responsibilities
            for i in 0..n_samples {
                for k in 0..n_samples {
                    let mut max_val = f32::NEG_INFINITY;
                    for kp in 0..n_samples {
                        if kp != k {
                            max_val = max_val.max(a[i][kp] + s[i][kp]);
                        }
                    }
                    let new_r = s[i][k] - max_val;
                    r[i][k] = self.damping * r[i][k] + (1.0 - self.damping) * new_r;
                }
            }

            // Update availabilities
            for i in 0..n_samples {
                for k in 0..n_samples {
                    if i == k {
                        let mut sum = 0.0f32;
                        for ip in 0..n_samples {
                            if ip != k {
                                sum += r[ip][k].max(0.0);
                            }
                        }
                        let new_a = sum;
                        a[i][k] = self.damping * a[i][k] + (1.0 - self.damping) * new_a;
                    } else {
                        let mut sum = 0.0f32;
                        for ip in 0..n_samples {
                            if ip != i && ip != k {
                                sum += r[ip][k].max(0.0);
                            }
                        }
                        let new_a = (r[k][k] + sum).min(0.0);
                        a[i][k] = self.damping * a[i][k] + (1.0 - self.damping) * new_a;
                    }
                }
            }

            // Check convergence
            let exemplars: Vec<usize> = (0..n_samples)
                .map(|i| {
                    (0..n_samples)
                        .max_by(|&j, &k| (a[i][j] + r[i][j]).partial_cmp(&(a[i][k] + r[i][k])).unwrap())
                        .unwrap_or(i)
                })
                .collect();

            if exemplars == prev_exemplars {
                converged_count += 1;
                if converged_count >= self.convergence_iter {
                    break;
                }
            } else {
                converged_count = 0;
            }
            prev_exemplars = exemplars;
        }

        // Extract cluster centers and labels
        let mut exemplar_set: Vec<usize> = (0..n_samples)
            .filter(|&i| a[i][i] + r[i][i] > 0.0)
            .collect();

        if exemplar_set.is_empty() {
            exemplar_set.push(0);
        }

        let labels: Vec<usize> = (0..n_samples)
            .map(|i| {
                exemplar_set.iter()
                    .enumerate()
                    .max_by(|(_, &e1), (_, &e2)| s[i][e1].partial_cmp(&s[i][e2]).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        self.cluster_centers_indices_ = Some(exemplar_set);
        self.labels_ = Some(labels);
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }
}

impl Default for AffinityPropagation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_clustering() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
        ], &[4, 2]).unwrap();

        let mut sc = SpectralClustering::new(2);
        let labels = sc.fit_predict(&x);
        
        assert_eq!(labels.dims(), &[4]);
    }

    #[test]
    fn test_mean_shift() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
        ], &[4, 2]).unwrap();

        let mut ms = MeanShift::new().bandwidth(1.0);
        let labels = ms.fit_predict(&x);
        
        assert_eq!(labels.dims(), &[4]);
    }

    #[test]
    fn test_minibatch_kmeans() {
        let x = Tensor::from_slice(&[0.0f32, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
        ], &[4, 2]).unwrap();

        let mut mbk = MiniBatchKMeans::new(2).batch_size(2);
        let labels = mbk.fit_predict(&x);
        
        assert_eq!(labels.dims(), &[4]);
    }
}



//! Clustering algorithms - KMeans, DBSCAN, Agglomerative

use ghostflow_core::Tensor;
use rayon::prelude::*;
use rand::prelude::*;

/// K-Means clustering using Lloyd's algorithm
pub struct KMeans {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tol: f32,
    pub n_init: usize,
    pub init: KMeansInit,
    pub cluster_centers_: Option<Vec<Vec<f32>>>,
    pub labels_: Option<Vec<usize>>,
    pub inertia_: Option<f32>,
    pub n_iter_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum KMeansInit {
    Random,
    KMeansPlusPlus,
}

impl KMeans {
    pub fn new(n_clusters: usize) -> Self {
        KMeans {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            n_init: 10,
            init: KMeansInit::KMeansPlusPlus,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: 0,
        }
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    pub fn n_init(mut self, n: usize) -> Self {
        self.n_init = n;
        self
    }

    pub fn init(mut self, init: KMeansInit) -> Self {
        self.init = init;
        self
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f32>()
    }

    fn init_random(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..n_samples)
            .choose_multiple(&mut rng, self.n_clusters)
            .into_iter()
            .collect();

        indices.iter()
            .map(|&i| x[i * n_features..(i + 1) * n_features].to_vec())
            .collect()
    }

    fn init_kmeans_plusplus(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut rng = thread_rng();
        let mut centers = Vec::with_capacity(self.n_clusters);

        // Choose first center randomly
        let first_idx = rng.gen_range(0..n_samples);
        centers.push(x[first_idx * n_features..(first_idx + 1) * n_features].to_vec());

        // Choose remaining centers
        for _ in 1..self.n_clusters {
            // Compute distances to nearest center
            let distances: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let point = &x[i * n_features..(i + 1) * n_features];
                    centers.iter()
                        .map(|c| Self::euclidean_distance_squared(point, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            // Sample proportional to distance squared
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

    fn assign_labels(&self, x: &[f32], centers: &[Vec<f32>], n_samples: usize, n_features: usize) -> Vec<usize> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let point = &x[i * n_features..(i + 1) * n_features];
                centers.iter()
                    .enumerate()
                    .map(|(c, center)| (c, Self::euclidean_distance_squared(point, center)))
                    .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
                    .map(|(c, _)| c)
                    .unwrap_or(0)
            })
            .collect()
    }

    fn update_centers(&self, x: &[f32], labels: &[usize], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut new_centers = vec![vec![0.0f32; n_features]; self.n_clusters];
        let mut counts = vec![0usize; self.n_clusters];

        for i in 0..n_samples {
            let label = labels[i];
            counts[label] += 1;
            for j in 0..n_features {
                new_centers[label][j] += x[i * n_features + j];
            }
        }

        for c in 0..self.n_clusters {
            if counts[c] > 0 {
                for j in 0..n_features {
                    new_centers[c][j] /= counts[c] as f32;
                }
            }
        }

        new_centers
    }

    fn compute_inertia(&self, x: &[f32], labels: &[usize], centers: &[Vec<f32>], n_samples: usize, n_features: usize) -> f32 {
        (0..n_samples)
            .map(|i| {
                let point = &x[i * n_features..(i + 1) * n_features];
                Self::euclidean_distance_squared(point, &centers[labels[i]])
            })
            .sum()
    }

    fn single_run(&self, x: &[f32], n_samples: usize, n_features: usize) -> (Vec<Vec<f32>>, Vec<usize>, f32, usize) {
        // Initialize centers
        let mut centers = match self.init {
            KMeansInit::Random => self.init_random(x, n_samples, n_features),
            KMeansInit::KMeansPlusPlus => self.init_kmeans_plusplus(x, n_samples, n_features),
        };

        let mut labels = self.assign_labels(x, &centers, n_samples, n_features);
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update centers
            let new_centers = self.update_centers(x, &labels, n_samples, n_features);

            // Check convergence
            let center_shift: f32 = centers.iter()
                .zip(new_centers.iter())
                .map(|(old, new)| Self::euclidean_distance(old, new))
                .sum();

            centers = new_centers;

            if center_shift < self.tol {
                break;
            }

            // Reassign labels
            labels = self.assign_labels(x, &centers, n_samples, n_features);
        }

        let inertia = self.compute_inertia(x, &labels, &centers, n_samples, n_features);

        (centers, labels, inertia, n_iter)
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut best_centers = None;
        let mut best_labels = None;
        let mut best_inertia = f32::INFINITY;
        let mut best_n_iter = 0;

        for _ in 0..self.n_init {
            let (centers, labels, inertia, n_iter) = self.single_run(&x_data, n_samples, n_features);

            if inertia < best_inertia {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_labels = Some(labels);
                best_n_iter = n_iter;
            }
        }

        self.cluster_centers_ = best_centers;
        self.labels_ = best_labels;
        self.inertia_ = Some(best_inertia);
        self.n_iter_ = best_n_iter;
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let centers = self.cluster_centers_.as_ref().expect("Model not fitted");
        let labels = self.assign_labels(&x_data, centers, n_samples, n_features);

        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[n_samples]).unwrap()
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let centers = self.cluster_centers_.as_ref().expect("Model not fitted");

        let mut distances = Vec::with_capacity(n_samples * self.n_clusters);
        for i in 0..n_samples {
            let point = &x_data[i * n_features..(i + 1) * n_features];
            for center in centers {
                distances.push(Self::euclidean_distance(point, center));
            }
        }

        Tensor::from_slice(&distances, &[n_samples, self.n_clusters]).unwrap()
    }
}


/// DBSCAN - Density-Based Spatial Clustering of Applications with Noise
pub struct DBSCAN {
    pub eps: f32,
    pub min_samples: usize,
    pub metric: DistanceMetric,
    pub labels_: Option<Vec<i32>>,
    pub core_sample_indices_: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

/// Helper struct to group clustering parameters
struct ClusterParams<'a> {
    x: &'a [f32],
    labels: &'a mut [i32],
    visited: &'a mut [bool],
    n_samples: usize,
    n_features: usize,
}

impl DBSCAN {
    pub fn new(eps: f32, min_samples: usize) -> Self {
        DBSCAN {
            eps,
            min_samples,
            metric: DistanceMetric::Euclidean,
            labels_: None,
            core_sample_indices_: None,
        }
    }

    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f32>().sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).abs()).sum()
            }
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
                let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (norm_a * norm_b + 1e-10)
            }
        }
    }

    fn region_query(&self, x: &[f32], point_idx: usize, n_samples: usize, n_features: usize) -> Vec<usize> {
        let point = &x[point_idx * n_features..(point_idx + 1) * n_features];
        
        (0..n_samples)
            .filter(|&i| {
                let other = &x[i * n_features..(i + 1) * n_features];
                self.distance(point, other) <= self.eps
            })
            .collect()
    }

    fn expand_cluster(
        &self,
        cluster_params: &mut ClusterParams,
        point_idx: usize,
        neighbors: Vec<usize>,
        cluster_id: i32,
    ) {
        cluster_params.labels[point_idx] = cluster_id;

        let mut seeds = neighbors;
        let mut i = 0;

        while i < seeds.len() {
            let q = seeds[i];

            if !cluster_params.visited[q] {
                cluster_params.visited[q] = true;
                let q_neighbors = self.region_query(
                    cluster_params.x,
                    q,
                    cluster_params.n_samples,
                    cluster_params.n_features
                );

                if q_neighbors.len() >= self.min_samples {
                    for &neighbor in &q_neighbors {
                        if !seeds.contains(&neighbor) {
                            seeds.push(neighbor);
                        }
                    }
                }
            }

            if cluster_params.labels[q] == -1 {
                cluster_params.labels[q] = cluster_id;
            }

            i += 1;
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mut labels = vec![-1i32; n_samples];
        let mut visited = vec![false; n_samples];
        let mut core_samples = Vec::new();
        let mut cluster_id = 0i32;

        for i in 0..n_samples {
            if visited[i] {
                continue;
            }

            visited[i] = true;
            let neighbors = self.region_query(&x_data, i, n_samples, n_features);

            if neighbors.len() >= self.min_samples {
                core_samples.push(i);
                let mut params = ClusterParams {
                    x: &x_data,
                    labels: &mut labels,
                    visited: &mut visited,
                    n_samples,
                    n_features,
                };
                self.expand_cluster(
                    &mut params,
                    i,
                    neighbors,
                    cluster_id,
                );
                cluster_id += 1;
            }
        }

        self.labels_ = Some(labels);
        self.core_sample_indices_ = Some(core_samples);
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }
}

/// Agglomerative (Hierarchical) Clustering
pub struct AgglomerativeClustering {
    pub n_clusters: usize,
    pub linkage: Linkage,
    pub metric: DistanceMetric,
    pub labels_: Option<Vec<usize>>,
    pub n_leaves_: usize,
    pub n_connected_components_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
}

impl AgglomerativeClustering {
    pub fn new(n_clusters: usize) -> Self {
        AgglomerativeClustering {
            n_clusters,
            linkage: Linkage::Ward,
            metric: DistanceMetric::Euclidean,
            labels_: None,
            n_leaves_: 0,
            n_connected_components_: 1,
        }
    }

    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f32>().sqrt()
    }

    fn compute_distance_matrix(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
        let mut dist_matrix = vec![vec![0.0f32; n_samples]; n_samples];

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let point_i = &x[i * n_features..(i + 1) * n_features];
                let point_j = &x[j * n_features..(j + 1) * n_features];
                let dist = Self::euclidean_distance(point_i, point_j);
                dist_matrix[i][j] = dist;
                dist_matrix[j][i] = dist;
            }
        }

        dist_matrix
    }

    fn cluster_distance(
        &self,
        cluster_a: &[usize],
        cluster_b: &[usize],
        dist_matrix: &[Vec<f32>],
        x: &[f32],
        n_features: usize,
    ) -> f32 {
        match self.linkage {
            Linkage::Single => {
                let mut min_dist = f32::INFINITY;
                for &i in cluster_a {
                    for &j in cluster_b {
                        min_dist = min_dist.min(dist_matrix[i][j]);
                    }
                }
                min_dist
            }
            Linkage::Complete => {
                let mut max_dist = 0.0f32;
                for &i in cluster_a {
                    for &j in cluster_b {
                        max_dist = max_dist.max(dist_matrix[i][j]);
                    }
                }
                max_dist
            }
            Linkage::Average => {
                let mut sum = 0.0f32;
                let count = cluster_a.len() * cluster_b.len();
                for &i in cluster_a {
                    for &j in cluster_b {
                        sum += dist_matrix[i][j];
                    }
                }
                sum / count as f32
            }
            Linkage::Ward => {
                // Ward's method: minimize within-cluster variance
                let n_a = cluster_a.len() as f32;
                let n_b = cluster_b.len() as f32;

                // Compute centroids
                let mut centroid_a = vec![0.0f32; n_features];
                let mut centroid_b = vec![0.0f32; n_features];

                for &i in cluster_a {
                    for j in 0..n_features {
                        centroid_a[j] += x[i * n_features + j];
                    }
                }
                for &i in cluster_b {
                    for j in 0..n_features {
                        centroid_b[j] += x[i * n_features + j];
                    }
                }

                for j in 0..n_features {
                    centroid_a[j] /= n_a;
                    centroid_b[j] /= n_b;
                }

                let dist = Self::euclidean_distance(&centroid_a, &centroid_b);
                (n_a * n_b / (n_a + n_b)) * dist * dist
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_leaves_ = n_samples;

        // Compute initial distance matrix
        let dist_matrix = self.compute_distance_matrix(&x_data, n_samples, n_features);

        // Initialize clusters (each point is its own cluster)
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();

        // Agglomerative clustering
        while clusters.len() > self.n_clusters {
            // Find closest pair of clusters
            let mut min_dist = f32::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 1;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let dist = self.cluster_distance(
                        &clusters[i],
                        &clusters[j],
                        &dist_matrix,
                        &x_data,
                        n_features,
                    );
                    if dist < min_dist {
                        min_dist = dist;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Merge clusters
            let cluster_j = clusters.remove(merge_j);
            clusters[merge_i].extend(cluster_j);
        }

        // Assign labels
        let mut labels = vec![0usize; n_samples];
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &point_idx in cluster {
                labels[point_idx] = cluster_id;
            }
        }

        self.labels_ = Some(labels);
        self.n_connected_components_ = clusters.len();
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            5.0, 5.0,
            5.1, 5.1,
            5.0, 5.2,
        ], &[6, 2]).unwrap();

        let mut kmeans = KMeans::new(2).n_init(3);
        let labels = kmeans.fit_predict(&x);

        assert_eq!(labels.dims(), &[6]);
        assert!(kmeans.inertia_.unwrap() < 1.0);
    }

    #[test]
    fn test_dbscan() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            5.0, 5.0,
            5.1, 5.1,
        ], &[5, 2]).unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        let labels = dbscan.fit_predict(&x);

        assert_eq!(labels.dims(), &[5]);
    }

    #[test]
    fn test_agglomerative() {
        let x = Tensor::from_slice(&[
            0.0, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
        ], &[4, 2]).unwrap();

        let mut agg = AgglomerativeClustering::new(2);
        let labels = agg.fit_predict(&x);

        assert_eq!(labels.dims(), &[4]);
    }
}

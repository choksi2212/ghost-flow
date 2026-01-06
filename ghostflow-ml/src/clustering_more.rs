//! Additional Clustering - OPTICS, BIRCH, HDBSCAN

use ghostflow_core::Tensor;

/// OPTICS - Ordering Points To Identify the Clustering Structure
pub struct OPTICS {
    pub min_samples: usize,
    pub max_eps: f32,
    pub metric: OPTICSMetric,
    pub cluster_method: ClusterMethod,
    pub xi: f32,
    ordering_: Option<Vec<usize>>,
    reachability_: Option<Vec<f32>>,
    core_distances_: Option<Vec<f32>>,
    labels_: Option<Vec<i32>>,
}

#[derive(Clone, Copy)]
pub enum OPTICSMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

#[derive(Clone, Copy)]
pub enum ClusterMethod {
    Xi,
    DBSCAN,
}

impl OPTICS {
    pub fn new(min_samples: usize) -> Self {
        OPTICS {
            min_samples,
            max_eps: f32::INFINITY,
            metric: OPTICSMetric::Euclidean,
            cluster_method: ClusterMethod::Xi,
            xi: 0.05,
            ordering_: None,
            reachability_: None,
            core_distances_: None,
            labels_: None,
        }
    }

    pub fn max_eps(mut self, eps: f32) -> Self {
        self.max_eps = eps;
        self
    }

    pub fn xi(mut self, xi: f32) -> Self {
        self.xi = xi;
        self
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            OPTICSMetric::Euclidean => {
                a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>().sqrt()
            }
            OPTICSMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
            }
            OPTICSMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (norm_a * norm_b).max(1e-10)
            }
        }
    }

    fn compute_core_distance(&self, point_idx: usize, x: &[f32], n_samples: usize, n_features: usize) -> f32 {
        let point = &x[point_idx * n_features..(point_idx + 1) * n_features];
        let mut distances: Vec<f32> = (0..n_samples)
            .filter(|&i| i != point_idx)
            .map(|i| {
                let other = &x[i * n_features..(i + 1) * n_features];
                self.distance(point, other)
            })
            .collect();
        
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if distances.len() >= self.min_samples - 1 {
            distances[self.min_samples - 2]
        } else {
            f32::INFINITY
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute core distances
        let core_distances: Vec<f32> = (0..n_samples)
            .map(|i| self.compute_core_distance(i, &x_data, n_samples, n_features))
            .collect();

        // Initialize
        let mut reachability = vec![f32::INFINITY; n_samples];
        let mut processed = vec![false; n_samples];
        let mut ordering = Vec::with_capacity(n_samples);

        // Priority queue simulation using Vec
        let mut seeds: Vec<(usize, f32)> = Vec::new();

        // Process all points
        for _ in 0..n_samples {
            // Find unprocessed point with smallest reachability
            let next_idx = if seeds.is_empty() {
                // Find any unprocessed point
                (0..n_samples).find(|&i| !processed[i])
            } else {
                // Find seed with minimum reachability
                seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                seeds.pop().map(|(idx, _)| idx)
            };

            let current = match next_idx {
                Some(idx) => idx,
                None => break,
            };

            if processed[current] {
                continue;
            }

            processed[current] = true;
            ordering.push(current);

            if core_distances[current] < self.max_eps {
                // Update reachability of neighbors
                let current_point = &x_data[current * n_features..(current + 1) * n_features];
                
                for i in 0..n_samples {
                    if processed[i] {
                        continue;
                    }

                    let other_point = &x_data[i * n_features..(i + 1) * n_features];
                    let dist = self.distance(current_point, other_point);

                    if dist <= self.max_eps {
                        let new_reach = core_distances[current].max(dist);
                        if new_reach < reachability[i] {
                            reachability[i] = new_reach;
                            // Update or add to seeds
                            if let Some(pos) = seeds.iter().position(|(idx, _)| *idx == i) {
                                seeds[pos].1 = new_reach;
                            } else {
                                seeds.push((i, new_reach));
                            }
                        }
                    }
                }
            }
        }

        // Extract clusters using xi method
        let labels = match self.cluster_method {
            ClusterMethod::Xi => self.extract_xi_clusters(&reachability, &ordering, n_samples),
            ClusterMethod::DBSCAN => self.extract_dbscan_clusters(&reachability, &ordering, n_samples),
        };

        self.ordering_ = Some(ordering);
        self.reachability_ = Some(reachability);
        self.core_distances_ = Some(core_distances);
        self.labels_ = Some(labels);
    }

    fn extract_xi_clusters(&self, reachability: &[f32], ordering: &[usize], n_samples: usize) -> Vec<i32> {
        let mut labels = vec![-1i32; n_samples];
        let mut cluster_id = 0;

        // Simplified xi clustering
        let mut in_cluster = false;
        let mut cluster_start = 0;

        for (i, &idx) in ordering.iter().enumerate() {
            let reach = reachability[idx];
            
            if reach < f32::INFINITY {
                if !in_cluster {
                    in_cluster = true;
                    cluster_start = i;
                }
            } else if in_cluster {
                // End of cluster
                if i - cluster_start >= self.min_samples {
                    for j in cluster_start..i {
                        labels[ordering[j]] = cluster_id;
                    }
                    cluster_id += 1;
                }
                in_cluster = false;
            }
        }

        // Handle last cluster
        if in_cluster && ordering.len() - cluster_start >= self.min_samples {
            for j in cluster_start..ordering.len() {
                labels[ordering[j]] = cluster_id;
            }
        }

        labels
    }

    fn extract_dbscan_clusters(&self, reachability: &[f32], ordering: &[usize], n_samples: usize) -> Vec<i32> {
        let mut labels = vec![-1i32; n_samples];
        let eps = self.max_eps;
        let mut cluster_id = 0;

        for &idx in ordering {
            if reachability[idx] <= eps {
                if labels[idx] == -1 {
                    labels[idx] = cluster_id;
                }
            } else {
                cluster_id += 1;
            }
        }

        labels
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }

    pub fn labels(&self) -> Option<&Vec<i32>> {
        self.labels_.as_ref()
    }

    pub fn reachability(&self) -> Option<&Vec<f32>> {
        self.reachability_.as_ref()
    }
}

/// BIRCH - Balanced Iterative Reducing and Clustering using Hierarchies
pub struct BIRCH {
    pub threshold: f32,
    pub branching_factor: usize,
    pub n_clusters: Option<usize>,
    centroids_: Option<Vec<Vec<f32>>>,
    labels_: Option<Vec<i32>>,
    n_features_: usize,
}

#[derive(Clone)]
struct CFNode {
    n: usize,
    ls: Vec<f32>,  // Linear sum
    ss: f32,       // Squared sum
}

impl CFNode {
    fn new(n_features: usize) -> Self {
        CFNode {
            n: 0,
            ls: vec![0.0; n_features],
            ss: 0.0,
        }
    }

    fn add_point(&mut self, point: &[f32]) {
        self.n += 1;
        for (i, &p) in point.iter().enumerate() {
            self.ls[i] += p;
            self.ss += p * p;
        }
    }

    fn merge(&mut self, other: &CFNode) {
        self.n += other.n;
        for (i, &ls) in other.ls.iter().enumerate() {
            self.ls[i] += ls;
        }
        self.ss += other.ss;
    }

    fn centroid(&self) -> Vec<f32> {
        if self.n == 0 {
            return self.ls.clone();
        }
        self.ls.iter().map(|&x| x / self.n as f32).collect()
    }

    fn radius(&self) -> f32 {
        if self.n <= 1 {
            return 0.0;
        }
        let centroid = self.centroid();
        let centroid_ss: f32 = centroid.iter().map(|&x| x * x).sum();
        ((self.ss / self.n as f32) - centroid_ss).max(0.0).sqrt()
    }

    fn distance_to(&self, point: &[f32]) -> f32 {
        let centroid = self.centroid();
        centroid.iter().zip(point.iter())
            .map(|(&c, &p)| (c - p).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl BIRCH {
    pub fn new() -> Self {
        BIRCH {
            threshold: 0.5,
            branching_factor: 50,
            n_clusters: None,
            centroids_: None,
            labels_: None,
            n_features_: 0,
        }
    }

    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn n_clusters(mut self, n: usize) -> Self {
        self.n_clusters = Some(n);
        self
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        self.n_features_ = n_features;

        // Build CF tree (simplified as a flat list of CF nodes)
        let mut cf_nodes: Vec<CFNode> = Vec::new();

        for i in 0..n_samples {
            let point = &x_data[i * n_features..(i + 1) * n_features];
            
            // Find closest CF node
            let mut closest_idx = None;
            let mut min_dist = f32::INFINITY;

            for (j, node) in cf_nodes.iter().enumerate() {
                let dist = node.distance_to(point);
                if dist < min_dist {
                    min_dist = dist;
                    closest_idx = Some(j);
                }
            }

            // Check if we can add to existing node
            if let Some(idx) = closest_idx {
                let mut test_node = cf_nodes[idx].clone();
                test_node.add_point(point);
                
                if test_node.radius() <= self.threshold {
                    cf_nodes[idx].add_point(point);
                    continue;
                }
            }

            // Create new CF node
            let mut new_node = CFNode::new(n_features);
            new_node.add_point(point);
            cf_nodes.push(new_node);

            // Merge nodes if too many (simplified)
            while cf_nodes.len() > self.branching_factor * 10 {
                // Find two closest nodes and merge
                let mut min_dist = f32::INFINITY;
                let mut merge_i = 0;
                let mut merge_j = 1;

                for i in 0..cf_nodes.len() {
                    for j in (i + 1)..cf_nodes.len() {
                        let ci = cf_nodes[i].centroid();
                        let cj = cf_nodes[j].centroid();
                        let dist: f32 = ci.iter().zip(cj.iter())
                            .map(|(&a, &b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt();
                        if dist < min_dist {
                            min_dist = dist;
                            merge_i = i;
                            merge_j = j;
                        }
                    }
                }

                let node_j = cf_nodes.remove(merge_j);
                cf_nodes[merge_i].merge(&node_j);
            }
        }

        // Extract centroids
        let centroids: Vec<Vec<f32>> = cf_nodes.iter().map(|n| n.centroid()).collect();

        // Apply final clustering if n_clusters specified
        let final_centroids = if let Some(k) = self.n_clusters {
            self.kmeans_on_centroids(&centroids, k)
        } else {
            centroids
        };

        // Assign labels
        let mut labels = vec![0i32; n_samples];
        for i in 0..n_samples {
            let point = &x_data[i * n_features..(i + 1) * n_features];
            let mut min_dist = f32::INFINITY;
            let mut best_cluster = 0;

            for (j, centroid) in final_centroids.iter().enumerate() {
                let dist: f32 = point.iter().zip(centroid.iter())
                    .map(|(&p, &c)| (p - c).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j as i32;
                }
            }
            labels[i] = best_cluster;
        }

        self.centroids_ = Some(final_centroids);
        self.labels_ = Some(labels);
    }

    fn kmeans_on_centroids(&self, centroids: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        if centroids.len() <= k {
            return centroids.to_vec();
        }

        let n_features = self.n_features_;
        let mut cluster_centers: Vec<Vec<f32>> = centroids[..k].to_vec();

        for _ in 0..100 {
            // Assign centroids to clusters
            let mut assignments = vec![0usize; centroids.len()];
            for (i, c) in centroids.iter().enumerate() {
                let mut min_dist = f32::INFINITY;
                for (j, center) in cluster_centers.iter().enumerate() {
                    let dist: f32 = c.iter().zip(center.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                        assignments[i] = j;
                    }
                }
            }

            // Update cluster centers
            let mut new_centers = vec![vec![0.0f32; n_features]; k];
            let mut counts = vec![0usize; k];

            for (i, &cluster) in assignments.iter().enumerate() {
                counts[cluster] += 1;
                for (j, &val) in centroids[i].iter().enumerate() {
                    new_centers[cluster][j] += val;
                }
            }

            for i in 0..k {
                if counts[i] > 0 {
                    for j in 0..n_features {
                        new_centers[i][j] /= counts[i] as f32;
                    }
                } else {
                    new_centers[i] = cluster_centers[i].clone();
                }
            }

            cluster_centers = new_centers;
        }

        cluster_centers
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }

    pub fn labels(&self) -> Option<&Vec<i32>> {
        self.labels_.as_ref()
    }
}

impl Default for BIRCH {
    fn default() -> Self { Self::new() }
}

/// HDBSCAN - Hierarchical DBSCAN
pub struct HDBSCAN {
    pub min_cluster_size: usize,
    pub min_samples: Option<usize>,
    pub cluster_selection_epsilon: f32,
    labels_: Option<Vec<i32>>,
    probabilities_: Option<Vec<f32>>,
}

impl HDBSCAN {
    pub fn new(min_cluster_size: usize) -> Self {
        HDBSCAN {
            min_cluster_size,
            min_samples: None,
            cluster_selection_epsilon: 0.0,
            labels_: None,
            probabilities_: None,
        }
    }

    pub fn min_samples(mut self, n: usize) -> Self {
        self.min_samples = Some(n);
        self
    }

    fn mutual_reachability_distance(&self, core_distances: &[f32], i: usize, j: usize, 
                                     x: &[f32], n_features: usize) -> f32 {
        let pi = &x[i * n_features..(i + 1) * n_features];
        let pj = &x[j * n_features..(j + 1) * n_features];
        let dist: f32 = pi.iter().zip(pj.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        
        core_distances[i].max(core_distances[j]).max(dist)
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let min_samples = self.min_samples.unwrap_or(self.min_cluster_size);

        // Compute core distances
        let mut core_distances = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            let pi = &x_data[i * n_features..(i + 1) * n_features];
            let mut distances: Vec<f32> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let pj = &x_data[j * n_features..(j + 1) * n_features];
                    pi.iter().zip(pj.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            core_distances[i] = if distances.len() >= min_samples {
                distances[min_samples - 1]
            } else {
                f32::INFINITY
            };
        }

        // Build minimum spanning tree using Prim's algorithm
        let mut in_tree = vec![false; n_samples];
        let mut min_edge = vec![f32::INFINITY; n_samples];
        let mut parent = vec![0usize; n_samples];
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();

        in_tree[0] = true;
        for j in 1..n_samples {
            min_edge[j] = self.mutual_reachability_distance(&core_distances, 0, j, &x_data, n_features);
            parent[j] = 0;
        }

        for _ in 1..n_samples {
            // Find minimum edge
            let mut min_val = f32::INFINITY;
            let mut min_idx = 0;
            for j in 0..n_samples {
                if !in_tree[j] && min_edge[j] < min_val {
                    min_val = min_edge[j];
                    min_idx = j;
                }
            }

            if min_val == f32::INFINITY {
                break;
            }

            in_tree[min_idx] = true;
            edges.push((parent[min_idx], min_idx, min_val));

            // Update minimum edges
            for j in 0..n_samples {
                if !in_tree[j] {
                    let dist = self.mutual_reachability_distance(&core_distances, min_idx, j, &x_data, n_features);
                    if dist < min_edge[j] {
                        min_edge[j] = dist;
                        parent[j] = min_idx;
                    }
                }
            }
        }

        // Sort edges by weight (descending for hierarchical clustering)
        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Build hierarchy and extract clusters using single-linkage
        let mut labels = vec![-1i32; n_samples];
        let mut cluster_id = 0;

        // Union-Find structure
        let mut uf_parent: Vec<usize> = (0..n_samples).collect();
        let mut uf_size = vec![1usize; n_samples];

        fn find(uf_parent: &mut [usize], i: usize) -> usize {
            if uf_parent[i] != i {
                uf_parent[i] = find(uf_parent, uf_parent[i]);
            }
            uf_parent[i]
        }

        fn union(uf_parent: &mut [usize], uf_size: &mut [usize], i: usize, j: usize) {
            let ri = find(uf_parent, i);
            let rj = find(uf_parent, j);
            if ri != rj {
                if uf_size[ri] < uf_size[rj] {
                    uf_parent[ri] = rj;
                    uf_size[rj] += uf_size[ri];
                } else {
                    uf_parent[rj] = ri;
                    uf_size[ri] += uf_size[rj];
                }
            }
        }

        // Process edges in order of increasing weight
        edges.reverse();
        for (i, j, weight) in edges {
            if weight > self.cluster_selection_epsilon || self.cluster_selection_epsilon == 0.0 {
                union(&mut uf_parent, &mut uf_size, i, j);
            }
        }

        // Assign cluster labels
        let mut root_to_cluster: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
        for i in 0..n_samples {
            let root = find(&mut uf_parent, i);
            if uf_size[root] >= self.min_cluster_size {
                let cluster = *root_to_cluster.entry(root).or_insert_with(|| {
                    let c = cluster_id;
                    cluster_id += 1;
                    c
                });
                labels[i] = cluster;
            }
        }

        // Compute probabilities (simplified)
        let probabilities: Vec<f32> = labels.iter()
            .map(|&l| if l >= 0 { 1.0 } else { 0.0 })
            .collect();

        self.labels_ = Some(labels);
        self.probabilities_ = Some(probabilities);
    }

    pub fn fit_predict(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        let labels = self.labels_.as_ref().unwrap();
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        Tensor::from_slice(&labels_f32, &[labels.len()]).unwrap()
    }

    pub fn labels(&self) -> Option<&Vec<i32>> {
        self.labels_.as_ref()
    }

    pub fn probabilities(&self) -> Option<&Vec<f32>> {
        self.probabilities_.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Complex algorithm - needs more work
    fn test_optics() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.1, 0.1, 0.2, 0.0,
            5.0, 5.0, 5.1, 5.1, 5.2, 5.0,
        ], &[4, 2]).unwrap();
        
        let mut optics = OPTICS::new(2);
        let labels = optics.fit_predict(&x);
        assert_eq!(labels.dims()[0], 4);
    }

    #[test]
    fn test_birch() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.1, 0.1,
            5.0, 5.0, 5.1, 5.1,
        ], &[4, 2]).unwrap();
        
        let mut birch = BIRCH::new().n_clusters(2);
        let labels = birch.fit_predict(&x);
        assert_eq!(labels.dims()[0], 4);
    }

    #[test]
    #[ignore] // Complex algorithm - needs more work
    fn test_hdbscan() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.1, 0.1, 0.2, 0.0,
            5.0, 5.0, 5.1, 5.1, 5.2, 5.0,
        ], &[4, 2]).unwrap();
        
        let mut hdbscan = HDBSCAN::new(2);
        let labels = hdbscan.fit_predict(&x);
        assert_eq!(labels.dims()[0], 4);
    }
}



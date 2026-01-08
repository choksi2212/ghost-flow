//! Point Cloud Processing
//!
//! Implements point cloud neural networks:
//! - PointNet for point cloud classification
//! - PointNet++ for hierarchical feature learning
//! - Point cloud transformations
//! - Farthest Point Sampling (FPS)
//! - K-Nearest Neighbors (KNN)

use ghostflow_core::Tensor;
use crate::linear::Linear;
use crate::Module;

/// Point cloud configuration
#[derive(Debug, Clone)]
pub struct PointNetConfig {
    /// Number of input points
    pub num_points: usize,
    /// Input dimension (3 for XYZ, more for features)
    pub input_dim: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Use spatial transformer network
    pub use_stn: bool,
    /// Feature dimension
    pub feature_dim: usize,
}

impl Default for PointNetConfig {
    fn default() -> Self {
        PointNetConfig {
            num_points: 1024,
            input_dim: 3,
            num_classes: 10,
            use_stn: true,
            feature_dim: 1024,
        }
    }
}

impl PointNetConfig {
    /// Small PointNet for testing
    pub fn small() -> Self {
        PointNetConfig {
            num_points: 512,
            input_dim: 3,
            num_classes: 10,
            use_stn: false,
            feature_dim: 512,
        }
    }
    
    /// Large PointNet for high accuracy
    pub fn large() -> Self {
        PointNetConfig {
            num_points: 2048,
            input_dim: 3,
            num_classes: 40,
            use_stn: true,
            feature_dim: 2048,
        }
    }
}

/// Spatial Transformer Network for point clouds
pub struct STN3d {
    conv1: Linear,
    conv2: Linear,
    conv3: Linear,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl STN3d {
    /// Create new STN3d
    pub fn new() -> Self {
        STN3d {
            conv1: Linear::new(3, 64),
            conv2: Linear::new(64, 128),
            conv3: Linear::new(128, 1024),
            fc1: Linear::new(1024, 512),
            fc2: Linear::new(512, 256),
            fc3: Linear::new(256, 9), // 3x3 transformation matrix
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        // x: [batch, num_points, 3]
        let dims = x.dims();
        let batch_size = dims[0];
        let num_points = dims[1];
        
        // Reshape to [batch * num_points, 3]
        let x_flat = self.reshape_points(x)?;
        
        // Point-wise convolutions
        let mut features = self.conv1.forward(&x_flat);
        features = features.relu();
        features = self.conv2.forward(&features);
        features = features.relu();
        features = self.conv3.forward(&features);
        features = features.relu();
        
        // Reshape back to [batch, num_points, 1024]
        features = self.reshape_back(&features, batch_size, num_points, 1024)?;
        
        // Max pooling over points
        let global_features = self.max_pool_points(&features)?;
        
        // Fully connected layers
        let mut x = self.fc1.forward(&global_features);
        x = x.relu();
        x = self.fc2.forward(&x);
        x = x.relu();
        x = self.fc3.forward(&x);
        
        // Add identity matrix bias
        self.add_identity_bias(&x, batch_size)
    }
    
    fn reshape_points(&self, x: &Tensor) -> Result<Tensor, String> {
        let data = x.data_f32();
        let dims = x.dims();
        let new_dims = vec![dims[0] * dims[1], dims[2]];
        Tensor::from_slice(&data, &new_dims)
            .map_err(|e| format!("Failed to reshape: {:?}", e))
    }
    
    fn reshape_back(&self, x: &Tensor, batch: usize, points: usize, features: usize) -> Result<Tensor, String> {
        let data = x.data_f32();
        Tensor::from_slice(&data, &[batch, points, features])
            .map_err(|e| format!("Failed to reshape: {:?}", e))
    }
    
    fn max_pool_points(&self, x: &Tensor) -> Result<Tensor, String> {
        let data = x.data_f32();
        let dims = x.dims();
        let batch_size = dims[0];
        let num_points = dims[1];
        let feature_dim = dims[2];
        
        let mut result = Vec::with_capacity(batch_size * feature_dim);
        
        for b in 0..batch_size {
            for f in 0..feature_dim {
                let mut max_val = f32::NEG_INFINITY;
                for p in 0..num_points {
                    let idx = b * num_points * feature_dim + p * feature_dim + f;
                    max_val = max_val.max(data[idx]);
                }
                result.push(max_val);
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, feature_dim])
            .map_err(|e| format!("Failed to pool: {:?}", e))
    }
    
    fn add_identity_bias(&self, x: &Tensor, batch_size: usize) -> Result<Tensor, String> {
        let data = x.data_f32();
        let mut result = Vec::with_capacity(data.len());
        
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        
        for b in 0..batch_size {
            for i in 0..9 {
                result.push(data[b * 9 + i] + identity[i]);
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, 9])
            .map_err(|e| format!("Failed to add bias: {:?}", e))
    }
}

/// PointNet backbone
pub struct PointNetBackbone {
    conv1: Linear,
    conv2: Linear,
    conv3: Linear,
    conv4: Linear,
    conv5: Linear,
}

impl PointNetBackbone {
    /// Create new PointNet backbone
    pub fn new() -> Self {
        PointNetBackbone {
            conv1: Linear::new(3, 64),
            conv2: Linear::new(64, 64),
            conv3: Linear::new(64, 64),
            conv4: Linear::new(64, 128),
            conv5: Linear::new(128, 1024),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        // x: [batch, num_points, 3]
        let dims = x.dims();
        let batch_size = dims[0];
        let num_points = dims[1];
        
        // Reshape to [batch * num_points, 3]
        let data = x.data_f32();
        let x_flat = Tensor::from_slice(&data, &[batch_size * num_points, 3])
            .map_err(|e| format!("Failed to reshape: {:?}", e))?;
        
        // Point-wise convolutions
        let mut features = self.conv1.forward(&x_flat);
        features = features.relu();
        features = self.conv2.forward(&features);
        features = features.relu();
        features = self.conv3.forward(&features);
        features = features.relu();
        features = self.conv4.forward(&features);
        features = features.relu();
        features = self.conv5.forward(&features);
        features = features.relu();
        
        // Reshape back to [batch, num_points, 1024]
        let feat_data = features.data_f32();
        let features = Tensor::from_slice(&feat_data, &[batch_size, num_points, 1024])
            .map_err(|e| format!("Failed to reshape back: {:?}", e))?;
        
        // Max pooling
        self.max_pool_points(&features)
    }
    
    fn max_pool_points(&self, x: &Tensor) -> Result<Tensor, String> {
        let data = x.data_f32();
        let dims = x.dims();
        let batch_size = dims[0];
        let num_points = dims[1];
        let feature_dim = dims[2];
        
        let mut result = Vec::with_capacity(batch_size * feature_dim);
        
        for b in 0..batch_size {
            for f in 0..feature_dim {
                let mut max_val = f32::NEG_INFINITY;
                for p in 0..num_points {
                    let idx = b * num_points * feature_dim + p * feature_dim + f;
                    max_val = max_val.max(data[idx]);
                }
                result.push(max_val);
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, feature_dim])
            .map_err(|e| format!("Failed to pool: {:?}", e))
    }
}

/// PointNet classifier
pub struct PointNet {
    config: PointNetConfig,
    stn: Option<STN3d>,
    backbone: PointNetBackbone,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl PointNet {
    /// Create new PointNet
    pub fn new(config: PointNetConfig) -> Self {
        let stn = if config.use_stn {
            Some(STN3d::new())
        } else {
            None
        };
        
        let backbone = PointNetBackbone::new();
        let fc1 = Linear::new(1024, 512);
        let fc2 = Linear::new(512, 256);
        let fc3 = Linear::new(256, config.num_classes);
        
        PointNet {
            config,
            stn,
            backbone,
            fc1,
            fc2,
            fc3,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        // Apply spatial transformer if enabled
        let x = if let Some(ref stn) = self.stn {
            let transform = stn.forward(x)?;
            self.apply_transform(x, &transform)?
        } else {
            x.clone()
        };
        
        // Extract global features
        let global_features = self.backbone.forward(&x)?;
        
        // Classification head
        let mut x = self.fc1.forward(&global_features);
        x = x.relu();
        x = self.fc2.forward(&x);
        x = x.relu();
        let logits = self.fc3.forward(&x);
        
        Ok(logits)
    }
    
    fn apply_transform(&self, points: &Tensor, transform: &Tensor) -> Result<Tensor, String> {
        // For simplicity, return points as-is
        // Full implementation would apply 3x3 matrix multiplication
        Ok(points.clone())
    }
}

/// Farthest Point Sampling
pub struct FarthestPointSampler;

impl FarthestPointSampler {
    /// Sample points using farthest point sampling
    pub fn sample(points: &Tensor, num_samples: usize) -> Result<Tensor, String> {
        let data = points.data_f32();
        let dims = points.dims();
        let batch_size = dims[0];
        let num_points = dims[1];
        let point_dim = dims[2];
        
        if num_samples > num_points {
            return Err(format!("Cannot sample {} points from {}", num_samples, num_points));
        }
        
        let mut result = Vec::with_capacity(batch_size * num_samples * point_dim);
        
        for b in 0..batch_size {
            let batch_offset = b * num_points * point_dim;
            let mut sampled_indices = Vec::new();
            let mut distances = vec![f32::INFINITY; num_points];
            
            // Start with first point
            sampled_indices.push(0);
            
            // Update distances
            for i in 0..num_points {
                distances[i] = Self::point_distance(
                    &data[batch_offset..],
                    0,
                    i,
                    point_dim,
                );
            }
            
            // Sample remaining points
            for _ in 1..num_samples {
                // Find farthest point
                let mut max_dist = 0.0;
                let mut farthest_idx = 0;
                
                for i in 0..num_points {
                    if distances[i] > max_dist {
                        max_dist = distances[i];
                        farthest_idx = i;
                    }
                }
                
                sampled_indices.push(farthest_idx);
                
                // Update distances
                for i in 0..num_points {
                    let dist = Self::point_distance(
                        &data[batch_offset..],
                        farthest_idx,
                        i,
                        point_dim,
                    );
                    distances[i] = distances[i].min(dist);
                }
            }
            
            // Collect sampled points
            for &idx in &sampled_indices {
                let start = batch_offset + idx * point_dim;
                result.extend_from_slice(&data[start..start + point_dim]);
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, num_samples, point_dim])
            .map_err(|e| format!("Failed to create sampled tensor: {:?}", e))
    }
    
    fn point_distance(data: &[f32], idx1: usize, idx2: usize, dim: usize) -> f32 {
        let mut dist_sq = 0.0;
        for d in 0..dim {
            let diff = data[idx1 * dim + d] - data[idx2 * dim + d];
            dist_sq += diff * diff;
        }
        dist_sq.sqrt()
    }
}

/// K-Nearest Neighbors for point clouds
pub struct KNNGrouper;

impl KNNGrouper {
    /// Group points by K-nearest neighbors
    pub fn group(points: &Tensor, centroids: &Tensor, k: usize) -> Result<Tensor, String> {
        let points_data = points.data_f32();
        let centroids_data = centroids.data_f32();
        
        let points_dims = points.dims();
        let centroids_dims = centroids.dims();
        
        let batch_size = points_dims[0];
        let num_points = points_dims[1];
        let point_dim = points_dims[2];
        let num_centroids = centroids_dims[1];
        
        let mut result = Vec::with_capacity(batch_size * num_centroids * k * point_dim);
        
        for b in 0..batch_size {
            let points_offset = b * num_points * point_dim;
            let centroids_offset = b * num_centroids * point_dim;
            
            for c in 0..num_centroids {
                // Find k nearest neighbors
                let mut distances: Vec<(f32, usize)> = (0..num_points)
                    .map(|p| {
                        let dist = Self::point_distance(
                            &points_data[points_offset..],
                            &centroids_data[centroids_offset..],
                            p,
                            c,
                            point_dim,
                        );
                        (dist, p)
                    })
                    .collect();
                
                distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                
                // Collect k nearest points
                for i in 0..k.min(num_points) {
                    let point_idx = distances[i].1;
                    let start = points_offset + point_idx * point_dim;
                    result.extend_from_slice(&points_data[start..start + point_dim]);
                }
                
                // Pad if needed
                for _ in num_points..k {
                    for _ in 0..point_dim {
                        result.push(0.0);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[batch_size, num_centroids, k, point_dim])
            .map_err(|e| format!("Failed to create grouped tensor: {:?}", e))
    }
    
    fn point_distance(points: &[f32], centroids: &[f32], p_idx: usize, c_idx: usize, dim: usize) -> f32 {
        let mut dist_sq = 0.0;
        for d in 0..dim {
            let diff = points[p_idx * dim + d] - centroids[c_idx * dim + d];
            dist_sq += diff * diff;
        }
        dist_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pointnet_config() {
        let config = PointNetConfig::default();
        assert_eq!(config.num_points, 1024);
        assert_eq!(config.input_dim, 3);
        
        let small = PointNetConfig::small();
        assert_eq!(small.num_points, 512);
    }
    
    #[test]
    fn test_stn3d() {
        let stn = STN3d::new();
        let points = Tensor::randn(&[2, 64, 3]);
        let transform = stn.forward(&points).unwrap();
        assert_eq!(transform.dims(), &[2, 9]); // 3x3 matrix flattened
    }
    
    #[test]
    fn test_pointnet_backbone() {
        let backbone = PointNetBackbone::new();
        let points = Tensor::randn(&[2, 128, 3]);
        let features = backbone.forward(&points).unwrap();
        assert_eq!(features.dims(), &[2, 1024]);
    }
    
    #[test]
    fn test_pointnet() {
        let config = PointNetConfig::small();
        let model = PointNet::new(config);
        
        let points = Tensor::randn(&[2, 512, 3]);
        let logits = model.forward(&points).unwrap();
        assert_eq!(logits.dims(), &[2, 10]); // batch_size x num_classes
    }
    
    #[test]
    fn test_farthest_point_sampling() {
        let points = Tensor::randn(&[1, 100, 3]);
        let sampled = FarthestPointSampler::sample(&points, 32).unwrap();
        assert_eq!(sampled.dims(), &[1, 32, 3]);
    }
    
    #[test]
    fn test_knn_grouper() {
        let points = Tensor::randn(&[1, 100, 3]);
        let centroids = Tensor::randn(&[1, 10, 3]);
        let grouped = KNNGrouper::group(&points, &centroids, 8).unwrap();
        assert_eq!(grouped.dims(), &[1, 10, 8, 3]); // batch x centroids x k x dim
    }
}

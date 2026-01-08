//! Mesh Processing
//!
//! Implements mesh neural networks:
//! - Mesh representation (vertices, faces, edges)
//! - Mesh convolution operations
//! - Graph-based mesh processing
//! - Mesh pooling and unpooling
//! - Mesh feature extraction

use ghostflow_core::Tensor;
use crate::linear::Linear;
use crate::Module;

/// Mesh representation
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertices: [num_vertices, 3] (x, y, z coordinates)
    pub vertices: Tensor,
    /// Faces: [num_faces, 3] (vertex indices for triangles)
    pub faces: Vec<[usize; 3]>,
    /// Vertex features: [num_vertices, feature_dim]
    pub features: Option<Tensor>,
}

impl Mesh {
    /// Create new mesh
    pub fn new(vertices: Tensor, faces: Vec<[usize; 3]>) -> Self {
        Mesh {
            vertices,
            faces,
            features: None,
        }
    }
    
    /// Create mesh with features
    pub fn with_features(vertices: Tensor, faces: Vec<[usize; 3]>, features: Tensor) -> Self {
        Mesh {
            vertices,
            faces,
            features: Some(features),
        }
    }
    
    /// Get number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.dims()[0]
    }
    
    /// Get number of faces
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
    
    /// Compute adjacency list
    pub fn compute_adjacency(&self) -> Vec<Vec<usize>> {
        let num_verts = self.num_vertices();
        let mut adjacency = vec![Vec::new(); num_verts];
        
        for face in &self.faces {
            // Add edges for each triangle
            adjacency[face[0]].push(face[1]);
            adjacency[face[0]].push(face[2]);
            adjacency[face[1]].push(face[0]);
            adjacency[face[1]].push(face[2]);
            adjacency[face[2]].push(face[0]);
            adjacency[face[2]].push(face[1]);
        }
        
        // Remove duplicates and sort
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        
        adjacency
    }
    
    /// Compute face normals
    pub fn compute_face_normals(&self) -> Result<Tensor, String> {
        let verts_data = self.vertices.data_f32();
        let mut normals = Vec::with_capacity(self.num_faces() * 3);
        
        for face in &self.faces {
            let v0 = &verts_data[face[0] * 3..face[0] * 3 + 3];
            let v1 = &verts_data[face[1] * 3..face[1] * 3 + 3];
            let v2 = &verts_data[face[2] * 3..face[2] * 3 + 3];
            
            // Compute edges
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            
            // Cross product
            let normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            
            // Normalize
            let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if length > 1e-8 {
                normals.push(normal[0] / length);
                normals.push(normal[1] / length);
                normals.push(normal[2] / length);
            } else {
                normals.extend_from_slice(&[0.0, 0.0, 1.0]);
            }
        }
        
        Tensor::from_slice(&normals, &[self.num_faces(), 3])
            .map_err(|e| format!("Failed to create normals: {:?}", e))
    }
}

/// Mesh convolution layer
pub struct MeshConv {
    in_features: usize,
    out_features: usize,
    weight: Linear,
}

impl MeshConv {
    /// Create new mesh convolution
    pub fn new(in_features: usize, out_features: usize) -> Self {
        MeshConv {
            in_features,
            out_features,
            weight: Linear::new(in_features, out_features),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, features: &Tensor, adjacency: &[Vec<usize>]) -> Result<Tensor, String> {
        let feat_data = features.data_f32();
        let dims = features.dims();
        let num_vertices = dims[0];
        
        // Aggregate neighbor features
        let mut aggregated = Vec::with_capacity(num_vertices * self.in_features);
        
        for v in 0..num_vertices {
            let neighbors = &adjacency[v];
            
            if neighbors.is_empty() {
                // No neighbors, use self features
                let start = v * self.in_features;
                aggregated.extend_from_slice(&feat_data[start..start + self.in_features]);
            } else {
                // Average neighbor features
                let mut avg_features = vec![0.0; self.in_features];
                
                for &neighbor in neighbors {
                    let start = neighbor * self.in_features;
                    for i in 0..self.in_features {
                        avg_features[i] += feat_data[start + i];
                    }
                }
                
                let num_neighbors = neighbors.len() as f32;
                for feat in &mut avg_features {
                    *feat /= num_neighbors;
                }
                
                aggregated.extend_from_slice(&avg_features);
            }
        }
        
        let aggregated_tensor = Tensor::from_slice(&aggregated, &[num_vertices, self.in_features])
            .map_err(|e| format!("Failed to create aggregated tensor: {:?}", e))?;
        
        // Apply linear transformation
        Ok(self.weight.forward(&aggregated_tensor))
    }
}

/// Mesh pooling (vertex decimation)
pub struct MeshPool;

impl MeshPool {
    /// Pool mesh by selecting every nth vertex
    pub fn pool(mesh: &Mesh, stride: usize) -> Result<Mesh, String> {
        let verts_data = mesh.vertices.data_f32();
        let num_verts = mesh.num_vertices();
        
        // Select vertices
        let mut new_verts = Vec::new();
        let mut vertex_map = vec![None; num_verts];
        let mut new_idx = 0;
        
        for i in (0..num_verts).step_by(stride) {
            vertex_map[i] = Some(new_idx);
            new_verts.extend_from_slice(&verts_data[i * 3..i * 3 + 3]);
            new_idx += 1;
        }
        
        // Update faces
        let mut new_faces = Vec::new();
        for face in &mesh.faces {
            if let (Some(v0), Some(v1), Some(v2)) = (vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]) {
                new_faces.push([v0, v1, v2]);
            }
        }
        
        let new_vertices = Tensor::from_slice(&new_verts, &[new_idx, 3])
            .map_err(|e| format!("Failed to create pooled vertices: {:?}", e))?;
        
        Ok(Mesh::new(new_vertices, new_faces))
    }
}

/// Mesh encoder
pub struct MeshEncoder {
    conv1: MeshConv,
    conv2: MeshConv,
    conv3: MeshConv,
    fc: Linear,
}

impl MeshEncoder {
    /// Create new mesh encoder
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        MeshEncoder {
            conv1: MeshConv::new(input_dim, hidden_dim),
            conv2: MeshConv::new(hidden_dim, hidden_dim * 2),
            conv3: MeshConv::new(hidden_dim * 2, hidden_dim * 4),
            fc: Linear::new(hidden_dim * 4, output_dim),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, mesh: &Mesh) -> Result<Tensor, String> {
        let adjacency = mesh.compute_adjacency();
        
        // Use vertex positions as initial features if no features provided
        let features = if let Some(ref feat) = mesh.features {
            feat.clone()
        } else {
            mesh.vertices.clone()
        };
        
        // Apply mesh convolutions
        let mut x = self.conv1.forward(&features, &adjacency)?;
        x = x.relu();
        
        x = self.conv2.forward(&x, &adjacency)?;
        x = x.relu();
        
        x = self.conv3.forward(&x, &adjacency)?;
        x = x.relu();
        
        // Global pooling (max over vertices)
        let pooled = self.global_max_pool(&x)?;
        
        // Final linear layer
        Ok(self.fc.forward(&pooled))
    }
    
    fn global_max_pool(&self, x: &Tensor) -> Result<Tensor, String> {
        let data = x.data_f32();
        let dims = x.dims();
        let num_vertices = dims[0];
        let feature_dim = dims[1];
        
        let mut result = vec![f32::NEG_INFINITY; feature_dim];
        
        for v in 0..num_vertices {
            for f in 0..feature_dim {
                let val = data[v * feature_dim + f];
                result[f] = result[f].max(val);
            }
        }
        
        Tensor::from_slice(&result, &[1, feature_dim])
            .map_err(|e| format!("Failed to pool: {:?}", e))
    }
}

/// Mesh utilities
pub struct MeshUtils;

impl MeshUtils {
    /// Create a simple cube mesh
    pub fn create_cube() -> Mesh {
        let vertices = vec![
            -1.0f32, -1.0, -1.0,  // 0
             1.0, -1.0, -1.0,  // 1
             1.0,  1.0, -1.0,  // 2
            -1.0,  1.0, -1.0,  // 3
            -1.0, -1.0,  1.0,  // 4
             1.0, -1.0,  1.0,  // 5
             1.0,  1.0,  1.0,  // 6
            -1.0,  1.0,  1.0,  // 7
        ];
        
        let faces = vec![
            // Front
            [0, 1, 2], [0, 2, 3],
            // Back
            [4, 6, 5], [4, 7, 6],
            // Left
            [0, 3, 7], [0, 7, 4],
            // Right
            [1, 5, 6], [1, 6, 2],
            // Top
            [3, 2, 6], [3, 6, 7],
            // Bottom
            [0, 4, 5], [0, 5, 1],
        ];
        
        let verts_tensor = Tensor::from_slice(&vertices, &[8, 3]).unwrap();
        Mesh::new(verts_tensor, faces)
    }
    
    /// Create a simple tetrahedron mesh
    pub fn create_tetrahedron() -> Mesh {
        let vertices = vec![
            0.0f32, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0,
            0.5, 0.5, 1.0,
        ];
        
        let faces = vec![
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ];
        
        let verts_tensor = Tensor::from_slice(&vertices, &[4, 3]).unwrap();
        Mesh::new(verts_tensor, faces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mesh_creation() {
        let cube = MeshUtils::create_cube();
        assert_eq!(cube.num_vertices(), 8);
        assert_eq!(cube.num_faces(), 12);
    }
    
    #[test]
    fn test_mesh_adjacency() {
        let cube = MeshUtils::create_cube();
        let adjacency = cube.compute_adjacency();
        
        assert_eq!(adjacency.len(), 8);
        // Each vertex in a cube connects to multiple neighbors through triangulated faces
        for neighbors in &adjacency {
            assert!(neighbors.len() >= 3, "Each vertex should have at least 3 neighbors");
        }
    }
    
    #[test]
    fn test_face_normals() {
        let cube = MeshUtils::create_cube();
        let normals = cube.compute_face_normals().unwrap();
        assert_eq!(normals.dims(), &[12, 3]); // 12 faces, 3D normals
    }
    
    #[test]
    fn test_mesh_conv() {
        let conv = MeshConv::new(3, 16);
        let cube = MeshUtils::create_cube();
        let adjacency = cube.compute_adjacency();
        
        let output = conv.forward(&cube.vertices, &adjacency).unwrap();
        assert_eq!(output.dims(), &[8, 16]); // 8 vertices, 16 features
    }
    
    #[test]
    fn test_mesh_pool() {
        let cube = MeshUtils::create_cube();
        let pooled = MeshPool::pool(&cube, 2).unwrap();
        
        assert!(pooled.num_vertices() <= cube.num_vertices());
        assert!(pooled.num_faces() <= cube.num_faces());
    }
    
    #[test]
    fn test_mesh_encoder() {
        let encoder = MeshEncoder::new(3, 16, 128);
        let cube = MeshUtils::create_cube();
        
        let features = encoder.forward(&cube).unwrap();
        assert_eq!(features.dims(), &[1, 128]); // Global features
    }
    
    #[test]
    fn test_tetrahedron() {
        let tetra = MeshUtils::create_tetrahedron();
        assert_eq!(tetra.num_vertices(), 4);
        assert_eq!(tetra.num_faces(), 4);
    }
}

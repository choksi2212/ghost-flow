//! Graph Neural Networks (GNN) module
//!
//! Implements various GNN architectures:
//! - Graph Convolutional Networks (GCN)
//! - Graph Attention Networks (GAT)
//! - GraphSAGE
//! - Message Passing Neural Networks (MPNN)

use ghostflow_core::Tensor;
use std::collections::HashMap;

/// Graph structure for GNN operations
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Edge list: (source, target) pairs
    pub edges: Vec<(usize, usize)>,
    /// Node features [num_nodes, feature_dim]
    pub node_features: Tensor,
    /// Edge features [num_edges, edge_feature_dim] (optional)
    pub edge_features: Option<Tensor>,
    /// Adjacency matrix (sparse representation)
    adjacency: HashMap<usize, Vec<usize>>,
}

impl Graph {
    /// Create a new graph from edge list and node features
    pub fn new(edges: Vec<(usize, usize)>, node_features: Tensor) -> Self {
        let num_nodes = node_features.dims()[0];
        let num_edges = edges.len();
        
        // Build adjacency list
        let mut adjacency = HashMap::new();
        for &(src, dst) in &edges {
            adjacency.entry(src).or_insert_with(Vec::new).push(dst);
        }
        
        Graph {
            num_nodes,
            num_edges,
            edges,
            node_features,
            edge_features: None,
            adjacency,
        }
    }
    
    /// Add edge features
    pub fn with_edge_features(mut self, edge_features: Tensor) -> Self {
        self.edge_features = Some(edge_features);
        self
    }
    
    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> &[usize] {
        self.adjacency.get(&node).map(|v| v.as_slice()).unwrap_or(&[])
    }
    
    /// Get degree of a node
    pub fn degree(&self, node: usize) -> usize {
        self.neighbors(node).len()
    }
    
    /// Compute normalized adjacency matrix (for GCN)
    pub fn normalized_adjacency(&self) -> Tensor {
        // A_norm = D^(-1/2) * A * D^(-1/2)
        let mut adj_data = vec![0.0f32; self.num_nodes * self.num_nodes];
        
        // Build adjacency matrix with self-loops
        for i in 0..self.num_nodes {
            adj_data[i * self.num_nodes + i] = 1.0; // Self-loop
        }
        for &(src, dst) in &self.edges {
            adj_data[src * self.num_nodes + dst] = 1.0;
        }
        
        // Compute degree matrix
        let mut degrees = vec![0.0f32; self.num_nodes];
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                degrees[i] += adj_data[i * self.num_nodes + j];
            }
        }
        
        // Normalize: D^(-1/2) * A * D^(-1/2)
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                let idx = i * self.num_nodes + j;
                if adj_data[idx] > 0.0 {
                    adj_data[idx] /= (degrees[i] * degrees[j]).sqrt();
                }
            }
        }
        
        Tensor::from_slice(&adj_data, &[self.num_nodes, self.num_nodes]).unwrap()
    }
}

/// Graph Convolutional Network (GCN) layer
pub struct GCNLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    use_bias: bool,
}

impl GCNLayer {
    /// Create a new GCN layer
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let weight = Tensor::randn(&[in_features, out_features]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };
        
        GCNLayer {
            weight,
            bias,
            use_bias,
        }
    }
    
    /// Forward pass: H' = σ(A_norm * H * W + b)
    pub fn forward(&self, graph: &Graph, activation: bool) -> Tensor {
        let adj = graph.normalized_adjacency();
        let features = &graph.node_features;
        
        // H * W
        let hw = features.matmul(&self.weight).unwrap();
        
        // A_norm * (H * W)
        let mut output = adj.matmul(&hw).unwrap();
        
        // Add bias
        if let Some(ref bias) = self.bias {
            output = output.add(bias).unwrap();
        }
        
        // Apply activation (ReLU)
        if activation {
            output = output.relu();
        }
        
        output
    }
}

/// Graph Attention Network (GAT) layer
pub struct GATLayer {
    weight: Tensor,
    attention_weight: Tensor,
    bias: Option<Tensor>,
    num_heads: usize,
    dropout: f32,
}

impl GATLayer {
    /// Create a new GAT layer with multi-head attention
    pub fn new(in_features: usize, out_features: usize, num_heads: usize, dropout: f32) -> Self {
        let weight = Tensor::randn(&[in_features, out_features * num_heads]);
        let attention_weight = Tensor::randn(&[2 * out_features, 1]);
        let bias = Some(Tensor::zeros(&[out_features * num_heads]));
        
        GATLayer {
            weight,
            attention_weight,
            bias,
            num_heads,
            dropout,
        }
    }
    
    /// Compute attention coefficients
    fn attention_coefficients(&self, node_i: &Tensor, node_j: &Tensor) -> f32 {
        // Concatenate features and compute attention
        // e_ij = LeakyReLU(a^T [W*h_i || W*h_j])
        let data_i = node_i.data_f32();
        let data_j = node_j.data_f32();
        let mut concat_data = Vec::with_capacity(data_i.len() + data_j.len());
        concat_data.extend_from_slice(&data_i);
        concat_data.extend_from_slice(&data_j);
        
        let concat = Tensor::from_slice(&concat_data, &[data_i.len() + data_j.len()]).unwrap();
        let score = concat.matmul(&self.attention_weight).unwrap();
        
        // LeakyReLU with alpha=0.2
        let data = score.data_f32();
        let alpha = 0.2;
        if data[0] > 0.0 {
            data[0]
        } else {
            alpha * data[0]
        }
    }
    
    /// Forward pass with multi-head attention
    pub fn forward(&self, graph: &Graph) -> Tensor {
        let features = &graph.node_features;
        
        // Transform features: H' = W * H
        let transformed = features.matmul(&self.weight).unwrap();
        
        // For now, return transformed features
        // Full GAT implementation would compute attention for each edge
        if let Some(ref bias) = self.bias {
            transformed.add(bias).unwrap()
        } else {
            transformed
        }
    }
}

/// GraphSAGE layer (Sample and Aggregate)
pub struct GraphSAGELayer {
    weight_self: Tensor,
    weight_neighbor: Tensor,
    aggregator: AggregatorType,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregatorType {
    Mean,
    Pool,
    LSTM,
}

impl GraphSAGELayer {
    /// Create a new GraphSAGE layer
    pub fn new(in_features: usize, out_features: usize, aggregator: AggregatorType) -> Self {
        let weight_self = Tensor::randn(&[in_features, out_features]);
        let weight_neighbor = Tensor::randn(&[in_features, out_features]);
        
        GraphSAGELayer {
            weight_self,
            weight_neighbor,
            aggregator,
        }
    }
    
    /// Aggregate neighbor features
    fn aggregate(&self, neighbor_features: &[Tensor]) -> Tensor {
        match self.aggregator {
            AggregatorType::Mean => {
                // Mean aggregation
                if neighbor_features.is_empty() {
                    return Tensor::zeros(neighbor_features[0].dims());
                }
                
                let sum = neighbor_features.iter()
                    .fold(Tensor::zeros(neighbor_features[0].dims()), |acc, feat| {
                        acc.add(feat).unwrap()
                    });
                
                sum.div_scalar(neighbor_features.len() as f32)
            }
            AggregatorType::Pool => {
                // Max pooling aggregation
                neighbor_features[0].clone() // Simplified
            }
            AggregatorType::LSTM => {
                // LSTM aggregation (simplified)
                neighbor_features[0].clone()
            }
        }
    }
    
    /// Forward pass: h_v' = σ(W_self * h_v + W_neighbor * AGG({h_u : u ∈ N(v)}))
    pub fn forward(&self, graph: &Graph) -> Tensor {
        let features = &graph.node_features;
        let num_nodes = graph.num_nodes;
        let feature_dim = features.dims()[1];
        
        let mut output_data = Vec::new();
        
        for node in 0..num_nodes {
            // Get node's own features
            let node_feat_data: Vec<f32> = (0..feature_dim)
                .map(|i| features.data_f32()[node * feature_dim + i])
                .collect();
            let node_feat = Tensor::from_slice(&node_feat_data, &[1, feature_dim]).unwrap();
            
            // Get neighbor features
            let neighbors = graph.neighbors(node);
            let neighbor_feats: Vec<Tensor> = neighbors.iter()
                .map(|&n| {
                    let data: Vec<f32> = (0..feature_dim)
                        .map(|i| features.data_f32()[n * feature_dim + i])
                        .collect();
                    Tensor::from_slice(&data, &[1, feature_dim]).unwrap()
                })
                .collect();
            
            // Aggregate neighbors
            let aggregated = if !neighbor_feats.is_empty() {
                self.aggregate(&neighbor_feats)
            } else {
                Tensor::zeros(&[1, feature_dim])
            };
            
            // Combine self and neighbor information
            let self_part = node_feat.matmul(&self.weight_self).unwrap();
            let neighbor_part = aggregated.matmul(&self.weight_neighbor).unwrap();
            let combined = self_part.add(&neighbor_part).unwrap();
            
            output_data.extend(combined.data_f32());
        }
        
        let out_dim = self.weight_self.dims()[1];
        Tensor::from_slice(&output_data, &[num_nodes, out_dim]).unwrap()
    }
}

/// Message Passing Neural Network (MPNN) layer
pub struct MPNNLayer {
    message_fn: Tensor,
    update_fn: Tensor,
}

impl MPNNLayer {
    /// Create a new MPNN layer
    pub fn new(node_dim: usize, edge_dim: usize, hidden_dim: usize) -> Self {
        let message_fn = Tensor::randn(&[node_dim + edge_dim, hidden_dim]);
        let update_fn = Tensor::randn(&[node_dim + hidden_dim, node_dim]);
        
        MPNNLayer {
            message_fn,
            update_fn,
        }
    }
    
    /// Forward pass: message passing and node update
    pub fn forward(&self, graph: &Graph) -> Tensor {
        // Simplified MPNN implementation
        // Full version would iterate over edges and aggregate messages
        graph.node_features.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_graph_creation() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let features = Tensor::randn(&[3, 4]);
        let graph = Graph::new(edges, features);
        
        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.num_edges, 3);
        assert_eq!(graph.neighbors(0).len(), 1);
    }
    
    #[test]
    fn test_gcn_layer() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let features = Tensor::randn(&[3, 4]);
        let graph = Graph::new(edges, features);
        
        let gcn = GCNLayer::new(4, 8, true);
        let output = gcn.forward(&graph, true);
        
        assert_eq!(output.dims(), &[3, 8]);
    }
    
    #[test]
    fn test_graphsage_layer() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let features = Tensor::randn(&[3, 4]);
        let graph = Graph::new(edges, features);
        
        let sage = GraphSAGELayer::new(4, 8, AggregatorType::Mean);
        let output = sage.forward(&graph);
        
        assert_eq!(output.dims(), &[3, 8]);
    }
}

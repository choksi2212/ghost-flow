//! Graph Neural Network Architectures - GCN, GAT, GraphSAGE, GIN, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::Dense;
use crate::deep::activations::ReLU;

/// Graph Convolutional Network (GCN) Layer
pub struct GCNLayer {
    weight: Dense,
    in_features: usize,
    out_features: usize,
}

impl GCNLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        GCNLayer {
            weight: Dense::new(in_features, out_features),
            in_features,
            out_features,
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        // H' = σ(AHW)
        let hw = self.weight.forward(x, training);
        self.sparse_matmul(adj, &hw)
    }

    fn sparse_matmul(&self, adj: &Tensor, x: &Tensor) -> Tensor {
        let adj_data = adj.data_f32();
        let x_data = x.data_f32();
        
        let num_nodes = adj.dims()[0];
        let out_features = x.dims()[1];
        
        let mut result = vec![0.0f32; num_nodes * out_features];
        
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let adj_val = adj_data[i * num_nodes + j];
                if adj_val != 0.0 {
                    for k in 0..out_features {
                        result[i * out_features + k] += adj_val * x_data[j * out_features + k];
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, &[num_nodes, out_features]).unwrap()
    }
}

/// GCN Model
pub struct GCN {
    layer1: GCNLayer,
    layer2: GCNLayer,
    layer3: GCNLayer,
}

impl GCN {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Self {
        GCN {
            layer1: GCNLayer::new(in_features, hidden_features),
            layer2: GCNLayer::new(hidden_features, hidden_features),
            layer3: GCNLayer::new(hidden_features, out_features),
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let mut out = self.layer1.forward(x, adj, training);
        out = ReLU::new().forward(&out);
        
        out = self.layer2.forward(&out, adj, training);
        out = ReLU::new().forward(&out);
        
        self.layer3.forward(&out, adj, training)
    }
}

/// Graph Attention Network (GAT) Layer
pub struct GATLayer {
    weight: Dense,
    attention: Dense,
    in_features: usize,
    out_features: usize,
    num_heads: usize,
}

impl GATLayer {
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        GATLayer {
            weight: Dense::new(in_features, out_features * num_heads),
            attention: Dense::new(out_features * 2, 1),
            in_features,
            out_features,
            num_heads,
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let h = self.weight.forward(x, training);
        let attention_scores = self.compute_attention(&h, adj, training);
        self.aggregate_with_attention(&h, &attention_scores)
    }

    fn compute_attention(&mut self, h: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let num_nodes = h.dims()[0];
        let features = h.dims()[1];
        let h_data = h.data_f32();
        
        let mut attention_scores = vec![0.0f32; num_nodes * num_nodes];
        
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                // Concatenate features
                let mut concat = Vec::new();
                for k in 0..features {
                    concat.push(h_data[i * features + k]);
                }
                for k in 0..features {
                    concat.push(h_data[j * features + k]);
                }
                
                let concat_tensor = Tensor::from_slice(&concat, &[1, features * 2]).unwrap();
                let score = self.attention.forward(&concat_tensor, training);
                attention_scores[i * num_nodes + j] = score.data_f32()[0];
            }
        }
        
        // Apply softmax
        self.softmax_rows(&Tensor::from_slice(&attention_scores, &[num_nodes, num_nodes]).unwrap())
    }

    fn softmax_rows(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let rows = x.dims()[0];
        let cols = x.dims()[1];
        
        let mut result = vec![0.0f32; data.len()];
        
        for i in 0..rows {
            let offset = i * cols;
            
            // Find max
            let mut max_val = data[offset];
            for j in 1..cols {
                max_val = max_val.max(data[offset + j]);
            }
            
            // Compute exp and sum
            let mut sum = 0.0f32;
            for j in 0..cols {
                let exp_val = (data[offset + j] - max_val).exp();
                result[offset + j] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for j in 0..cols {
                result[offset + j] /= sum;
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn aggregate_with_attention(&self, h: &Tensor, attention: &Tensor) -> Tensor {
        let num_nodes = h.dims()[0];
        let features = h.dims()[1];
        let h_data = h.data_f32();
        let att_data = attention.data_f32();
        
        let mut result = vec![0.0f32; num_nodes * features];
        
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let att_val = att_data[i * num_nodes + j];
                for k in 0..features {
                    result[i * features + k] += att_val * h_data[j * features + k];
                }
            }
        }
        
        Tensor::from_slice(&result, &[num_nodes, features]).unwrap()
    }
}

/// GAT Model
pub struct GAT {
    layer1: GATLayer,
    layer2: GATLayer,
}

impl GAT {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize, num_heads: usize) -> Self {
        GAT {
            layer1: GATLayer::new(in_features, hidden_features, num_heads),
            layer2: GATLayer::new(hidden_features * num_heads, out_features, 1),
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let mut out = self.layer1.forward(x, adj, training);
        out = ReLU::new().forward(&out);
        self.layer2.forward(&out, adj, training)
    }
}

/// GraphSAGE Layer
pub struct GraphSAGELayer {
    weight_self: Dense,
    weight_neigh: Dense,
    in_features: usize,
    out_features: usize,
}

impl GraphSAGELayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        GraphSAGELayer {
            weight_self: Dense::new(in_features, out_features),
            weight_neigh: Dense::new(in_features, out_features),
            in_features,
            out_features,
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let h_self = self.weight_self.forward(x, training);
        let h_neigh = self.aggregate_neighbors(x, adj);
        let h_neigh_transformed = self.weight_neigh.forward(&h_neigh, training);
        
        self.add_tensors(&h_self, &h_neigh_transformed)
    }

    fn aggregate_neighbors(&self, x: &Tensor, adj: &Tensor) -> Tensor {
        let num_nodes = x.dims()[0];
        let features = x.dims()[1];
        let x_data = x.data_f32();
        let adj_data = adj.data_f32();
        
        let mut result = vec![0.0f32; num_nodes * features];
        
        for i in 0..num_nodes {
            let mut count = 0.0f32;
            for j in 0..num_nodes {
                if adj_data[i * num_nodes + j] != 0.0 {
                    count += 1.0;
                    for k in 0..features {
                        result[i * features + k] += x_data[j * features + k];
                    }
                }
            }
            
            // Mean aggregation
            if count > 0.0 {
                for k in 0..features {
                    result[i * features + k] /= count;
                }
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

/// GraphSAGE Model
pub struct GraphSAGE {
    layer1: GraphSAGELayer,
    layer2: GraphSAGELayer,
}

impl GraphSAGE {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Self {
        GraphSAGE {
            layer1: GraphSAGELayer::new(in_features, hidden_features),
            layer2: GraphSAGELayer::new(hidden_features, out_features),
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let mut out = self.layer1.forward(x, adj, training);
        out = ReLU::new().forward(&out);
        self.layer2.forward(&out, adj, training)
    }
}

/// Graph Isomorphism Network (GIN) Layer
pub struct GINLayer {
    mlp: Vec<Dense>,
    epsilon: f32,
}

impl GINLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        GINLayer {
            mlp: vec![
                Dense::new(in_features, out_features),
                Dense::new(out_features, out_features),
            ],
            epsilon: 0.0,
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let aggregated = self.aggregate_neighbors(x, adj);
        let combined = self.combine_with_self(x, &aggregated);
        
        let mut out = combined;
        for (i, layer) in self.mlp.iter_mut().enumerate() {
            out = layer.forward(&out, training);
            if i < self.mlp.len() - 1 {
                out = ReLU::new().forward(&out);
            }
        }
        
        out
    }

    fn aggregate_neighbors(&self, x: &Tensor, adj: &Tensor) -> Tensor {
        let num_nodes = x.dims()[0];
        let features = x.dims()[1];
        let x_data = x.data_f32();
        let adj_data = adj.data_f32();
        
        let mut result = vec![0.0f32; num_nodes * features];
        
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if adj_data[i * num_nodes + j] != 0.0 {
                    for k in 0..features {
                        result[i * features + k] += x_data[j * features + k];
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn combine_with_self(&self, x: &Tensor, aggregated: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let agg_data = aggregated.data_f32();
        
        let result: Vec<f32> = x_data.iter()
            .zip(agg_data.iter())
            .map(|(&x_val, &agg_val)| (1.0 + self.epsilon) * x_val + agg_val)
            .collect();
        
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// GIN Model
pub struct GIN {
    layer1: GINLayer,
    layer2: GINLayer,
    layer3: GINLayer,
}

impl GIN {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Self {
        GIN {
            layer1: GINLayer::new(in_features, hidden_features),
            layer2: GINLayer::new(hidden_features, hidden_features),
            layer3: GINLayer::new(hidden_features, out_features),
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let mut out = self.layer1.forward(x, adj, training);
        out = ReLU::new().forward(&out);
        
        out = self.layer2.forward(&out, adj, training);
        out = ReLU::new().forward(&out);
        
        self.layer3.forward(&out, adj, training)
    }
}

/// Message Passing Neural Network (MPNN) Layer
pub struct MPNNLayer {
    message_fn: Dense,
    update_fn: Dense,
}

impl MPNNLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        MPNNLayer {
            message_fn: Dense::new(in_features * 2, out_features),
            update_fn: Dense::new(in_features + out_features, out_features),
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let messages = self.compute_messages(x, adj, training);
        self.update_nodes(x, &messages, training)
    }

    fn compute_messages(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let num_nodes = x.dims()[0];
        let features = x.dims()[1];
        let x_data = x.data_f32();
        let adj_data = adj.data_f32();
        
        let mut messages = vec![0.0f32; num_nodes * features];
        
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if adj_data[i * num_nodes + j] != 0.0 {
                    // Concatenate node features
                    let mut concat = Vec::new();
                    for k in 0..features {
                        concat.push(x_data[i * features + k]);
                        concat.push(x_data[j * features + k]);
                    }
                    
                    let concat_tensor = Tensor::from_slice(&concat, &[1, features * 2]).unwrap();
                    let message = self.message_fn.forward(&concat_tensor, training);
                    let msg_data = message.data_f32();
                    
                    for k in 0..features {
                        messages[i * features + k] += msg_data[k];
                    }
                }
            }
        }
        
        Tensor::from_slice(&messages, &[num_nodes, features]).unwrap()
    }

    fn update_nodes(&mut self, x: &Tensor, messages: &Tensor, training: bool) -> Tensor {
        let num_nodes = x.dims()[0];
        let features = x.dims()[1];
        let x_data = x.data_f32();
        let msg_data = messages.data_f32();
        
        let mut result = Vec::new();
        
        for i in 0..num_nodes {
            let mut concat = Vec::new();
            for k in 0..features {
                concat.push(x_data[i * features + k]);
            }
            for k in 0..features {
                concat.push(msg_data[i * features + k]);
            }
            
            let concat_tensor = Tensor::from_slice(&concat, &[1, features * 2]).unwrap();
            let updated = self.update_fn.forward(&concat_tensor, training);
            result.extend_from_slice(updated.data_f32());
        }
        
        Tensor::from_slice(&result, &[num_nodes, features]).unwrap()
    }
}

/// MPNN Model
pub struct MPNN {
    layer1: MPNNLayer,
    layer2: MPNNLayer,
}

impl MPNN {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Self {
        MPNN {
            layer1: MPNNLayer::new(in_features, hidden_features),
            layer2: MPNNLayer::new(hidden_features, out_features),
        }
    }

    pub fn forward(&mut self, x: &Tensor, adj: &Tensor, training: bool) -> Tensor {
        let mut out = self.layer1.forward(x, adj, training);
        out = ReLU::new().forward(&out);
        self.layer2.forward(&out, adj, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn() {
        let mut gcn = GCN::new(10, 16, 7);
        let x = Tensor::from_slice(&vec![0.5f32; 5 * 10], &[5, 10]).unwrap();
        let adj = Tensor::from_slice(&vec![1.0f32; 5 * 5], &[5, 5]).unwrap();
        let output = gcn.forward(&x, &adj, false);
        assert_eq!(output.dims(), &[5, 7]);
    }

    #[test]
    fn test_gat() {
        let mut gat = GAT::new(10, 8, 7, 4);
        let x = Tensor::from_slice(&vec![0.5f32; 5 * 10], &[5, 10]).unwrap();
        let adj = Tensor::from_slice(&vec![1.0f32; 5 * 5], &[5, 5]).unwrap();
        let output = gat.forward(&x, &adj, false);
        assert_eq!(output.dims()[0], 5);
    }

    #[test]
    fn test_graphsage() {
        let mut sage = GraphSAGE::new(10, 16, 7);
        let x = Tensor::from_slice(&vec![0.5f32; 5 * 10], &[5, 10]).unwrap();
        let adj = Tensor::from_slice(&vec![1.0f32; 5 * 5], &[5, 5]).unwrap();
        let output = sage.forward(&x, &adj, false);
        assert_eq!(output.dims(), &[5, 7]);
    }

    #[test]
    fn test_gin() {
        let mut gin = GIN::new(10, 16, 7);
        let x = Tensor::from_slice(&vec![0.5f32; 5 * 10], &[5, 10]).unwrap();
        let adj = Tensor::from_slice(&vec![1.0f32; 5 * 5], &[5, 5]).unwrap();
        let output = gin.forward(&x, &adj, false);
        assert_eq!(output.dims(), &[5, 7]);
    }
}



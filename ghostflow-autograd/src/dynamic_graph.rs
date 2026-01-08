//! Dynamic Computation Graph
//!
//! Implements dynamic computation graphs (like PyTorch) where the graph
//! is built on-the-fly during forward pass.

use ghostflow_core::Tensor;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Node in the dynamic computation graph
#[derive(Clone)]
pub struct GraphNode {
    /// Unique node ID
    pub id: usize,
    /// Operation name
    pub op: String,
    /// Input node IDs
    pub inputs: Vec<usize>,
    /// Output tensor
    pub output: Tensor,
    /// Gradient function
    pub backward_fn: Option<Arc<dyn Fn(&[Tensor]) -> Vec<Tensor> + Send + Sync>>,
}

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphNode")
            .field("id", &self.id)
            .field("op", &self.op)
            .field("inputs", &self.inputs)
            .field("output", &self.output)
            .field("backward_fn", &self.backward_fn.is_some())
            .finish()
    }
}

/// Dynamic computation graph
#[derive(Debug)]
pub struct DynamicGraph {
    /// All nodes in the graph
    nodes: Arc<Mutex<HashMap<usize, GraphNode>>>,
    /// Next node ID
    next_id: Arc<Mutex<usize>>,
    /// Whether to record operations
    recording: Arc<Mutex<bool>>,
}

impl DynamicGraph {
    /// Create a new dynamic graph
    pub fn new() -> Self {
        DynamicGraph {
            nodes: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
            recording: Arc::new(Mutex::new(true)),
        }
    }
    
    /// Start recording operations
    pub fn start_recording(&self) {
        *self.recording.lock().unwrap() = true;
    }
    
    /// Stop recording operations
    pub fn stop_recording(&self) {
        *self.recording.lock().unwrap() = false;
    }
    
    /// Check if recording
    pub fn is_recording(&self) -> bool {
        *self.recording.lock().unwrap()
    }
    
    /// Add a node to the graph
    pub fn add_node(&self, op: String, inputs: Vec<usize>, output: Tensor) -> usize {
        if !self.is_recording() {
            return 0;
        }
        
        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;
        
        let node = GraphNode {
            id,
            op,
            inputs,
            output,
            backward_fn: None,
        };
        
        self.nodes.lock().unwrap().insert(id, node);
        id
    }
    
    /// Get a node by ID
    pub fn get_node(&self, id: usize) -> Option<GraphNode> {
        self.nodes.lock().unwrap().get(&id).cloned()
    }
    
    /// Clear the graph
    pub fn clear(&self) {
        self.nodes.lock().unwrap().clear();
        *self.next_id.lock().unwrap() = 0;
    }
    
    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.lock().unwrap().len()
    }
    
    /// Perform backward pass from a node
    pub fn backward(&self, node_id: usize, grad: Tensor) -> HashMap<usize, Tensor> {
        let mut gradients: HashMap<usize, Tensor> = HashMap::new();
        gradients.insert(node_id, grad);
        
        // Topological sort (simplified - assumes DAG)
        let nodes = self.nodes.lock().unwrap();
        let mut sorted_ids: Vec<usize> = nodes.keys().cloned().collect();
        sorted_ids.sort_by(|a, b| b.cmp(a)); // Reverse order
        
        for &id in &sorted_ids {
            if let Some(grad) = gradients.get(&id) {
                if let Some(node) = nodes.get(&id) {
                    // Compute gradients for inputs
                    if let Some(ref backward_fn) = node.backward_fn {
                        let input_grads = backward_fn(&[grad.clone()]);
                        
                        for (i, input_id) in node.inputs.iter().enumerate() {
                            if i < input_grads.len() {
                                let input_grad = &input_grads[i];
                                gradients.entry(*input_id)
                                    .and_modify(|g| *g = g.add(input_grad).unwrap())
                                    .or_insert_with(|| input_grad.clone());
                            }
                        }
                    }
                }
            }
        }
        
        gradients
    }
}

impl Default for DynamicGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for dynamic graph operations
pub struct DynamicContext {
    graph: Arc<DynamicGraph>,
}

impl DynamicContext {
    /// Create a new context
    pub fn new() -> Self {
        DynamicContext {
            graph: Arc::new(DynamicGraph::new()),
        }
    }
    
    /// Get the graph
    pub fn graph(&self) -> &Arc<DynamicGraph> {
        &self.graph
    }
    
    /// Execute a function with gradient tracking
    pub fn with_grad<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.graph.start_recording();
        let result = f();
        result
    }
    
    /// Execute a function without gradient tracking
    pub fn no_grad<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.graph.stop_recording();
        let result = f();
        self.graph.start_recording();
        result
    }
}

impl Default for DynamicContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for tensors in dynamic graph
#[derive(Debug, Clone)]
pub struct DynamicTensor {
    /// The actual tensor
    pub tensor: Tensor,
    /// Node ID in the graph
    pub node_id: Option<usize>,
    /// Reference to the graph
    pub graph: Option<Arc<DynamicGraph>>,
}

impl DynamicTensor {
    /// Create a new dynamic tensor
    pub fn new(tensor: Tensor, graph: Arc<DynamicGraph>) -> Self {
        let node_id = graph.add_node("input".to_string(), vec![], tensor.clone());
        
        DynamicTensor {
            tensor,
            node_id: Some(node_id),
            graph: Some(graph),
        }
    }
    
    /// Create from tensor without graph
    pub fn from_tensor(tensor: Tensor) -> Self {
        DynamicTensor {
            tensor,
            node_id: None,
            graph: None,
        }
    }
    
    /// Add two dynamic tensors
    pub fn add(&self, other: &DynamicTensor) -> DynamicTensor {
        let result = self.tensor.add(&other.tensor).unwrap();
        
        if let (Some(graph), Some(id1), Some(id2)) = (&self.graph, self.node_id, other.node_id) {
            let node_id = graph.add_node("add".to_string(), vec![id1, id2], result.clone());
            
            DynamicTensor {
                tensor: result,
                node_id: Some(node_id),
                graph: Some(graph.clone()),
            }
        } else {
            DynamicTensor::from_tensor(result)
        }
    }
    
    /// Multiply two dynamic tensors
    pub fn mul(&self, other: &DynamicTensor) -> DynamicTensor {
        let result = self.tensor.mul(&other.tensor).unwrap();
        
        if let (Some(graph), Some(id1), Some(id2)) = (&self.graph, self.node_id, other.node_id) {
            let node_id = graph.add_node("mul".to_string(), vec![id1, id2], result.clone());
            
            DynamicTensor {
                tensor: result,
                node_id: Some(node_id),
                graph: Some(graph.clone()),
            }
        } else {
            DynamicTensor::from_tensor(result)
        }
    }
    
    /// Matrix multiplication
    pub fn matmul(&self, other: &DynamicTensor) -> DynamicTensor {
        let result = self.tensor.matmul(&other.tensor).unwrap();
        
        if let (Some(graph), Some(id1), Some(id2)) = (&self.graph, self.node_id, other.node_id) {
            let node_id = graph.add_node("matmul".to_string(), vec![id1, id2], result.clone());
            
            DynamicTensor {
                tensor: result,
                node_id: Some(node_id),
                graph: Some(graph.clone()),
            }
        } else {
            DynamicTensor::from_tensor(result)
        }
    }
    
    /// ReLU activation
    pub fn relu(&self) -> DynamicTensor {
        let result = self.tensor.relu();
        
        if let (Some(graph), Some(id)) = (&self.graph, self.node_id) {
            let node_id = graph.add_node("relu".to_string(), vec![id], result.clone());
            
            DynamicTensor {
                tensor: result,
                node_id: Some(node_id),
                graph: Some(graph.clone()),
            }
        } else {
            DynamicTensor::from_tensor(result)
        }
    }
    
    /// Compute gradients
    pub fn backward(&self) -> HashMap<usize, Tensor> {
        if let (Some(graph), Some(node_id)) = (&self.graph, self.node_id) {
            let grad = Tensor::ones(self.tensor.dims());
            graph.backward(node_id, grad)
        } else {
            HashMap::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dynamic_graph() {
        let graph = DynamicGraph::new();
        assert_eq!(graph.num_nodes(), 0);
        
        let t1 = Tensor::ones(&[2, 2]);
        let id = graph.add_node("test".to_string(), vec![], t1);
        
        assert_eq!(graph.num_nodes(), 1);
        assert!(graph.get_node(id).is_some());
    }
    
    #[test]
    fn test_dynamic_context() {
        let ctx = DynamicContext::new();
        
        ctx.with_grad(|| {
            assert!(ctx.graph().is_recording());
        });
        
        ctx.no_grad(|| {
            assert!(!ctx.graph().is_recording());
        });
    }
    
    #[test]
    fn test_dynamic_tensor() {
        let graph = Arc::new(DynamicGraph::new());
        let t1 = Tensor::ones(&[2, 2]);
        let t2 = Tensor::ones(&[2, 2]);
        
        let dt1 = DynamicTensor::new(t1, graph.clone());
        let dt2 = DynamicTensor::new(t2, graph.clone());
        
        let result = dt1.add(&dt2);
        assert_eq!(result.tensor.data_f32()[0], 2.0);
        assert_eq!(graph.num_nodes(), 3); // 2 inputs + 1 add
    }
}

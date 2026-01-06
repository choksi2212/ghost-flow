//! Kernel fusion engine for optimizing computation graphs
//!
//! This module provides automatic fusion of operations to reduce memory bandwidth
//! and improve performance.

use std::collections::HashMap;

/// Fusion pattern for combining operations
#[derive(Debug, Clone, PartialEq)]
pub enum FusionPattern {
    /// Element-wise operations (add, mul, relu, etc.)
    ElementWise(Vec<String>),
    /// Reduction operations (sum, mean, max, etc.)
    Reduction(String),
    /// Matrix operations (matmul, conv, etc.)
    MatrixOp(String),
    /// Custom fusion pattern
    Custom(String, Vec<String>),
}

/// Computational graph node
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub op_type: String,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub fusible: bool,
}

/// Computational graph for fusion analysis
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    nodes: Vec<GraphNode>,
    next_id: usize,
}

impl ComputeGraph {
    /// Create a new empty compute graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, op_type: String, inputs: Vec<usize>, fusible: bool) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        // Update outputs of input nodes first
        for &input_id in &inputs {
            if let Some(node) = self.nodes.iter_mut().find(|n| n.id == input_id) {
                node.outputs.push(id);
            }
        }

        // Then add the new node
        self.nodes.push(GraphNode {
            id,
            op_type,
            inputs,
            outputs: Vec::new(),
            fusible,
        });

        id
    }

    /// Get all nodes
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Get a node by ID
    pub fn get_node(&self, id: usize) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion engine for optimizing computation graphs
pub struct FusionEngine {
    patterns: Vec<FusionPattern>,
    fused_ops: HashMap<String, Vec<String>>,
}

impl FusionEngine {
    /// Create a new fusion engine
    pub fn new() -> Self {
        let mut engine = Self {
            patterns: Vec::new(),
            fused_ops: HashMap::new(),
        };

        // Register default fusion patterns
        engine.register_default_patterns();
        engine
    }

    /// Register default fusion patterns
    fn register_default_patterns(&mut self) {
        // Conv + BatchNorm + ReLU
        self.add_pattern(FusionPattern::Custom(
            "ConvBNReLU".to_string(),
            vec!["Conv2d".to_string(), "BatchNorm".to_string(), "ReLU".to_string()],
        ));

        // Linear + ReLU
        self.add_pattern(FusionPattern::Custom(
            "LinearReLU".to_string(),
            vec!["Linear".to_string(), "ReLU".to_string()],
        ));

        // MatMul + Add (GEMM)
        self.add_pattern(FusionPattern::Custom(
            "GEMM".to_string(),
            vec!["MatMul".to_string(), "Add".to_string()],
        ));

        // Add + ReLU
        self.add_pattern(FusionPattern::Custom(
            "AddReLU".to_string(),
            vec!["Add".to_string(), "ReLU".to_string()],
        ));

        // Mul + Add (FMA - Fused Multiply-Add)
        self.add_pattern(FusionPattern::Custom(
            "FMA".to_string(),
            vec!["Mul".to_string(), "Add".to_string()],
        ));

        // BatchNorm + ReLU
        self.add_pattern(FusionPattern::Custom(
            "BNReLU".to_string(),
            vec!["BatchNorm".to_string(), "ReLU".to_string()],
        ));
    }

    /// Add a fusion pattern
    pub fn add_pattern(&mut self, pattern: FusionPattern) {
        self.patterns.push(pattern);
    }

    /// Analyze a compute graph and find fusion opportunities
    pub fn analyze(&mut self, graph: &ComputeGraph) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();

        // Check each pattern against the graph
        for pattern in &self.patterns {
            if let FusionPattern::Custom(name, ops) = pattern {
                opportunities.extend(self.find_pattern_matches(graph, name, ops));
            }
        }

        opportunities
    }

    /// Find matches for a specific pattern in the graph
    fn find_pattern_matches(
        &self,
        graph: &ComputeGraph,
        pattern_name: &str,
        ops: &[String],
    ) -> Vec<FusionOpportunity> {
        let mut matches = Vec::new();

        // Simple pattern matching: look for consecutive operations
        for i in 0..graph.nodes().len() {
            if self.matches_pattern_at(graph, i, ops) {
                let node_ids: Vec<usize> = (i..i + ops.len()).collect();
                matches.push(FusionOpportunity {
                    pattern_name: pattern_name.to_string(),
                    nodes: node_ids,
                    estimated_speedup: self.estimate_speedup(ops),
                });
            }
        }

        matches
    }

    /// Check if a pattern matches at a specific position
    fn matches_pattern_at(&self, graph: &ComputeGraph, start: usize, ops: &[String]) -> bool {
        if start + ops.len() > graph.nodes().len() {
            return false;
        }

        for (i, op) in ops.iter().enumerate() {
            if let Some(node) = graph.get_node(start + i) {
                if &node.op_type != op || !node.fusible {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Estimate speedup from fusing operations
    fn estimate_speedup(&self, ops: &[String]) -> f32 {
        // Simple heuristic: more ops fused = better speedup
        match ops.len() {
            2 => 1.3,  // 30% speedup
            3 => 1.5,  // 50% speedup
            4 => 1.7,  // 70% speedup
            _ => 1.2,  // 20% speedup
        }
    }

    /// Apply fusion to a graph
    pub fn fuse(&mut self, graph: &mut ComputeGraph, opportunities: &[FusionOpportunity]) {
        for opp in opportunities {
            self.fused_ops.insert(
                opp.pattern_name.clone(),
                opp.nodes.iter().map(|&id| {
                    graph.get_node(id).map(|n| n.op_type.clone()).unwrap_or_default()
                }).collect(),
            );
        }
    }

    /// Get fused operations
    pub fn get_fused_ops(&self) -> &HashMap<String, Vec<String>> {
        &self.fused_ops
    }
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion opportunity found in the graph
#[derive(Debug, Clone)]
pub struct FusionOpportunity {
    pub pattern_name: String,
    pub nodes: Vec<usize>,
    pub estimated_speedup: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_graph() {
        let mut graph = ComputeGraph::new();
        
        let n1 = graph.add_node("Input".to_string(), vec![], false);
        let n2 = graph.add_node("Conv2d".to_string(), vec![n1], true);
        let n3 = graph.add_node("ReLU".to_string(), vec![n2], true);
        
        assert_eq!(graph.nodes().len(), 3);
        assert_eq!(graph.get_node(n2).unwrap().op_type, "Conv2d");
    }

    #[test]
    fn test_fusion_engine() {
        let mut engine = FusionEngine::new();
        let mut graph = ComputeGraph::new();
        
        let n1 = graph.add_node("Input".to_string(), vec![], false);
        let n2 = graph.add_node("Linear".to_string(), vec![n1], true);
        let n3 = graph.add_node("ReLU".to_string(), vec![n2], true);
        
        let opportunities = engine.analyze(&graph);
        
        // Should find LinearReLU fusion
        assert!(!opportunities.is_empty());
    }

    #[test]
    fn test_fusion_patterns() {
        let engine = FusionEngine::new();
        
        // Should have default patterns registered
        assert!(!engine.patterns.is_empty());
    }
}

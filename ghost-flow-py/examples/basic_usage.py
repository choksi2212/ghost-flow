"""
Basic usage examples for GhostFlow Python bindings
"""

import ghost_flow as gf

def tensor_operations():
    """Basic tensor operations"""
    print("=== Tensor Operations ===")
    
    # Create tensors
    x = gf.randn([3, 4])
    y = gf.randn([4, 5])
    
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    
    # Matrix multiplication
    z = x @ y
    print(f"z = x @ y, shape: {z.shape}")
    
    # Element-wise operations
    a = gf.ones([3, 3])
    b = gf.ones([3, 3])
    c = a + b
    print(f"ones + ones = {c.tolist()[:3]}")  # First 3 elements

def neural_network():
    """Neural network example"""
    print("\n=== Neural Network ===")
    
    # Create a simple network
    linear = gf.nn.Linear(10, 5)
    relu = gf.nn.ReLU()
    
    # Forward pass
    x = gf.randn([32, 10])  # Batch of 32
    hidden = linear(x)
    output = relu(hidden)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

def activations():
    """Activation functions"""
    print("\n=== Activations ===")
    
    x = gf.Tensor([[-1.0, 0.0, 1.0, 2.0]], [1, 4])
    
    relu_out = x.relu()
    sigmoid_out = x.sigmoid()
    gelu_out = x.gelu()
    
    print(f"Input: {x.tolist()}")
    print(f"ReLU: {relu_out.tolist()}")
    print(f"Sigmoid: {sigmoid_out.tolist()}")
    print(f"GELU: {gelu_out.tolist()}")

if __name__ == "__main__":
    print("GhostFlow Python Bindings - Examples\n")
    tensor_operations()
    neural_network()
    activations()
    print("\n✅ All examples completed!")

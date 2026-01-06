//! Performance Optimization Demo
//!
//! This example demonstrates all performance optimization features:
//! - SIMD operations
//! - Kernel fusion
//! - Memory optimization
//! - Profiling tools

use ghostflow_core::{
    Tensor,
    simd_add_f32, simd_mul_f32, simd_dot_f32, simd_relu_f32,
    MemoryPool, MemoryLayoutOptimizer, TrackedAllocator,
    Profiler, Benchmark,
    FusionEngine, ComputeGraph,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GhostFlow Performance Optimization Demo ===\n");

    // 1. SIMD Operations
    println!("1. SIMD Operations");
    demo_simd()?;

    // 2. Kernel Fusion
    println!("\n2. Kernel Fusion");
    demo_fusion()?;

    // 3. Memory Optimization
    println!("\n3. Memory Optimization");
    demo_memory()?;

    // 4. Profiling
    println!("\n4. Profiling");
    demo_profiling()?;

    // 5. Benchmarking
    println!("\n5. Benchmarking");
    demo_benchmarking()?;

    println!("\n=== All Performance Features Demonstrated! ===");
    Ok(())
}

fn demo_simd() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing SIMD-optimized operations...");
    
    // Create test vectors
    let size = 1024;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    let mut result = vec![0.0f32; size];

    // SIMD addition
    simd_add_f32(&a, &b, &mut result);
    println!("   ✓ SIMD addition: {} elements", size);

    // SIMD multiplication
    simd_mul_f32(&a, &b, &mut result);
    println!("   ✓ SIMD multiplication: {} elements", size);

    // SIMD dot product
    let dot = simd_dot_f32(&a, &b);
    println!("   ✓ SIMD dot product: {:.2}", dot);

    // SIMD ReLU
    let input: Vec<f32> = (-512..512).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; 1024];
    simd_relu_f32(&input, &mut output);
    println!("   ✓ SIMD ReLU: {} elements", 1024);

    Ok(())
}

fn demo_fusion() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Analyzing computation graph for fusion opportunities...");
    
    let mut engine = FusionEngine::new();
    let mut graph = ComputeGraph::new();

    // Build a sample computation graph
    let input = graph.add_node("Input".to_string(), vec![], false);
    let conv = graph.add_node("Conv2d".to_string(), vec![input], true);
    let bn = graph.add_node("BatchNorm".to_string(), vec![conv], true);
    let relu = graph.add_node("ReLU".to_string(), vec![bn], true);
    let linear = graph.add_node("Linear".to_string(), vec![relu], true);
    let relu2 = graph.add_node("ReLU".to_string(), vec![linear], true);

    println!("   Graph nodes: {}", graph.nodes().len());

    // Analyze for fusion opportunities
    let opportunities = engine.analyze(&graph);
    println!("   Fusion opportunities found: {}", opportunities.len());

    for opp in &opportunities {
        println!("   - {}: {:.1}x speedup estimated", 
                 opp.pattern_name, opp.estimated_speedup);
    }

    // Apply fusion
    engine.fuse(&mut graph, &opportunities);
    println!("   ✓ Fusion applied");

    Ok(())
}

fn demo_memory() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing memory optimization features...");
    
    // Memory pool
    let mut pool = MemoryPool::new();
    
    println!("   Memory Pool:");
    let buf1 = pool.allocate(1024 * 1024); // 1MB
    println!("   - Allocated 1MB");
    
    pool.deallocate(buf1);
    println!("   - Deallocated 1MB");
    
    let buf2 = pool.allocate(1024 * 1024); // Reuse
    println!("   - Allocated 1MB (reused)");
    
    let stats = pool.stats();
    println!("   - Total allocations: {}", stats.total_allocations);
    println!("   - Reused allocations: {}", stats.reused_allocations);
    println!("   - Reuse rate: {:.1}%", stats.reuse_rate());
    println!("   - Peak memory: {:.2} MB", stats.peak_mb());

    pool.deallocate(buf2);

    // Memory layout optimizer
    println!("\n   Memory Layout Optimizer:");
    let optimizer = MemoryLayoutOptimizer::default();
    
    let layout = optimizer.optimize_layout(&[128, 256]);
    println!("   - Original size: {} bytes", layout.original_size);
    println!("   - Aligned size: {} bytes", layout.aligned_size);
    println!("   - Padding: {} bytes", layout.padding);
    println!("   - Stride: {:?}", layout.stride);

    // Tracked allocator
    println!("\n   Tracked Allocator:");
    let allocator = TrackedAllocator::new();
    
    let _buf = allocator.allocate(2 * 1024 * 1024); // 2MB
    println!("   - Allocated 2MB");
    
    let alloc_stats = allocator.stats();
    println!("   - Allocations: {}", alloc_stats.allocations);
    println!("   - Current memory: {:.2} MB", alloc_stats.current_mb());
    println!("   - Peak memory: {:.2} MB", alloc_stats.peak_mb());

    Ok(())
}

fn demo_profiling() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Profiling operations...");
    
    let profiler = Profiler::new();

    // Profile some operations
    for _ in 0..10 {
        let _scope = profiler.start("tensor_add");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3])?;
        let _c = &a + &b;
    }

    for _ in 0..5 {
        let _scope = profiler.start("tensor_mul");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3])?;
        let _c = &a * &b;
    }

    for _ in 0..3 {
        let _scope = profiler.start("tensor_matmul");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2])?;
        let _c = a.matmul(&b)?;
    }

    // Print summary
    profiler.print_summary();

    Ok(())
}

fn demo_benchmarking() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Running benchmarks...");
    
    // Benchmark tensor addition
    let result = Benchmark::new("Tensor Addition")
        .warmup(5)
        .iterations(100)
        .run(|| {
            let a = Tensor::from_slice(&[1.0f32; 1000], &[1000]).unwrap();
            let b = Tensor::from_slice(&[2.0f32; 1000], &[1000]).unwrap();
            let _c = &a + &b;
        });
    
    result.print();

    // Benchmark matrix multiplication
    let result = Benchmark::new("Matrix Multiplication")
        .warmup(5)
        .iterations(50)
        .run(|| {
            let a = Tensor::from_slice(&[1.0f32; 10000], &[100, 100]).unwrap();
            let b = Tensor::from_slice(&[2.0f32; 10000], &[100, 100]).unwrap();
            let _c = a.matmul(&b).unwrap();
        });
    
    result.print();

    // Benchmark SIMD operations
    let a = vec![1.0f32; 10000];
    let b = vec![2.0f32; 10000];
    let mut result_vec = vec![0.0f32; 10000];

    let result = Benchmark::new("SIMD Addition")
        .warmup(5)
        .iterations(1000)
        .run(|| {
            simd_add_f32(&a, &b, &mut result_vec);
        });
    
    result.print();

    Ok(())
}

//! Build script for CUDA compilation

fn main() {
    // Check if CUDA is available
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/kernels/");
        
        // Find CUDA toolkit
        let cuda_path = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudnn");
        
        // Compile CUDA kernels using nvcc
        // cc::Build::new()
        //     .cuda(true)
        //     .file("src/kernels/elementwise.cu")
        //     .file("src/kernels/matmul.cu")
        //     .file("src/kernels/attention.cu")
        //     .compile("ghostflow_kernels");
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("cargo:warning=Building without CUDA support");
    }
}

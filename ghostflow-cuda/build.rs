//! Build script for CUDA compilation

fn main() {
    println!("cargo:rerun-if-changed=src/optimized_kernels.cu");
    
    // Try to detect if nvcc is available
    let nvcc_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok();
    
    if !nvcc_available {
        println!("cargo:warning=NVCC not found - building without CUDA support. GPU operations will use CPU fallback.");
        println!("cargo:warning=To enable CUDA: Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads");
        return;
    }
    
    // Check if CUDA is available
    #[cfg(feature = "cuda")]
    {
        // Find CUDA toolkit
        let cuda_path = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        
        // Compile CUDA kernels using nvcc
        match cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_70") // Volta and newer
            .flag("-gencode=arch=compute_70,code=sm_70")
            .flag("-gencode=arch=compute_75,code=sm_75")
            .flag("-gencode=arch=compute_80,code=sm_80")
            .flag("-gencode=arch=compute_86,code=sm_86")
            .flag("--use_fast_math")
            .flag("-O3")
            .file("src/optimized_kernels.cu")
            .try_compile("ghostflow_cuda_kernels")
        {
            Ok(_) => println!("cargo:warning=Compiled CUDA kernels successfully"),
            Err(e) => {
                println!("cargo:warning=Failed to compile CUDA kernels: {}. Using CPU fallback.", e);
            }
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("cargo:warning=Building without CUDA support - GPU operations will use CPU fallback");
    }
}

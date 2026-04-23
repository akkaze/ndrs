use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("kernel");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    // 监听文件变化
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("include").display()
    );

    // 1. 编译 CPU 部分 (C++)
    let mut build = cc::Build::new();
    build
        .include(&kernel_dir.join("include"))
        .file(kernel_dir.join("src/cpu/ops.cpp"))
        .flag_if_supported("-std=c++14")
        .flag_if_supported("-fopenmp")
        .opt_level(if env::var("PROFILE").unwrap() == "release" {
            3
        } else {
            0
        });

    if cfg!(target_os = "linux") {
        build.flag("-pthread");
        println!("cargo:rustc-link-lib=gomp");
    }
    build.compile("ndrs_kernel_cpu");

    let cpu_lib = out_dir.join("libndrs_kernel_cpu.a");
    let final_lib = out_dir.join("libndrs_kernel.a");

    if cuda_enabled {
        // 编译 CUDA 部分
        use cudaforge::KernelBuilder;

        let cuda_files = vec![kernel_dir.join("src/cuda/ops.cu")];
        let cuda_include = kernel_dir.join("include");
        let cuda_lib_path = out_dir.join("libndrs_kernel_cuda.a");

        KernelBuilder::new()
            .source_files(cuda_files)
            .include_path(cuda_include)
            .arg("-O3")
            .arg("-std=c++17")
            .arg("--use_fast_math")
            .build_lib(cuda_lib_path.to_str().unwrap())
            .expect("Failed to build CUDA kernel");

        // 分别链接两个库
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=ndrs_kernel_cpu");
        println!("cargo:rustc-link-lib=static=ndrs_kernel_cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");

        // 设置 CUDA 库路径
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
    } else {
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=ndrs_kernel_cpu");
    }
}

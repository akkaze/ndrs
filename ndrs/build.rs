use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernel_dir = PathBuf::from(&manifest_dir)
        .parent()
        .unwrap()
        .join("kernel");
    let build_dir = kernel_dir.join("build");
    if !build_dir.exists() {
        fs::create_dir_all(&build_dir).unwrap();
    }
    let status = Command::new("cmake")
        .arg(&kernel_dir)
        .current_dir(&build_dir)
        .status()
        .expect("Failed to run cmake");
    assert!(status.success());
    let status = Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--config")
        .arg("Debug")
        .current_dir(&build_dir)
        .status()
        .expect("Failed to build kernel");
    assert!(status.success());

    let kernel_lib_src = build_dir.join("libndrs_kernel.a");

    // 复制到 cargo 输出目录
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = PathBuf::from(&out_dir)
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let kernel_lib_dst = target_dir.join("libndrs_kernel.a");
    fs::copy(&kernel_lib_src, &kernel_lib_dst)
        .expect("Failed to copy kernel library to target dir");
    println!("cargo:rerun-if-changed={}", kernel_lib_src.display());

    // 关键：复制到 python/ndrs/ 目录
    let python_ndrs_dir = PathBuf::from(&manifest_dir)
        .parent()
        .unwrap()
        .join("python/ndrs");
    if !python_ndrs_dir.exists() {
        fs::create_dir_all(&python_ndrs_dir).unwrap();
    }
    let python_kernel_dst = python_ndrs_dir.join("libndrs_kernel.a");
    fs::copy(&kernel_lib_src, &python_kernel_dst)
        .expect("Failed to copy kernel library to python/ndrs/");
    println!(
        "cargo:warning=Kernel library copied to {}",
        python_kernel_dst.display()
    );

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=ndrs_kernel");

    // 添加 CUDA 库路径
    if let Ok(cuda_home) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
        let lib_dir = PathBuf::from(cuda_home).join("lib64");
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    } else {
        // 尝试默认路径
        let default_cuda = PathBuf::from("/usr/local/cuda/lib64");
        if default_cuda.exists() {
            println!("cargo:rustc-link-search=native={}", default_cuda.display());
        }
    }

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=gomp");
}

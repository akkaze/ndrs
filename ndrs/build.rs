use find_cuda_helper::find_cuda_root;
use std::env;
use std::path::PathBuf;
#[cfg(target_env = "msvc")]
use vcvars::Vcvars;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("kernel");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    // ---------- 检测 CUDA 是否可用 ----------
    let cuda_available = if cuda_feature_enabled {
        find_cuda_root().is_some()
    } else {
        false
    };
    let cuda_enabled = cuda_feature_enabled && cuda_available;

    if cuda_feature_enabled && !cuda_available {
        eprintln!(
            "cargo:warning=CUDA feature enabled but CUDA toolkit not found. Falling back to CPU-only mode."
        );
    }

    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("include").display()
    );

    // ---------- 获取 MSVC 环境变量（仅 Windows MSVC）----------
    #[cfg(target_env = "msvc")]
    let (vc_include_dirs, vc_lib_dirs): (Vec<PathBuf>, Vec<PathBuf>) = {
        let include_dirs = {
            let mut vcvars = Vcvars::new();
            let vc_include = vcvars.get_cached("INCLUDE").unwrap_or_default();
            env::split_paths(&*vc_include).map(PathBuf::from).collect()
        };
        let lib_dirs = {
            let mut vcvars = Vcvars::new();
            let vc_lib = vcvars.get_cached("LIB").unwrap_or_default();
            env::split_paths(&*vc_lib).map(PathBuf::from).collect()
        };
        (include_dirs, lib_dirs)
    };

    // ---------- 编译 CPU 部分 (C++) ----------
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include(kernel_dir.join("include"))
        .file(kernel_dir.join("src").join("cpu").join("ops.cpp"))
        .opt_level(if env::var("PROFILE").unwrap() == "release" {
            3
        } else {
            0
        });

    #[cfg(target_env = "msvc")]
    for dir in &vc_include_dirs {
        build.include(dir);
    }

    let compiler = build.get_compiler();
    let clang_path = if compiler.is_like_msvc() {
        compiler.path().parent().map(|p| p.to_path_buf())
    } else {
        None
    };

    if compiler.is_like_msvc() {
        build.flag("/std:c++14");
    } else {
        build.flag_if_supported("-std=c++14");
    }

    if cfg!(target_os = "linux") {
        build.flag("-fopenmp").flag("-pthread");
        println!("cargo:rustc-link-lib=gomp");
    } else if cfg!(target_os = "windows") && compiler.is_like_msvc() {
        build.flag("/openmp");
        println!("cargo:rustc-link-lib=vcomp");
    }

    build.compile("ndrs_kernel_cpu");

    // ---------- 可选 CUDA 部分 ----------
    if cuda_enabled {
        use cudaforge::KernelBuilder;
        use find_cuda_helper::find_cuda_lib_dirs;

        let cuda_root = find_cuda_root().expect("CUDA root not found despite availability check");
        eprintln!("cargo:warning=Found CUDA at: {}", cuda_root.display());

        // 让 nvcc 能找到 cl.exe
        if let Some(cl_dir) = &clang_path {
            let cl_dir_str = cl_dir.to_str().expect("Invalid cl.exe directory");
            unsafe {
                env::set_var("NVCC_CCBIN", cl_dir_str);
                env::set_var("CUDA_CCBIN", cl_dir_str);
            }
        }

        let cuda_file = kernel_dir.join("src").join("cuda").join("ops.cu");
        let cuda_include = kernel_dir.join("include");

        let mut cuda_builder = KernelBuilder::new()
            .source_files(vec![cuda_file])
            .include_path(cuda_include)
            .cuda_root(&cuda_root) // 显式指定 CUDA 根目录
            .arg("-O3")
            .arg("-std=c++17")
            .arg("--use_fast_math");

        // 显式传递 -ccbin 参数
        if let Some(cl_dir) = &clang_path {
            let cl_dir_str = cl_dir.to_str().unwrap();
            cuda_builder = cuda_builder.arg("-ccbin").arg(cl_dir_str);
        }

        // 传递 MSVC 标准库头文件路径给 nvcc
        #[cfg(target_env = "msvc")]
        for dir in &vc_include_dirs {
            let dir_str = dir.to_str().expect("Invalid include path");
            cuda_builder = cuda_builder.arg(&format!("-I{}", dir_str));
        }

        let cuda_lib_path = out_dir.join("libndrs_kernel_cuda.a");
        cuda_builder
            .build_lib(cuda_lib_path.to_str().unwrap())
            .expect("Failed to build CUDA kernel");

        // 输出 CUDA 库搜索路径（使用 find_cuda_helper 获得的库目录）
        for lib_dir in find_cuda_lib_dirs() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }

        // 链接顺序：先 CUDA 静态库，再 CPU 静态库（因为 CPU 库依赖 CUDA 符号）
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=ndrs_kernel_cuda");
        println!("cargo:rustc-link-lib=static=ndrs_kernel_cpu");
        println!("cargo:rustc-link-lib=dylib=cudart");

        // 添加 MSVC 运行时库路径（Windows）
        #[cfg(target_env = "msvc")]
        for dir in &vc_lib_dirs {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    } else {
        // 仅 CPU 模式
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=ndrs_kernel_cpu");
    }
}

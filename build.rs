use cmake::Config;

fn main() {
    // 构建 kernel 动态库
    let dst = Config::new("kernel")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build();
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=tensor_kernel");
    // 让 Rust 链接时能找到这个库
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dst.join("build").display());
}
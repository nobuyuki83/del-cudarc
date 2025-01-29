fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/sort_by_key_u32.cu");
    println!("cargo:rerun-if-changed=src/sort_by_key_u64.cu");
    println!("cargo:rerun-if-changed=src/cumsum.cu");
    println!("cargo:rerun-if-changed=src/sort_u32.cu");
    println!("cargo:rerun-if-changed=src/sort_u6.cu");
    println!("cargo:rerun-if-changed=src/util.cu");

    let builder = bindgen_cuda::Builder::default().include_paths_glob("../cpp_header/*");
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}

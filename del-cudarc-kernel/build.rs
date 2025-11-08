fn main() {
    //use std::{env, path::PathBuf};
    //let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    //println!("cargo:warning=out dir: {:?}", out_dir);
    //println!("cargo:rustc-env=PTX_DIR={}", out_dir.display());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/sort_by_key_u32.cu");
    println!("cargo:rerun-if-changed=src/sort_by_key_u64.cu");
    println!("cargo:rerun-if-changed=src/cumsum.cu");
    println!("cargo:rerun-if-changed=src/sort_u32.cu");
    println!("cargo:rerun-if-changed=src/sort_u64.cu");
    println!("cargo:rerun-if-changed=src/array1d.cu");
    println!("cargo:rerun-if-changed=src/sorted_array1d.cu");
    println!("cargo:rerun-if-changed=src/offset_array.cu");
    println!("cargo:rerun-if-changed=src/util.cu");
    //println!("cargo:rustc-env={}",out_dir);
    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
        #[cfg(not(target_os = "linux"))]
        {
            panic!("CUDA_PATH is not set.");
        }
        #[cfg(target_os = "linux")]
        {
            "/usr/local/cuda".to_string()
        }
    });
    let cuda_path = PathBuf::from(cuda_path);

    let nvcc = if cfg!(target_os = "windows") {
        cuda_path.join("bin").join("nvcc.exe")
    } else {
        cuda_path.join("bin").join("nvcc")
    };

    let cuda_dir = manifest_dir.join("cuda");
    let mut cu_files = Vec::<PathBuf>::new();
    for entry in fs::read_dir(&cuda_dir).expect("failed to read cuda dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map(|e| e == "cu").unwrap_or(false) {
            println!("cargo:rerun-if-changed={}", path.display());
            cu_files.push(path);
        }
    }

    let lib_path = if cfg!(target_os = "windows") {
        out_dir.join("del-cudarc-sys-thrust.lib")
    } else {
        out_dir.join("del-cudarc-sys-thrust.a")
    };

    let mut cmd = Command::new(&nvcc);
    cmd.current_dir(&manifest_dir);
    cmd.arg("-std=c++17");
    if cfg!(target_os = "windows") {
        cmd.args(["-Xcompiler", "/MD"]);
    } else {
        cmd.args(["-Xcompiler", "-fPIC"]);
    }
    cmd.arg("--lib");
    for cu in &cu_files {
        cmd.arg(cu.to_str().unwrap());
    }
    cmd.args(["-o", lib_path.to_str().unwrap()]);
    cmd.args([
        "-gencode",
        "arch=compute_80,code=sm_80",
        "-gencode",
        "arch=compute_80,code=compute_80",
        "-gencode",
        "arch=compute_89,code=sm_89",
        "-gencode",
        "arch=compute_89,code=compute_89",
    ]);

    let status = cmd.status().expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=del-cudarc-sys-thrust");

    let cuda_lib = if cfg!(target_os = "windows") {
        cuda_path.join("lib").join("x64")
    } else {
        cuda_path.join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
}

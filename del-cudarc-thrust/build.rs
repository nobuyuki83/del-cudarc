use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    // cargo が教えてくれるパス
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // CUDA_PATH を取得（Windows では必須）
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
        #[cfg(target_os = "linux")]
        {
            "/usr/local/cuda".to_string()
        }
        #[cfg(not(target_os = "linux"))]
        {
            panic!("CUDA_PATH is not set. 環境変数 CUDA_PATH を CUDA のインストール先に設定してください。");
        }
    });
    let cuda_path = PathBuf::from(cuda_path);

    // nvcc の場所
    let nvcc = if cfg!(target_os = "windows") {
        cuda_path.join("bin").join("nvcc.exe")
    } else {
        cuda_path.join("bin").join("nvcc")
    };

    // cuda ディレクトリ
    let cuda_dir = manifest_dir.join("cuda");
    if !cuda_dir.exists() {
        panic!("CUDA source directory not found: {}", cuda_dir.display());
    }

    // cuda フォルダ直下の *.cu を全部集める
    let mut cu_files = Vec::<PathBuf>::new();
    for entry in fs::read_dir(&cuda_dir).expect("failed to read cuda dir") {
        let entry = entry.expect("failed to read dir entry");
        let path = entry.path();
        if path.extension().map(|ext| ext == "cu").unwrap_or(false) {
            cu_files.push(path);
        }
    }

    println!("cargo:warning=cuda files = {:?}", cu_files);

    if cu_files.is_empty() {
        panic!("No .cu files found in {}", cuda_dir.display());
    }

    // 出力するライブラリ名
    let lib_path = if cfg!(target_os = "windows") {
        out_dir.join("del-cudarc-thrust.lib")
    } else {
        out_dir.join("del-cudarc-thrust.a")
    };

    // 以前のファイルが残っていたら一応消しておく
    let _ = fs::remove_file(&lib_path);

    // ----- nvcc コマンドを組み立て -----
    let mut cmd = Command::new(&nvcc);
    cmd.current_dir(&manifest_dir);

    cmd.arg("-std=c++17");

    // ホストコンパイラオプション
    if cfg!(target_os = "windows") {
        // MSVC /MD (ランタイム設定は環境に合わせて調整可)
        cmd.args(["-Xcompiler", "/MD"]);
    } else {
        cmd.args(["-Xcompiler", "-fPIC"]);
    }

    // static ライブラリとしてビルド
    cmd.arg("--lib");

    // ここで cuda/*.cu を全部追加
    for cu in &cu_files {
        cmd.arg(cu.to_str().unwrap());
    }

    cmd.args(["-o", lib_path.to_str().unwrap()]);

    // 複数アーキをサポート（必要に応じて調整）
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

    println!("cargo:warning=running nvcc: {:?}", cmd);

    let status = cmd.status().expect("failed to run nvcc");
    if !status.success() {
        panic!("nvcc failed with status: {status}");
    }

    // ここで本当に lib ができているかチェック
    if !lib_path.exists() {
        panic!(
            "nvcc finished, but library not found: {}",
            lib_path.display()
        );
    } else {
        println!("cargo:warning=created CUDA library: {}", lib_path.display());
    }

    // ----- Rust にリンク情報を伝える -----

    // out_dir をライブラリ検索パスに追加
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // del-cudarc-thrust.lib / .a → "del-cudarc-thrust"
    println!("cargo:rustc-link-lib=static=del-cudarc-thrust");

    // CUDA ランタイムへのパス
    let cuda_lib_dir = if cfg!(target_os = "windows") {
        cuda_path.join("lib").join("x64")
    } else {
        cuda_path.join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");

    // 各 .cu が変わったら再ビルド
    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());
    }
}

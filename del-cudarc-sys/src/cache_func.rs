use crate::cu;

// --- 生ハンドル薄ラッパ ---
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CuFunction(pub cu::CUfunction);

#[repr(transparent)]
struct CuModule(cu::CUmodule);

// ★ 安全性は呼び出し側で担保する前提で明示
unsafe impl Send for CuFunction {}
unsafe impl Sync for CuFunction {}
unsafe impl Send for CuModule {}
unsafe impl Sync for CuModule {}

use dashmap::DashMap;
use std::{
    ffi::CString,
    sync::{Arc, OnceLock},
};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct KeyModule {
    ctx_id: usize, // CUcontext を usize 化してキーに
    module_name: &'static str,
}

pub struct ModuleEntry {
    module: CuModule,
    funcs: DashMap<&'static str, CuFunction>, // ← ★ ここを *mut CUfunc_st ではなく
}

pub type Cache = DashMap<KeyModule, Arc<ModuleEntry>>;
pub static MOD_CACHE: OnceLock<Cache> = OnceLock::new();
#[inline]
fn cache() -> &'static Cache {
    MOD_CACHE.get_or_init(DashMap::new)
}
pub fn clear() {
    MOD_CACHE.get_or_init(DashMap::new).clear();
}

fn current_ctx_id() -> Result<cu::CUcontext, String> {
    let mut ctx: cu::CUcontext = std::ptr::null_mut();
    crate::cuda_check!(cu::cuCtxGetCurrent(&mut ctx))?;
    if ctx.is_null() {
        return Err("no current CUDA context".to_string());
    }
    Ok(ctx)
}

fn load_module_once(name: &'static str, fatbin: &'static [u8]) -> Result<Arc<ModuleEntry>, String> {
    let ctx = current_ctx_id().unwrap();
    let key = KeyModule {
        ctx_id: ctx as usize,
        module_name: name,
    };
    if let Some(e) = cache().get(&key) {
        return Ok(e.value().clone());
    }
    let mut m: cu::CUmodule = std::ptr::null_mut();
    crate::cuda_check!(cu::cuModuleLoadData(&mut m, fatbin.as_ptr() as *const _))?;
    let entry = Arc::new(ModuleEntry {
        module: CuModule(m),
        funcs: DashMap::new(),
    });
    cache().insert(key, entry.clone());
    Ok(entry)
}

pub fn get_function_cached(
    mod_name: &'static str,
    fatbin: &'static [u8],
    func_name: &'static str,
) -> Result<cu::CUfunction, String> {
    let entry = load_module_once(mod_name, fatbin)
        .map_err(|e| format!("failure in load_module_once: {} {} {e}", file!(), line!()))?;
    if let Some(f) = entry.funcs.get(func_name) {
        return Ok(f.0);
    }
    let mut cu_func: cu::CUfunction = std::ptr::null_mut();
    let c_func_name = CString::new(func_name).unwrap();
    crate::cuda_check!(cu::cuModuleGetFunction(
        &mut cu_func,
        entry.module.0,
        c_func_name.as_ptr()
    ))
    .map_err(|e| {
        format!(
            "failure in cuModuleGetFunction: {}, {}, {e}",
            file!(),
            line!()
        )
    })?;
    let f = CuFunction(cu_func);
    entry.funcs.insert(func_name, f);
    Ok(f.0)
}

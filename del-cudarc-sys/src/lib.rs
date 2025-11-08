use cudarc::driver::sys::CUfunction;
use std::marker::PhantomData;

pub mod array1d;
pub mod cumsum;

pub mod offset_array;
pub mod sort_by_key_u32;
pub mod sorted_array1d;

pub mod cu {
    pub use cudarc::driver::sys::*;
}

/*
#[macro_export]
macro_rules! cuda_check {
    ($e:expr) => {
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $e };
        if res != cu::CUresult::CUDA_SUCCESS {
            panic!("CUDA Error: {:?}", res);
        }
    };
}
 */

/// put "use del_cudarc_sys::cu" before using this macro
#[macro_export]
macro_rules! cuda_check {
    ($e:expr) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $e };
        if res != cu::CUresult::CUDA_SUCCESS {
            // エラーならErr(String)を返す
            Err(format!("CUDA Error: {:?}", res))
        } else {
            // 成功ならOk(())
            Ok(())
        }
    }};
}

pub fn load_get_function(file_name: &str, function_name: &str) -> Result<CUfunction, String> {
    let fatbin = del_cudarc_kernels::get(file_name).ok_or("missing add.fatbin in kernels")?;
    let mut module: cu::CUmodule = std::ptr::null_mut();
    cuda_check!(cu::cuModuleLoadData(
        &mut module as *mut _,
        fatbin.as_ptr() as *const _
    ))?;
    let cname = std::ffi::CString::new(function_name).unwrap();
    let mut f: cu::CUfunction = std::ptr::null_mut();
    cuda_check!(cu::cuModuleGetFunction(&mut f, module, cname.as_ptr()))?;
    Ok(f)
}

pub fn create_stream_in_current_context() -> Result<cu::CUstream, String> {
    let mut stream: cu::CUstream = std::ptr::null_mut();
    cuda_check!(cu::cuStreamCreate(&mut stream, 0))?;
    Ok(stream)
}

pub fn stream_from_u64(stream_ptr: u64) -> cu::CUstream {
    assert_eq!(size_of::<usize>(), 8);
    stream_ptr as usize as *mut std::ffi::c_void as cu::CUstream
}

pub fn get_current_context() -> Result<cu::CUcontext, String> {
    let mut raw: cu::CUcontext = std::ptr::null_mut();
    cuda_check!(cu::cuCtxGetCurrent(&mut raw))?;
    assert!(!raw.is_null(), "no current CUDA context");
    Ok(raw)
}

pub fn init_cuda_and_make_context(device_id: i32) -> Result<(cu::CUdevice, cu::CUcontext), String> {
    cuda_check!(cu::cuInit(0))?;
    let mut dev: cu::CUdevice = 0;
    cuda_check!(cu::cuDeviceGet(&mut dev, device_id))?;
    let mut ctx: cu::CUcontext = std::ptr::null_mut();
    cuda_check!(cu::cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
    cuda_check!(cu::cuCtxSetCurrent(ctx))?;
    Ok((dev, ctx))
}

/// do not implement "Copy" trait.
pub struct CuVec<T> {
    pub dptr: cu::CUdeviceptr,
    pub n: usize,
    pub is_free_at_drop: bool,
    pub phantom: PhantomData<T>,
}

impl<T> CuVec<T> {
    pub fn new(dptr: cu::CUdeviceptr, n: usize, is_free_at_drop: bool) -> Self {
        Self {
            dptr,
            n,
            is_free_at_drop,
            phantom: PhantomData,
        }
    }

    pub fn with_capacity(n: usize) -> Result<Self, String> {
        let mut dptr: cu::CUdeviceptr = 0;
        cuda_check!(cu::cuMemAlloc_v2(&mut dptr, n * size_of::<T>()))?;
        Ok(Self {
            dptr,
            n,
            is_free_at_drop: true,
            phantom: PhantomData,
        })
    }

    pub fn from_slice(s: &[T]) -> Result<Self, String> {
        let n = s.len();
        let mut dptr: cu::CUdeviceptr = 0;
        let bytes = size_of_val(s);
        cuda_check!(cu::cuMemAlloc_v2(&mut dptr, bytes))?;
        cuda_check!(cu::cuMemcpyHtoD_v2(
            dptr,
            s.as_ptr() as *const std::ffi::c_void,
            bytes,
        ))?;
        Ok(Self {
            dptr,
            n,
            is_free_at_drop: true,
            phantom: PhantomData,
        })
    }

    pub fn alloc_zeros(n: usize, stream: cu::CUstream) -> Result<Self, String> {
        let a = CuVec::<T>::with_capacity(n)?;
        a.set_zeros(stream)?;
        Ok(a)
    }

    pub fn from_dptr(dptr: cu::CUdeviceptr, n: usize) -> Self {
        crate::CuVec::<T> {
            dptr,
            n,
            is_free_at_drop: false,
            phantom: PhantomData,
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_zeros(&self, stream: cu::CUstream) -> Result<(), String> {
        cuda_check!(cu::cuMemsetD8Async(
            self.dptr,
            0u8,
            self.n * size_of::<T>(),
            stream
        ))?;
        cuda_check!(cu::cuStreamSynchronize(stream))?;
        Ok(())
    }

    pub fn copy_to_host(&self) -> Result<Vec<T>, String> {
        let mut v: Vec<T> = Vec::with_capacity(self.n);
        cuda_check!(cu::cuMemcpyDtoH_v2(
            v.as_mut_ptr() as *mut std::ffi::c_void,
            self.dptr,
            self.n * size_of::<T>(),
        ))?;
        unsafe { v.set_len(self.n) };
        Ok(v)
    }

    pub fn last(&self) -> Result<u32, String> {
        let dptr0 = self.dptr + ((self.n - 1) * size_of::<T>()) as u64;
        let mut last = 0u32;
        cuda_check!(cu::cuMemcpyDtoH_v2(
            (&mut last as *mut u32) as *mut std::ffi::c_void,
            dptr0,
            size_of::<T>(),
        ))?;
        Ok(last)
    }
}

impl<T> Drop for CuVec<T> {
    fn drop(&mut self) {
        if self.is_free_at_drop {
            assert_ne!(self.dptr, 0);
            cuda_check!(cu::cuMemFree_v2(self.dptr)).unwrap();
            self.dptr = 0;
        }
    }
}

pub fn malloc_device<T>(n: usize) -> Result<cu::CUdeviceptr, String> {
    let mut dptr: cu::CUdeviceptr = 0;
    cuda_check!(cu::cuMemAlloc_v2(&mut dptr, n * size_of::<T>()))?;
    Ok(dptr)
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn memset_zeros_32(
    dptr: cu::CUdeviceptr,
    n: usize,
    stream: cu::CUstream,
) -> Result<(), String> {
    cuda_check!(cu::cuMemsetD8Async(dptr, 0u8, n * 4, stream))?;
    cuda_check!(cu::cuStreamSynchronize(stream))?;
    Ok(())
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn memcpy_d2d_32(
    d_dst: cu::CUdeviceptr,
    d_src: cu::CUdeviceptr,
    n: usize,
    stream: cu::CUstream,
) -> Result<(), String> {
    // ---- Device→Device コピー（非同期）----
    cuda_check!(cu::cuMemcpyDtoDAsync_v2(d_dst, d_src, n * 4, stream))?;
    // 必要に応じて同期
    cuda_check!(cu::cuStreamSynchronize(stream))?;
    Ok(())
}

pub struct Builder {
    pub cu_stream: cu::CUstream,
    pub args: Vec<*mut std::ffi::c_void>,
    pub vec_u32: Vec<u32>,
    pub vec_f32: Vec<f32>,
    pub vec_u8: Vec<u8>,
    pub vec_dptr: Vec<Box<cu::CUdeviceptr>>, // to get stable pointer to the CUdeviceptr
}

impl Builder {
    pub fn new(cu_stream: cudarc::driver::sys::CUstream) -> Builder {
        Builder {
            cu_stream,
            args: vec![],
            vec_u32: vec![],
            vec_f32: vec![],
            vec_u8: vec![],
            vec_dptr: vec![],
        }
    }

    pub fn arg_data(&mut self, ptr: *const *mut std::ffi::c_void) {
        let ptr = (ptr as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    pub fn arg_dptr(&mut self, dptr: cu::CUdeviceptr) {
        let mut slot = Box::new(dptr); // put dptr on the heap (the address won't change)
        let ptr: *mut std::ffi::c_void =
            (&mut *slot as *mut cu::CUdeviceptr) as *mut std::ffi::c_void;
        self.args.push(ptr);
        self.vec_dptr.push(slot);
    }

    pub fn arg_u32(&mut self, val: u32) {
        self.vec_u32.push(val);
        let val_ref = self.vec_u32.last().unwrap();
        let ptr = (val_ref as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    pub fn arg_f32(&mut self, val: f32) {
        self.vec_f32.push(val);
        let val_ref = self.vec_f32.last().unwrap();
        let ptr = (val_ref as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    pub fn arg_bool(&mut self, val: bool) {
        assert_eq!(size_of::<bool>(), 1);
        self.vec_u8.push(val as u8);
        let val_ref = self.vec_u8.last().unwrap();
        let ptr = (val_ref as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    /// # Safety
    /// unde
    /// fined if function is invalid
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch_kernel(
        &mut self,
        function: cu::CUfunction,
        cfg: LaunchConfig,
    ) -> Result<(), String> {
        cuda_check!(cu::cuLaunchKernel(
            function,
            cfg.grid_dim.0,
            cfg.grid_dim.1,
            cfg.grid_dim.2,
            cfg.block_dim.0,
            cfg.block_dim.1,
            cfg.block_dim.2,
            cfg.shared_mem_bytes,
            self.cu_stream,
            self.args.as_mut_ptr(),
            std::ptr::null_mut(),
        ))?;
        cuda_check!(cu::cuStreamSynchronize(self.cu_stream))?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a [cudarc::driver::LaunchConfig] with:
    /// - block_dim == `1024`
    /// - grid_dim == `(n + 1023) / 1024`
    /// - shared_mem_bytes == `0`
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = n.div_ceil(NUM_THREADS);
        Self {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

pub fn get_ptx_compiler_version() -> (i32, i32, i32) {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    let a = {
        let func = load_get_function("util", "get_version").unwrap();
        /*
        let (func, _mdl) =
            crate::load_function_in_module(del_cudarc_kernel::UTIL, "get_version").unwrap();
         */
        let a = CuVec::<i32>::with_capacity(3).unwrap();
        let mut builder = crate::Builder::new(stream);
        builder.arg_dptr(a.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(1))
            .unwrap();
        a.copy_to_host().unwrap()
    };
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
    (a[0], a[1], a[2])
}

#[test]
fn test_get_ptx_compiler_version() {
    let (major, minor, build) = get_ptx_compiler_version();
    dbg!(major, minor, build);
}

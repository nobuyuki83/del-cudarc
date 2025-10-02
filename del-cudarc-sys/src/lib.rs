use std::marker::PhantomData;

pub mod cumsum;
pub mod util;

pub mod cu {
    pub use cudarc::driver::sys::*;
}

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

pub fn load_function_in_module(ptx: &str, func_name: &str) -> (cu::CUfunction, cu::CUmodule) {
    let ptx = std::ffi::CString::new(ptx).unwrap();
    let mut module: cu::CUmodule = std::ptr::null_mut();
    cuda_check!(cu::cuModuleLoadDataEx(
        &mut module,
        ptx.as_ptr() as *const _,
        0,
        std::ptr::null::<u32>() as *mut cu::CUjit_option,
        std::ptr::null::<u32>() as *mut *mut std::ffi::c_void,
    ));
    let mut function: cu::CUfunction = std::ptr::null_mut();
    let func_name_str = std::ffi::CString::new(func_name).unwrap();
    cuda_check!(cu::cuModuleGetFunction(
        &mut function,
        module,
        func_name_str.as_ptr()
    ));
    (function, module)
}

pub fn create_stream_in_current_context() -> cu::CUstream {
    let mut stream: cu::CUstream = std::ptr::null_mut();
    cuda_check!(cu::cuStreamCreate(&mut stream, 0));
    stream
}

pub fn stream_from_u64(stream_ptr: u64) -> cu::CUstream {
    assert_eq!(size_of::<usize>(), 8);
    stream_ptr as usize as *mut std::ffi::c_void as cu::CUstream
}

pub fn get_current_context() -> cu::CUcontext {
    let mut raw: cu::CUcontext = std::ptr::null_mut();
    cuda_check!(cu::cuCtxGetCurrent(&mut raw));
    assert!(!raw.is_null(), "no current CUDA context");
    raw
}

pub fn init_cuda_and_make_context(device_id: i32) -> (cu::CUdevice, cu::CUcontext) {
    cuda_check!(cu::cuInit(0));
    let mut dev: cu::CUdevice = 0;
    cuda_check!(cu::cuDeviceGet(&mut dev, device_id));
    let mut ctx: cu::CUcontext = std::ptr::null_mut();
    cuda_check!(cu::cuDevicePrimaryCtxRetain(&mut ctx, dev));
    cuda_check!(cu::cuCtxSetCurrent(ctx));
    (dev, ctx)
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

    pub fn with_capacity(n: usize) -> Self {
        let mut dptr: cu::CUdeviceptr = 0;
        cuda_check!(cu::cuMemAlloc_v2(&mut dptr, n * size_of::<T>()));
        Self {
            dptr,
            n,
            is_free_at_drop: true,
            phantom: PhantomData,
        }
    }

    pub fn from_slice(s: &[T]) -> Self {
        let n = s.len();
        let mut dptr: cu::CUdeviceptr = 0;
        let bytes = size_of_val(s);
        cuda_check!(cu::cuMemAlloc_v2(&mut dptr, bytes));
        cuda_check!(cu::cuMemcpyHtoD_v2(
            dptr,
            s.as_ptr() as *const std::ffi::c_void,
            bytes,
        ));
        Self {
            dptr,
            n,
            is_free_at_drop: true,
            phantom: PhantomData,
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_zeros(&self, stream: cu::CUstream) {
        cuda_check!(cu::cuMemsetD8Async(
            self.dptr,
            0u8,
            self.n * size_of::<T>(),
            stream
        ));
        cuda_check!(cu::cuStreamSynchronize(stream));
    }

    pub fn copy_to_host(&self) -> Vec<T> {
        let mut v: Vec<T> = Vec::with_capacity(self.n);
        cuda_check!(cu::cuMemcpyDtoH_v2(
            v.as_mut_ptr() as *mut std::ffi::c_void,
            self.dptr,
            self.n * size_of::<T>(),
        ));
        unsafe { v.set_len(self.n) };
        v
    }

    pub fn last(&self) -> u32 {
        let dptr0 = self.dptr + ((self.n - 1) * size_of::<T>()) as u64;
        let mut last = 0u32;
        cuda_check!(cu::cuMemcpyDtoH_v2(
            (&mut last as *mut u32) as *mut std::ffi::c_void,
            dptr0,
            size_of::<T>(),
        ));
        last
    }
}

impl<T> Drop for CuVec<T> {
    fn drop(&mut self) {
        if self.is_free_at_drop {
            assert_ne!(self.dptr, 0);
            cuda_check!(cu::cuMemFree_v2(self.dptr));
            self.dptr = 0;
        }
    }
}

pub fn malloc_device<T>(n: usize) -> cu::CUdeviceptr {
    let mut dptr: cu::CUdeviceptr = 0;
    cuda_check!(cu::cuMemAlloc_v2(&mut dptr, n * size_of::<T>()));
    dptr
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn memset_zeros_32(dptr: cu::CUdeviceptr, n: usize, stream: cu::CUstream) {
    cuda_check!(cu::cuMemsetD8Async(dptr, 0u8, n * 4, stream));
    cuda_check!(cu::cuStreamSynchronize(stream));
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn memcpy_d2d_32(
    d_dst: cu::CUdeviceptr,
    d_src: cu::CUdeviceptr,
    n: usize,
    stream: cu::CUstream,
) {
    // ---- Device→Device コピー（非同期）----
    cuda_check!(cu::cuMemcpyDtoDAsync_v2(d_dst, d_src, n * 4, stream));
    // 必要に応じて同期
    cuda_check!(cu::cuStreamSynchronize(stream));
}

pub struct Builder {
    pub cu_stream: cu::CUstream,
    pub args: Vec<*mut std::ffi::c_void>,
    pub vec_i32: Vec<i32>,
    pub vec_f32: Vec<f32>,
    pub keep_dptr: Vec<Box<cu::CUdeviceptr>>, // to get stable pointer to the CUdeviceptr
}

impl Builder {
    pub fn new(cu_stream: cudarc::driver::sys::CUstream) -> Builder {
        Builder {
            cu_stream,
            args: vec![],
            vec_i32: vec![],
            vec_f32: vec![],
            keep_dptr: vec![],
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
        self.keep_dptr.push(slot);
    }

    pub fn arg_i32(&mut self, val: i32) {
        self.vec_i32.push(val);
        let val_ref = self.vec_i32.last().unwrap();
        let ptr = (val_ref as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    pub fn arg_f32(&mut self, val: f32) {
        self.vec_f32.push(val);
        let val_ref = self.vec_f32.last().unwrap();
        let ptr = (val_ref as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    /// # Safety
    /// unde
    /// fined if function is invalid
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch_kernel(&mut self, function: cu::CUfunction, cfg: LaunchConfig) {
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
        ));
        cuda_check!(cu::cuStreamSynchronize(self.cu_stream));
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

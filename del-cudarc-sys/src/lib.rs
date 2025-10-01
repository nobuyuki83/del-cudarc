pub mod util;

pub mod cu {
    pub use cudarc::driver::sys::*;
}

#[macro_export]
macro_rules! cuda_check {
    ($e:expr) => {
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

pub fn get_current_context() -> cu::CUcontext {
    let mut raw: cu::CUcontext = std::ptr::null_mut();
    cuda_check!(cu::cuCtxGetCurrent(&mut raw));
    assert!(!raw.is_null(), "no current CUDA context");
    raw
}

pub fn init_cuda_and_make_context(device_id: i32) -> cu::CUcontext {
    cuda_check!(cu::cuInit(0));
    let mut dev: cu::CUdevice = 0;
    cuda_check!(cu::cuDeviceGet(&mut dev, device_id));
    let mut ctx: cu::CUcontext = std::ptr::null_mut();
    cuda_check!(cu::cuCtxCreate_v2(&mut ctx, 0, dev));
    ctx
}

pub fn malloc_device<T>(n: usize) -> cu::CUdeviceptr {
    let mut dptr: cu::CUdeviceptr = 0;
    cuda_check!(cu::cuMemAlloc_v2(&mut dptr, n * size_of::<T>()));
    dptr
}

pub struct Builder {
    pub cu_stream: cu::CUstream,
    pub args: Vec<*mut std::ffi::c_void>,
    pub vec_i32: Vec<i32>,
    pub vec_f32: Vec<f32>,
}

impl Builder {
    pub fn new(cu_stream: cudarc::driver::sys::CUstream) -> Builder {
        Builder {
            cu_stream,
            args: vec![],
            vec_i32: vec![],
            vec_f32: vec![],
        }
    }

    pub fn arg_data(&mut self, ptr: *const *mut std::ffi::c_void) {
        let ptr = (ptr as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    pub fn arg_dptr(&mut self, dptr: &mut cu::CUdeviceptr) {
        let ptr: *mut std::ffi::c_void = (dptr as *mut cu::CUdeviceptr) as *mut std::ffi::c_void;
        self.args.push(ptr);
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
    /// undefined if function is invalid
    pub fn launch_kernel(&mut self, function: cudarc::driver::sys::CUfunction, n: u32) {
        let grid_dim = n.div_ceil(256); // n + 255) / 256;
        cuda_check!(cu::cuLaunchKernel(
            function,
            grid_dim,
            1,
            1,
            256,
            1,
            1,
            0,
            self.cu_stream,
            self.args.as_mut_ptr(),
            std::ptr::null_mut(),
        ));
        cuda_check!(cu::cuStreamSynchronize(self.cu_stream));
    }
}

pub fn dtoh_vec<T>(dptr: cu::CUdeviceptr, n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    cuda_check!(cu::cuMemcpyDtoH_v2(
        v.as_mut_ptr() as *mut std::ffi::c_void,
        dptr,
        n * size_of::<T>(),
    ));
    unsafe { v.set_len(n) };
    v
}

use std::os::raw::c_void;

pub fn load_function_in_module(
    ptx: &str,
    func_name: &str,
) -> (
    cudarc::driver::sys::CUfunction,
    cudarc::driver::sys::CUmodule,
) {
    use cudarc::driver::sys::{cudaError_enum::CUDA_SUCCESS, *};
    use std::ffi::CString;
    let ptx = CString::new(ptx).unwrap();
    let mut module: CUmodule = std::ptr::null_mut();
    let res = unsafe {
        cuModuleLoadDataEx(
            &mut module,
            ptx.as_ptr() as *const _,
            0,
            std::ptr::null::<u32>() as *mut CUjit_option,
            std::ptr::null::<u32>() as *mut *mut std::ffi::c_void,
        )
    };
    assert_eq!(res, CUDA_SUCCESS);
    let mut function: CUfunction = std::ptr::null_mut();
    let func_name_str = CString::new(func_name).unwrap();
    let res = unsafe { cuModuleGetFunction(&mut function, module, func_name_str.as_ptr()) };
    assert_eq!(res, CUDA_SUCCESS);
    (function, module)
}

pub fn create_stream_in_current_context() -> cudarc::driver::sys::CUstream {
    use cudarc::driver::sys::{CUresult, CUstream, cuStreamCreate, cudaError_enum::CUDA_SUCCESS};
    let mut stream: CUstream = std::ptr::null_mut();
    let res: CUresult = unsafe { cuStreamCreate(&mut stream, 0) };
    assert_eq!(res, CUDA_SUCCESS);
    println!("stream ptr: {:?}", stream);
    stream
}

pub fn get_current_context() -> cudarc::driver::sys::CUcontext {
    use cudarc::driver::sys::{CUcontext, CUresult, cuCtxGetCurrent, cudaError_enum::CUDA_SUCCESS};
    let mut raw: CUcontext = std::ptr::null_mut();
    let res: CUresult = unsafe { cuCtxGetCurrent(&mut raw) };
    assert_eq!(res, CUDA_SUCCESS);
    assert!(!raw.is_null(), "no current CUDA context");
    dbg!(raw);
    raw
}

pub struct Builder {
    pub cu_stream: cudarc::driver::sys::CUstream,
    pub args: Vec<*mut c_void>,
    pub vec_i32: Vec<i32>,
}

impl Builder {
    pub fn new(cu_stream: cudarc::driver::sys::CUstream) -> Builder {
        Builder {
            cu_stream,
            args: vec![],
            vec_i32: vec![],
        }
    }

    pub fn arg_data(&mut self, ptr: *const *mut c_void) {
        let ptr = (ptr as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }

    pub fn arg_i32(&mut self, val: i32) {
        self.vec_i32.push(val);
        let val_ref = self.vec_i32.last().unwrap();
        let ptr = (val_ref as *const _) as *mut std::ffi::c_void;
        self.args.push(ptr);
    }


    /// # Safety
    /// undefined if function is invalid
    pub unsafe fn launch_kernel(&mut self, function: cudarc::driver::sys::CUfunction, n: u32) {
        let grid_dim = n.div_ceil(256);// n + 255) / 256;
        unsafe {
            cudarc::driver::sys::cuLaunchKernel(
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
            );
            cudarc::driver::sys::cuStreamSynchronize(self.cu_stream);
        }
    }
}

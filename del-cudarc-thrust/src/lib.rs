extern "C" {
    pub fn thrust_sort_u64_inplace(d_data: *mut u64, n: u32, stream: *mut std::ffi::c_void);

    pub fn thrust_sort_by_key_u64_u32(
        keys: *mut u64,
        vals: *mut u32,
        n: u32,
        stream: *mut std::ffi::c_void,
    );
}

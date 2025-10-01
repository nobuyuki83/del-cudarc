use crate::cu;

pub fn set_consecutive_sequence(
    stream: cu::CUstream,
    d_in: &mut cu::CUdeviceptr,
    num_d_in: usize
) {
    let (func, _mdl) = crate::load_function_in_module(
        del_cudarc_kernel::UTIL,
        "gpu_set_consecutive_sequence");
    assert_eq!(size_of::<usize>(), 8);
    let mut builder = crate::Builder::new(stream);
    builder.arg_dptr(d_in);
    builder.arg_i32(num_d_in as i32);
    builder.launch_kernel(func, num_d_in as u32);
}


#[test]
fn test_set_consecutive_sequence(){
    crate::init_cuda_and_make_context(0);

    let n = 1025usize;
    let mut dptr = crate::malloc_device::<u32>(n);
    let stream = crate::create_stream_in_current_context();
    set_consecutive_sequence(stream, &mut dptr, n);
    {
        let h_v0 = crate::dtoh_vec::<u32>(dptr, n);
        let h_v1 = (0..n).map(|v| v as u32).collect::<Vec<u32>>();
        assert_eq!(h_v0, h_v1);
    }
    crate::cuda_check!( cu::cuMemFree_v2(dptr) );
    crate::cuda_check!( cu::cuStreamDestroy_v2(stream) );
}

use crate::{CuVec, LaunchConfig, cu, cuda_check};

pub fn set_consecutive_sequence(stream: cu::CUstream, din: &CuVec<u32>) {
    /*
    let (func, _mdl) =
        crate::load_function_in_module(del_cudarc_kernel::ARRAY1D, "set_consecutive_sequence")
            .unwrap();
     */
    let func = crate::load_get_function("array1d", "set_consecutive_sequence").unwrap();
    let mut builder = crate::Builder::new(stream);
    builder.arg_u32(din.n as u32);
    builder.arg_dptr(din.dptr);
    builder
        .launch_kernel(func, LaunchConfig::for_num_elems(din.n as u32))
        .unwrap();
}

#[test]
fn test_set_consecutive_sequence() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let din: CuVec<u32> = CuVec::with_capacity(1025).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    set_consecutive_sequence(stream, &din);
    {
        let h_v0 = din.copy_to_host().unwrap();
        let h_v1 = (0..1025).map(|v| v as u32).collect::<Vec<u32>>();
        assert_eq!(h_v0, h_v1);
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    drop(din);
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

pub fn shift_array_right(stream: cu::CUstream, din: &CuVec<u32>) -> CuVec<u32> {
    /*
    let (func, _mdl) =
        crate::load_function_in_module(del_cudarc_kernel::ARRAY1D, "shift_array_right").unwrap();
    */
    let func = crate::load_get_function("array1d", "shift_array_right").unwrap();
    let dout: CuVec<u32> = CuVec::with_capacity(din.n).unwrap();
    {
        let mut builder = crate::Builder::new(stream);
        builder.arg_u32(din.n as u32);
        builder.arg_dptr(din.dptr);
        builder.arg_dptr(dout.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(din.n as u32))
            .unwrap();
    }
    cuda_check!(cu::cuMemsetD32_v2(dout.dptr, 0, 1)).unwrap();
    dout
}

#[test]
fn test_shift_right() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let n = 257;
    let vin = (1..n).map(|v| v as u32).collect::<Vec<u32>>();
    let din: CuVec<u32> = CuVec::from_slice(&vin).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    set_consecutive_sequence(stream, &din);
    {
        let h_v0 = din.copy_to_host().unwrap();
        let h_v1 = (0..n - 1).map(|v| v as u32).collect::<Vec<u32>>();
        assert_eq!(h_v0, h_v1);
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    drop(din); // drop before destroy context
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

pub fn permute(
    stream: cu::CUstream,
    new2data: &CuVec<u32>,
    new2old: &CuVec<u32>,
    old2data: &CuVec<u32>,
) {
    let num_new = new2data.n;
    assert!(new2old.n >= num_new); // inequality in case for new2old is an array of offset
    let func = crate::load_get_function("array1d", "permute").unwrap();
    let mut builder = crate::Builder::new(stream);
    builder.arg_u32(num_new as u32);
    builder.arg_dptr(new2data.dptr);
    builder.arg_dptr(new2old.dptr);
    builder.arg_dptr(old2data.dptr);
    builder
        .launch_kernel(func, LaunchConfig::for_num_elems(num_new as u32))
        .unwrap();
}

#[test]
fn test_permute() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    {
        let old2data = CuVec::from_slice(&[15u32, 13, 11, 14, 12]).unwrap();
        let new2old = CuVec::from_slice(&[4u32, 2, 0, 1, 3]).unwrap();
        let new2data = CuVec::<u32>::with_capacity(old2data.n).unwrap();
        permute(stream, &new2data, &new2old, &old2data);
        dbg!(new2data.copy_to_host().unwrap());
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

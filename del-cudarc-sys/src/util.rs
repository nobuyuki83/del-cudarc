use crate::{CuVec, LaunchConfig, cu, cuda_check};

pub fn set_consecutive_sequence(stream: cu::CUstream, din: &CuVec<u32>) {
    let (func, _mdl) =
        crate::load_function_in_module(del_cudarc_kernel::UTIL, "set_consecutive_sequence");
    let mut builder = crate::Builder::new(stream);
    builder.arg_i32(din.n as i32);
    builder.arg_dptr(din.dptr);
    builder.launch_kernel(func, LaunchConfig::for_num_elems(din.n as u32));
}

#[test]
fn test_set_consecutive_sequence() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0);
    let din: CuVec<u32> = CuVec::with_capacity(1025);
    let stream = crate::create_stream_in_current_context();
    set_consecutive_sequence(stream, &din);
    {
        let h_v0 = din.copy_to_host();
        let h_v1 = (0..1025).map(|v| v as u32).collect::<Vec<u32>>();
        assert_eq!(h_v0, h_v1);
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream));
    drop(din);
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev));
}

pub fn shift_array_right(stream: cu::CUstream, din: &CuVec<u32>) -> CuVec<u32> {
    let (func, _mdl) = crate::load_function_in_module(del_cudarc_kernel::UTIL, "shift_array_right");
    let dout: CuVec<u32> = CuVec::with_capacity(din.n);
    {
        let mut builder = crate::Builder::new(stream);
        builder.arg_i32(din.n as i32);
        builder.arg_dptr(din.dptr);
        builder.arg_dptr(dout.dptr);
        builder.launch_kernel(func, LaunchConfig::for_num_elems(din.n as u32));
    }
    cuda_check!(cu::cuMemsetD32_v2(dout.dptr, 0, 1));
    dout
}

#[test]
fn test_shift_right() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0);
    let n = 257;
    let vin = (1..n).map(|v| v as u32).collect::<Vec<u32>>();
    let din: CuVec<u32> = CuVec::from_slice(&vin);
    let stream = crate::create_stream_in_current_context();
    set_consecutive_sequence(stream, &din);
    {
        let h_v0 = din.copy_to_host();
        let h_v1 = (0..n - 1).map(|v| v as u32).collect::<Vec<u32>>();
        assert_eq!(h_v0, h_v1);
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream));
    drop(din); // drop before destroy context
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev));
}

pub fn sort_indexed_array(stream: cu::CUstream, p2idx: &CuVec<u32>, idx2q: &CuVec<u32>) {
    let np = p2idx.n - 1;
    let (func, _mdl) =
        crate::load_function_in_module(del_cudarc_kernel::UTIL, "sort_indexed_array");
    let mut builder = crate::Builder::new(stream);
    builder.arg_i32(np as i32);
    builder.arg_dptr(p2idx.dptr);
    builder.arg_dptr(idx2q.dptr);
    builder.launch_kernel(func, LaunchConfig::for_num_elems(np as u32));
}

#[test]
fn test_sort_indexed_array() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0);
    let stream = crate::create_stream_in_current_context();
    let n = 3;
    let m = 10;
    let idx2q_in = (0..n * m).map(|v| v).rev().collect::<Vec<u32>>();
    let p2idx = (0..n + 1).map(|v| v * m).collect::<Vec<u32>>();
    let idx2q_trg = (0..n)
        .rev()
        .flat_map(|v| (v * m..v * m + m))
        .collect::<Vec<u32>>();
    let p2idx = CuVec::from_slice(&p2idx);
    let idx2q = CuVec::from_slice(&idx2q_in);
    sort_indexed_array(stream, &p2idx, &idx2q);
    let idx2q_out = idx2q.copy_to_host();
    assert_eq!(idx2q_out, idx2q_trg);
    cuda_check!(cu::cuStreamDestroy_v2(stream));
    drop(p2idx); // drop before destroy context
    drop(idx2q); // drop before destroy context
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev));
}

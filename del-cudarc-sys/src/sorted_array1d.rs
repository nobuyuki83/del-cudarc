use crate::{CuVec, LaunchConfig, cu};

#[cfg(test)]
use crate::cuda_check;

pub fn has_duplicates(stream: cu::CUstream, vals: &CuVec<u32>) -> bool {
    /*
    let (func, _mdl) =
        crate::load_function_in_module(del_cudarc_kernel::SORTED_ARRAY1D, "has_duplicates")
            .unwrap();
     */
    let func = crate::load_get_function("sorted_array1d", "has_duplicates").unwrap();
    let is_duplicate = CuVec::<u32>::with_capacity(1).unwrap();
    is_duplicate.set_zeros(stream).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (4, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = crate::Builder::new(stream);
    builder.arg_dptr(vals.dptr);
    builder.arg_u32(vals.n as u32);
    builder.arg_dptr(is_duplicate.dptr);
    builder.launch_kernel(func, cfg).unwrap();
    let flg = is_duplicate.copy_to_host().unwrap();
    flg[0] == 1
}

#[test]
fn test_has_duplicate() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    {
        let vals = CuVec::from_slice(&[1u32, 2, 3, 4, 5, 6]).unwrap();
        let is_duplicate = has_duplicates(stream, &vals);
        assert!(!is_duplicate);
    }
    {
        let vals = CuVec::from_slice(&[1u32, 2, 3, 4, 4, 6]).unwrap();
        let is_duplicate = has_duplicates(stream, &vals);
        assert!(is_duplicate);
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

pub fn unique(stream: cu::CUstream, idx2val: &CuVec<u32>, idx2jdx: &CuVec<u32>) {
    let num_idx = idx2val.n;
    assert_eq!(idx2jdx.n, num_idx);
    let idx2isdiff = CuVec::<u32>::with_capacity(num_idx).unwrap();
    {
        /*
        let (func, _mdl) =
            crate::load_function_in_module(del_cudarc_kernel::SORTED_ARRAY1D, "idx2isdiff")
                .unwrap();
         */
        let func = crate::load_get_function("sorted_array1d", "idx2isdiff").unwrap();
        let mut builder = crate::Builder::new(stream);
        builder.arg_u32(num_idx as u32);
        builder.arg_dptr(idx2val.dptr);
        builder.arg_dptr(idx2isdiff.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(num_idx as u32))
            .unwrap();
    }
    crate::cumsum::exclusive_scan(stream, &idx2isdiff, idx2jdx);
}

#[test]
fn test_unique() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    {
        let idx2val = CuVec::from_slice(&[1u32, 2, 3, 5, 5, 6]).unwrap();
        let idx2jdx = CuVec::<u32>::with_capacity(idx2val.n).unwrap();
        crate::sorted_array1d::unique(stream, &idx2val, &idx2jdx);
        assert_eq!(idx2jdx.copy_to_host().unwrap(), vec!(0, 1, 2, 3, 3, 4));
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

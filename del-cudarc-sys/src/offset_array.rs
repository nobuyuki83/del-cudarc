use crate::{CuVec, LaunchConfig, cu};

#[cfg(test)]
use crate::cuda_check;

pub fn sort(stream: cu::CUstream, p2idx: &CuVec<u32>, idx2q: &CuVec<u32>) {
    let np = p2idx.n - 1;
    /*
    let (func, _mdl) =
        crate::load_function_in_module(del_cudarc_kernel::OFFSET_ARRAY, "sort").unwrap();
     */
    let func = crate::load_get_function("offset_array", "sort").unwrap();
    let mut builder = crate::Builder::new(stream);
    builder.arg_u32(np as u32);
    builder.arg_dptr(p2idx.dptr);
    builder.arg_dptr(idx2q.dptr);
    builder
        .launch_kernel(func, LaunchConfig::for_num_elems(np as u32))
        .unwrap();
}

#[test]
fn test_sort_indexed_array() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    let n = 3;
    let m = 10;
    let idx2q_in = (0..n * m).rev().collect::<Vec<u32>>();
    let p2idx = (0..n + 1).map(|v| v * m).collect::<Vec<u32>>();
    let idx2q_trg = (0..n)
        .rev()
        .flat_map(|v| (v * m..v * m + m))
        .collect::<Vec<u32>>();
    let p2idx = CuVec::from_slice(&p2idx).unwrap();
    let idx2q = CuVec::from_slice(&idx2q_in).unwrap();
    sort(stream, &p2idx, &idx2q);
    let idx2q_out = idx2q.copy_to_host().unwrap();
    assert_eq!(idx2q_out, idx2q_trg);
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    drop(p2idx); // drop before destroy context
    drop(idx2q); // drop before destroy context
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

/// "idx2jdx" should be sorted
/// "jdx2idx_offset" is array of offset
/// output "jdx2idx_offset" is also sorted
pub fn inverse_map(stream: cu::CUstream, idx2jdx: &CuVec<u32>, jdx2idx_offset: &CuVec<u32>) {
    let num_idx = idx2jdx.n;
    let num_jdx = jdx2idx_offset.n - 1;
    {
        /*
        let (func, _mdl) =
            crate::load_function_in_module(del_cudarc_kernel::OFFSET_ARRAY, "inverse_map").unwrap();
         */
        let func = crate::load_get_function("offset_array", "inverse_map").unwrap();
        let mut builder = crate::Builder::new(stream);
        builder.arg_u32(num_jdx as u32);
        builder.arg_dptr(jdx2idx_offset.dptr);
        builder.arg_u32(num_idx as u32);
        builder.arg_dptr(idx2jdx.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(num_jdx as u32 + 1))
            .unwrap();
    }
}

#[test]
pub fn test_inverse_map() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    //
    {
        let idx2jdx = CuVec::<u32>::from_slice(&[0, 1, 2, 3, 3, 4]).unwrap();
        let num_jdx = idx2jdx.last().unwrap() + 1;
        assert_eq!(num_jdx, 5);
        let jdx2idx = CuVec::<u32>::with_capacity(num_jdx as usize + 1).unwrap();
        inverse_map(stream, &idx2jdx, &jdx2idx);
        assert_eq!(jdx2idx.copy_to_host().unwrap(), vec!(0, 1, 2, 3, 5, 6));
    }
    //
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    for _i in 0..10 {
        let num_idx = 1_000_000;
        let mut idx2jdx: Vec<u32> = (0..num_idx).map(|_| rng.random_range(0..100)).collect();
        idx2jdx.sort();
        let d_idx2jdx = CuVec::<u32>::from_slice(&idx2jdx).unwrap();
        let num_jdx = idx2jdx.last().unwrap() + 1;
        let d_jdx2idx = CuVec::<u32>::with_capacity(num_jdx as usize + 1).unwrap();
        inverse_map(stream, &d_idx2jdx, &d_jdx2idx);
        let jdx2idx = d_jdx2idx.copy_to_host().unwrap();
        assert_eq!(jdx2idx[0], 0);
        assert_eq!(*jdx2idx.last().unwrap(), num_idx as u32);
        for jdx in 0..num_jdx as usize {
            let idx0 = jdx2idx[jdx] as usize;
            let idx1 = jdx2idx[jdx + 1] as usize;
            assert!(idx0 <= idx1);
            idx2jdx[idx0..idx1]
                .iter()
                .for_each(|&idx| assert_eq!(idx as usize, jdx));
        }
    }
    cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

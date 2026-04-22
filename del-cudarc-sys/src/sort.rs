use crate::{CuVec, cu};

unsafe extern "C" {
    pub fn thrust_sort_u64_inplace(d_data: *mut u64, n: u32, stream: *mut std::ffi::c_void);
}

pub fn sort_u64_inplace(stream: cu::CUstream, vals: &CuVec<u64>) {
    unsafe {
        thrust_sort_u64_inplace(
            vals.dptr as usize as *mut u64,
            vals.n as u32,
            stream as *mut std::ffi::c_void,
        );
    }
}

#[test]
fn test_sort_u64_inplace() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    let ns = [
        13usize,
        1023,
        1024,
        1024 * 1024 - 1,
        1024 * 1024,
        1024 * 1024 + 1,
    ];
    for n in ns {
        let mut rng = rand_chacha::ChaChaRng::from_seed([0; 32]);
        let vals: Vec<u64> = (0..n).map(|_| rng.random()).collect();
        let mut vals_cpu = vals.clone();
        vals_cpu.sort();
        let stream = crate::create_stream_in_current_context().unwrap();
        let vals_dev = CuVec::<u64>::from_slice(&vals).unwrap();
        sort_u64_inplace(stream, &vals_dev);
        let vals_out = vals_dev.copy_to_host().unwrap();
        vals_out
            .iter()
            .zip(vals_cpu.iter())
            .for_each(|(a, b)| assert_eq!(a, b));
        crate::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    }
    crate::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

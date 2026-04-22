use crate::{CuVec, cu};

unsafe extern "C" {
    pub fn thrust_sort_by_key_u64_u32(
        keys: *mut u64,
        vals: *mut u32,
        n: u32,
        stream: *mut std::ffi::c_void,
    );
}

pub fn sort_u64_by_u32_key(
    stream: cu::CUstream,
    keys: &CuVec<u64>,
    vals: &CuVec<u32>,
) -> Result<(), String> {
    unsafe {
        thrust_sort_by_key_u64_u32(
            keys.dptr as usize as *mut u64,
            vals.dptr as usize as *mut u32,
            keys.n as u32,
            stream as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

pub fn sort_u32_by_u32_key(
    stream: cu::CUstream,
    keys: &CuVec<u32>,
    vals: &CuVec<u32>,
) -> Result<(), String> {
    crate::sort_by_key_u32::radix_sort_by_key_u32(stream, keys, vals).map_err(|e| format!("{e:?}"))
}

#[test]
fn test_sort_u64_by_u32_key() -> Result<(), String> {
    crate::cache_func::clear();
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
        let keys: Vec<u64> = (0..n).map(|_| rng.random()).collect();
        let vals: Vec<u32> = (0u32..n as u32).collect();
        let keys_out_cpu = {
            let mut v = keys.clone();
            v.sort();
            v
        };
        let stream = crate::create_stream_in_current_context().unwrap();
        let keys_dev = CuVec::<u64>::from_slice(&keys).unwrap();
        let vals_dev = CuVec::<u32>::from_slice(&vals).unwrap();
        unsafe { sort_u64_by_u32_key(stream, &keys_dev, &vals_dev) }.unwrap();
        let keys_out = keys_dev.copy_to_host().unwrap();
        let vals_out = vals_dev.copy_to_host().unwrap();
        keys_out
            .iter()
            .zip(keys_out_cpu.iter())
            .for_each(|(a, b)| assert_eq!(a, b));
        for jdx in 1..vals_out.len() {
            assert!(keys[vals_out[jdx - 1] as usize] <= keys[vals_out[jdx] as usize]);
        }
        crate::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    }
    crate::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
    Ok(())
}

#[test]
fn test_sort_u32_by_u32_key() -> Result<(), String> {
    crate::cache_func::clear();
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
        let keys: Vec<u32> = (0..n).map(|_| rng.random()).collect();
        let vals: Vec<u32> = (0u32..n as u32).collect();
        let keys_out_cpu = {
            let mut v = keys.clone();
            v.sort();
            v
        };
        let stream = crate::create_stream_in_current_context().unwrap();
        let keys_dev = CuVec::<u32>::from_slice(&keys).unwrap();
        let vals_dev = CuVec::<u32>::from_slice(&vals).unwrap();
        unsafe { sort_u32_by_u32_key(stream, &keys_dev, &vals_dev) }.unwrap();
        let keys_out = keys_dev.copy_to_host().unwrap();
        let vals_out = vals_dev.copy_to_host().unwrap();
        keys_out
            .iter()
            .zip(keys_out_cpu.iter())
            .for_each(|(a, b)| assert_eq!(a, b));
        for jdx in 1..vals_out.len() {
            assert!(keys[vals_out[jdx - 1] as usize] <= keys[vals_out[jdx] as usize]);
        }
        crate::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    }
    crate::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
    Ok(())
}

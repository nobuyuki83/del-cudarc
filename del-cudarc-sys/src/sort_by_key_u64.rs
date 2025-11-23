use crate::{CuVec, cu};

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
pub fn radix_sort_by_key_u64(
    stream: cu::CUstream,
    d_in: &CuVec<u64>,
    d_idx_in: &CuVec<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let d_in_len = d_in.n as u32;
    const MAX_BLOCK_SZ: u32 = 128;
    let block_sz: u32 = MAX_BLOCK_SZ;
    let max_elems_per_block = block_sz;
    let grid_sz: u32 = {
        let mut grid_sz = d_in_len / max_elems_per_block;
        // Take advantage of the fact that integer division drops the decimals
        if d_in_len % max_elems_per_block != 0 {
            grid_sz += 1;
        }
        grid_sz
    };
    let d_out = CuVec::<u64>::alloc_zeros(d_in.n, stream).unwrap();
    let d_prefix_sums = CuVec::<u32>::alloc_zeros(d_in.n, stream).unwrap();
    let d_idx_out = CuVec::<u32>::alloc_zeros(d_in.n, stream).unwrap();
    //
    let d_block_sums_len = 4 * grid_sz; // 4-way split
    let d_block_sums = CuVec::<u32>::alloc_zeros(d_block_sums_len as usize, stream).unwrap();
    let d_scan_block_sums = CuVec::<u32>::with_capacity(d_block_sums.n).unwrap();

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    let s_data_len = max_elems_per_block;
    let s_mask_out_len = max_elems_per_block + 1;
    let s_merged_scan_mask_out_len = max_elems_per_block;
    let s_mask_out_sums_len = 4; // 4-way split
    let s_scan_mask_out_sums_len = 4;
    let shmem_sz = s_data_len * 2
        + s_mask_out_len
        + s_merged_scan_mask_out_len
        + s_mask_out_sums_len
        + s_scan_mask_out_sums_len;

    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for shift_width in (0..=62).step_by(2) {
        {
            let cfg = crate::LaunchConfig {
                grid_dim: (grid_sz, 1, 1),
                block_dim: (block_sz, 1, 1),
                shared_mem_bytes: shmem_sz * (u32::BITS / 8),
            };
            let d_in_len = d_in.n as u32;
            //
            let func = crate::cache_func::get_function_cached(
                "del_cudarc::sort_by_key_u64",
                del_cudarc_kernels::get("sort_by_key_u64").unwrap(),
                "gpu_radix_sort_local",
            )
            .unwrap();
            {
                let mut builder = crate::Builder::new(stream);
                builder.arg_dptr(d_out.dptr);
                builder.arg_dptr(d_prefix_sums.dptr);
                builder.arg_dptr(d_block_sums.dptr);
                builder.arg_u32(shift_width);
                builder.arg_dptr(d_in.dptr);
                builder.arg_u32(d_in_len);
                builder.arg_u32(max_elems_per_block);
                builder.arg_dptr(d_idx_in.dptr);
                builder.arg_dptr(d_idx_out.dptr);
                builder.launch_kernel(func, cfg).unwrap();
            }
        }

        // scan global block sum array
        crate::cumsum::exclusive_scan(stream, &d_block_sums, &d_scan_block_sums);

        {
            let d_in_len = d_in.n as u32;
            let cfg = crate::LaunchConfig {
                grid_dim: (grid_sz, 1, 1),
                block_dim: (block_sz, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = crate::cache_func::get_function_cached(
                "del_cudarc::sort_by_key_u64",
                del_cudarc_kernels::get("sort_by_key_u64").unwrap(),
                "gpu_glbl_shuffle",
            )
            .unwrap();
            let mut builder = crate::Builder::new(stream);
            builder.arg_dptr(d_in.dptr);
            builder.arg_dptr(d_out.dptr);
            builder.arg_dptr(d_scan_block_sums.dptr);
            builder.arg_dptr(d_prefix_sums.dptr);
            builder.arg_u32(shift_width);
            builder.arg_u32(d_in_len);
            builder.arg_u32(max_elems_per_block);
            builder.arg_dptr(d_idx_out.dptr);
            builder.arg_dptr(d_idx_in.dptr);
            builder.launch_kernel(func, cfg).unwrap();
        }
    }
    Ok(())
}

/// # Safety
pub unsafe fn radix_sort_by_key_u64_u32(
    stream: cu::CUstream,
    keys: &CuVec<u64>,
    vals: &CuVec<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let n = keys.n;
    assert_eq!(vals.n, n);
    let stream_ptr: *mut std::ffi::c_void = stream as *mut std::ffi::c_void;
    let keys_ptr: *mut u64 = keys.dptr as usize as *mut u64;
    let vals_ptr: *mut u32 = vals.dptr as usize as *mut u32;
    unsafe {
        del_cudarc_thrust::thrust_sort_by_key_u64_u32(keys_ptr, vals_ptr, n as u32, stream_ptr)
    };
    let res = unsafe { cu::cuStreamSynchronize(stream) };
    unsafe { crate::check_cu_error(res, "thrust_sort_by_key_u64_u32 + cuCtxSynchronize") };
    Ok(())
}

#[test]
fn test_u64() -> Result<(), cudarc::driver::DriverError> {
    crate::cache_func::clear();
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let ns = [
        13usize,
        1023,
        1024,
        1024 * 1024 - 1,
        1024 * 1024,
        1024 * 1024 + 1,
    ];
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    for n in ns {
        let mut rng = rand_chacha::ChaChaRng::from_seed([0; 32]);
        let keys = {
            let mut vin: Vec<u64> = vec![];
            (0..n).for_each(|_| vin.push(rng.random()));
            vin
        };
        let vals: Vec<u32> = (0u32..n as u32).collect::<Vec<_>>();
        let keys_out_cpu = {
            // naive cpu computation
            let mut vout0 = keys.clone();
            vout0.sort();
            vout0
        };
        let vals_out_cpu = {
            let mut vals_out0 = vals.clone();
            vals_out0.sort_by(|&a, &b| keys[a as usize].cmp(&keys[b as usize]));
            vals_out0
        };
        {
            let stream = crate::create_stream_in_current_context().unwrap();
            let keys_dev = CuVec::<u64>::from_slice(&keys).unwrap();
            let vals_dev = CuVec::<u32>::from_slice(&vals).unwrap();
            unsafe { radix_sort_by_key_u64_u32(stream, &keys_dev, &vals_dev) }.unwrap();
            let keys_out = keys_dev.copy_to_host().unwrap();
            let vals_out = vals_dev.copy_to_host().unwrap();
            keys_out.iter().zip(keys_out_cpu.iter()).for_each(|(a, b)| {
                assert_eq!(a, b, "{} {}", a, b);
            });
            for jdx in 1..vals_out.len() {
                assert!(keys[vals_out[jdx - 1] as usize] <= keys[vals_out[jdx] as usize]);
            }
            vals_out
                .iter()
                .zip(vals_out_cpu.iter())
                .for_each(|(&a, &b)| {
                    assert_eq!(a, b);
                });
            crate::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
        }
    }
    crate::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
    Ok(())
}

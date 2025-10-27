use crate::{CuVec, cu};

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
pub fn radix_sort_by_key_u32(
    stream: cu::CUstream,
    d_in: &CuVec<u32>,
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
    let d_out = CuVec::<u32>::alloc_zeros(d_in.n, stream).unwrap();
    let d_prefix_sums = CuVec::<u32>::alloc_zeros(d_in.n, stream).unwrap();
    let d_idx_out = CuVec::<u32>::alloc_zeros(d_in.n, stream).unwrap();
    //
    let d_block_sums_len = 4 * grid_sz; // 4-way split
    let d_block_sums = CuVec::<u32>::alloc_zeros(d_block_sums_len as usize, stream).unwrap();

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    let s_data_len = max_elems_per_block;
    let s_mask_out_len = max_elems_per_block + 1;
    let s_merged_scan_mask_out_len = max_elems_per_block;
    let s_mask_out_sums_len = 4; // 4-way split
    let s_scan_mask_out_sums_len = 4;
    let shmem_sz = s_data_len
        + s_mask_out_len
        + s_merged_scan_mask_out_len
        + s_mask_out_sums_len
        + s_scan_mask_out_sums_len;

    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for shift_width in (0..=30).step_by(2) {
        {
            let cfg = crate::LaunchConfig {
                grid_dim: (grid_sz, 1, 1),
                block_dim: (block_sz, 1, 1),
                shared_mem_bytes: shmem_sz * (u32::BITS / 8),
            };
            let d_in_len = d_in.n as u32;
            let (func, _mdl) = crate::load_function_in_module(
                del_cudarc_kernel::SORT_BY_KEY_U32,
                "gpu_radix_sort_local",
            ).unwrap();
            {
                let mut builder = crate::Builder::new(stream);
                builder.arg_dptr(d_out.dptr);
                builder.arg_dptr(d_prefix_sums.dptr);
                builder.arg_dptr(d_block_sums.dptr);
                builder.arg_i32(shift_width);
                builder.arg_dptr(d_in.dptr);
                builder.arg_i32(d_in_len as i32);
                builder.arg_i32(max_elems_per_block as i32);
                builder.arg_dptr(d_idx_in.dptr);
                builder.arg_dptr(d_idx_out.dptr);
                builder.launch_kernel(func, cfg).unwrap();
            }
        }

        // scan global block sum array
        let d_scan_block_sums = CuVec::<u32>::with_capacity(d_block_sums.n).unwrap();
        crate::cumsum::exclusive_scan(stream, &d_block_sums, &d_scan_block_sums);

        {
            let d_in_len = d_in.n as u32;
            let cfg = crate::LaunchConfig {
                grid_dim: (grid_sz, 1, 1),
                block_dim: (block_sz, 1, 1),
                shared_mem_bytes: 0,
            };
            let (func, _mdl) = crate::load_function_in_module(
                del_cudarc_kernel::SORT_BY_KEY_U32,
                "gpu_glbl_shuffle",
            ).unwrap();
            let mut builder = crate::Builder::new(stream);
            builder.arg_dptr(d_in.dptr);
            builder.arg_dptr(d_out.dptr);
            builder.arg_dptr(d_scan_block_sums.dptr);
            builder.arg_dptr(d_prefix_sums.dptr);
            builder.arg_i32(shift_width);
            builder.arg_i32(d_in_len as i32);
            builder.arg_i32(max_elems_per_block as i32);
            builder.arg_dptr(d_idx_out.dptr);
            builder.arg_dptr(d_idx_in.dptr);
            builder.launch_kernel(func, cfg).unwrap();
        }
    }
    Ok(())
}

#[test]
fn test_u32() -> Result<(), cudarc::driver::DriverError> {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let ns = [
        13usize,
        1023,
        1024,
        1025,
        1024 * 1024 - 1,
        1024 * 1024,
        1024 * 1024 + 1,
    ];
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    for n in ns {
        let stream = crate::create_stream_in_current_context().unwrap();
        let mut rng = rand_chacha::ChaChaRng::from_seed([0; 32]);
        let vin = {
            let mut vin: Vec<u32> = vec![];
            (0..n).for_each(|_| vin.push(rng.random()));
            vin
        };
        let idxin: Vec<u32> = (0u32..n as u32).collect::<Vec<_>>();
        let idxin_dev = crate::CuVec::<u32>::from_slice(&idxin).unwrap();
        // dbg!(dev.dtoh_sync_copy(&idxin_dev));
        // let mut idxout_dev = dev.alloc_zeros(idxin_dev.len())?;
        let vio_dev = crate::CuVec::<u32>::from_slice(&vin).unwrap();
        radix_sort_by_key_u32(stream, &vio_dev, &idxin_dev)?;
        let vout0 = {
            // naive cpu computation
            let mut vout0 = vin.clone();
            vout0.sort();
            vout0
        };
        let vout = vio_dev.copy_to_host().unwrap();
        vout.iter().zip(vout0.iter()).for_each(|(a, b)| {
            // println!("{} {}",a,b);
            assert_eq!(a, b, "{} {}", a, b);
        });
        let idxout = idxin_dev.copy_to_host().unwrap();
        for jdx in 1..idxout.len() {
            assert!(
                vin[idxout[jdx - 1] as usize] <= vin[idxout[jdx] as usize],
                "{} {} {}",
                n,
                1024 * 1024 - 1,
                jdx
            );
        }
        let mut idxout0 = idxin.clone();
        idxout0.sort_by(|&a, &b| vin[a as usize].cmp(&vin[b as usize]));
        idxout.iter().zip(idxout0.iter()).for_each(|(&a, &b)| {
            // println!("{} {}",a,b);
            assert_eq!(a, b);
        });
        crate::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    }
    crate::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
    Ok(())
}

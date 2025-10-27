use crate::{CuVec, cu};

/// Add each block's total sum to its scan output
/// in order to get the final, global scanned array
pub fn add_block_sums_u32(
    stream: cu::CUstream,
    vout_dev: &CuVec<u32>,
    d_block_sums: &CuVec<u32>,
    num_elem: u32,
    grid_sz: u32,
    block_sz: u32,
) {
    let cfg = crate::LaunchConfig {
        grid_dim: (grid_sz, 1, 1),
        block_dim: (block_sz, 1, 1),
        shared_mem_bytes: 0,
    };
    let (func, _mdl) =
        crate::load_function_in_module(del_cudarc_kernel::CUMSUM, "gpu_add_block_sums").unwrap();
    let mut builder = crate::Builder::new(stream);
    builder.arg_dptr(vout_dev.dptr);
    builder.arg_dptr(d_block_sums.dptr);
    builder.arg_i32(num_elem as i32);
    builder.launch_kernel(func, cfg).unwrap();
}

/// # Returns
/// Ok(grid_size, block_size, d_block_sums)
/// d_block_sums is size of "grid_size"
pub fn block_sums(
    stream: cu::CUstream,
    d_out: &CuVec<u32>,
    d_in: &CuVec<u32>,
) -> (u32, u32, CuVec<u32>) {
    let num_elem = d_out.n as u32;
    assert_eq!(d_in.n as u32, num_elem);

    const MAX_BLOCK_SZ: u32 = 1024;
    // const NUM_BANKS: u32 = 32;
    const LOG_NUM_BANKS: u32 = 5;

    // Set up number of threads and blocks
    let block_size = MAX_BLOCK_SZ / 2;
    let max_elems_per_block = 2 * block_size; // due to binary tree nature of algorithm

    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
    //unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
    // UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically
    //  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
    let mut grid_size = num_elem / max_elems_per_block;

    // Take advantage of the fact that integer division drops the decimals
    if num_elem % max_elems_per_block != 0 {
        grid_size += 1;
    }

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    let shmem_size = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);
    //let shmem_size = max_elems_per_block + ((max_elems_per_block) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    let d_block_sums = CuVec::<u32>::with_capacity(grid_size as usize).unwrap();
    d_block_sums.set_zeros(stream).unwrap();

    d_out.set_zeros(stream).unwrap();

    // Sum scan data allocated to each block
    {
        let stream2 = stream; //stream.fork()?;
        let cfg = crate::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (u32::BITS / 8u32) * shmem_size,
        };
        let (fnc, _mdl) = crate::load_function_in_module(del_cudarc_kernel::CUMSUM, "gpu_prescan").unwrap();
        let mut builder = crate::Builder::new(stream2);
        builder.arg_dptr(d_out.dptr);
        builder.arg_dptr(d_in.dptr);
        builder.arg_dptr(d_block_sums.dptr);
        builder.arg_i32(num_elem as i32);
        builder.arg_i32(shmem_size as i32);
        builder.arg_i32(max_elems_per_block as i32);
        builder.launch_kernel(fnc, cfg).unwrap();
    }
    // dbg!(d_out.copy_to_host().unwrap());

    // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    //  (This requires only one block to do the scan)
    if grid_size <= max_elems_per_block {
        let stream3 = stream; // stream.fork()?;
        let d_dummy_blocks_sums: CuVec<u32> = CuVec::with_capacity(1).unwrap();
        d_dummy_blocks_sums.set_zeros(stream3).unwrap();
        let cfg = crate::LaunchConfig {
            grid_dim: (1u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: u32::BITS / 8u32 * shmem_size,
        };
        let gpu_prescan = crate::load_function_in_module(del_cudarc_kernel::CUMSUM, "gpu_prescan").unwrap();
        let mut builder = crate::Builder::new(stream3);
        builder.arg_dptr(d_block_sums.dptr);
        builder.arg_dptr(d_block_sums.dptr);
        builder.arg_dptr(d_dummy_blocks_sums.dptr);
        builder.arg_i32(grid_size as i32);
        builder.arg_i32(shmem_size as i32);
        builder.arg_i32(max_elems_per_block as i32);
        builder.launch_kernel(gpu_prescan.0, cfg).unwrap();

    } else {
        let stream3 = stream; // stream.fork()?;
        // println!("prefix sum of blocks using recursive");
        // Else, recurse on this same function as you'll need the full-blown scan
        //  for the block sums
        let d_in_block_sums: CuVec<u32> = CuVec::with_capacity(grid_size as usize).unwrap();
        crate::memcpy_d2d_32(
            d_in_block_sums.dptr,
            d_block_sums.dptr,
            grid_size as usize,
            stream3,
        ).unwrap();
        let (grid_sz1, block_sz1, d_block_sums1) =
            block_sums(stream, &d_block_sums, &d_in_block_sums);
        add_block_sums_u32(
            stream,
            &d_block_sums,
            &d_block_sums1,
            num_elem,
            grid_sz1,
            block_sz1,
        );
    }
    (grid_size, block_size, d_block_sums)
}

/// exclusive scan
/// `vout_dev` does not need to be zeros
pub fn exclusive_scan(stream: cu::CUstream, vin: &CuVec<u32>, vout: &CuVec<u32>) {
    let n = vin.n;
    assert_eq!(vout.n, n);
    let (grid_sz, block_sz, d_block_sums) = block_sums(stream, vout, vin);
    add_block_sums_u32(stream, vout, &d_block_sums, n as u32, grid_sz, block_sz);
}

#[test]
fn test_hoge() {
    let (dev, _ctx) = crate::init_cuda_and_make_context(0).unwrap();
    let stream = crate::create_stream_in_current_context().unwrap();
    let nvs = [
        (1024, 1024),
        (1023, 1),
        (1024, 1),
        (1025, 1),
        (1024 * 1024 - 1, 1),
        (1024 * 1024, 1),
        (1024 * 1024 + 1, 1),
        (1024 * 1024 + 1, 2),
    ];
    for (n, v) in nvs {
        let h_vin: Vec<u32> = {
            let mut vin = vec![v; n];
            vin.push(0u32); // push 0 at the end
            vin
        };
        let d_vin = CuVec::from_slice(&h_vin).unwrap();
        let d_vout = CuVec::with_capacity(d_vin.n).unwrap();
        let d_buff = CuVec::<u32>::with_capacity(1000).unwrap();
        d_buff.set_zeros(stream).unwrap();
        exclusive_scan(stream, &d_vin, &d_vout);
        let h_vout = d_vout.copy_to_host().unwrap();
        assert_eq!(h_vout[0], 0);
        assert_eq!(h_vout.len(), n + 1);
        for i in 0..n {
            assert_eq!(
                h_vout[i + 1] as i64 - h_vout[i] as i64,
                h_vin[i] as i64,
                "{}-->{} {}",
                i,
                h_vout[i],
                h_vout[i + 1]
            );
        }
        assert_eq!(h_vout[n], (n as u32) * v);
        assert_eq!(d_buff.copy_to_host().unwrap(), vec!(0;1000), "illegal memory access");
    }
    crate::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    crate::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
}

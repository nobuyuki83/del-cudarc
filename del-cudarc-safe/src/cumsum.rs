use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

fn block_sums(
    stream: &std::sync::Arc<CudaStream>,
    d_out: &mut CudaSlice<u32>,
    d_in: &CudaSlice<u32>,
    num_elem: u32,
) -> Result<(u32, u32, CudaSlice<u32>), cudarc::driver::DriverError> {
    const MAX_BLOCK_SZ: u32 = 1024;
    // const NUM_BANKS: u32 = 32;
    const LOG_NUM_BANKS: u32 = 5;

    // Zero out d_out
    stream.memset_zeros(d_out)?;

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

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    let mut d_block_sums = stream.alloc_zeros::<u32>(grid_size as usize)?;

    // Sum scan data allocated to each block
    //gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
    {
        let stream2 = stream.fork()?;
        let cfg = {
            cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: (u32::BITS / 8u32) * shmem_size,
            }
        };
        let gpu_prescan =
            crate::get_or_load_func(stream.context(), "gpu_prescan", del_cudarc_kernel::CUMSUM)?;
        let mut builder = stream2.launch_builder(&gpu_prescan);
        builder.arg(d_out);
        builder.arg(d_in);
        builder.arg(&mut d_block_sums);
        builder.arg(&num_elem);
        builder.arg(&shmem_size);
        builder.arg(&max_elems_per_block);
        unsafe { builder.launch(cfg) }?;
    }

    // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    //  (This requires only one block to do the scan)
    if grid_size <= max_elems_per_block {
        let stream3 = stream.fork()?;
        let d_dummy_blocks_sums = stream3.alloc_zeros::<u32>(1)?;
        //gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
        let cfg = {
            cudarc::driver::LaunchConfig {
                grid_dim: (1u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: u32::BITS / 8u32 * shmem_size,
            }
        };
        let gpu_prescan =
            crate::get_or_load_func(stream.context(), "gpu_prescan", del_cudarc_kernel::CUMSUM)?;
        let mut builder = stream3.launch_builder(&gpu_prescan);
        builder.arg(&d_block_sums);
        builder.arg(&d_block_sums);
        builder.arg(&d_dummy_blocks_sums);
        builder.arg(&grid_size);
        builder.arg(&shmem_size);
        builder.arg(&max_elems_per_block);
        unsafe { builder.launch(cfg) }?;
    } else {
        let stream3 = stream.fork()?;
        // println!("prefix sum of blocks using recursive");
        // Else, recurse on this same function as you'll need the full-blown scan
        //  for the block sums
        let mut d_in_block_sums = unsafe { stream3.alloc::<u32>(grid_size as usize)? };
        stream3.memcpy_dtod(&d_block_sums, &mut d_in_block_sums)?;
        assert_eq!(d_block_sums.len(), d_in_block_sums.len());
        let (grid_sz1, block_sz1, d_block_sums1) =
            block_sums(stream, &mut d_block_sums, &d_in_block_sums, grid_size)?;
        add_block_sums(
            stream,
            &mut d_block_sums,
            &d_block_sums1,
            num_elem,
            grid_sz1,
            block_sz1,
        )?;
    }
    Ok((grid_size, block_size, d_block_sums))
}

/// Add each block's total sum to its scan output
/// in order to get the final, global scanned array
fn add_block_sums(
    stream: &std::sync::Arc<CudaStream>,
    vout_dev: &mut CudaSlice<u32>,
    d_block_sums: &CudaSlice<u32>,
    num_elem: u32,
    grid_sz: u32,
    block_sz: u32,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    let cfg = {
        cudarc::driver::LaunchConfig {
            grid_dim: (grid_sz, 1, 1),
            block_dim: (block_sz, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    // let param = (vout_dev, d_block_sums, num_elem);
    let gpu_add_block_sums = crate::get_or_load_func(
        stream.context(),
        "gpu_add_block_sums",
        del_cudarc_kernel::CUMSUM,
    )?;
    let mut builder = stream.launch_builder(&gpu_add_block_sums);
    builder.arg(vout_dev);
    builder.arg(d_block_sums);
    builder.arg(&num_elem);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// `vout_dev` does not need to be zeros
pub fn sum_scan_blelloch(
    stream: &std::sync::Arc<CudaStream>,
    vout_dev: &mut CudaSlice<u32>,
    vin_dev: &CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let n = vin_dev.len();
    assert_eq!(vout_dev.len(), n);
    let (grid_sz, block_sz, d_block_sums) = block_sums(stream, vout_dev, vin_dev, n as u32)?;
    add_block_sums(stream, vout_dev, &d_block_sums, n as u32, grid_sz, block_sz)?;
    Ok(())
}

#[test]
fn test() -> Result<(), cudarc::driver::DriverError> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
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
        let vin: Vec<u32> = {
            let mut vin = vec![v; n];
            vin.push(0u32);
            vin
        };
        let stream = ctx.default_stream();
        let vin_dev = stream.memcpy_stod(&vin)?;
        let mut vout_dev = stream.alloc_zeros::<u32>(vin_dev.len())?;
        sum_scan_blelloch(&stream, &mut vout_dev, &vin_dev)?;
        let vout = stream.memcpy_dtov(&vout_dev)?;
        assert_eq!(vout[0], 0);
        assert_eq!(vout.len(), n + 1);
        for i in 0..n {
            assert_eq!(
                vout[i + 1] as i64 - vout[i] as i64,
                vin[i] as i64,
                "{}-->{} {}",
                i,
                vout[i],
                vout[i + 1]
            );
        }
        assert_eq!(vout[n], (n as u32) * v);
    }
    Ok(())
}

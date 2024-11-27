use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

pub fn set_consecutive_sequence(
    dev: &std::sync::Arc<CudaDevice>,
    d_in: &mut CudaSlice<u32>,
) -> anyhow::Result<()> {
    let num_d_in = d_in.len() as u32;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_d_in);
    let param = (d_in, num_d_in);
    use cudarc::driver::LaunchAsync;
    let func = crate::get_or_load_func(dev, "gpu_set_consecutive_sequence", kernels::UTIL)?;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

pub fn permute(
    dev: &std::sync::Arc<CudaDevice>,
    new2data: &mut CudaSlice<u32>,
    new2old: &CudaSlice<u32>,
    old2data: &CudaSlice<u32>,
) -> anyhow::Result<()> {
    let n = new2data.len();
    assert_eq!(new2old.len(), n);
    assert_eq!(old2data.len(), n);
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    let param = (n, new2data, new2old, old2data);
    use cudarc::driver::LaunchAsync;
    let func = crate::get_or_load_func(dev, "permute", kernels::UTIL)?;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

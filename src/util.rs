#[allow(unused_imports)]
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DeviceSlice};

pub fn set_consecutive_sequence(
    dev: &std::sync::Arc<CudaDevice>,
    d_in: &mut cudarc::driver::CudaViewMut<u32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    let num_d_in = d_in.len() as u32;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_d_in);
    let param = (d_in, num_d_in);
    use cudarc::driver::LaunchAsync;
    let func =
        crate::get_or_load_func(dev, "gpu_set_consecutive_sequence", del_cudarc_kernel::UTIL)?;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

pub fn permute(
    dev: &std::sync::Arc<CudaDevice>,
    new2data: &mut CudaSlice<u32>,
    new2old: &CudaSlice<u32>,
    old2data: &CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let n = new2data.len();
    assert_eq!(new2old.len(), n);
    assert_eq!(old2data.len(), n);
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    let param = (n, new2data, new2old, old2data);
    use cudarc::driver::LaunchAsync;
    let func = crate::get_or_load_func(dev, "permute", del_cudarc_kernel::UTIL)?;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

pub fn set_value_at_mask(
    dev: &std::sync::Arc<CudaDevice>,
    elem2value: &mut CudaSlice<f32>,
    set_value: f32,
    elem2mask: &CudaSlice<u32>,
    mask: u32,
    is_set_value_where_mask_value_equal: bool,
) -> Result<(), cudarc::driver::DriverError> {
    let n = elem2value.len();
    assert_eq!(elem2mask.len(), n);
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    let param = (
        n,
        elem2value,
        set_value,
        elem2mask,
        mask,
        is_set_value_where_mask_value_equal,
    );
    use cudarc::driver::LaunchAsync;
    let func = crate::get_or_load_func(dev, "set_value_at_mask", del_cudarc_kernel::UTIL)?;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

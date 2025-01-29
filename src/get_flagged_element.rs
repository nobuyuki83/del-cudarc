use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

/// the size of the element is `ielem2flg.len()-1`
/// the last element of `ielem2flg` is ignored.
pub fn get_flagged_element(
    device: &std::sync::Arc<CudaDevice>,
    ielem2val: &CudaSlice<u32>,
    ielem2flg: &CudaSlice<u32>,
) -> std::result::Result<CudaSlice<u32>, cudarc::driver::DriverError> {
    let num_ielem = ielem2flg.len() - 1;
    let num_dim = ielem2val.len() / num_ielem;
    assert_eq!(ielem2val.len(), num_ielem * num_dim);
    let mut cumsum_ielem2flg = unsafe { device.alloc::<u32>(num_ielem + 1) }?;
    crate::cumsum::sum_scan_blelloch(device, &mut cumsum_ielem2flg, ielem2flg)?;
    // dbg!(device.dtoh_sync_copy(&cumsum_ielem2flg)?);
    let num_oelem = cumsum_ielem2flg.slice(num_ielem..num_ielem + 1);
    let num_oelem = device.dtoh_sync_copy(&num_oelem)?[0] as usize;
    let mut oelem2val = device.alloc_zeros::<u32>(num_oelem * num_dim)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_oelem as u32);
    let param = (
        num_oelem,
        &mut oelem2val,
        num_dim,
        num_ielem,
        &cumsum_ielem2flg,
        ielem2val,
    );
    use cudarc::driver::LaunchAsync;
    let func = crate::get_or_load_func(
        device,
        "get_element_from_cumsum_flag",
        del_cudarc_kernel::GET_FLAGGED_ELEMENT,
    )?;
    unsafe { func.launch(cfg, param) }?;
    Ok(oelem2val)
}

#[test]
fn test_get_flagged_element() -> Result<(), cudarc::driver::result::DriverError> {
    let device = cudarc::driver::CudaDevice::new(0)?;
    let ielem2val = device.htod_copy(vec![7u32, 3, 1, 5])?;
    let ielem2flag = device.htod_copy(vec![1u32, 1, 1, 1, 0])?;
    let oelem2val = get_flagged_element(&device, &ielem2val, &ielem2flag)?;
    let oelem2val_cpu = device.dtoh_sync_copy(&oelem2val)?;
    dbg!(oelem2val_cpu);
    Ok(())
}

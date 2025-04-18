use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

/// the size of the element is `ielem2flg.len()-1`
/// the last element of `ielem2flg` is ignored.
pub fn get_flagged_element(
    stream: &std::sync::Arc<CudaStream>,
    ielem2val: &CudaSlice<u32>,
    ielem2flg: &CudaSlice<u32>,
) -> std::result::Result<CudaSlice<u32>, cudarc::driver::DriverError> {
    let num_ielem = ielem2flg.len() - 1;
    let num_dim = ielem2val.len() / num_ielem;
    assert_eq!(ielem2val.len(), num_ielem * num_dim);
    let mut cumsum_ielem2flg = unsafe { stream.alloc::<u32>(num_ielem + 1) }?;
    crate::cumsum::sum_scan_blelloch(stream, &mut cumsum_ielem2flg, ielem2flg)?;
    // dbg!(device.dtoh_sync_copy(&cumsum_ielem2flg)?);
    let num_oelem = cumsum_ielem2flg.slice(num_ielem..num_ielem + 1);
    let num_oelem_hst = stream.memcpy_dtov(&num_oelem)?[0] as usize;
    let mut oelem2val = stream.alloc_zeros::<u32>(num_oelem_hst * num_dim)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_oelem_hst as u32);
    let func = crate::get_or_load_func(
        stream.context(),
        "get_element_from_cumsum_flag",
        del_cudarc_kernel::GET_FLAGGED_ELEMENT,
    )?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&num_oelem_hst);
    builder.arg(&mut oelem2val);
    builder.arg(&num_dim);
    builder.arg(&num_ielem);
    builder.arg(&cumsum_ielem2flg);
    builder.arg(ielem2val);
    unsafe { builder.launch(cfg) }?;
    Ok(oelem2val)
}

#[test]
fn test_get_flagged_element() -> Result<(), cudarc::driver::result::DriverError> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ielem2val = stream.memcpy_stod(&vec![7u32, 3, 1, 5])?;
    let ielem2flag = stream.memcpy_stod(&vec![1u32, 0, 1, 1, 0])?;
    let oelem2val = get_flagged_element(&stream, &ielem2val, &ielem2flag)?;
    let oelem2val_cpu = stream.memcpy_dtov(&oelem2val)?;
    assert_eq!(oelem2val_cpu, [7, 1, 5]);
    Ok(())
}

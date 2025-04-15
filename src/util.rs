use cudarc::driver::PushKernelArg;
#[allow(unused_imports)]
use cudarc::driver::{CudaContext, CudaSlice, CudaView, CudaViewMut, DeviceSlice};

pub fn set_consecutive_sequence(
    ctx: &std::sync::Arc<CudaContext>,
    d_in: &mut CudaViewMut<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let stream = ctx.default_stream();
    let num_d_in = d_in.len() as u32;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_d_in);
    let func =
        crate::get_or_load_func(ctx, "gpu_set_consecutive_sequence", del_cudarc_kernel::UTIL)?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(d_in);
    builder.arg(&num_d_in);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

pub fn permute(
    ctx: &std::sync::Arc<CudaContext>,
    new2data: &mut CudaSlice<u32>,
    new2old: &CudaSlice<u32>,
    old2data: &CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let n = new2data.len();
    assert_eq!(new2old.len(), n);
    assert_eq!(old2data.len(), n);
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    let func = crate::get_or_load_func(ctx, "permute", del_cudarc_kernel::UTIL)?;
    let stream = ctx.default_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&n);
    builder.arg(new2data);
    builder.arg(new2old);
    builder.arg(old2data);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

pub fn set_value_at_mask(
    ctx: &std::sync::Arc<CudaContext>,
    elem2value: &mut CudaSlice<f32>,
    set_value: f32,
    elem2mask: &CudaSlice<u32>,
    mask: u32,
    is_set_value_where_mask_value_equal: bool,
) -> Result<(), cudarc::driver::DriverError> {
    let n = elem2value.len();
    assert_eq!(elem2mask.len(), n);
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    let stream = ctx.default_stream();
    let func = crate::get_or_load_func(ctx, "set_value_at_mask", del_cudarc_kernel::UTIL)?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&n);
    builder.arg(elem2value);
    builder.arg(&set_value);
    builder.arg(elem2mask);
    builder.arg(&mask);
    builder.arg(&is_set_value_where_mask_value_equal);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// trick from https://github.com/coreylowman/cudarc/issues/295
/// # Safety
/// This is unsafe because it uses the `transmute`
pub unsafe fn from_raw_parts<'a, T>(
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ptr: cudarc::driver::sys::CUdeviceptr,
    len: usize,
) -> CudaView<'a, T> {
    let slice = unsafe { stream.upgrade_device_ptr(ptr, len) };
    let temp_view = slice.slice(..);
    // extend lifetime
    let view = unsafe { std::mem::transmute::<CudaView<'_, T>, CudaView<'a, T>>(temp_view) };
    // don't free the fake slice
    slice.leak();
    view
}

/// # Safety
pub unsafe fn from_raw_parts_mut<'a, T>(
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ptr: cudarc::driver::sys::CUdeviceptr,
    len: usize,
) -> CudaViewMut<'a, T> {
    let slice = unsafe { stream.upgrade_device_ptr(ptr, len) };
    let temp_view = slice.slice(..);
    // extend lifetime
    let view = unsafe { std::mem::transmute::<CudaView<'_, T>, CudaViewMut<'a, T>>(temp_view) };
    // don't free the fake slice
    slice.leak();
    view
}

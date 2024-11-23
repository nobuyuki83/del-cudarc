use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

pub fn from_trimesh_with_bvhnodes(
    dev: &std::sync::Arc<CudaDevice>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    bvhnodes: &CudaSlice<u32>,
    bvhnode2aabb: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    let num_tri = tri2vtx.len() / 3;
    let num_bvhnode = bvhnodes.len() / 3;
    assert_eq!(num_bvhnode, 2 * num_tri - 1);
    let num_branch = num_tri - 1;
    let mut bvhbranch2flag = dev.alloc_zeros::<u32>(num_branch)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_tri as u32);
    let param = (
        bvhnode2aabb,
        &mut bvhbranch2flag,
        num_bvhnode,
        bvhnodes,
        num_tri as u32,
        tri2vtx,
        vtx2xyz,
        0.,
    );
    let from_trimsh = del_cudarc_util::get_or_load_func(dev, "from_trimesh3", kernel_bvh::BVHNODE2AABB)?;
    use cudarc::driver::LaunchAsync;
    unsafe { from_trimsh.launch(cfg, param) }?;
    Ok(())
}

pub fn tri2cntr_from_trimesh3(
    dev: &std::sync::Arc<CudaDevice>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    tri2cntr: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    let num_tri = tri2vtx.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_tri as u32);
    let param = (tri2cntr, num_tri as u32, tri2vtx, vtx2xyz);
    let from_trimsh = del_cudarc_util::get_or_load_func(dev, "tri2cntr", kernel_bvh::BVHNODES_MORTON)?;
    use cudarc::driver::LaunchAsync;
    unsafe { from_trimsh.launch(cfg, param) }?;
    Ok(())
}

pub fn vtx2morton(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2xyz: &CudaSlice<f32>,
    transform_cntr2uni: &CudaSlice<f32>,
    vtx2morton: &mut CudaSlice<u32>,
) -> anyhow::Result<()> {
    let num_vtx = vtx2xyz.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_vtx as u32);
    let param = (num_vtx, vtx2xyz, transform_cntr2uni, vtx2morton);
    let func = del_cudarc_util::get_or_load_func(dev, "vtx2morton", kernel_bvh::BVHNODES_MORTON)?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

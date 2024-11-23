use cudarc::driver::DeviceSlice;
use cudarc::driver::LaunchAsync;

fn main() -> anyhow::Result<()> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let (tri2vtx, vtx2xyz, _vtx2uv) = {
        let mut obj = del_msh_core::io_obj::WavefrontObj::<u32, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        obj.unified_xyz_uv_as_trimesh()
    };
    let tri2vtx_dev = dev.htod_copy(tri2vtx.clone())?;
    let vtx2xyz_dev = dev.htod_copy(vtx2xyz.clone())?;
    // -----------
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnodes_dev = dev.htod_copy(bvhnodes.clone())?;
    {
        let num_tri = tri2vtx.len() / 3;
        let mut tri2cntr_dev = dev.alloc_zeros::<f32>(num_tri * 3)?;
        del_canvas_cuda::bvh::tri2cntr_from_trimesh3(
            &dev,
            &tri2vtx_dev,
            &vtx2xyz_dev,
            &mut tri2cntr_dev,
        )?;
        let tri2cntr =
            del_msh_core::elem2center::from_uniform_mesh_as_points(&tri2vtx, 3, &vtx2xyz, 3);
        {
            // check tri2cntr
            let tri2cntr_hst = dev.dtoh_sync_copy(&tri2cntr_dev)?;
            for i in 0..tri2cntr.len() {
                let diff = (tri2cntr[i] - tri2cntr_hst[i]).abs();
                // assert!(diff<=f32::EPSILON, "{} {}", diff, f32::EPSILON);
                assert_eq!(tri2cntr[i], tri2cntr_hst[i], "{}", diff);
            }
        }
        // get aabb
        let aabb = del_msh_core::vtx2xyz::aabb3(&tri2cntr, 0f32);
        let transform_cntr2uni =
            del_geo_core::mat4_col_major::from_aabb3_fit_into_unit_preserve_asp(&aabb);
        // del_geo_core::mat4_col_major::from_aabb3_fit_into_unit(&aabb);
        let mut idx2tri = vec![0usize; num_tri];
        let mut idx2morton = vec![0u32; num_tri];
        let mut tri2morton = vec![0u32; num_tri];
        del_msh_core::bvhnodes_morton::sorted_morten_code3(
            &mut idx2tri,
            &mut idx2morton,
            &mut tri2morton,
            &tri2cntr,
            &transform_cntr2uni,
        );
        let mut tri2morton_dev = dev.alloc_zeros(num_tri)?;
        let transform_cntr2uni_dev = dev.htod_copy(transform_cntr2uni.to_vec())?;
        del_canvas_cuda::bvh::vtx2morton(
            &dev,
            &tri2cntr_dev,
            &transform_cntr2uni_dev,
            &mut tri2morton_dev,
        )?;
        let tri2morton_hst = dev.dtoh_sync_copy(&tri2morton_dev)?;
        for i in 0..tri2morton.len() {
            assert_eq!(
                tri2morton_hst[i], tri2morton[i],
                "{} {}",
                tri2morton_hst[i], tri2morton[i]
            );
        }
        let mut idx2tri_dev = dev.alloc_zeros(num_tri)?;
        del_cudarc_util::sort_by_key_u64::set_consecutive_sequence(&dev, &mut idx2tri_dev)?;
        // del_canvas_cuda::sort_by_key_u64::radix_sort_by_key_u64(&dev, &mut tri2morton_dev, &mut idx2tri_dev);
    }
    // ------------
    let bvhnode2aabb = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    let bvhnode2aabb_dev = {
        let num_tri = tri2vtx.len() / 3;
        let mut bvhnode2aabb_dev = dev.alloc_zeros::<f32>(bvhnodes_dev.len() / 3 * 6)?;
        del_canvas_cuda::bvh::from_trimesh_with_bvhnodes(
            &dev,
            &tri2vtx_dev,
            &vtx2xyz_dev,
            &bvhnodes_dev,
            &mut bvhnode2aabb_dev,
        )?;
        let bvhnode2aabb_from_gpu = dev.dtoh_sync_copy(&bvhnode2aabb_dev)?;
        for i in 0..(num_tri * 2 - 1) * 6 {
            assert_eq!(bvhnode2aabb_from_gpu[i], bvhnode2aabb[i]);
        }
        bvhnode2aabb_dev
    };
    let img_size = {
        const TILE_SIZE: usize = 16;
        (TILE_SIZE * 28, TILE_SIZE * 28)
    };
    let cam_projection = del_geo_core::mat4_col_major::camera_perspective_blender(
        img_size.0 as f32 / img_size.1 as f32,
        24f32,
        0.5,
        3.0,
        true,
    );
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.], 0., 0., 0.);

    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
    //
    dev.load_ptx(kernel_bvh::PIX2TRI.into(), "my_module", &["pix_to_tri"])?;
    let pix_to_tri = dev.get_func("my_module", "pix_to_tri").unwrap();
    //
    let mut pix2tri_dev = dev.alloc_zeros::<u32>(img_size.1 * img_size.0)?;
    let transform_ndc2world_dev = dev.htod_copy(transform_ndc2world.to_vec())?;
    let now = std::time::Instant::now();
    let cfg = {
        let num_threads = 256;
        let num_blocks = (img_size.0 * img_size.1) / num_threads + 1;
        cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    //for_num_elems((img_size.0 * img_size.1).try_into()?);
    let param = (
        &mut pix2tri_dev,
        tri2vtx.len() / 3,
        &tri2vtx_dev,
        &vtx2xyz_dev,
        img_size.0,
        img_size.1,
        &transform_ndc2world_dev,
        &bvhnodes_dev,
        &bvhnode2aabb_dev,
    );
    unsafe { pix_to_tri.launch(cfg, param) }?;
    let pix2tri = dev.dtoh_sync_copy(&pix2tri_dev)?;
    println!("   Elapsed pix2tri: {:.2?}", now.elapsed());
    let pix2flag: Vec<f32> = pix2tri
        .iter()
        .map(|v| if *v == u32::MAX { 0f32 } else { 1f32 })
        .collect();
    del_canvas_image::write_png_from_float_image_grayscale(
        "target/raycast_trimesh3_silhouette.png",
        &img_size,
        &pix2flag,
    )?;
    dbg!(tri2vtx.len());
    Ok(())
}

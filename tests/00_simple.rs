use cudarc::driver::PushKernelArg;

#[test]
fn test_if_gpu_working() -> Result<(), cudarc::driver::DriverError> {
    let ptx = cudarc::nvrtc::compile_ptx(
        "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}",
    )
    .unwrap();

    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    //
    let inp0 = stream.memcpy_stod(&vec![1.0f32; 100])?;
    let inp1 = stream.memcpy_stod(&vec![2.0f32; 100])?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(100);
    {
        // test sine kernel
        let module = ctx.load_module(ptx)?;
        let sin_kernel = module.load_function("sin_kernel")?;
        let n = 100usize;
        let mut out = stream.alloc_zeros::<f32>(n)?;
        let mut builder = stream.launch_builder(&sin_kernel);
        builder.arg(&mut out);
        builder.arg(&inp0);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }?;
        // unsafe { sin_kernel.launch(cfg, (&mut out, &inp0, 100usize)) }?;
        let out_host: Vec<f32> = stream.memcpy_dtov(&out)?;
        out_host
            .iter()
            .for_each(|&v| assert!((v - 1.0f32.sin()) < 1.0e-5));
    }
    {
        let module = ctx.load_module(del_cudarc_kernel::SIMPLE.into())?;
        let vector_add = module.load_function("vector_add")?;
        let n = 100usize;
        let mut out = stream.alloc_zeros::<f32>(n)?;
        let mut builder = stream.launch_builder(&vector_add);
        builder.arg(&mut out);
        builder.arg(&inp0);
        builder.arg(&inp1);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }?;
        // unsafe { vector_add.launch(cfg, (&mut out, &inp0, &inp1, 100usize)) }?;
        let out_host: Vec<f32> = stream.memcpy_dtov(&out)?;
        out_host.iter().for_each(|&v| assert_eq!(v, 3.0f32));
    }
    Ok(())
}

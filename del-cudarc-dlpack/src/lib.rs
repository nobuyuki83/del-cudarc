use dlpack::ManagedTensor;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyCapsule};

/// Pythonから渡された PyCapsule を Rust 側で読み取る
#[pyfunction]
fn set_consecutive_sequence(_py: Python, obj: &pyo3::Bound<'_, PyAny>) -> PyResult<()> {
    let capsule = obj.downcast::<PyCapsule>()?;

    println!("Capsule name: {}", capsule.name()?.unwrap().to_str()?);

    // DLPack を unsafe にアンラップ
    unsafe {
        let ptr = capsule.pointer() as *mut ManagedTensor;
        if ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Null ManagedTensor",
            ));
        }
        let tensor = &(*ptr).dl_tensor;
        let ndim = tensor.ndim as usize;
        let shape = std::slice::from_raw_parts(tensor.shape, ndim);
        let total_elements = shape.iter().product::<i64>() as usize;
        println!(
            "DLPack tensor shape: {:?}, {:?}, {:?}",
            ndim, shape, total_elements
        );
        match tensor.ctx.device_type {
            dlpack::device_type_codes::CPU => {
                let data_ptr = tensor.data as *mut u32;
                let data = std::slice::from_raw_parts_mut(data_ptr, total_elements);
                data.iter_mut().enumerate().for_each(|(i, v)| *v = i as u32);
            }
            dlpack::device_type_codes::GPU => {
                use cudarc::driver::sys::*;
                println!("GPU_{}", tensor.ctx.device_id);
                let (function, _module) = del_cudarc_sys::load_function_in_module(
                    del_cudarc_kernel::UTIL,
                    "gpu_set_consecutive_sequence",
                );
                let stream = del_cudarc_sys::create_stream_in_current_context();
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_data(&tensor.data);
                builder.arg_i32(total_elements as i32);
                builder.launch_kernel(function, total_elements as u32);
                cuStreamDestroy_v2(stream);
            }
            _ => println!("Unknown device type {}", tensor.ctx.device_type),
        }
    }
    Ok(())
}

/// Pythonモジュールに登録
#[pymodule]
#[pyo3(name = "del_cudarc_dlpack")]
fn del_fem_dlpack_(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_consecutive_sequence, m)?)?;
    Ok(())
}

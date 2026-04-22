#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" void thrust_sort_u64_inplace(
    uint64_t* d_data,
    uint32_t n,
    cudaStream_t stream)
{
    thrust::device_ptr<uint64_t> begin(d_data);
    thrust::sort(thrust::cuda::par.on(stream), begin, begin + n);
}

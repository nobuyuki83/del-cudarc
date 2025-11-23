// thrust_sort.cu
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

extern "C" void thrust_sort_by_key_u64_u32(
    std::uint64_t* keys,
    std::uint32_t* vals,
    std::uint32_t n,
    cudaStream_t stream)
{
    thrust::device_ptr<uint64_t> k(keys);
    thrust::device_ptr<uint32_t> v(vals);
    thrust::sort_by_key(thrust::cuda::par.on(stream), k, k + n, v);
}
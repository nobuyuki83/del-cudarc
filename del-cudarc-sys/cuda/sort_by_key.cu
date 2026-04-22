#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <stdint.h>

extern "C" void thrust_sort_by_key_u64_u32(
    uint64_t* keys,
    uint32_t* vals,
    uint32_t n,
    cudaStream_t stream)
{
    thrust::device_ptr<uint64_t> k(keys);
    thrust::device_ptr<uint32_t> v(vals);
    thrust::sort_by_key(thrust::cuda::par.on(stream), k, k + n, v);
}

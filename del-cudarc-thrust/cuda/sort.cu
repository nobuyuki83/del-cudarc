// cuda/thrust_sort.cu
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <stdint.h>

// stream を受け取って thrust::sort を呼ぶラッパ
extern "C"
void thrust_sort_u64_inplace(
    uint64_t* d_data,
    uint32_t  n,
    cudaStream_t stream
)
{
    using T = uint64_t;
    thrust::device_ptr<T> begin(d_data);
    thrust::device_ptr<T> end  = begin + n;
    thrust::sort(thrust::cuda::par.on(stream), begin, end);
}

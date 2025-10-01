#include <stdint.h>

extern "C" {

__global__
void gpu_set_consecutive_sequence(
    uint32_t* d_out,
    uint32_t num_d_out)
{
    int i_d_out = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_d_out >= num_d_out ){ return; }
    //
    d_out[i_d_out] = i_d_out;
}

__global__
void permute(
    unsigned int n,
    uint32_t* new2data,
    const uint32_t* new2old,
    const uint32_t* old2data)
{
    int i_new = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_new >= n ){ return; }
    //
    int i_old = new2old[i_new];
    assert(i_new < n);
    new2data[i_new] = old2data[i_old];
}

__global__
void set_value_at_mask(
    unsigned int n,
    float* elem2value,
    float set_value,
    const uint32_t* elem2mask,
    uint32_t mask,
    bool is_set_value_at_mask_value_equal)
{
    int i_elem = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_elem >= n ){ return; }
    //
    if( (elem2mask[i_elem] == mask) == is_set_value_at_mask_value_equal) {
        elem2value[i_elem] = set_value;
    }
}

}
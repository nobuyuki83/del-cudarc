#include <stdint.h>

extern "C" {

__global__
void set_consecutive_sequence(
    uint32_t num_d_out,
    uint32_t* d_out)
{
    int i_d_out = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_d_out >= num_d_out ){ return; }
    //
    d_out[i_d_out] = i_d_out;
}

__global__
void shift_array_right(
    uint32_t n,
    uint32_t* din,
    uint32_t* dout
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i >= n ){ return; }
    //
    if( i == 0 ){ return; }
    dout[i] = din[i-1];
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

__global__
void sort_indexed_array(
    unsigned int n,
    const uint32_t* p2idx,
    uint32_t* idx2q
)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if( p >= n ){ return; }
    //
    const uint32_t idx0 = p2idx[p];
    const uint32_t idx1 = p2idx[p+1];
    for(uint32_t idx=idx0;idx<idx1;++idx){
        uint32_t idx_min = idx;
        for(uint32_t jdx=idx+1;jdx<idx1;++jdx){
            if( idx2q[jdx] < idx2q[idx_min] ){
                idx_min = jdx;
            }
        }
        const uint32_t tmp = idx2q[idx];
        idx2q[idx] = idx2q[idx_min];
        idx2q[idx_min] = tmp;
    }

}


}
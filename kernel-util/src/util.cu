extern "C" {

__global__
void gpu_set_consecutive_sequence(
    uint32_t* d_out,
    uint32_t num_d_out)
{
    int i_d_out = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_d_out >= num_d_out ){ return; }
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

}
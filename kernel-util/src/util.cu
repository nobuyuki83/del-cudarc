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

}
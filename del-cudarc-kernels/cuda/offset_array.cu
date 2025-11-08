#include <stdint.h>

extern "C" {

__global__
void sort(
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

/// the number of threads should be "num_jdx + 1"
/// idx2jdx should be sorted (increasing order)
__global__
void inverse_map(
    uint32_t num_jdx,
    uint32_t* jdx2idx_offset,
    uint32_t num_idx,
    const uint32_t* idx2jdx)
{
    const uint32_t jdx = blockDim.x * blockIdx.x + threadIdx.x;
    if( jdx > num_jdx ){ return; }
    // -----
    if( num_idx  == 0 ){
      jdx2idx_offset[0] = 0;
      return;
    }
    uint32_t idx0 = 0;
    uint32_t idx2 = num_idx;
    while(idx0<idx2){
      const uint32_t idx1 = idx0 + ((idx2 - idx0) >> 1); // to avoid overflow
      // assert(idx1<num_idx);
      if( idx2jdx[idx1] < jdx ){
          idx0 = idx1 + 1;
      }
      else {
          idx2 = idx1;
      }
    }
    jdx2idx_offset[jdx] = idx0;
}

__global__
void aggregate(
    uint32_t num_idx,
    const uint32_t* idx2jdx_offset,
    const uint32_t* jdx2kdx,
    const uint32_t num_dim,
    const float* kdx2val,
    float* idx2aggval)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if( idx >= num_idx ){ return; }
    //
    for(int i_dim=0;i_dim<num_dim;++i_dim){
        idx2aggval[idx*num_dim+i_dim] = 0.0;
    }
    for(int jdx=idx2jdx_offset[idx];jdx<idx2jdx_offset[idx+1];++jdx){
        int kdx = jdx2kdx[jdx];
        for(int i_dim=0;i_dim<num_dim;++i_dim){
             idx2aggval[idx*num_dim+i_dim] += kdx2val[kdx*num_dim+i_dim];
        }
    }
}

}
#include <stdint.h>

extern "C" {

__global__ void get_element_from_cumsum_flag(
    uint32_t num_oelem,
    uint32_t* oelem2val,
    uint32_t num_dim,
    uint32_t num_ielem,
    const uint32_t* cumsum_ielem2flag,
    const uint32_t* ielem2val
)
{
    int i_oelem = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_oelem >= num_oelem) {
        return;
    }
    // ----------------------
    int i_ielem_l = 0;
    int i_ielem_u =  num_ielem;
    // find the index `i_ielem`
    // `cumsum_elem2flag[i_ielem] == i_oelem` and `cumsum_ielem2flag[i_ielem+1] == i_oelem+1`
    for(int iter=0;iter<32;++iter) {
        if( i_ielem_u - i_ielem_l == 1 ){ break; }
        int i_ielem_h = (i_ielem_l + i_ielem_u) / 2;
        if( cumsum_ielem2flag[i_ielem_h] > i_oelem ) {
            i_ielem_u = i_ielem_h;
        }
        else {
            i_ielem_l = i_ielem_h;
        }
    }
    int i_ielem = i_ielem_l;
    assert(cumsum_ielem2flag[i_ielem+1]==cumsum_ielem2flag[i_ielem]+1);
    for(int i_dim=0;i_dim<num_dim;++i_dim){
        oelem2val[i_oelem*num_dim+i_dim] = ielem2val[i_ielem*num_dim+i_dim];
    }
}

}
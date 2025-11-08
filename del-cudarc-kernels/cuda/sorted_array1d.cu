#include <stdint.h>

extern "C" {

/// check if there is a duplicated element
/// arr need to be sorted
__global__
void has_duplicates(const uint32_t* arr, int n, uint32_t* has_dup) {
  const int lane = threadIdx.x & 31;
  int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if( i0 >= n - 1 ){ return; }
  // ---------------
  // grid-stride loop
  for (int i = i0;
    i < n - 1;
    i += blockDim.x * gridDim.x)
  {
    if (atomicAdd(has_dup, 0)) return;
    int pred = (arr[i] == arr[i + 1]);
    // warp 内に1つでも pred があれば true
    unsigned mask = __activemask();
    if (__any_sync(mask, pred)) {
      if (lane == 0) atomicOr(has_dup, 1);
      return;
    }
  }
}

__global__
void idx2isdiff(uint32_t num_idx, const uint32_t* idx2val, uint32_t* idx2isdiff) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if( i >= num_idx-1 ){ return; }
  // ---------------
  idx2isdiff[i] = ( idx2val[i] == idx2val[i+1] ) ? 0 : 1;
}

}
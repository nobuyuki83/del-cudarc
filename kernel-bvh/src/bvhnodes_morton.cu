#include "mat4_col_major.h"

__device__
uint32_t device_ExpandBits(uint32_t v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__
unsigned int device_MortonCode(float x, float y, float z)
{
  auto ix = (uint32_t)fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
  auto iy = (uint32_t)fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
  auto iz = (uint32_t)fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
  //  std::cout << std::bitset<10>(ix) << " " << std::bitset<10>(iy) << " " << std::bitset<10>(iz) << std::endl;
  ix = device_ExpandBits(ix);
  iy = device_ExpandBits(iy);
  iz = device_ExpandBits(iz);
  //  std::cout << std::bitset<30>(ix) << " " << std::bitset<30>(iy) << " " << std::bitset<30>(iz) << std::endl;
  return ix * 4 + iy * 2 + iz;
}


extern "C" {

__global__
void tri2cntr(
  float* tri2cntr,
  uint32_t num_tri,
  const uint32_t* tri2vtx,
  const float* vtx2xyz)
{
  const unsigned int i_tri = blockDim.x * blockIdx.x + threadIdx.x;
  if (i_tri >= num_tri) return;
  //
  const uint32_t* node2vtx = tri2vtx + i_tri*3;
  const float ratio = 1.0/3.0;
  for(int i_dim=0;i_dim<3;++i_dim){
    float sum = 0.;
    for(uint32_t i_node=0;i_node<3;++i_node){
        sum += vtx2xyz[node2vtx[i_node]*3+i_dim];
    }
    tri2cntr[i_tri*3+i_dim] = sum * ratio;
  }
}

__global__
void vtx2morton(
    uint32_t num_vtx,
    const float* vtx2xyz,
    const float* transform_cntr2uni,
    uint32_t* vtx2morton)
{
  const unsigned int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
  if (i_vtx >= num_vtx) return;
  //
  const float* xyz0 = vtx2xyz + i_vtx * 3;
  auto xyz1 = mat4_col_major::transform_homogeneous(transform_cntr2uni, xyz0);
  // printf("%f %f %f\n", xyz1[0], xyz1[1], xyz1[2]);
  assert(xyz1[0]>-1.0e-7f && xyz1[0]<1.0f+1.e-7f);
  assert(xyz1[1]>-1.0e-7f && xyz1[1]<1.0f+1.e-7f);
  assert(xyz1[2]>-1.0e-7f && xyz1[2]<1.0f+1.e-7f);
  uint32_t mc = device_MortonCode(xyz1[0], xyz1[1], xyz1[2]);
  vtx2morton[i_vtx] = mc;
}


}



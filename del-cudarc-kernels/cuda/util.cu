
extern "C" __global__
void get_version(int* out) {
    out[0] = __CUDACC_VER_MAJOR__;
    out[1] = __CUDACC_VER_MINOR__;
    #ifdef __CUDACC_VER_BUILD__
        out[2] = __CUDACC_VER_BUILD__;
    #else
        out[2] = -1;
    #endif
}
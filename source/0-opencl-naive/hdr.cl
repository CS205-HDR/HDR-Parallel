__kernel void
hdr(__global __read_only float *gpu_0,
           __global __read_only float *gpu_1,
           __global __read_only float *gpu_2,
           __global __read_only float *gpu_3,
           __global __write_only float *gpu_out,
           int w, int h)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);


    if ((x < w) && (y < h)) {

        gpu_out[w*y+x] = (gpu_0[w*y+x]+gpu_1[w*y+x]+gpu_2[w*y+x]+gpu_3[w*y+x])/4;
    }

}

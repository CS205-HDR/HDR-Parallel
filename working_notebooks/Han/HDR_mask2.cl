
// This is the mask calculation version


float get_right_pix(__global __read_only float *labels,
           int w, int h,
           int x, int y);

// for given point in buffer, make sure it is within bounds of workgroup

//float get_right_pix(__global __read_only float *gpu_in,
//           int w, int h,
//           int curr_x, int curr_y) {
//    if (curr_x < 0) {
//        curr_x = 0;
//    }
//    if (curr_x > w-1) {
//        curr_x = w-1;
//    }
//    if (curr_y < 0) {
//        curr_y = 0;
//    }
//    if (curr_y > h-1) {
//        curr_y = h-1;
//    }
//    return gpu_in[curr_y * w + curr_x];
//}

float
get_right_pix(__global __read_only float *gpu_in,
                  int w, int h,
                  int x, int y)
{
    if ((x < 0) || (x >= w) || (y < 0) || (y >= h))
        return w * h;
    return gpu_in[y * w + x];
}





__kernel void
mask(__global __read_only float *gpu_in,
                 __global __read_only float *gpu_mask,
                 __global __write_only float *gpu_out,
                 __local float *buffer,
                 int w, int h,
                 int buf_w, int buf_h,
                 const int halo)
{
    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;


    //Create buffer
    //buffer[0,0] = gpu_in[]

    // Load the relevant labels to a local buffer with a halo
    if (idx_1D < buf_w) {
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = get_right_pix(gpu_in, w, h,
                                                         buf_corner_x + idx_1D,
                                                         buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);



    // write outputs: matrix convolution
    if ((y < h) && (x < w)) {
        //float out_val = buffer[(buf_y - 1)*buf_w + buf_x - 1] * gpu_mask[(buf_y - 1)*buf_w + buf_x - 1] +
        //          buffer[(buf_y - 1)*buf_w + buf_x] * gpu_mask[(buf_y - 1)*buf_w + buf_x] +
        //          buffer[(buf_y - 1)*buf_w + buf_x + 1] * gpu_mask[(buf_y - 1)*buf_w + buf_x + 1] +
        //          buffer[buf_y*buf_w + buf_x - 1] * gpu_mask[buf_y*buf_w + buf_x - 1] +
        //          buffer[buf_y*buf_w + buf_x] * gpu_mask[buf_y*buf_w + buf_x] +
        //          buffer[buf_y*buf_w + buf_x + 1] * gpu_mask[buf_y*buf_w + buf_x + 1] +
        //          buffer[(buf_y + 1)*buf_w + buf_x - 1] * gpu_mask[(buf_y + 1)*buf_w + buf_x - 1] +
        //          buffer[(buf_y + 1)*buf_w + buf_x] * gpu_mask[(buf_y + 1)*buf_w + buf_x] +
        //          buffer[(buf_y + 1)*buf_w + buf_x + 1] * gpu_mask[(buf_y + 1)*buf_w + buf_x + 1];

        float out_val = buffer[(buf_y - 1)*buf_w + buf_x - 1] * gpu_mask[0] +
                  buffer[(buf_y - 1)*buf_w + buf_x] * gpu_mask[1] +
                  buffer[(buf_y - 1)*buf_w + buf_x + 1] * gpu_mask[2] +
                  buffer[buf_y*buf_w + buf_x - 1] * gpu_mask[3] +
                  buffer[buf_y*buf_w + buf_x] * gpu_mask[4] +
                  buffer[buf_y*buf_w + buf_x + 1] * gpu_mask[5] +
                  buffer[(buf_y + 1)*buf_w + buf_x - 1] * gpu_mask[6] +
                  buffer[(buf_y + 1)*buf_w + buf_x] * gpu_mask[7] +
                  buffer[(buf_y + 1)*buf_w + buf_x + 1] * gpu_mask[8];





        if(y<200 && x<100){
            printf("%f", out_val);
        }
        //printf(buffer);
        //printf(gpu_mask);
        //printf(buf_x);
        //printf(buf_y);

        gpu_out[y*w+x] = out_val;

    }

}

__kernel void
mask_nobuffer(__global __read_only float *gpu_in,
                 __global __read_only float *gpu_mask,
                 __global __write_only float *gpu_out,
                 int w, int h,
                 int mask_w, int mask_h)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if((y < h)  && (x < w)){
        float out_val = 0;
        for(int mh=0;mh<mask_h;mh++)
            for(int mw=0;mw<mask_w;mw++)
                out_val += gpu_mask[mh*mask_w+mw]*gpu_in[(y-mask_h/2+mh)*w+x-mask_w/2+mw];

        gpu_out[x+w*y] = out_val;
    }
}


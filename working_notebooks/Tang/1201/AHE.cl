int findrank(float* a, float target){
  int rank=0;
  for(unsigned i=0;i<sizeof(a)/sizeof(a[0]);i++){
    if(a[i]<target){
      rank++;
    }
  }
  return rank;
}

__kernel void
AHE(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo,
           const int windowsize)
{
    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.


    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.


    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

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

    float current_pixel = in_values[x*w+y];


    int row;

    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {

            // Handle boundary case, use the closest pixel
            int this_x = buf_corner_x + idx_1D;
            int this_y = buf_corner_y + row;
            if(this_x>=w) this_x = 2*w-1-this_x;
            else if(this_x<0) this_x = -this_x;
            if(this_y>=h) this_y = 2*h-1-this_y;
            else if(this_y<0) this_y = -this_y;
            buffer[row * buf_w + idx_1D] = in_values[this_y * w + this_x];
        }

    float window [windowsize*windowsize];
    for(int i=0; i<windowsize;i++)
      for(int j=0;j<windowsize;j++){
        window[i*windowsize+j] = buffer[i*buf_w+j];
      }


    barrier(CLK_LOCAL_MEM_FENCE);
    int this_rank;
    // write output
    if((y<h) && (x<w)){  //stay in bound
      //Apply median filter within buffer
      this_rank = findrank(window,current_pixel);
      out_values[y*w+x] = (float)(this_rank)/(float)(windowsize*windowsize)*256;
    }


}


__kernel void
AHE_test(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo,
           const int windowsize){
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    out_values[y*w+x]=0;


}
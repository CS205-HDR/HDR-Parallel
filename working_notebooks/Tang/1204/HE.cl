__kernel void
AHE_buffer(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo,
           const int windowsize){
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

    const float current_pixel = in_values[y*w+x];

    int row;

    if (idx_1D < buf_w){
        for (row = 0; row < buf_h; row++) {

            int this_x = buf_corner_x + idx_1D;
            int this_y = buf_corner_y + row;
            if(this_x>=w) this_x = 2*w-1-this_x;
            else if(this_x<0) this_x = -this_x;
            if(this_y>=h) this_y = 2*h-1-this_y;
            else if(this_y<0) this_y = -this_y;
            buffer[row * buf_w + idx_1D] = in_values[this_y * w + this_x];

        }
      }
      
    barrier(CLK_LOCAL_MEM_FENCE);

    if((y<h) && (x<w)){
      int rank=0;
      for(int i=buf_x-windowsize/2; i<buf_x+windowsize/2+1;i++)
        for(int j=buf_y-windowsize/2; j<buf_y+windowsize/2+1;j++){
          if(buffer[j*buf_w+i]<current_pixel){
            rank++;
          }
        }
      out_values[y*w+x] = (float)(rank)/(float)(windowsize*windowsize)*256;
    }


}